import math
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


import torch.nn.functional as F
from .base import Estimator  # keep your existing Estimator base
from .lfbgs import batched_lbfgs
torch.set_float32_matmul_precision("high")
#torch.set_printoptions(profile="full")   # show full tensors


def _sigmoid_to_range(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    # Map R -> (lo,hi)
    return lo + (hi - lo) * torch.sigmoid(u)

def _range_to_sigmoid_x(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    # Inverse of _sigmoid_to_range (for seeding)
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)  # avoid 0/1
    return torch.log(z) - torch.log1p(-z)  # logit

def _tanh_to_range(u: torch.Tensor, max_abs: float) -> torch.Tensor:
    # Map R -> (-max_abs, max_abs)
    return max_abs * torch.tanh(u)

def _range_to_tanh_x(x: torch.Tensor, max_abs: float, eps: float = 1e-6) -> torch.Tensor:
    # Inverse of _tanh_to_range (for seeding)
    r = (x / max_abs).clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log1p(r) - 0.5 * torch.log1p(-r)  # atanh

class OptimizedMLE(Estimator):
    """
    Joint continuous MLE for theta = [L1, ReZF, ImZF, ReZL, ImZL].

    Parameterization:
      L1    in [0, L]           via sigmoid
      ReZF  in [1, 2000]        via sigmoid
      ImZF  in [-100, +100]     via tanh
      ReZL  in [1, 200]         via sigmoid
      ImZL  in [-100, +100]     via tanh

    Args
    ----
    fm: ForwardModel                (provides compute_H_complex(L1, ZF, ZL))
    likelihood:                     ComplexGaussianLik()
    L: float                        line length (upper bound for L1)
    device: torch.device
    n_starts: int                   number of random restarts per observation
    adam_steps: int
    adam_lr: float
    use_lbfgs: bool
    lbfgs_steps: int
    verbose: bool
    """
    def __init__(self,
                 fm,
                 likelihood,
                 L: float = 1000.0,
                 device="cuda",
                 adam_steps: int = 400,
                 adam_lr: float = 1e-3,
                 use_lbfgs: bool = True,
                 lbfgs_steps: int = 40,
                 verbose: bool = True,
                 # --- NEW: profiling/bracketing controls ---
                 profile_L1: bool = True,
                 L1_grid_points: int = 400,      # coarse scan resolution
                 inner_steps: int = 6,          # steps for Z reopt at each L1
                 inner_lr: float = 1e-2,         # LR for the short inner Z-optim
                 profile_topk: int = 3):
        self.fm = fm
        self.device = device
        self.lik = likelihood
        self.L = float(L)
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.use_lbfgs = bool(use_lbfgs)
        self.lbfgs_steps = int(lbfgs_steps)
        self.verbose = bool(verbose)

        # parameter range
        self.L1_lo, self.L1_hi = 1.0, self.L
        self.ReZF_lo, self.ReZF_hi = 1.0, 4000.0
        self.ImZF_max = 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

        # profiling config
        self.profile_L1 = bool(profile_L1)
        self.L1_grid_points = L1_grid_points
        self.inner_steps = int(inner_steps)
        self.inner_lr = float(inner_lr)
        self.topk = int(profile_topk)
    

    def _u_to_theta(self, u):
        """
        Only works for u of shape [..., 5].
        Map: u = [u0, u1, u2, u3, u4] from unconstained space -> theta = (L1, ZF, ZL) in constrained space
        """
        assert u.shape[-1] == 5, f"Expected last dim=5, got {u.shape}"
        u0, u1, u2, u3, u4 = u.unbind(dim=-1)  # each [... ]

        L1    = _sigmoid_to_range(u0, self.L1_lo, self.L1_hi)
        ReZF  = _sigmoid_to_range(u1, self.ReZF_lo, self.ReZF_hi)
        ImZF  = _tanh_to_range(u2, self.ImZF_max)
        ReZL  = _sigmoid_to_range(u3, self.ReZL_lo, self.ReZL_hi)
        ImZL  = _tanh_to_range(u4, self.ImZL_max)

        ZF = torch.complex(ReZF, ImZF)
        ZL = torch.complex(ReZL, ImZL)
        return L1, ZF, ZL #[...] each
    
    def _theta_to_u(self, L1, ZF, ZL):
        """
        Works for L1,ZF,ZL shaped as the same leading dims [...], returns u of shape [..., 5].
        Inverse map: theta -> u  (used for seeding)
        """
        ReZF, ImZF = ZF.real, ZF.imag
        ReZL, ImZL = ZL.real, ZL.imag

        u0 = _range_to_sigmoid_x(L1, self.L1_lo, self.L1_hi)
        u1 = _range_to_sigmoid_x(ReZF, self.ReZF_lo, self.ReZF_hi)
        u2 = _range_to_tanh_x(ImZF, self.ImZF_max)
        u3 = _range_to_sigmoid_x(ReZL, self.ReZL_lo, self.ReZL_hi)
        u4 = _range_to_tanh_x(ImZL, self.ImZL_max)

        return torch.stack([u0, u1, u2, u3, u4], dim=-1)  # [..., 5]
    
    
    def _uZ_init_mid(self, dev):
        """
        Midpoint init in u-space for ZF, ZL, returned as uZ = [u1..u4]
        """
        ReZF0 = 0.5*(self.ReZF_lo + self.ReZF_hi)
        ImZF0 = 0.0
        ReZL0 = 0.5*(self.ReZL_lo + self.ReZL_hi)
        ImZL0 = 0.0
        ZF0 = torch.complex(torch.tensor(ReZF0, device=dev), torch.tensor(ImZF0, device=dev))
        ZL0 = torch.complex(torch.tensor(ReZL0, device=dev), torch.tensor(ImZL0, device=dev))
        L1_dummy = torch.tensor(0.5*(self.L1_lo + self.L1_hi), device=dev) #Dummy L1 initialized middle of line
        u_all = self._theta_to_u(L1_dummy, ZF0, ZL0)
        return u_all[1:].detach().clone()  # u1..u4


    def _profile_all(self, y_f, var_f):
        """
        Vectorized profile:
          y_f,var_f: [M,F]
        Returns:
          L1_top: [M,topk]
          uZ_top: [M,topk,4]  (uZ snapshot at chosen L1)
          nll_top:[M,topk]
        """
        M,F = y_f.shape
        dev = self.device
        G = self.L1_grid_points

        L1_grid = torch.linspace(100, 900, G, device=dev, dtype=torch.float32)

        # uZ is initialized to middle of parameter range for all M runs
        uZ = torch.nn.Parameter(self._uZ_init_mid(dev).repeat(M, 1))  # [4] -> [M,4]
        opt = torch.optim.Adam([uZ], lr=self.inner_lr)
        nll_track = torch.empty(M, G, device=dev, dtype=torch.float32) #[M, G] NLL per run per grid
        uZ_snap = torch.empty(M, G, 4, device=dev, dtype=torch.float32) #[M, G, 4] best uZ per run per grid


        for gi in range(G):
            u0 = _range_to_sigmoid_x(L1_grid[gi], self.L1_lo, self.L1_hi).expand(M, 1)  # [M,1]

            #Adam at each grid point. u[0] (L1) always set to gridvalue. 
            for t in range(self.inner_steps):
                opt.zero_grad()
                U = torch.cat([u0, uZ], dim=1)  # Concatenate [M, 1] and [M,4] to form [M,5]
                L1, ZF, ZL = self._u_to_theta(U) # [M] each
                H = self.fm.compute_H_complex(L1, ZF, ZL) #[M, F]
                diff = y_f - H
                nll = ((diff.abs()**2) / var_f).sum(dim=1) #[M]
                # Key idea: If each entry in nll[m] depends ONLY on U[m,;] then summing doesn't couple anything because only
                #the nll[m]th term has a gradient with respect to U[m,;] the rest are constants
                loss = nll.mean()   
                loss.backward()
                opt.step()

            # Record profiled NLL and snapshot after this grid point
            with torch.no_grad():
                U = torch.cat([u0, uZ], dim=1)
                L1, ZF, ZL = self._u_to_theta(U) #[M]
                H = self.fm.compute_H_complex(L1, ZF, ZL)     # [M, F]
                diff = y_f - H 
                # At grid point gi, store the per-run NLL and best uZ for all M Monte Carlo runs in that column
                nll_track[:, gi] = ((diff.abs()**2) / var_f).sum(dim=1)  
                uZ_snap[:, gi, :] = uZ.detach()
        # Select top-K grid points per run [M, G] -> [M, K]
        _, top_idx = torch.topk(-nll_track, k=self.topk, dim=1) #max of the negative is same as min of positive
        L1_top  = L1_grid[top_idx]   # [N, K]
        
        true_L1  = 250.0
        wrong_L1 = 750.0
        tol = 5.0  
        col0 = L1_top[:, 0]  # [N]

        first_at_750 = torch.isclose(
            col0,
            torch.tensor(wrong_L1, device=L1_top.device),
            atol=tol
        ).sum().item()

        first_at_250 = torch.isclose(
            col0,
            torch.tensor(true_L1, device=L1_top.device),
            atol=tol
        ).sum().item()

        print(f"First-column seeds near 750 (wrong): {first_at_750}")
        print(f"First-column seeds near 250 (correct): {first_at_250}")
        print(f"First-column seeds at neither: {col0.numel() - first_at_750 - first_at_250}")

        
        arM = torch.arange(M, device=dev)
        uZ_top  = uZ_snap[arM[:, None], top_idx, :]  # [M, K, 4]

        return L1_top, uZ_top


    
    def predict(self, obs_tf, noise_var_f):
        """
        Jointly estimate parameters for M runs of N observations.
        Here M = 1000, N = 1
        Args
        ----
        obs_tf:      [M*N,F] complex tensor
        noise_var_f: [M*N,F] float tensor

        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [M],
              "ZF_re": [M], "ZF_im": [M],
              "ZL_re": [M], "ZL_im": [M]
            }
        """

        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [M,F]."
        M, F = obs_tf.shape

        print("Running L1 Profile...")
        # 1) L1 Profile all M in parallel
        L1_top, uZ_top = self._profile_all(obs_tf, noise_var_f) # [M,K], [M,K,4]
        print("Building Candidate Bank...")
        # 2) Build candidate bank U=[M,K,5]
        u0 = _range_to_sigmoid_x(L1_top, self.L1_lo, self.L1_hi).unsqueeze(-1) #[M,K,1]
        U = torch.cat([u0, uZ_top], dim=-1)   # [M,K,5]
        print("Running Adam in parallel...")
        # 3) Adam refine in parallel (opt vars are U) with top-k L1 seeds
        U = torch.nn.Parameter(U)
        opt = torch.optim.Adam([U], lr=self.adam_lr)
        for _ in range(self.adam_steps):
            opt.zero_grad()
            L1, ZF, ZL = self._u_to_theta(U) # [M,K] each
            H = self.fm.compute_H_complex(L1, ZF, ZL) #[M,K,F]
            diff = obs_tf.unsqueeze(1) - H #[M,1,F] - [M,K,F]
            #per-(m,k) NLL, then scalar loss for backprop
            nll = ((diff.abs()**2) / noise_var_f.unsqueeze(1)).sum(dim=-1) #[M,K,F] ->[M,K]
            loss = nll.mean()  
            loss.backward()
            opt.step()

        print("Running LFBGS in parallel...")
        # 4) L-BFGS polish (batched)
        if self.use_lbfgs and self.lbfgs_steps > 0:
            U_refined = batched_lbfgs(
                U0=U, y=obs_tf, nv=noise_var_f,
                fm=self.fm, u_to_theta=self._u_to_theta,
                history_size=12, iters=self.lbfgs_steps, alpha0=1.0
            )
        else:
            U_refined = U.detach()

        print("Evaluating final NLL...")
        # 5) Evaluate final NLL for each (m,k) and pick the best k per m
        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_refined)         # [M,K,5] -> [M,K]
            H = self.fm.compute_H_complex(L1, ZF, ZL)        # [M,K] -> [M,K,F]
            diff = obs_tf.unsqueeze(1) - H                   # [M,1,F] - [M,K,F]
            nv   = noise_var_f.unsqueeze(1)                  # [M,F] -> [M,1,F]
            final_nll = ((diff.abs() ** 2) / nv).sum(-1)     # [M,F,K] -> [M,K]

            # argmin across K, then gather best U row for each n
            i_best = torch.argmin(final_nll, dim=1)       #[M] containing indices of best k
            arM = torch.arange(M, device=U.device)
            U_best = U_refined[arM, i_best, :]               # [M,5]

        # 6) Map best U back to theta and return
        L1_best, ZF_best, ZL_best = self._u_to_theta(U_best) # each [M]
        out = {
            "L1":    L1_best.float().cpu().numpy(),
            "ZF_re": ZF_best.real.float().cpu().numpy(),
            "ZF_im": ZF_best.imag.float().cpu().numpy(),
            "ZL_re": ZL_best.real.float().cpu().numpy(),
            "ZL_im": ZL_best.imag.float().cpu().numpy(),
        }
        return out