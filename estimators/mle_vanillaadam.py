import torch
from .lfbgs import batched_lbfgs
from .base import Estimator  # keep your existing Estimator base
import time

torch.set_float32_matmul_precision("high")


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


class VanillaAdam(Estimator):
    """
    Adam + LBFGS + Restarts MLE for theta = [L1, ReZF, ImZF, ReZL, ImZL].

    Parameterization:
      L1    in [0, L]             via sigmoid
      ReZF  in [1, 2000]          via sigmoid
      ImZF  in [-100, +100]       via tanh
      ReZL  in [1, 200]           via sigmoid
      ImZL  in [-100, +100]       via tanh

    Args
    ----
    fm: ForwardModel                (provides compute_H_complex(L1, ZF, ZL))
    likelihood:                     e.g., ComplexGaussianLik()
    L: float                        line length (upper bound for L1)
    device: torch.device
    n_starts: int                   number of random restarts per observation
    adam_steps: int                 number of Adam optimization steps
    adam_lr: float                  adam learning rate
    verbose: boolng rate            
    """
    def __init__(self,
                 fm,
                 likelihood,
                 L: float = 1000.0,
                 device="cuda",
                 n_starts: int = 10,
                 adam_steps: int = 400,
                 adam_lr: float = 1e-3,
                 use_lbfgs: bool = True, 
                 lbfgs_steps: int = 60, 
                 lbfgs_lr: float = 1,
                 verbose: bool = True
                 ):
        self.fm = fm
        self.device = device
        self.lik = likelihood
        self.L = float(L)
        self.n_starts = int(n_starts)
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        #LBFGS hyperparametrs
        self.use_lbfgs = bool(use_lbfgs)
        self.lbfgs_steps = int(lbfgs_steps)
        self.lbfgs_lr = float(lbfgs_lr)
        self.verbose = bool(verbose)

        # parameter range
        self.L1_lo, self.L1_hi = 1.0, self.L
        self.ReZF_lo, self.ReZF_hi = 1.0, 2000.0
        self.ImZF_max = 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

    
    def _make_inits(self, N: int, R: int, dtype=torch.float32):
        """
        Deterministic init tensor [N, R, 5], independent of global RNG state.
        Uses a local torch.Generator seeded the same way every call.
        """
        g = torch.Generator(device=self.device)
        g.manual_seed(123456789)  # <- choose any fixed seed or expose via __init__
        # sample θ uniformly in the physical box, then map to u
        L1  = torch.empty((N, R), device=self.device, dtype=torch.float32).uniform_(self.L1_lo,   self.L1_hi,   generator=g)
        ReF = torch.empty((N, R), device=self.device, dtype=torch.float32).uniform_(self.ReZF_lo, self.ReZF_hi, generator=g)
        ImF = torch.empty((N, R), device=self.device, dtype=torch.float32).uniform_(-self.ImZF_max, self.ImZF_max, generator=g)
        ReL = torch.empty((N, R), device=self.device, dtype=torch.float32).uniform_(self.ReZL_lo, self.ReZL_hi, generator=g)
        ImL = torch.empty((N, R), device=self.device, dtype=torch.float32).uniform_(-self.ImZL_max, self.ImZL_max, generator=g)
        ZF = torch.complex(ReF, ImF)
        ZL = torch.complex(ReL, ImL)
        u = self._theta_to_u(L1, ZF, ZL).to(dtype)  # [N, R, 5]
        return u
    # ---------- parameterization ----------
    def _u_to_theta(self, u: torch.Tensor):
        """
        Works for u of shape [..., 5].
        Map: u = [u0, u1, u2, u3, u4] -> theta = (L1, ZF, ZL)
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
    
    def _theta_to_u(self, L1: torch.Tensor, ZF: torch.Tensor, ZL: torch.Tensor):
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
    
    def random_u(self, shape, dtype=torch.float32, device="cuda"):
        """
        Random init in unconstrained space by sampling physical θ uniformly in box,
        then mapping back with inverse transforms.
        """
        L1  = torch.empty(shape[:-1], device=device).uniform_(self.L1_lo, self.L1_hi)
        ReF = torch.empty(shape[:-1], device=device).uniform_(self.ReZF_lo, self.ReZF_hi)
        ImF = torch.empty(shape[:-1], device=device).uniform_(-self.ImZF_max, self.ImZF_max)
        ReL = torch.empty(shape[:-1], device=device).uniform_(self.ReZL_lo, self.ReZL_hi)
        ImL = torch.empty(shape[:-1], device=device).uniform_(-self.ImZL_max, self.ImZL_max)
        ZF = torch.complex(ReF, ImF)
        ZL = torch.complex(ReL, ImL)
        u = self._theta_to_u(L1, ZF, ZL).to(dtype)
        return u
    

    def _nll_one(self, y_f, var_f, u):
        """
        Negative log-likelihood for ONE observation:
          y_f:  [F] complex64  (obs)
          var_f:[F] float32              (per-frequency noise variances)
          u:    [5] float32              (unconstrained params)
        Returns scalar tensor (float32).
        """
        L1, ZF, ZL = self._u_to_theta(u)
        H = self.fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL)  # [1, F] complex
        # score_matrix expects batches [N,F]; add batch dim via unsqueeze(0)
        ll = self.lik.score_matrix(
            y_f.unsqueeze(0),        # [1,F]
            H,                       # [1,F]
            var_f.unsqueeze(0)      # [1,F]
        ).sum() #convert [1, 1] to scalar tensor by summing
        return -ll  # NLL
    
    def predict(self, obs_tf, noise_var_f, L1_true, ZF_true_re, ZF_true_im, ZL_true_re, ZL_true_im):
        """
        Jointly estimate parameters for a batch of N observations in series. 

        Args
        ----
        obs_tf:      [N,F] cfloat tensor
        noise_var_f: [N,F] float32 tensor


        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [N],
              "ZF_re": [N], "ZF_im": [N],
              "ZL_re": [N], "ZL_im": [N]
            }
        """


        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [N,F]."
        N, F = obs_tf.shape
        dev  = self.device
        R    = self.n_starts
        init_U = self._make_inits(N, R, dtype=torch.float32)  # [N, R, 5]

        out = {
            "L1":    torch.empty(N, dtype=torch.float32, device=dev),
            "ZF_re": torch.empty(N, dtype=torch.float32, device=dev),
            "ZF_im": torch.empty(N, dtype=torch.float32, device=dev),
            "ZL_re": torch.empty(N, dtype=torch.float32, device=dev),
            "ZL_im": torch.empty(N, dtype=torch.float32, device=dev),
        }

        for i in range(N):
            y_i = obs_tf[i]
            v_i = noise_var_f[i]
            best_nll = torch.tensor(float("inf"), device=dev)
            best_theta = None

            for r in range(R):
                u = torch.nn.Parameter(init_U[i, r, :].clone())  # <- deterministic per (i,r)
                # u = torch.nn.Parameter(self.random_u([5], dtype=torch.float32, device=dev))
                opt = torch.optim.Adam([u], lr=self.adam_lr)


                for step in range(self.adam_steps):
                    opt.zero_grad(set_to_none=True)
                    nll = self._nll_one(y_i, v_i, u) 
                    nll.backward()
                    opt.step()
                    
                if self.use_lbfgs and self.lbfgs_steps > 0:
                    lbfgs = torch.optim.LBFGS(
                        [u],
                        lr=self.lbfgs_lr,                 # e.g., 1.0
                        max_iter=self.lbfgs_steps,        # e.g., 40–80
                        line_search_fn="strong_wolfe",
                    )
                    def closure():
                        lbfgs.zero_grad()
                        loss = self._nll_one(y_i, v_i, u)   # scalar
                        loss.backward()
                        return loss
                    lbfgs.step(closure)


                with torch.no_grad():
                    if nll < best_nll:
                        best_nll = nll
                        best_theta = self._u_to_theta(u)

            # write out best
            L1, ZF, ZL = best_theta
            out["L1"][i]    = L1.float()
            out["ZF_re"][i] = ZF.real.float()
            out["ZF_im"][i] = ZF.imag.float()
            out["ZL_re"][i] = ZL.real.float()
            out["ZL_im"][i] = ZL.imag.float()

            if self.verbose:
                print(
                    f"[{i+1}/{N}], "
                    f"L1_pred={float(L1):.2f}, L1_true={L1_true[0]}, "
                    f"ZF_re_pred={float(ZF.real):.2f}, ZF_re_true={ZF_true_re[0]}, "
                    f"ZF_im_pred={float(ZF.imag):.2f}, ZF_im_true={ZF_true_im[0]}, "
                    f"ZL_re_pred={float(ZL.real):.2f}, ZL_re_true={ZL_true_re[0]}, "
                    f"ZL_im_pred={float(ZL.imag):.2f}, ZL_im_true={ZL_true_im[0]}\n"
                )

        return {k: v.detach().cpu().numpy() for k, v in out.items()}
    

    def _nll_batch(self, y_f, var_f, U):
        """
        Negative log-likelihood for BATCH of observations:
          y_f:  [N,F] complex64  (obs)
          var_f:[N,F] float32              (per-frequency noise variances)
          U:    [N,R,5] float32              (unconstrained params)
        Returns NLL matrix of size [N,R]
        """
        N, F = y_f.shape
        _, R, _ = U.shape

        L1, ZF, ZL = self._u_to_theta(U)     # [N,R] each
        H = self.fm.compute_H_complex(L1, ZF, ZL)    # [N,R,F]
        diff = y_f.unsqueeze(1) - H                   # [N,1,F] - [N,R,F]
        ll   = -((diff.abs()**2) / var_f.unsqueeze(1)).sum(dim=-1)  # [N,R] Sum over F 
        return -ll                                     # NLL [N,R]




    def predict_batch(self, obs_tf, noise_var_f):
        """
        Parallel NxR optimization with per-candidate independence.
        Equivalent to running N*R separate Adam optimizers, but fused.
        """

        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [N,F]."
        N, F = obs_tf.shape
        dev  = self.device
        R    = self.n_starts
        print(f"Number of restarts is {R}")
        start_time = time.perf_counter()
        # init_U = self._make_inits(N, R, dtype=torch.float32)  # [N, R, 5]
        # U = torch.nn.Parameter(init_U.clone())                # <- same inits as serial

        # ---- Halving schedule (tweak as you like)
        rounds_steps = (30, 30, 30)         # Adam steps per round
        rounds_keep  = (0.25, 0.2, 0.5)      # keep fraction after each round
        # This turns 200 -> 50 -> 10 -> 5
        
        # 0) Intialize U to random locations in parameter range
        U = torch.nn.Parameter(self.random_u([N, R, 5], dtype=torch.float32, device=dev))
        opt = torch.optim.Adam([U], lr=self.adam_lr)

        print("Running Adam Optimization with successive halving...")
        # 1) Adam Optimization with successive halving
        Rcur = R
        for steps, kfrac in zip(rounds_steps, rounds_keep):
            opt = torch.optim.Adam([U], lr=self.adam_lr)
            print("U shape", U.shape)
            for i in range(steps):
                print(f"Current step is {i}/{steps}")
                opt.zero_grad(set_to_none=True)
                nll = self._nll_batch(obs_tf, noise_var_f, U)   # [N,Rcur] NLL per observation n and restart r
                #Key idea: If each entry in nll[n,r] depends ONLY on U[n,r,;] then summing doesn't couple anything because only
                #the nll[n,r]th term has a gradient with respect to U[n,r,;] the rest are constants
                loss = nll.sum()              
                loss.backward()
                opt.step()
            with torch.no_grad():
                scores = self._nll_batch(obs_tf, noise_var_f, U)  # [N,Rcur]
                k = max(1, int(Rcur*kfrac))
                _, top_idx = torch.topk(-scores, k=k, dim=1) #[N,k]
                arN = torch.arange(N, device=dev)[:, None]
                U = U.detach()[arN, top_idx, :].contiguous() #[N,Rcur,5] -> [N,k,5]
                U = torch.nn.Parameter(U) # new leaf for next round Rcur = k
                Rcur = k


        print("Size of U after successive halving should be [N, 5, 5]", U.shape)
        print("Running L-BFGS...")
        # 2) L-BFGS in parallel after
        #Can't run LBFGS in parallel like Adam have to do for loop over N and R...
        if self.use_lbfgs and self.lbfgs_steps > 0:
            U_refined = batched_lbfgs(
    U0=U, y=obs_tf, nv=noise_var_f,
    fm=self.fm, u_to_theta=self._u_to_theta,
    history_size=12, iters=40, alpha0=1.0
)

        print("Selecting best restart per observation...")
        # 3) Select best restart per observation
        with torch.no_grad():
            final_nll = self._nll_batch(obs_tf, noise_var_f, U_refined)      # [N,R]
            i_best = torch.argmin(final_nll, dim=1)    # [N]

            arangeN = torch.arange(N, device=dev)
            U_best  = U_refined[arangeN, i_best, :]            # [N,5]

            
            L1, ZF, ZL = self._u_to_theta(U_best) #[N,5] -> [N]

        print("how long does this take", )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Program took {elapsed_time:.4f} seconds to run.")
        return {
            "L1":    L1.float().cpu().numpy(),
            "ZF_re": ZF.real.float().cpu().numpy(),
            "ZF_im": ZF.imag.float().cpu().numpy(),
            "ZL_re": ZL.real.float().cpu().numpy(),
            "ZL_im": ZL.imag.float().cpu().numpy(),
        }

