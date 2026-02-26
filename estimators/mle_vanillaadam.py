import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from .lfbgs import batched_lbfgs
import matplotlib.pyplot as plt
from .base import Estimator  # keep your existing Estimator base
torch.set_float32_matmul_precision("high")


def _sigmoid_to_range(u, lo, hi):
    # Map u in R -> theta in (lo,hi) using sigmoid
    return lo + (hi - lo) * torch.sigmoid(u)

def _range_to_sigmoid_x(x, lo, hi, eps: float = 1e-6):
    # Inverse of _sigmoid_to_range for seeding
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)  # avoid 0/1
    return torch.log(z) - torch.log1p(-z)  # logit

def _tanh_to_range(u, max_abs):
    # Map u in R -> thetea in (-max_abs, max_abs) using tanh
    return max_abs * torch.tanh(u)

def _range_to_tanh_x(x, max_abs, eps: float = 1e-6):
    # Inverse of _tanh_to_range for seeding
    r = (x / max_abs).clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log1p(r) - 0.5 * torch.log1p(-r)  # atanh


class VanillaAdam(Estimator):
    """
    Adam + LBFGS + Restarts MLE for theta = [L1, ReZF, ImZF, ReZL, ImZL].

    Parameterization:
      L1    in [0, L]             via sigmoid
      ReZF  in [1, 4000]          via sigmoid
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
    """
    def __init__(self,
                 fm,
                 likelihood,
                 L: float = 1000.0,
                 device="cuda",
                 n_starts: int = 1,
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
        self.ReZF_lo, self.ReZF_hi = 1.0, 4000.0
        self.ImZF_min, self.ImZF_max = -100.0, 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

    
    def _u_to_theta(self, u):
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
    
    def random_u(self, shape, dtype=torch.float32, device="cuda"):
        """
        Random init in unconstrained space by sampling physical θ uniformly in box, then mapping back with inverse transforms.
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
    

    def _nll_batch(self, y_f, var_f, U):
        """
        Negative log-likelihood for BATCH of observations:
          y_f:  [M,F] complex
          var_f:[M,F] float32          
          U:    [M,R,5] float32              
        Returns NLL matrix of size [M,R]
        """
        L1, ZF, ZL = self._u_to_theta(U)     # [M,R] each
        H = self.fm.compute_H_complex(L1, ZF, ZL)    # [M,R,F]
        diff = y_f.unsqueeze(1) - H                   # [M,1,F] - [M,R,F]
        ll = -((diff.abs()**2) / var_f.unsqueeze(1)).sum(dim=-1)  # [M,R] Sum over F 
        return -ll   # NLL [M,R]

    def loss_fn(self, pred_tf, obs_tf, noise_var, eps=1e-12):
        """
        Negative log-likelihood
          obs_tf:[1,500] tensor cfloat/complex64
          pred_tf:[1,500] tensor cfloat/complex64
          noise_var: [1,500] float32             
        Returns NLL scalar
        """
        diff = obs_tf - pred_tf
        var = noise_var.clamp_min(eps)
        nll = (diff.abs().pow(2) / var).mean()   # correct
        return nll
    
    def loss_fn_magnitude_only(self, pred_tf, obs_tf):
        """Only fit magnitudes, ignore phase"""
        pred_mag = torch.abs(pred_tf)
        obs_mag = torch.abs(obs_tf)
        return torch.mean((pred_mag - obs_mag)**2)
    def loss_fn_smoothed(self, pred_tf, obs_tf, kernel_size=20):
        """Smooth magnitude spectra before comparing"""
        pred_mag = torch.abs(pred_tf).flatten()  # Flatten to 1D [F]
        obs_mag = torch.abs(obs_tf).flatten()    # Flatten to 1D [F]

        # Apply moving average to blur sharp features
        # conv1d expects [batch, channels, length] = [1, 1, F]
        kernel = torch.ones(1, 1, kernel_size, device=pred_mag.device) / kernel_size
        pred_smooth = torch.nn.functional.conv1d(pred_mag.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()
        obs_smooth = torch.nn.functional.conv1d(obs_mag.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()

        return torch.mean((pred_smooth - obs_smooth)**2)
    def loss_fn_statistics(self, pred_tf, obs_tf):
        """Compare statistical properties, not point-by-point"""
        pred_mag = torch.abs(pred_tf)
        obs_mag = torch.abs(obs_tf)
        
        # Compare mean and variance instead of each frequency
        loss_mean = (pred_mag.mean() - obs_mag.mean())**2
        loss_var = (pred_mag.var() - obs_mag.var())**2
        
        return loss_mean + loss_var
    def plot_loss_landscape(self, obs_tf, noise_var, save_path="loss_landscape_ZFim.png"):
        """
        Plot loss vs L1 to visualize the optimization landscape.
        ZF and ZL are fixed to true values.
        """
        # Fix ZF and ZL to true values
        L1 = torch.tensor(250.0, device=self.device, dtype=torch.float32)
        ReZF = torch.tensor(100.0, device=self.device, dtype=torch.float32)
        ZL = torch.tensor(100.0 - 5.0j, device=self.device, dtype=torch.cfloat)

        # Sweep L1 from 0 to 1 (normalized)
        ImZF_normalized = torch.linspace(0.01, 0.99, 199, device=self.device,
    dtype=torch.float32)
        ImZF_physical = self.ImZF_min + (self.ImZF_max - self.ImZF_min) * ImZF_normalized
        losses = []

        with torch.no_grad():
            for ImZF in ImZF_physical:
                ZF = torch.complex(ReZF, ImZF)
                pred_tf = self.fm.compute_H_complex(L1, ZF, ZL)
                loss = self.loss_fn(pred_tf, obs_tf, noise_var)
                losses.append(loss.item())

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(ImZF_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
        plt.axvline(x=0.25, color='r', linestyle='--', linewidth=2, label='True ImZF=0.25')
        plt.xlabel('ImZF (normalized)', fontsize=12)
        plt.ylabel('Loss (NLL)', fontsize=12)
        plt.title('Loss Landscape vs Imaginary Part of Fault Impedance ImZF', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved loss landscape plot to {save_path}")

        # Print some diagnostics
        min_idx = np.argmin(losses)
        print(f"\nLoss Landscape Diagnostics:")
        print(f"Minimum loss at ZF_normalized = {ImZF_normalized[min_idx]:.4f}")

    def predict2(self, obs_tf, noise_var):
        # First plot the loss landscape to diagnose the problem
        # self.plot_loss_landscape(obs_tf, noise_var)
        
        M, F = obs_tf.shape
        # Fix ZF and ZL to true 
        L1 = torch.tensor(250.0, device=self.device, dtype=torch.float32)
        ImZF = torch.tensor(-50.0, device=self.device, dtype=torch.float32)
        ZL = torch.tensor(100.0 - 5.0j, device=self.device, dtype=torch.cfloat)
        dev  = self.device

        # Only optimize ReZF, initialize to sigmoid(0) = 0.5
        ReZF_init = torch.logit(torch.tensor(0.5))
        U_raw = torch.nn.Parameter(
            torch.tensor(ReZF_init, dtype=torch.float32, device=dev)  # Scalar for ReZF only
        )
        opt = torch.optim.Adam([U_raw], lr=0.05)

        for i in range(20000):
            ReZF = _sigmoid_to_range(U_raw, self.ReZF_lo, self.ReZF_hi)  # U_raw is now scalar
            ZF = torch.complex(ReZF, ImZF)
            pred_tf = self.fm.compute_H_complex(L1, ZF, ZL) #[1, 500]
            loss = self.loss_fn(pred_tf, obs_tf, noise_var)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 500 == 0:
                ReZF_normalized = torch.sigmoid(U_raw)
                print(f"step {i} | loss {loss.item():.4f}")
                print(f"ReZF_normalized (sigmoid)={ReZF_normalized.item():.4f} | True value is 0.025")

    def predict3(self, obs_tf, noise_var):
        """
        SVI-based inference for L1 only (ZF, ZL fixed to true values).
        This tests whether SVI performs any better than MLE on the sharp loss landscape.
        Uses same setup as sensorspaper3.py but for the simple model.
        """
        # Fix ZF and ZL to true values
        ZF = torch.tensor(100.0 + 50.0j, device=self.device, dtype=torch.cfloat)
        ZL = torch.tensor(100.0 - 5.0j, device=self.device, dtype=torch.cfloat)

        # Flatten observation for likelihood
        obs_tf_flat = obs_tf.flatten()  # [F] complex
        noise_std = torch.sqrt(noise_var.flatten().mean())  # Scalar std

        # Define Pyro model
        def model(obs):
            # Prior on L1 (normalized to [0, 1])
            L1_normalized = pyro.sample("L1", dist.Uniform(0.0, 1.0))

            # Convert to physical value
            L1_physical = self.L1_lo + (self.L1_hi - self.L1_lo) * L1_normalized

            # Compute predicted TF
            pred_tf = self.fm.compute_H_complex(L1_physical, ZF, ZL).flatten()  # [F] complex

            # Split into real and imaginary for likelihood
            pred_real = pred_tf.real
            pred_imag = pred_tf.imag
            obs_real = obs.real
            obs_imag = obs.imag

            # Observe real and imaginary parts
            pyro.sample("obs_real", dist.Normal(pred_real, noise_std).to_event(1), obs=obs_real)
            pyro.sample("obs_imag", dist.Normal(pred_imag, noise_std).to_event(1), obs=obs_imag)

        # Use AutoNormal guide (same as sensorspaper3.py)
        guide = AutoNormal(model, init_scale=0.1)

        # Setup SVI with same lr as sensorspaper3.py
        pyro.clear_param_store()
        optimizer = pyro.optim.Adam({"lr": 0.2})
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))

        # Run inference
        num_steps = 500
        losses = []

        print("\n=== SVI Inference for L1 (simple model) ===")
        for step in range(num_steps):
            loss = svi.step(obs_tf_flat)
            losses.append(loss)

            if step % 50 == 0 or step == num_steps - 1:
                # Get current posterior mean
                param_store = pyro.get_param_store()
                loc = param_store["AutoNormal.locs.L1"].item()
                scale = param_store["AutoNormal.scales.L1"].item()
                L1_normalized = torch.sigmoid(torch.tensor(loc)).item()
                L1_physical = self.L1_lo + (self.L1_hi - self.L1_lo) * L1_normalized

                print(f"Step {step:3d} | ELBO: {loss:.2f} | L1_norm: {L1_normalized:.4f} | L1_phys: {L1_physical:.2f}m | scale: {scale:.4f}")

        # Final result
        param_store = pyro.get_param_store()
        loc = param_store["AutoNormal.locs.L1"].item()
        L1_final = torch.sigmoid(torch.tensor(loc)).item()
        L1_physical_final = self.L1_lo + (self.L1_hi - self.L1_lo) * L1_final

        print(f"\n=== SVI Result ===")
        print(f"Final L1 (normalized): {L1_final:.4f}")
        print(f"Final L1 (physical): {L1_physical_final:.2f}m")
        print(f"True L1 should be ~250m (normalized 0.25)")

        # Plot ELBO convergence
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.xlabel("SVI Step")
        plt.ylabel("ELBO Loss")
        plt.title("SVI Convergence (Simple Model, L1 only)")
        plt.grid(True)
        plt.savefig("svi_simple_model_L1.png", dpi=150)
        plt.close()
        print("Saved ELBO plot to svi_simple_model_L1.png")

    def predict(self, obs_tf, noise_var):
        """
        Estimate theta = [L1, ReZF, ImZF, ReZL, ImZL] for M runs of N observations. M=1000 and N=1 here so we use M instead of M*N for simplicity.
        This uses Adam + LBFGS + Restarts with Successive Halving. NO PROFILING. 
        Args
        ----
        obs_tf:      [M,F] cfloat tensor
        noise_var_f: [M,F] float32 tensor

        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [M],
              "ZF_re": [M], "ZF_im": [M],
              "ZL_re": [M], "ZL_im": [M]
            }
        """

        assert obs_tf.dim() == 2 and noise_var.dim() == 2, "Expect shapes [M,F]."
        M, F = obs_tf.shape
        dev  = self.device
        R    = self.n_starts
        print(f"Number of restarts is {R}")
        # ---- Halving schedule (tweak as you like)
        rounds_steps = (30, 30, 30)         # Adam steps per round
        rounds_keep  = (0.25, 0.2, 0.5)      # keep fraction after each round
        # This turns 200 -> 50 -> 10 -> 5
        
        # 0) Intialize U to random locations in parameter range
        U = torch.nn.Parameter(self.random_u([M, R, 5], dtype=torch.float32, device=dev)) #[M, R, 5]
        opt = torch.optim.Adam([U], lr=self.adam_lr)

        print("Running Adam Optimization with successive halving...")
        # 1) Adam Optimization with successive halving
        Rcur = R
        for steps, kfrac in zip(rounds_steps, rounds_keep):
            opt = torch.optim.Adam([U], lr=self.adam_lr)
            print("U shape", U.shape)
            for i in range(steps):
                print(f"Current step is {i}/{steps}")
                opt.zero_grad()
                nll = self._nll_batch(obs_tf, noise_var, U)   # [M,Rcur] NLL per run m and restart r
                #Key idea: If each entry in nll[m,r] depends ONLY on U[m,r,;] then summing doesn't couple anything because only
                #the nll[m,r]th term has a gradient with respect to U[m,r,;] the rest are constants
                loss = nll.mean()              
                loss.backward()
                opt.step()
            with torch.no_grad():
                scores = self._nll_batch(obs_tf, noise_var, U)  # [M,Rcur]
                k = max(1, int(Rcur*kfrac))
                _, top_idx = torch.topk(-scores, k=k, dim=1) #[N,k]
                arN = torch.arange(M, device=dev)[:, None]
                U = U.detach()[arN, top_idx, :].contiguous() #[N,Rcur,5] -> [N,k,5]
                U = torch.nn.Parameter(U) # new leaf for next round Rcur = k
                Rcur = k

        print("Running L-BFGS...")
        # 2) L-BFGS in parallel after
        if self.use_lbfgs and self.lbfgs_steps > 0:
            U_refined = batched_lbfgs(
            U0=U, y=obs_tf, nv=noise_var,
            fm=self.fm, u_to_theta=self._u_to_theta,
            history_size=12, iters=40, alpha0=1.0
        )

        print("Selecting best restart per observation...")
        # 3) Select best restart per run m
        with torch.no_grad():
            final_nll = self._nll_batch(obs_tf, noise_var, U_refined)      # [M,R]
            i_best = torch.argmin(final_nll, dim=1)    # [M]
            arangeM = torch.arange(M, device=dev)
            U_best = U_refined[arangeM, i_best, :] # [M,5,5] -> [M,5]
            L1, ZF, ZL = self._u_to_theta(U_best) #[M,5] -> [M] each

        return {
            "L1":    L1.float().cpu().numpy(),
            "ZF_re": ZF.real.float().cpu().numpy(),
            "ZF_im": ZF.imag.float().cpu().numpy(),
            "ZL_re": ZL.real.float().cpu().numpy(),
            "ZL_im": ZL.imag.float().cpu().numpy(),
        }

