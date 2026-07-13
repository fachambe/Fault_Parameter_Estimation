import torch
import numpy as np
from .base import Estimator, u_to_theta_sigmoid, u_to_theta_tanh
from estimators.bfgs import BFGSEstimator
from estimators.LM import LMEstimator
from estimators.Newton import NewtonEstimator
from torch.func import hessian, vmap

torch.set_float32_matmul_precision("high")


class GradientMLE(Estimator):
    """
    Gradient-based MLE for theta = [L1, ReZF, ImZF, ReZL, ImZL].
    Supports Adam, SGD, or other PyTorch optimizers + optional BFGS/LM refinement.

    Parameterization:
      L1    in [0, L]             via sigmoid
      ReZF  in [1, 4000]          via sigmoid
      ImZF  in [-100, +100]       via tanh
      ReZL  in [1, 400]           via sigmoid
      ImZL  in [-100, +100]       via tanh

    Args
    ----
    fm: ForwardModel (provides compute_H_complex(L1, ZF, ZL) and L)
    likelihood: e.g., ComplexGaussianLik()
    target: "L1" | "ZF_re" | "ZF_im" | "ZL_re" | "ZL_im"
    fixed: dict of fixed params, e.g. {"ZF": {"re": 100, "im": -5}, "ZL": {...}, "L1": 500.0}
    true_range: parameter ranges from config
    device: default: "cuda"
    """
    def __init__(self,
                 fm,
                 likelihood,
                 target,
                 fixed,
                 true_range,
                 mode,
                 device="cuda",
                 adam_steps: int = 20000,
                 adam_lr: float = 1e-2,
                 adam_betas: tuple = (0.7, 0.9),  # Lower betas for oscillatory landscapes
                 use_bfgs: bool = False,
                 bfgs_steps: int = 60,
                 bfgs_lr: float = 1,
                 use_LM: bool = False,
                 use_Newton: bool = False,
                 verbose: bool = True
                 ):
        super().__init__(fm, likelihood, target, fixed, true_range, mode, device)

        # Adam hyperparameters
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.adam_betas = adam_betas
        self.verbose = bool(verbose)

        # BFGS hyperparameters
        self.use_bfgs = bool(use_bfgs)
        self.bfgs_steps = int(bfgs_steps)
        self.bfgs_lr = float(bfgs_lr)

        # LM hyperparameters
        self.use_LM = bool(use_LM)

        self.use_Newton = use_Newton
    
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
    
    def predict(self, obs_tf, noise_var):
        if self.mode == "1d":
            return self._predict_1d(obs_tf, noise_var)
        else:
            return self._predict_joint(obs_tf, noise_var)

    def _predict_1d(self, obs_tf, noise_var):
        """
        Batched 1D gradient optimization. Estimate self.target for all N observations in parallel.

        obs_tf:    [N, F] complex64 tensor
        noise_var: [N, 1] or [N, F] float32 tensor
        Returns:   dict with key=target, value= numpy array [N] of estimates
        """
        N = obs_tf.shape[0]
        dev = self.device

        # Get parameter bounds based on target
        if self.target == "L1":
            lo, hi = self.L1_lo, self.L1_hi
        elif self.target == "ZF_re":
            lo, hi = self.ReZF_lo, self.ReZF_hi
        elif self.target == "ZF_im":
            lo, hi = -self.ImZF_max, self.ImZF_max
        elif self.target == "ZL_re":
            lo, hi = self.ReZL_lo, self.ReZL_hi
        elif self.target == "ZL_im":
            lo, hi = -self.ImZL_max, self.ImZL_max
        else:
            raise ValueError(f"Unknown target: {self.target}")

        # Initialize U: one unconstrained parameter per observation [N]
        # Start at midpoint (sigmoid(0) = 0.5, tanh(0) = 0)
        U = torch.nn.Parameter(torch.zeros(N, dtype=torch.float32, device=dev))
        opt = torch.optim.Adam([U], lr=self.adam_lr)

        for step in range(self.adam_steps):
            opt.zero_grad()

            # Map unconstrained U to physical parameters [N]
            if self.target.endswith("_im"):
                max_abs = hi
                theta = u_to_theta_tanh(U, max_abs)
            else:
                theta = u_to_theta_sigmoid(U, lo, hi)

            # Build full parameter set - theta is [N], fixed params broadcast
            if self.target == "L1":
                L1 = theta  # [N]
                ZF = self.ZF_fix.expand(N)  # [N]
                ZL = self.ZL_fix.expand(N)  # [N]
            elif self.target == "ZF_re":
                L1 = self.L1_fix.expand(N)
                ZF = torch.complex(theta, self.ZF_fix.imag.expand(N))
                ZL = self.ZL_fix.expand(N)
            elif self.target == "ZF_im":
                L1 = self.L1_fix.expand(N)
                ZF = torch.complex(self.ZF_fix.real.expand(N), theta)
                ZL = self.ZL_fix.expand(N)
            elif self.target == "ZL_re":
                L1 = self.L1_fix.expand(N)
                ZF = self.ZF_fix.expand(N)
                ZL = torch.complex(theta, self.ZL_fix.imag.expand(N))
            elif self.target == "ZL_im":
                L1 = self.L1_fix.expand(N)
                ZF = self.ZF_fix.expand(N)
                ZL = torch.complex(self.ZL_fix.real.expand(N), theta)
            if step % 500 == 0:
                if self.target == "L1":
                    true_val = self.L1_fix.item()
                elif self.target == "ZF_re":
                    true_val = self.ZF_fix.real.item()
                elif self.target == "ZF_im":
                    true_val = self.ZF_fix.imag.item()
                elif self.target == "ZL_re":
                    true_val = self.ZL_fix.real.item()
                else:  # ZL_im
                    true_val = self.ZL_fix.imag.item()
                print(f"Step {step}/{self.adam_steps} | {self.target}: est_mean={theta.mean().item():.2f}, true={true_val:.2f}")
            # Forward model: [N, F]
            pred_tf = self.fm.compute_H_complex(L1, ZF, ZL)

            # Compute NLL per observation [N] - element-wise comparison
            nll_per_obs = self.lik.nll_elementwise(obs_tf, pred_tf, noise_var)

            # Sum and backprop - each nll_per_obs[n] depends only on U[n]
            loss = nll_per_obs.mean()
            loss.backward()
            opt.step()

        # Extract final estimates
        with torch.no_grad():
            if self.target.endswith("_im"):
                final_theta = u_to_theta_tanh(U, hi)
            else:
                final_theta = u_to_theta_sigmoid(U, lo, hi)

        return {self.target: final_theta.cpu().numpy()}

    def _predict_joint(self, obs_tf, noise_var):
        """
        Estimate theta = [L1, ReZF, ImZF, ReZL, ImZL] for all N observations in parallel with Adam + polishing. 
        Args
        ----
        obs_tf:      [N,F] cfloat tensor
        noise_var_f: [N,F] or [N,1] float32 tensor

        Returns
        -------
        dict of numpy arrays containing estimates for the parameter
            {
              "L1": [N],
              "ZF": [N], 
              "ZL": [N], 
            }
        """
        N, _ = obs_tf.shape
        dev  = self.device
        # 0) U initialization - random or center of param range
        # U = torch.nn.Parameter(self.random_u((N, 5), dtype=torch.float32, device=dev))
        U = torch.nn.Parameter(torch.zeros((N, 5), dtype=torch.float32, device=dev))
        opt = torch.optim.Adam([U], lr=self.adam_lr)


        for step in range(self.adam_steps):
            opt.zero_grad()

            L1, ZF, ZL = self._u_to_theta(U)

            # Forward model: [N, F]
            pred_tf = self.fm.compute_H_complex(L1, ZF, ZL)

            # Compute NLL per observation [N]
            nll_per_obs = self.lik.nll_elementwise(obs_tf, pred_tf, noise_var)

            # Backprop and step
            loss = nll_per_obs.mean()
            loss.backward()
            opt.step()

            if step % 500 == 0:

                with torch.no_grad():
                    rmse_L1 = torch.sqrt(((L1 - self.L1_fix) ** 2).mean())
                    rmse_ZF = torch.sqrt(((ZF - self.ZF_fix).abs() ** 2).mean())
                    rmse_ZL = torch.sqrt(((ZL - self.ZL_fix).abs() ** 2).mean())

                print(
                    f"Adam Step {step}/{self.adam_steps} | "
                    f"Loss: {loss.item():.2f} | "
                    f"RMSE: L1={rmse_L1.item():.3f}, "
                    f"ZF={rmse_ZF.item():.3f}, "
                    f"ZL={rmse_ZL.item():.3f} | "
                    f"L1: mean={L1.mean().item():.2f}, true={self.L1_fix.item():.2f} | "
                    f"ZF: mean={ZF.real.mean().item():.2f}{ZF.imag.mean().item():+.2f}j, "
                    f"true={self.ZF_fix.real.item():.2f}{self.ZF_fix.imag.item():+.2f}j | "
                    f"ZL: mean={ZL.real.mean().item():.2f}{ZL.imag.mean().item():+.2f}j, "
                    f"true={self.ZL_fix.real.item():.2f}{self.ZL_fix.imag.item():+.2f}j"
                )


        # Compute RMSE before polishing
        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U)
            rmse_L1_before = torch.sqrt(((L1 - self.L1_fix) ** 2).mean()).item()
            rmse_ZF_before = torch.sqrt(((ZF - self.ZF_fix).abs() ** 2).mean()).item()
            rmse_ZL_before = torch.sqrt(((ZL - self.ZL_fix).abs() ** 2).mean()).item()
            print(f"Before polishing | RMSE L1: {rmse_L1_before:.4f}, ZF: {rmse_ZF_before:.4f}, ZL: {rmse_ZL_before:.4f}")
        
        # Polishing
        if self.use_bfgs:
            bfgs_est = BFGSEstimator(fm=self.fm, likelihood=self.lik, target=self.target,
                        fixed=self.fixed, true_range=self.true_range, mode=self.mode, device=self.device,
                        bfgs_iters=10, alpha0=1.0, polishing=True, verbose=True
                        )
            U_refined = bfgs_est._batched_bfgs(U0=U, y=obs_tf, nv=noise_var)

        elif self.use_LM:
            lm_est = LMEstimator(fm=self.fm, likelihood=self.lik, target=self.target,
                        fixed=self.fixed, true_range=self.true_range, mode=self.mode, device=self.device,
                        max_iters=10
                        )
            U_refined = lm_est._batched_lm(U0=U, y=obs_tf, nv=noise_var)
        elif self.use_Newton:
            newton_est = NewtonEstimator(
                fm=self.fm, likelihood=self.lik, target=self.target,
                fixed=self.fixed, true_range=self.true_range, mode=self.mode, device=self.device,
                max_iters=10
            )
            U_refined = newton_est._batched_newton(U, obs_tf, noise_var)
        else:
            U_refined = U

        # Compute RMSE after polishing
        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_refined)
            print("L1 after", L1)
            print("ZF after", ZF)
            rmse_L1_after = torch.sqrt(((L1 - self.L1_fix) ** 2).mean()).item()
            rmse_ZF_after = torch.sqrt(((ZF - self.ZF_fix).abs() ** 2).mean()).item()
            rmse_ZL_after = torch.sqrt(((ZL - self.ZL_fix).abs() ** 2).mean()).item()
            print(f"After polishing  | RMSE L1: {rmse_L1_after:.4f}, ZF: {rmse_ZF_after:.4f}, ZL: {rmse_ZL_after:.4f}")
            print(f"Improvement      | ΔL1: {rmse_L1_before - rmse_L1_after:+.4f}, ΔZF: {rmse_ZF_before - rmse_ZF_after:+.4f}, ΔZL: {rmse_ZL_before - rmse_ZL_after:+.4f}")
            return {
                "L1": L1.cpu().numpy(),
                "ZF": ZF.cpu().numpy(),
                "ZL": ZL.cpu().numpy(),
            }



