# LM.py - Batched Levenberg-Marquardt Estimator
import torch
import numpy as np
from .base import Estimator

Tensor = torch.Tensor


# ---------------------------------------------------------------------
# Levenberg-Marquardt Estimator Class
# ---------------------------------------------------------------------
class LMEstimator(Estimator):
    """
    Batched Levenberg-Marquardt for least-squares MLE.

    LM solves: min sum_f |y_f - H(θ)_f|^2 / σ^2

    Uses Gauss-Newton with scaled adaptive damping (Seber & Wild, 2003):
        (J^T J + λ D²) δ = J^T r
    where D² = diag(J^T J) scales the trust region according to the
    local sensitivity of each optimization variable.
    """
    def __init__(self,
                 fm,
                 likelihood,
                 target,
                 fixed,
                 true_range,
                 mode,
                 device="cuda",
                 max_iters: int = 10,
                 lambda_init: float = 1e-3,
                 lambda_up: float = 10.0,
                 lambda_down: float = 0.1,
                 lambda_min: float = 1e-10,
                 lambda_max: float = 1e10,
                 tol: float = 1e-8,
                 verbose: bool = True):
        super().__init__(fm, likelihood, target, fixed, true_range, mode, device)
        self.max_iters = max_iters
        self.lambda_init = lambda_init
        self.lambda_up = lambda_up
        self.lambda_down = lambda_down
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.tol = tol
        self.verbose = verbose

    def _compute_residuals_and_jacobian(self, u, y, nv_sqrt):
        """
        Compute weighted residuals and Jacobian for LM.

        Args:
            u: [N, 5] unconstrained parameters
            y: [N, F] complex observations
            nv_sqrt: [N, F] sqrt of noise variance (for weighting)

        Returns:
            r: [N, 2F] real weighted residuals (real then imag)
            J: [N, 2F, 5] Jacobian dr/du
            cost: [N] sum of squared residuals
        """
        N, F = y.shape
        d = 5

        # Enable gradient tracking
        u = u.detach().clone().requires_grad_(True)

        # Forward pass
        L1, ZF, ZL = self._u_to_theta(u)
        H_pred = self.fm.compute_H_complex(L1, ZF, ZL)  # [N, F]

        # Weighted residuals: (y - H) / sqrt(var)
        diff = y - H_pred  # [N, F] complex
        r_real = diff.real / nv_sqrt  # [N, F]
        r_imag = diff.imag / nv_sqrt  # [N, F]
        r = torch.cat([r_real, r_imag], dim=-1)  # [N, 2F]

        # Cost = sum of squared residuals
        cost = (r ** 2).sum(dim=-1)  # [N]

        # Compute Jacobian using autograd
        # J[n, i, j] = dr[n, i] / du[n, j]
        J = torch.zeros(N, 2 * F, d, device=self.device)

        for i in range(2 * F):
            # Gradient of r[:, i] w.r.t. u
            grad_outputs = torch.zeros(N, 2 * F, device=self.device)
            grad_outputs[:, i] = 1.0

            grads = torch.autograd.grad(
                outputs=r,
                inputs=u,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False
            )[0]  # [N, 5]

            J[:, i, :] = grads

        return r.detach(), J.detach(), cost.detach()

    def _batched_lm(self, U0, y, nv):
        """
        Batched Levenberg-Marquardt optimization.

        Args:
            U0: [N, 5] initial unconstrained parameters
            y: [N, F] complex observations
            nv: [N, F] noise variance

        Returns:
            U_final: [N, 5] optimized parameters
        """
        N, F = y.shape

        nv_sqrt = torch.sqrt(nv)  # For weighting residuals

        x = U0.clone()
        lam = torch.full((N,), self.lambda_init, device=self.device)

        # Initial cost
        r, J, cost = self._compute_residuals_and_jacobian(x, y, nv_sqrt)

        for it in range(self.max_iters):
            if self.verbose and it % 1 == 0:
                L1, ZF, ZL = self._u_to_theta(x)
                print(f"LM Step {it}/{self.max_iters} | Cost: {cost.mean().item():.2f} | "
                      f"λ: {lam[0].item():.2e} | "
                      f"L1: est={L1[0].item():.2f}, true={self.L1_fix.item():.2f} | "
                      f"ZF: est={ZF[0].real.item():.2f}{ZF[0].imag.item():+.2f}j, true={self.ZF_fix.real.item():.2f}{self.ZF_fix.imag.item():+.2f}j | "
                      f"ZL: est={ZL[0].real.item():.2f}{ZL[0].imag.item():+.2f}j, true={self.ZL_fix.real.item():.2f}{self.ZL_fix.imag.item():+.2f}j")
                print(f" LM Step {it}/{self.max_iters} | Cost: {cost.mean().item():.2f} | "
                    f"L1 mean={L1.mean().item():.4f},"
                    f"ZF mean={ZF.real.mean().item():.4f}{ZF.imag.mean().item():+.4f}j, "
                    f"ZL mean={ZL.real.mean().item():.4f}{ZL.imag.mean().item():+.4f}j")
            # Compute LM step for each observation
            # (J^T J + λ D²) δ = J^T r  where D² = diag(J^T J)
            # Note: we want to minimize residuals, so δ = -(J^T J + λD²)^{-1} J^T r

            JTJ = torch.bmm(J.transpose(1, 2), J)  # [N, 5, 5]
            JTr = torch.bmm(J.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1)  # [N, 5]

            # Scaled trust region (Seber & Wild, 2003): D² = diag(J^T J)
            # This automatically scales regularization by local sensitivity
            D2 = torch.diagonal(JTJ, dim1=-2, dim2=-1)  # [N, 5]
            D2 = torch.clamp(D2, min=1e-10)  # Avoid zero diagonal elements

            # Add damping: J^T J + λ D²
            A = JTJ + lam.unsqueeze(-1).unsqueeze(-1) * torch.diag_embed(D2)  # [N, 5, 5]

            # Solve for step: A δ = J^T r
            # Using Cholesky for stability (A should be positive definite)
            try:
                L_chol = torch.linalg.cholesky(A)  # [N, 5, 5]
                delta = torch.cholesky_solve(JTr.unsqueeze(-1), L_chol).squeeze(-1)  # [N, 5]
            except RuntimeError:
                # Fallback to pseudo-inverse if Cholesky fails
                delta = torch.linalg.lstsq(A, JTr.unsqueeze(-1)).solution.squeeze(-1)

            # Trial step: x_new = x - (J^T J + λI)^{-1} J^T r
            # The minus sign is because we minimize ||r||² and gradient is 2 J^T r
            x_new = x - delta

            # Evaluate new cost
            r_new, J_new, cost_new = self._compute_residuals_and_jacobian(x_new, y, nv_sqrt)

            # Accept/reject step for each observation
            improved = cost_new < cost

            # Update x and cost where improved
            x = torch.where(improved.unsqueeze(-1), x_new, x)
            r = torch.where(improved.unsqueeze(-1), r_new, r)
            J = torch.where(improved.unsqueeze(-1).unsqueeze(-1), J_new, J)
            cost = torch.where(improved, cost_new, cost)

            # Update lambda
            # Decrease lambda if improved (trust more), increase if not
            lam = torch.where(improved,
                              torch.clamp(lam * self.lambda_down, min=self.lambda_min),
                              torch.clamp(lam * self.lambda_up, max=self.lambda_max))

            # Check convergence
            if (delta.abs().max() < self.tol).all():
                if self.verbose:
                    print(f"LM converged at iteration {it}")
                break

        return x

    def predict(self, obs_tf, noise_var):
        if self.mode == "1d":
            return self._predict_1d(obs_tf, noise_var)
        else:
            return self._predict_joint(obs_tf, noise_var)

    def _predict_1d(self, obs_tf, noise_var):
        """1D LM - not implemented, fallback to joint."""
        raise NotImplementedError("1D mode not implemented for LM, use mode='joint'")

    def _predict_joint(self, obs_tf, noise_var):
        """Joint LM optimization for all 5 parameters."""
        N = obs_tf.shape[0]

        # Initialize at center of parameter range (u=0)
        U0 = torch.zeros((N, 5), dtype=torch.float32, device=self.device)

        U_final = self._batched_lm(U0, obs_tf, noise_var)

        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_final)
            return {
                "L1": L1.cpu().numpy(),
                "ZF": ZF.cpu().numpy(),
                "ZL": ZL.cpu().numpy(),
            }
