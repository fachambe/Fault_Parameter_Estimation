# Newton.py - Modified Newton's Method with Eigenvalue Modification
import torch
from .base import Estimator
from torch.func import jacfwd, hessian, vmap, grad

class NewtonEstimator(Estimator):
    """
    Newton's method with eigenvalue modification for MLE.

    When the Hessian is not positive definite (has negative eigenvalues),
    we modify it via spectral decomposition:
        H = Q Λ Q^T
        B = Q Λ_mod Q^T, where Λ_mod replaces λ_i < δ with δ

    This ensures p = -B^{-1} ∇f is always a descent direction.

    Reference: Nocedal & Wright, "Numerical Optimization", Algorithm 3.2
    """
    def __init__(self,
                 fm,
                 likelihood,
                 target,
                 fixed,
                 true_range,
                 mode,
                 device="cuda",
                 max_iters: int = 10 ,
                 delta: float = 1e-4,  # Minimum eigenvalue threshold
                 alpha0: float = 1.0,  # Initial step size for line search
                 c1: float = 1e-4,     # Armijo condition constant
                 shrink: float = 0.5,  # Line search shrink factor
                 max_ls: int = 30,     # Max line search iterations
                 tol: float = 1e-8,    # Gradient norm tolerance
                 verbose: bool = True):
        super().__init__(fm, likelihood, target, fixed, true_range, mode, device)
        self.max_iters = max_iters
        self.delta = delta
        self.alpha0 = alpha0
        self.c1 = c1
        self.shrink = shrink
        self.max_ls = max_ls
        self.tol = tol
        self.verbose = verbose

    def _compute_batched_hessians(self, U, obs_tf, noise_var):
        """
        Compute Hessians, gradients, and NLL for all N observations.

        Args:
            U: [N, 5] unconstrained parameters
            obs_tf: [N, F] complex observations
            noise_var: [N, F] noise variance

        Returns:
            hess: [N, 5, 5] Hessian matrices
            grad: [N, 5] gradient vectors
            f: [N] function values (NLL)
        """
        def single_nll_vmap(u, obs_re, obs_im, nv):
            """NLL for single observation. u: [5], obs_re/obs_im: [F], nv: [F] -> scalar"""
            obs = torch.complex(obs_re, obs_im)
            L1, ZF, ZL = self._u_to_theta(u)
            pred = self.fm.compute_H_complex(L1, ZF, ZL).squeeze(0)  # [F]
            diff = obs - pred
            return (diff.abs()**2 / nv).sum()
        
        # Batch dimension is axis 0 for all inputs
        in_dims = (0, 0, 0, 0)

        # Function values: f(u) -> scalar, vmapped -> [N]
        batched_f = vmap(single_nll_vmap, in_dims=in_dims)
        f = batched_f(U, obs_tf.real, obs_tf.imag, noise_var).to(torch.float32)  # [N]

        # Gradients: grad(f)(u) -> [5], vmapped -> [N, 5]
        batched_grad = vmap(grad(single_nll_vmap), in_dims=in_dims)
        g = batched_grad(U, obs_tf.real, obs_tf.imag, noise_var).to(torch.float32)  # [N, 5]

        # Hessians: hessian(f)(u) -> [5, 5], vmapped -> [N, 5, 5]
        batched_hessian = vmap(hessian(single_nll_vmap), in_dims=in_dims)
        hess = batched_hessian(U, obs_tf.real, obs_tf.imag, noise_var)  # [N, 5, 5]
        hess = 0.5 * (hess + hess.transpose(-1, -2)).to(torch.float32)  # symmetrize

        return hess, g, f

    def _modify_hessians_batched(self, hess):
        """
        Apply eigenvalue modification to ensure positive definiteness.

        H = Q Λ Q^T
        Replace λ_i < δ with δ
        B = Q Λ_modified Q^T

        Args:
            hess: [N, 5, 5] Hessian matrices

        Returns:
            B: [N, 5, 5] modified positive definite matrices
            eigvals: [N, 5] original eigenvalues (for diagnostics)
            num_modified: number of observations with negative eigenvalues
        """
        # Batched eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(hess)  # [N, 5], [N, 5, 5]
        # Count how many have negative eigenvalues
        num_neg = (eigvals < 0).any(dim=-1).sum().item()
        print("number of negative eigenvalues", num_neg)
        # Replace small/negative eigenvalues with delta
        eigvals_mod = torch.clamp(eigvals, min=self.delta)

        # Reconstruct: B = Q diag(Λ_mod) Q^T
        # Using batch matrix multiplication: B[n] = Q[n] @ diag(λ_mod[n]) @ Q[n]^T
        B = torch.bmm(
            eigvecs * eigvals_mod.unsqueeze(1),  # [N, 5, 5] * [N, 1, 5] -> [N, 5, 5]
            eigvecs.transpose(1, 2)
        )  # Ensure same dtype as input (eigh may return float64)
        return B, eigvals, num_neg

    def _armijo_line_search_batched(self, x, f, g, p, obs_tf, noise_var):
        """
        Batched Armijo backtracking line search.

        Args:
            x: [N, 5] current parameters
            f: [N] current function values
            g: [N, 5] gradients
            p: [N, 5] search directions
            obs_tf: [N, F] observations
            noise_var: [N, F] noise variance

        Returns:
            alpha: [N] step sizes
            f_new: [N] new function values
            x_new: [N, 5] new parameters
        """
        N = x.shape[0]
        gTp = (g * p).sum(dim=-1)  # [N]

        alpha = torch.full((N,), self.alpha0, device=self.device)
        x_new = x.clone()
        f_new = f.clone()

        # Active set: observations still searching
        active = gTp < 0  # Only search if descent direction

        for _ in range(self.max_ls):
            if not active.any():
                break

            # Trial step
            trial_x = x + alpha.unsqueeze(-1) * p

            # Evaluate function at trial points
            with torch.no_grad():
                L1, ZF, ZL = self._u_to_theta(trial_x)
                pred = self.fm.compute_H_complex(L1, ZF, ZL)
                trial_f = self.lik.nll_elementwise(obs_tf, pred, noise_var)

            # Check Armijo condition: f(x + αp) ≤ f(x) + c1 α g^T p
            ok = active & (trial_f <= f + self.c1 * alpha * gTp)

            # Update accepted steps
            x_new = torch.where(ok.unsqueeze(-1), trial_x, x_new)
            f_new = torch.where(ok, trial_f, f_new)

            # Shrink step size for remaining active observations
            active = active & ~ok
            alpha = torch.where(active, alpha * self.shrink, alpha)

        return alpha, f_new, x_new

    def _batched_newton(self, U0, obs_tf, noise_var):
        """
        Batched modified Newton optimization.

        Args:
            U0: [N, 5] initial parameters
            obs_tf: [N, F] complex observations
            noise_var: [N, F] noise variance

        Returns:
            U_final: [N, 5] optimized parameters
        """
        N = U0.shape[0]
        x = U0.clone()

        for it in range(self.max_iters):
            # Compute Hessians and gradients for all observations
            hess, grad, f = self._compute_batched_hessians(x, obs_tf, noise_var)

            # Check convergence
            grad_norm = grad.norm(dim=-1)
            converged = grad_norm < self.tol

            if converged.all():
                if self.verbose:
                    print(f"Newton converged at iteration {it}")
                break

            # Modify Hessian to ensure positive definiteness
            B, eigvals, num_neg = self._modify_hessians_batched(hess)

            # Full Newton direction: p = -H^{-1} @ grad
            # Using Cholesky solve for stability (B is positive definite after modification)
            try:
                L_chol = torch.linalg.cholesky(B)  # [N, 5, 5]
                p = -torch.cholesky_solve(grad.unsqueeze(-1), L_chol).squeeze(-1)  # [N, 5]
            except RuntimeError:
                # Fallback to lstsq if Cholesky fails
                p = -torch.linalg.lstsq(B, grad.unsqueeze(-1)).solution.squeeze(-1)


            # Line search with Armijo condition
            alpha, f_new, x_new = self._armijo_line_search_batched(
                x, f, grad, p, obs_tf, noise_var
            )

            # NLL reduction (what Newton actually optimizes)
            nll_reduced = (f_new < f).sum().item()
            nll_delta = (f.mean() - f_new.mean()).item()

            if self.verbose and it % 1 == 0:
                print(f"Newton iter {it} | NLL: {f.mean():.2f} -> {f_new.mean():.2f} (Δ={nll_delta:+.2f}) | {nll_reduced}/{N} reduced")
            x = x_new
 

        return x

    def predict(self, obs_tf, noise_var):
        if self.mode == "1d":
            return self._predict_1d(obs_tf, noise_var)
        else:
            return self._predict_joint(obs_tf, noise_var)

    def _predict_1d(self, obs_tf, noise_var):
        """1D Newton - not implemented."""
        raise NotImplementedError("1D mode not implemented for Newton, use mode='joint'")

    def _predict_joint(self, obs_tf, noise_var):
        """Joint Newton optimization for all 5 parameters."""
        N = obs_tf.shape[0]

        # Initialize at center of parameter range (u=0)
        U0 = torch.zeros((N, 5), dtype=torch.float32, device=self.device)

        U_final = self._batched_newton(U0, obs_tf, noise_var)

        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_final)
            return {
                "L1": L1.cpu().numpy(),
                "ZF": ZF.cpu().numpy(),
                "ZL": ZL.cpu().numpy(),
            }
