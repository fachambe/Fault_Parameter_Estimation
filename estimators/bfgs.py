# bfgs.py
import torch
import numpy as np
from typing import Callable, Tuple
from .base import Estimator, u_to_theta_sigmoid, u_to_theta_tanh

Tensor = torch.Tensor


# ---------------------------------------------------------------------
# BFGS Estimator Class
# ---------------------------------------------------------------------
class BFGSEstimator(Estimator):
    """
    BFGS-based MLE for theta = [L1, ReZF, ImZF, ReZL, ImZL].

    Uses batched BFGS with Armijo line search for joint estimation
    of all 5 parameters.
    """
    def __init__(self,
                 fm,
                 likelihood,
                 target,
                 fixed,
                 true_range,
                 mode,
                 device="cuda",
                 bfgs_iters: int = 10,
                 alpha0: float = 1.0,  # Conservative initial step size
                 polishing: bool = True,
                 verbose: bool = True):
        super().__init__(fm, likelihood, target, fixed, true_range, mode, device)
        self.bfgs_iters = bfgs_iters
        self.alpha0 = alpha0
        self.polishing = polishing
        self.verbose = verbose


    def _armijo_line_search(self, x, f, g, p, eval_fg, c1=1e-4, shrink=0.5, max_ls=30):
        gTp = (g * p).sum(dim=-1)
        alpha = torch.full_like(f, self.alpha0)

        x_new = x
        f_new = f
        g_new = g

        active = gTp < 0

        for _ in range(max_ls):
            trial_x = x + alpha.unsqueeze(-1) * p
            trial_f, trial_g = eval_fg(trial_x)

            ok = active & (trial_f <= f + c1 * alpha * gTp)

            x_new = torch.where(ok.unsqueeze(-1), trial_x, x_new)
            f_new = torch.where(ok, trial_f, f_new)
            g_new = torch.where(ok.unsqueeze(-1), trial_g, g_new)

            active = active & ~ok
            if not active.any():
                break

            alpha = torch.where(active, alpha * shrink, alpha)

        return alpha, f_new, g_new, x_new

    def _batched_bfgs(self, U0, y, nv):
        """
        Batched full BFGS optimization for N independent observations.
        Since d=5, storing the full inverse Hessian is cheap.

        U0: [N, 5]
        y:  [N, F] complex observations
        nv: [N, F]
        Returns: U_final [N, 5]
        """
        assert U0.dim() == 2 and U0.size(-1) == 5

        N, d = U0.shape
        device = U0.device

        x = U0.clone().requires_grad_(True)

        def eval_fg(x_curr):
            x_curr = x_curr.detach().clone().requires_grad_(True)

            L1, ZF, ZL = self._u_to_theta(x_curr)
            pred_tf = self.fm.compute_H_complex(L1, ZF, ZL)
            nll = self.lik.nll_elementwise(y, pred_tf, nv)

            grad = torch.autograd.grad(
                outputs=nll,
                inputs=x_curr,
                grad_outputs=torch.ones_like(nll),
                retain_graph=False,
                create_graph=False
            )[0]

            return nll.detach(), grad.detach()

        f, g = eval_fg(x)

        # Initial inverse Hessian H0 with proper scaling
        # Scale by 1/||g|| so that first step p = -H*g has unit norm
        eye = torch.eye(d, device=device).unsqueeze(0).repeat(N, 1, 1)

        g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [N, 1]
        h_scale = 1.0 / g_norm  # [N, 1]
        H = h_scale.unsqueeze(-1) * eye  # [N, d, d] scaled identity

        for it in range(self.bfgs_iters):
            L1_old, ZF_old, ZL_old = self._u_to_theta(x)

            # Full BFGS search direction: p = -H g
            p = -torch.bmm(H, g.unsqueeze(-1)).squeeze(-1)

            # Ensure descent direction
            gTp = (g * p).sum(dim=-1)
            bad = gTp >= 0

            if bad.any():
                p = torch.where(bad.unsqueeze(-1), -g, p)

                # Reset bad Hessians to scaled identity
                g_norm_bad = g[bad].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                H[bad] = (1.0 / g_norm_bad).unsqueeze(-1) * eye[bad]

            alpha, f_new, g_new, x_new = self._armijo_line_search(
                x, f, g, p, eval_fg
            )

            s = (x_new - x).detach()
            yvec = (g_new - g).detach()

            sy = (s * yvec).sum(dim=-1)
            s_norm = s.norm(dim=-1)
            y_norm = yvec.norm(dim=-1)

            # Relative curvature condition
            valid = sy > 1e-6 * s_norm * y_norm

            if valid.any():
                idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)

                s_i = s[idx].unsqueeze(-1)       # [Nv, d, 1]
                y_i = yvec[idx].unsqueeze(-1)    # [Nv, d, 1]

                rho_i = 1.0 / sy[idx].clamp(min=1e-12)
                rho_i = rho_i.view(-1, 1, 1)

                I_i = eye[idx]
                H_i = H[idx]

                sy_outer = torch.bmm(s_i, y_i.transpose(1, 2))
                ys_outer = torch.bmm(y_i, s_i.transpose(1, 2))
                ss_outer = torch.bmm(s_i, s_i.transpose(1, 2))

                V = I_i - rho_i * sy_outer

                # Inverse BFGS update:
                # H_{k+1} = V H_k V^T + rho s s^T
                H_new = torch.bmm(torch.bmm(V, H_i), V.transpose(1, 2)) + rho_i * ss_outer

                # Symmetrize for numerical stability
                H_new = 0.5 * (H_new + H_new.transpose(1, 2))

                H[idx] = H_new

            # If curvature condition fails, reset that sample's H to scaled identity
            invalid = ~valid
            if invalid.any():
                g_norm_sq_inv = (g_new[invalid] * g_new[invalid]).sum(dim=-1, keepdim=True).clamp(min=1e-8)
                H[invalid] = (1.0 / g_norm_sq_inv).unsqueeze(-1) * eye[invalid]

            L1_new, ZF_new, ZL_new = self._u_to_theta(x_new)

            if self.verbose and it % 2 == 0:
                loss_drop = f - f_new
                grad_norm = g.norm(dim=-1)
                step_norm = s.norm(dim=-1)
                p_norm = p.norm(dim=-1)

                dL1 = (L1_new - L1_old).abs()
                dZF = (ZF_new - ZF_old).abs()
                dZL = (ZL_new - ZL_old).abs()

                print(
                    f"BFGS Step {it}/{self.bfgs_iters} | "
                    f"Loss: {f.mean().item():.4f} -> {f_new.mean().item():.4f} | "
                    f"drop mean={loss_drop.mean().item():.3e}, "
                    f"max={loss_drop.max().item():.3e}"
                )
                worse = f_new > f
                print("worse fraction:", worse.float().mean().item())
                print("max loss increase:", (f_new - f).max().item())

                print(
                    f"  alpha mean={alpha.mean().item():.3e}, "
                    f"min={alpha.min().item():.3e}, "
                    f"max={alpha.max().item():.3e}"
                )

                print(
                    "grad:",
                    torch.quantile(grad_norm, 0.0).item(),
                    torch.quantile(grad_norm, 0.25).item(),
                    torch.quantile(grad_norm, 0.50).item(),
                    torch.quantile(grad_norm, 0.75).item(),
                    torch.quantile(grad_norm, 1.00).item(),
                )
                print(
                    f"  p norm mean={p_norm.mean().item():.3e}, "
                    f"step norm mean={step_norm.mean().item():.3e}, "
                    f"max={step_norm.max().item():.3e}"
                )
                print(
                    f"  curvature valid={valid.float().mean().item()*100:.1f}% | "
                    f"sy mean={sy.mean().item():.3e}, "
                    f"min={sy.min().item():.3e}"
                )
                print(
                    f"  |ΔL1| mean={dL1.mean().item():.3e}, "
                    f"|ΔZF| mean={dZF.mean().item():.3e}, "
                    f"|ΔZL| mean={dZL.mean().item():.3e}"
                )
                print(
                    f"  L1 mean={L1_new.mean().item():.4f}, "
                    f"ZF mean={ZF_new.real.mean().item():.4f}"
                    f"{ZF_new.imag.mean().item():+.4f}j, "
                    f"ZL mean={ZL_new.real.mean().item():.4f}"
                    f"{ZL_new.imag.mean().item():+.4f}j"
                )
                gTp = (g * p).sum(dim=-1)

                print(
                    "gTp mean =", gTp.mean().item(),
                    "min =", gTp.min().item(),
                    "max =", gTp.max().item()
                )

            x, f, g = x_new.detach().requires_grad_(True), f_new, g_new

        return x.detach()

    def predict(self, obs_tf, noise_var):
        if self.mode == "1d":
            return self._predict_1d(obs_tf, noise_var)
        else:
            return self._predict_joint(obs_tf, noise_var)

    def _predict_1d(self, obs_tf, noise_var):
        """1D BFGS - not implemented, fallback to joint."""
        raise NotImplementedError("1D mode not implemented for BFGS, use mode='joint'")

    def _predict_joint(self, obs_tf, noise_var):
        """Joint BFGS optimization for all 5 parameters."""
        N = obs_tf.shape[0]

        # Initialize at center of parameter range (u=0)
        U0 = torch.zeros((N, 5), dtype=torch.float32, device=self.device)

        U_final = self._batched_bfgs(U0, obs_tf, noise_var)

        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_final)
            return {
                "L1": L1.cpu().numpy(),
                "ZF": ZF.cpu().numpy(),
                "ZL": ZL.cpu().numpy(),
            }
