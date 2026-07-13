import torch
from .base import Estimator, theta_to_u_sigmoid
torch.set_float32_matmul_precision("high")


class L1ProfileMLE(Estimator):
    """
    Joint MLE with L1 Profile grid search for initialization.

    Algorithm:
    1. Grid search over L1 with cold restarts for ZF/ZL at each grid point
    2. Select top-K L1 seeds per observation based on profile likelihood
    3. Run Adam optimization from each seed
    4. Optional BFGS polishing
    5. Pick best result per observation

    Args
    ----
    fm: ForwardModel                (provides compute_H_complex(L1, ZF, ZL) and L)
    likelihood:                     ComplexGaussianLik()
    num_grid_points: int            Number of L1 grid points for profile
    topk: int                       Number of top L1 seeds to refine
    inner_steps: int                Adam steps per grid point for ZF/ZL optimization
    inner_lr: float                 Learning rate for inner ZF/ZL optimization
    adam_steps: int                 Adam steps for final joint optimization
    adam_lr: float                  Learning rate for final Adam
    """
    def __init__(self,
                 fm,
                 likelihood,
                 target,
                 fixed,
                 true_range,
                 mode,
                 device="cuda",
                 num_grid_points: int = 250,
                 topk: int = 3,
                 inner_steps: int = 50,
                 inner_lr: float = 1e-2,
                 adam_steps: int = 5000,
                 adam_lr: float = 1e-2,
                 adam_betas: tuple = (0.7, 0.9),
                 use_bfgs: bool = False,
                 bfgs_steps: int = 60,
                 bfgs_lr: float = 1.0,
                 verbose: bool = True
                 ):
        super().__init__(fm, likelihood, target, fixed, true_range, mode, device)

        # L1 profile grid settings
        self.G = num_grid_points
        self.topk = topk
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

        # Adam hyperparameters
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.adam_betas = adam_betas
        self.verbose = bool(verbose)

        # BFGS hyperparameters
        self.use_bfgs = bool(use_bfgs)
        self.bfgs_steps = int(bfgs_steps)
        self.bfgs_lr = float(bfgs_lr)

    def _profile_cold(self, obs_tf, noise_var):
        """
        Profile L1 with cold restarts: re-initialize ZF/ZL at each grid point.

        Args:
            obs_tf: [N, F] complex observations
            noise_var: [N, F] or [N, 1] noise variance

        Returns:
            L1_top: [N, topk] - top-K L1 values per observation
            uZ_top: [N, topk, 4] - corresponding uZ snapshots
        """
        N, _ = obs_tf.shape
        dev = self.device
        G = self.G

        # L1 grid from range
        L1_grid = torch.linspace(self.L1_lo, self.L1_hi, G, device=dev, dtype=torch.float32)

        # Storage
        nll_track = torch.empty(N, G, device=dev, dtype=torch.float32)
        uZ_snap = torch.empty(N, G, 4, device=dev, dtype=torch.float32)

        for gi in range(G):
            # Map L1 grid value to u-space
            u0 = theta_to_u_sigmoid(L1_grid[gi], self.L1_lo, self.L1_hi).expand(N, 1)  # [N, 1]

            # Cold restart: re-initialize uZ and optimizer at each grid point
            uZ = torch.nn.Parameter(torch.zeros((N, 4), dtype=torch.float32, device=dev)) #[N, 4]
            opt = torch.optim.Adam([uZ], lr=self.inner_lr)

            # Optimize ZF/ZL with L1 fixed to grid value
            for t in range(self.inner_steps):
                opt.zero_grad()
                U = torch.cat([u0, uZ], dim=1)  # [N, 5]
                L1, ZF, ZL = self._u_to_theta(U)
                # Forward model: [N, F]
                pred_tf = self.fm.compute_H_complex(L1, ZF, ZL)

                # Compute NLL per observation [N]
                nll_per_obs = self.lik.nll_elementwise(obs_tf, pred_tf, noise_var)

                # Backprop and step
                loss = nll_per_obs.mean()
                loss.backward()
                opt.step()

            # Log progress
            if self.verbose:
                with torch.no_grad():
                    U = torch.cat([u0, uZ], dim=1)
                    L1, ZF, ZL = self._u_to_theta(U)
                    print(f"[Grid {gi}/{G}] L1={L1_grid[gi]:.2f}  "
                          f"ZF={ZF.real.mean():.1f}{ZF.imag.mean():+.1f}j  "
                          f"ZL={ZL.real.mean():.1f}{ZL.imag.mean():+.1f}j  "
                          f"loss={loss.item():.2f}")

            # Record NLL and uZ snapshot
            with torch.no_grad():
                U = torch.cat([u0, uZ], dim=1)
                L1, ZF, ZL = self._u_to_theta(U)
                H = self.fm.compute_H_complex(L1, ZF, ZL)
                diff = obs_tf - H
                nll_track[:, gi] = ((diff.abs() ** 2) / noise_var).sum(dim=1)
                uZ_snap[:, gi, :] = uZ.detach()

        # Select top-K grid points per observation (minimum NLL)
        _, top_idx = torch.topk(-nll_track, k=self.topk, dim=1)  # [N, topk]
        L1_top = L1_grid[top_idx]  # [N, topk]

        # Debug output
        if self.verbose:
            col0 = L1_top[:, 0]
            print(f"L1 profile: best seeds range should be all near 250 for all observe [{col0.min():.1f}, {col0.max():.1f}]")

        # Gather uZ for top-K indices
        arN = torch.arange(N, device=dev)
        uZ_top = uZ_snap[arN[:, None], top_idx, :]  # [N, topk, 4]

        return L1_top, uZ_top

    def _profile_warm(self, obs_tf, noise_var):
        """
        Profile L1 with warm restarts: ZF/ZL carry over between grid points.

        Args:
            obs_tf: [N, F] complex observations
            noise_var: [N, F] or [N, 1] noise variance

        Returns:
            L1_top: [N, topk] - top-K L1 values per observation
            uZ_top: [N, topk, 4] - corresponding uZ snapshots
        """
        N, F = obs_tf.shape
        dev = self.device
        G = self.G

        L1_grid = torch.linspace(self.L1_lo, self.L1_hi, G, device=dev, dtype=torch.float32)

        # Warm restart: initialize once before loop
        uZ = torch.nn.Parameter(torch.zeros((N, 4), dtype=torch.float32, device=dev)) 
        opt = torch.optim.Adam([uZ], lr=self.inner_lr)

        nll_track = torch.empty(N, G, device=dev, dtype=torch.float32)
        uZ_snap = torch.empty(N, G, 4, device=dev, dtype=torch.float32)

        for gi in range(G):
            u0 = theta_to_u_sigmoid(L1_grid[gi], self.L1_lo, self.L1_hi).expand(N, 1)

            # Optimize ZF/ZL (optimizer state carries over)
            for t in range(self.inner_steps):
                opt.zero_grad()
                U = torch.cat([u0, uZ], dim=1)
                L1, ZF, ZL = self._u_to_theta(U)
                H = self.fm.compute_H_complex(L1, ZF, ZL)
                diff = obs_tf - H
                nll = ((diff.abs() ** 2) / noise_var).sum(dim=1)
                loss = nll.mean()
                loss.backward()
                opt.step()

            # Record NLL and uZ snapshot
            with torch.no_grad():
                U = torch.cat([u0, uZ], dim=1)
                L1, ZF, ZL = self._u_to_theta(U)
                H = self.fm.compute_H_complex(L1, ZF, ZL)
                diff = obs_tf - H
                nll_track[:, gi] = ((diff.abs() ** 2) / noise_var).sum(dim=1)
                uZ_snap[:, gi, :] = uZ.detach()
            if self.verbose:
                with torch.no_grad():
                    U = torch.cat([u0, uZ], dim=1)
                    L1, ZF, ZL = self._u_to_theta(U)
                    print(f"[Grid {gi}/{G}] L1={L1_grid[gi]:.2f}  "
                          f"ZF={ZF.real.mean():.1f}{ZF.imag.mean():+.1f}j  "
                          f"ZL={ZL.real.mean():.1f}{ZL.imag.mean():+.1f}j  "
                          f"loss={loss.item():.2f}")

        # Select top-K
        _, top_idx = torch.topk(-nll_track, k=self.topk, dim=1)
        L1_top = L1_grid[top_idx]

        arN = torch.arange(N, device=dev)
        uZ_top = uZ_snap[arN[:, None], top_idx, :]

        return L1_top, uZ_top

    def predict(self, obs_tf, noise_var):
        """
        Jointly estimate [L1, ZF, ZL] using L1 profile + multi-start Adam.

        Args:
            obs_tf: [N, F] complex tensor
            noise_var: [N, F] float tensor

        Returns:
            dict with "L1": [N], "ZF": [N] complex, "ZL": [N] complex
        """
        N, _ = obs_tf.shape

        if self.verbose:
            print("Running L1 Profile (cold restarts)...")

        # 1) Profile L1 grid
        L1_top, uZ_top = self._profile_warm(obs_tf, noise_var)
        print("L1 top", L1_top)
        # adad
        # L1_top, uZ_top = self._profile_cold(obs_tf, noise_var)
        # print("L1_top", L1_top)
        # adad
        if self.verbose:
            print(f"Seeding from top-{self.topk} L1 values per observation...")

        # 2) Build initial U from top-K seeds: [N, K, 5]
        u0 = theta_to_u_sigmoid(L1_top, self.L1_lo, self.L1_hi).unsqueeze(-1)  # [N, K, 1]
        U = torch.cat([u0, uZ_top], dim=-1)  # [N, K, 5]

        if self.verbose:
            print("Running Adam optimization...")

        # 3) Adam refinement
        U = torch.nn.Parameter(U)
        opt = torch.optim.Adam([U], lr=self.adam_lr)

        for step in range(self.adam_steps):
            opt.zero_grad()
            L1, ZF, ZL = self._u_to_theta(U)  # [N, K] each
            H = self.fm.compute_H_complex(L1, ZF, ZL)  # [N, K, F]
            diff = obs_tf.unsqueeze(1) - H  # [N, 1, F] - [N, K, F]
            nll = ((diff.abs() ** 2) / noise_var.unsqueeze(1)).sum(dim=-1)  # [N, K]
            loss = nll.mean()
            loss.backward()
            opt.step()

            if step % 500 == 0:
                # L1, ZF, ZL have shape [N, K] - show mean across N for best seed (k=0)
                print(
                    f"Adam Step {step}/{self.adam_steps} | "
                    f"Loss: {loss.item():.2f} | "
                    f"L1[:,0] mean={L1[:, 0].mean().item():.2f}, true={self.L1_fix.item():.2f} | "
                    f"ZF[:,0] mean={ZF[:, 0].real.mean().item():.2f}{ZF[:, 0].imag.mean().item():+.2f}j, "
                    f"true={self.ZF_fix.real.item():.2f}{self.ZF_fix.imag.item():+.2f}j | "
                    f"ZL[:,0] mean={ZL[:, 0].real.mean().item():.2f}{ZL[:, 0].imag.mean().item():+.2f}j, "
                    f"true={self.ZL_fix.real.item():.2f}{self.ZL_fix.imag.item():+.2f}j"
                )



        # 4) Optional BFGS polish
        # if self.use_bfgs and self.bfgs_steps > 0:
        #     if self.verbose:
        #         print("Running BFGS polishing...")
        #     U_refined = batched_bfgs(
        #         U0=U, y=obs_tf, nv=noise_var,
        #         fm=self.fm, u_to_theta=self._u_to_theta,
        #         history_size=12, iters=self.bfgs_steps, alpha0=self.bfgs_lr
        #     )
        # else:
        #     U_refined = U.detach()
        U_refined = U.detach() 
        # 5) Pick best seed per observation
        with torch.no_grad():
            L1, ZF, ZL = self._u_to_theta(U_refined)  # [N, K]
            H = self.fm.compute_H_complex(L1, ZF, ZL)  # [N, K, F]
            diff = obs_tf.unsqueeze(1) - H
            final_nll = ((diff.abs() ** 2) / noise_var.unsqueeze(1)).sum(dim=-1)  # [N, K]

            i_best = torch.argmin(final_nll, dim=1)  # [N]
            arN = torch.arange(N, device=self.device)
            U_best = U_refined[arN, i_best, :]  # [N, 5]

        # 6) Return best estimates
        L1_best, ZF_best, ZL_best = self._u_to_theta(U_best)
        print("L1 after", L1)
        print("ZF after", ZF)
        return {
            "L1": L1_best.cpu().numpy(),
            "ZF": ZF_best.cpu().numpy(),
            "ZL": ZL_best.cpu().numpy(),
        }
