# estimators/mle_gridsearch.py
import torch
import numpy as np
from estimators.base import Estimator

class GridSearchMLE(Estimator):
    def __init__(self, fm, likelihood, grid, target, fixed, device, batch_size=1000):
        """
        Grid search MLE estimator with batching for memory efficiency.

        Args:
            fm: Forward model
            likelihood: Likelihood function
            grid: [K] float candidates for the target parameter
            target: Parameter to estimate ("L1", "ZF_re", etc.)
            fixed: Dict with fixed parameter values
            device: torch device
            batch_size: Number of observations to process at once (default 500)
        """
        super().__init__(fm, likelihood, target, fixed, device=device)
        self.grid = grid
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, obs_tf, noise_var):
        """
        Estimate target parameter for each observation using grid search.

        Args:
            obs_tf: [N, F] complex64 tensor of observations
            noise_var: [N, 1] or [N, F] float32 tensor of noise variances

        Returns:
            dict with key=target, value=numpy array [N] of estimates
        """
        K = self.grid.numel()
        N = obs_tf.shape[0]

        # Build parameter arrays [K] for all candidates (done once)
        if self.target == "L1":
            L1 = self.grid.to(dtype=torch.float32, device=self.device)
            ZF = self.ZF_fix.expand(K)
            ZL = self.ZL_fix.expand(K)
        elif self.target == "ZF_re":
            L1 = self.L1_fix.expand(K)
            ZF = torch.complex(self.grid, self.ZF_fix.imag.expand(K).to(torch.float32))
            ZL = self.ZL_fix.expand(K)
        elif self.target == "ZF_im":
            L1 = self.L1_fix.expand(K)
            ZF = torch.complex(self.ZF_fix.real.expand(K).to(torch.float32), self.grid)
            ZL = self.ZL_fix.expand(K)
        elif self.target == "ZL_re":
            L1 = self.L1_fix.expand(K)
            ZF = self.ZF_fix.expand(K)
            ZL = torch.complex(self.grid, self.ZL_fix.imag.expand(K).to(torch.float32))
        elif self.target == "ZL_im":
            L1 = self.L1_fix.expand(K)
            ZF = self.ZF_fix.expand(K)
            ZL = torch.complex(self.ZL_fix.real.expand(K).to(torch.float32), self.grid)
        else:
            raise ValueError(f"Unknown target: {self.target}")

        # Forward model: H[K, F] - computed once for all observations
        H = self.fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL)

        # Process observations in batches to avoid OOM
        best_vals = np.empty(N, dtype=np.float32)

        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            obs_batch = obs_tf[start:end]  # [B, F]
            var_batch = noise_var[start:end]  # [B, 1] or [B, F]

            # NLL matrix: [K, B]
            nll_KB = self.lik.nll_matrix(obs_batch, H, var_batch)

            # Best candidate per observation in batch
            best_idx = nll_KB.argmin(dim=0)  # [B]
            best_vals[start:end] = self.grid[best_idx].cpu().numpy()

        return {self.target: best_vals}
