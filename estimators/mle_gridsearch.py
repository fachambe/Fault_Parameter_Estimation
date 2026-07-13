# estimators/mle_gridsearch.py
import torch
from estimators.base import Estimator

class GridSearchMLE(Estimator):
    def __init__(self, fm, likelihood, grid, target, fixed, device):
        """
        grid:  [K] float candidates for the target parameter
        """
        super().__init__(fm, likelihood, target, fixed, device=device)
        self.grid = grid

    @torch.no_grad()
    def predict(self, obs_tf, noise_var):
        """
        obs_tf:    [N, F] complex64 tensor
        noise_var: [N, 1] or [N, F] float32 tensor
        Returns:   dict with key=target, value=numpy array [N] of estimates
        """
        K = self.grid.numel()

        # Build parameter arrays [K] for all candidates
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

        # Forward model: H[K, F]
        H = self.fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL)

        # NLL matrix: [K, N]
        nll_KN = self.lik.nll_matrix(obs_tf, H, noise_var)

        # Best candidate per observation (lowest NLL)
        best_idx = nll_KN.argmin(dim=0)  # [N]
        best_vals = self.grid[best_idx].detach().cpu().numpy()

        return {self.target: best_vals}
