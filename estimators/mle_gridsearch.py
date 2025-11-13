# estimators/mle.py
import torch
from estimators.base import Estimator

class GridSearchMLE(Estimator):
    def __init__(self, forward_model, likelihood, grid,
                 target, fixed, device,
                 cand_batch=4096, obs_batch=64, estimate=None):
        """
        target: "L1" | "ZF" | "ZL"
        fixed: dict of fixed params, e.g. {"ZF": 1000-5j, "ZL": 100-5j, "L1": 500.0}
        grid:  [K] candidates (float for L1, complex for ZF/ZL)
               - float for L1
               - float for Re/Im when estimate="real"/"imag" for ZF/ZL
               - complex if estimate is None for ZF/ZL 
        estimate: None | "real" | "imag"
            - For ZF/ZL: choose which component to estimate. Other component held fixed.
        """
        self.fm, self.lik = forward_model, likelihood   # likelihood instance
        self.target = target
        self.fixed = fixed
        self.device = device
        self.grid = grid
        self.cand_batch_size = int(cand_batch) #[c] 
        self.obs_batch_size  = int(obs_batch) #[n] N = 1000 per SNR usually
        self.estimate = estimate  # None/"real"/"imag"

        fx = {k: (complex(fixed[k]["re"], fixed[k]["im"]) if isinstance(fixed[k], dict) else fixed[k])
              for k in fixed}
        self.L1_fix = torch.as_tensor(fx["L1"], device=self.device, dtype=torch.float32)
        self.ZF_fix = torch.as_tensor(fx["ZF"], device=self.device, dtype=torch.cfloat)
        self.ZL_fix = torch.as_tensor(fx["ZL"], device=self.device, dtype=torch.cfloat)

    @torch.no_grad()
    def predict(self, obs_tf, noise_var):
        """
        obs_tf:      [N,F] complex64 tensor
        noise_var: [N,F] float32 tensor
         Returns:     numpy array of estimates for parameter [N]
            - float32 for L1 and component-only ZF/ZL
            - complex64 for full ZF/ZL
        """

        N, F = obs_tf.shape
        K = self.grid.numel()
        device = self.device

        best_ll  = torch.full((N,), -float("inf"), device=device)
        best_idx = torch.zeros(N, dtype=torch.long, device=device)

        for s in range(0, K, self.cand_batch_size):
            cand = self.grid[s:s+self.cand_batch_size]         # [c]
            c = cand.numel()           # [c]
            # Build parameter triplets [c] for this batch
            if self.target == "L1":
                L1 = cand.to(dtype=torch.float32, device=device)
                ZF = self.ZF_fix.expand(c) 
                ZL = self.ZL_fix.expand(c)
            elif self.target == "ZF":
                L1 = self.L1_fix.expand(c)
                if self.estimate == "real":
                    ZF = torch.complex(cand, self.ZF_fix.imag.expand(c).to(torch.float32))
                elif self.estimate == "imag":
                    ZF = torch.complex(self.ZF_fix.real.expand(c).to(torch.float32), cand)
                else:  # full complex
                    ZF = cand.to(dtype=torch.cfloat, device=device)
                ZL = self.ZL_fix.expand(c)
            else: #ZL
                L1 = self.L1_fix.expand(c)
                ZF = self.ZF_fix.expand(c)
                if self.estimate == "real":
                    ZL = torch.complex(cand, self.ZL_fix.imag.expand(c).to(torch.float32))
                elif self.estimate == "imag":
                    ZL = torch.complex(self.ZL_fix.real.expand(c).to(torch.float32), cand)
                else:  # full complex
                    ZL = cand.to(dtype=torch.cfloat, device=device)
            
            # Forward: H[c,F]
            H = self.fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL)  # [c,F] cfloat
            for t in range(0, N, self.obs_batch_size):
                obs_batch = obs_tf[t:t+self.obs_batch_size] #[n,F] take first n observations from N observations per SNR
                var_batch = noise_var[t:t+self.obs_batch_size] #[n, F]
                ll_c_n = self.lik.score_matrix(obs_batch, H, var_batch) #scores per (candidate, obs) -> [c, n]
                # best candidate per obs in this chunk
                ll_max, idx_local = ll_c_n.max(dim=0)        # [n]
                # update global bests
                cur = slice(t, t+obs_batch.shape[0])
                take = ll_max > best_ll[cur] #update n with these c 
                best_ll[cur]  = torch.where(take, ll_max, best_ll[cur]) #Updates the running best scores only where take is True
                best_idx[cur] = torch.where(take, s + idx_local, best_idx[cur]) #Update the running best indices only where take is True
        
        return self.grid[best_idx].detach().cpu().numpy()    # [N] best candidates from grid for all N observations