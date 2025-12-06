# estimators/elbo1d.py
import torch
from torch import distributions as dist
import numpy as np
import math
from estimators.base import Estimator

def _range_to_sigmoid_x(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    # Inverse of _sigmoid_to_range (for seeding)
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)  # avoid 0/1
    return torch.log(z) - torch.log1p(-z)  # logit

class ELBOArgmaxMu1D(Estimator):
    """
    Single-parameter VI via ELBO argmax over μ (sigmoid-Normal guide on x∈(0,1),
    affine-mapped to θ∈[lo,hi]). Returns posterior-median θ per observation.
    Works for L1 or a single real/imaginary component of ZF/ZL.
    M = # of Monte Carlo samples for each mu in mu_grid of size [G] from variational distribution q 
    K = # of Monte Carlo samples for the posterior to calculate posterior mean (which is our estimate for the parameter)
    """
    
    def __init__(self, fm, target, estimate, L1_grid, fixed_sigma, fixed,
                 M, K, device=None):
        self.device = device
        self.fm = fm
        self.target = str(target).upper()
        self.estimate = estimate  # None for L1; "real"/"imag" for ZF/ZL
        self.L1_grid = L1_grid
        self.fixed_sigma = float(fixed_sigma)
        self.lo, self.hi = 1.0, 1000.0
        self.M, self.K = int(M), int(K)

        fx = {k: (complex(fixed[k]["re"], fixed[k]["im"]) if isinstance(fixed[k], dict) else fixed[k])
              for k in fixed}
        self.L1_fix = torch.as_tensor(fx["L1"], device=self.device, dtype=torch.float32)
        self.ZF_fix = torch.as_tensor(fx["ZF"], device=self.device, dtype=torch.cfloat)
        self.ZL_fix = torch.as_tensor(fx["ZL"], device=self.device, dtype=torch.cfloat)

    @torch.no_grad()
    def predict(self, obs_tf, noise_var):
        """
        obs_tf:    [F] or [N,F] (torch.cfloat)
        noise_var: [F] or [N,F] (torch.float32) 
        returns:   np.ndarray [N] (float32) of posterior means of θ
        """
        obs_tf   = self._ensure_batch(obs_tf).to(self.device)
        noise_var = self._ensure_batch(noise_var).to(self.device, dtype=torch.float32)
        N, F = obs_tf.shape

        out = np.empty(N, dtype=np.float32)
        mu_grid = _range_to_sigmoid_x(self.L1_grid, 1.0, 1000.0)
        G = mu_grid.numel() #G = num of elements in mu grid
        base = dist.Normal(mu_grid, self.fixed_sigma)  #[G]
        q = dist.TransformedDistribution(
           base,
           [
             dist.transforms.SigmoidTransform(),
             dist.transforms.AffineTransform(loc=self.lo, scale=(self.hi - self.lo))
           ]
      )

        samples = q.rsample((self.M,))           # [M,G], now these are theta directly
        theta   = samples.transpose(0, 1)        # [G,M]
        log_q   = q.log_prob(samples).transpose(0, 1)  # [G,M], this is log q_theta(theta)

        # ---------- forward ONCE (independent of n) ----------
        flat = theta.reshape(-1)                                           # [G*M]
        if self.target == "L1":
            L1 = flat.to(torch.float32)
            ZF = self.ZF_fix.expand(flat.numel())
            ZL = self.ZL_fix.expand(flat.numel())
        elif self.target == "ZF":
            L1 = self.L1_fix.expand(flat.numel())
            if self.estimate == "real":
                ZF = torch.complex(flat.to(torch.float32), self.ZF_fix.imag.expand(flat.numel()))
            else:
                ZF = torch.complex(self.ZF_fix.real.expand(flat.numel()), flat.to(torch.float32))
            ZL = self.ZL_fix.expand(flat.numel())
        else:  # "ZL"
            L1 = self.L1_fix.expand(flat.numel())
            ZF = self.ZF_fix.expand(flat.numel())
            if self.estimate == "real":
                ZL = torch.complex(flat.to(torch.float32), self.ZL_fix.imag.expand(flat.numel()))
            else:
                ZL = torch.complex(self.ZL_fix.real.expand(flat.numel()), flat.to(torch.float32))

        H = self.fm.compute_H_complex(L1=L1, ZF=ZF, ZL=ZL).reshape(G, self.M, F)  # [G,M,F]
        #print("H", H)
        #print("H shape should be [300, 500, 500]", H.shape)
        H_real = torch.view_as_real(H).to(torch.float32)                           # [G,M,F,2]

        elbo_debug = None  # ELBO(G) for obs 0
        ll_debug   = None  # LL(G)   for obs 0
        for n in range(N):
            obs_real = torch.view_as_real(obs_tf[n]).to(torch.float32)            # [F,2]
            sigma = torch.sqrt(noise_var[n] / 2.0).to(torch.float32)              # [F]
            sigma = sigma.unsqueeze(-1).expand(F, 2)                              # [F,2]
            
            diff  = H_real - obs_real.unsqueeze(0).unsqueeze(0)                      # [G,M,F,2] - [1,1,F,2]
            var2  = (sigma * sigma).unsqueeze(0).unsqueeze(0)                        # [1,1,F,2]
            log_lik = -0.5 * (diff*diff/var2 + math.log(2*math.pi) + torch.log(var2)).sum(dim=(2,3)) #[G,M] sum over [F,2]
            # dist_obs = dist.Independent(dist.Normal(H_real, sigma), 2)
            # log_lik  = dist_obs.log_prob(obs_real)                                # [G,M]
            elbo = (log_lik - log_q).mean(dim=1)                              # [G] avg over [M] Monte carlo for each grid point

            if n == 0:
                elbo_debug = elbo.detach().cpu()

            k = int(torch.argmax(elbo).item())
            mu_star  = mu_grid[k]

            # Posterior median
            x_med = torch.sigmoid(mu_star)     
            theta_med = self.lo + (self.hi - self.lo) * x_med         
            out[n] = theta_med.item()

            #Posterior mean
            # base_star = dist.Normal(mu_star, self.fixed_sigma)
            # z = base_star.rsample((self.K,)).to(self.device)                      # [K]
            # x_star = torch.sigmoid(z).clamp_(1e-6, 1-1e-6)                        # [K]
            # theta_samples = self.lo + (self.hi - self.lo) * x_star                # [K]
            # out[n] = theta_samples.mean().item()
        
        # with torch.no_grad():
        #     L1_grid = self.L1_grid.to(self.device).to(torch.float32)  # [G]
        #     ZF_grid = self.ZF_fix.expand(G)
        #     ZL_grid = self.ZL_fix.expand(G)

        #     H_mle = self.fm.compute_H_complex(L1=L1_grid, ZF=ZF_grid, ZL=ZL_grid)  # [G,F]
        #     H_mle_real = torch.view_as_real(H_mle).to(torch.float32)               # [G,F,2]

        #     obs0_real = torch.view_as_real(obs_tf[0]).to(torch.float32)           # [F,2]
        #     sigma0 = torch.sqrt(noise_var[0] / 2.0).to(torch.float32)             # [F]
        #     sigma0 = sigma0.unsqueeze(-1).expand(F, 2)                            # [F,2]

        #     diff0 = H_mle_real - obs0_real.unsqueeze(0)                           # [G,F,2]
        #     var2_0 = (sigma0 * sigma0).unsqueeze(0)                               # [1,F,2]

        #     loglik_grid = -0.5 * (
        #         diff0 * diff0 / var2_0 + math.log(2 * math.pi) + torch.log(var2_0)
        #     ).sum(dim=(1, 2))                                                     # [G]

        #     ll_debug = loglik_grid.detach().cpu()
        # import matplotlib.pyplot as plt

        # # --- Plot ELBO vs L1 and LL vs L1 on same figure for obs 0 ---
        # if elbo_debug is not None and ll_debug is not None:
        #     L1_plot = self.L1_grid.detach().cpu().numpy()
        #     elbo_np = elbo_debug.numpy()
        #     ll_np   = ll_debug.numpy()

        #     # optional: shift both so their max is 0 for nicer plotting
        #     elbo_np = elbo_np - elbo_np.max()
        #     ll_np   = ll_np   - ll_np.max()

        #     idx_ll   = np.argmax(ll_np)
        #     idx_elbo = np.argmax(elbo_np)

        #     L1_ll_max   = L1_plot[idx_ll]
        #     L1_elbo_max = L1_plot[idx_elbo]

        #     plt.figure(figsize=(8, 5))
        #     plt.plot(L1_plot, ll_np,   label="log-likelihood f(L1)", linewidth=2)
        #     plt.plot(L1_plot, elbo_np, label="ELBO(μ) (fixed σ)", linewidth=2)
        #     plt.axvline(L1_ll_max, color = 'r', linewidth=1.5, label=f"argmax f(L1) ≈ {L1_ll_max:.1f} m")
        #     plt.axvline(L1_elbo_max, color = 'r',linewidth=1.5, label=f"argmax ELBO ≈ {L1_elbo_max:.1f} m")   
        #     plt.xlabel("L1 (m)")
        #     plt.ylabel("Value (shifted)")
        #     plt.title("ELBO vs μ (mapped to L1) vs raw log-likelihood f(L1)\nfor observation 0")
        #     plt.legend()
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()
        
        
        return {
            "L1": out,  
        }
        return out  # [N], float32
