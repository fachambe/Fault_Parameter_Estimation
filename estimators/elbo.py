# estimators/elbo1d.py
import torch
from torch import distributions as dist
import numpy as np
import math
from estimators.base import Estimator

class ELBOArgmaxMu1D(Estimator):
    """
    Single-parameter VI via ELBO argmax over μ (sigmoid-Normal guide on x∈(0,1),
    affine-mapped to θ∈[lo,hi]). Returns posterior-mean θ per observation.
    Works for L1 or a single real/imaginary component of ZF/ZL.
    M = # of Monte Carlo samples for each mu in mu_grid of size [G] from variational distribution q 
    K = # of Monte Carlo samples for the posterior to calculate posterior mean (which is our estimate for the parameter)
    """
    def __init__(self, fm, target, estimate, mu_grid, fixed_sigma, bounds, fixed,
                 M, K, device=None):
        self.device = device
        self.fm = fm
        self.target = str(target).upper()
        self.estimate = estimate  # None for L1; "real"/"imag" for ZF/ZL
        self.mu_grid = torch.as_tensor(mu_grid, dtype=torch.float32, device=self.device)
        self.fixed_sigma = float(fixed_sigma)
        self.lo, self.hi = float(bounds[0]), float(bounds[1])
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

        G = self.mu_grid.numel() #G = num of elements in mu grid
        base = dist.Normal(self.mu_grid, self.fixed_sigma)  #[G]
        q    = dist.TransformedDistribution(base, [dist.transforms.SigmoidTransform()])


        samples = q.rsample((self.M,))                                     # [M,G]
        x = samples.transpose(0, 1)                                        # [G,M]
        theta = self.lo + (self.hi - self.lo) * x                          # [G,M]
        log_q = q.log_prob(samples).transpose(0, 1)                        # [G,M]

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

        for n in range(N):
            obs_real = torch.view_as_real(obs_tf[n]).to(torch.float32)            # [F,2]
            sigma = torch.sqrt(noise_var[n] / 2.0).to(torch.float32)              # [F]
            sigma = sigma.unsqueeze(-1).expand(F, 2)                              # [F,2]
            
            diff  = H_real - obs_real.unsqueeze(0).unsqueeze(0)                      # [G,M,F,2] - [1,1,F,2]
            var2  = (sigma * sigma).unsqueeze(0).unsqueeze(0)                        # [1,1,F,2]
            log_lik = -0.5 * (diff*diff/var2 + math.log(2*math.pi) + torch.log(var2)).sum(dim=(2,3)) #[G,M] sum over [F,2]
            # dist_obs = dist.Independent(dist.Normal(H_real, sigma), 2)
            # log_lik  = dist_obs.log_prob(obs_real)                                # [G,M]
            elbo     = (log_lik - log_q).mean(dim=1)                              # [G] avg over [M] Monte carlo for each grid point
            k        = int(torch.argmax(elbo).item())
            mu_star  = self.mu_grid[k]

            # posterior mean under q_{mu*}
            base_star = dist.Normal(mu_star, self.fixed_sigma)
            z = base_star.rsample((self.K,)).to(self.device)                      # [K]
            x_star = torch.sigmoid(z).clamp_(1e-6, 1-1e-6)                        # [K]
            theta_samples = self.lo + (self.hi - self.lo) * x_star                # [K]
            out[n] = theta_samples.mean().item()
        
        return out  # [N], float32
