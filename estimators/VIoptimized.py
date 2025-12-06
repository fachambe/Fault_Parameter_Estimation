import math
import time
import torch
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import pyro.poutine as poutine
from pyro.distributions import Delta
from torch.distributions import constraints, SigmoidTransform, TanhTransform, AffineTransform
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np


from estimators.base import Estimator  # your base
torch.set_float32_matmul_precision("high")
#torch.set_printoptions(profile="full")   # show full tensors

def _sigmoid_to_range(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(u)

def _tanh_to_range(u: torch.Tensor, max_abs: float) -> torch.Tensor:
    return max_abs * torch.tanh(u)

def _range_to_sigmoid_x(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    # Inverse of _sigmoid_to_range (for seeding)
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)  # avoid 0/1
    return torch.log(z) - torch.log1p(-z)  # logit

class VIoptimized(Estimator):
    """
    Per-observation Variational Inference:
      theta_n = [L1_n, ReZF_n, ImZF_n, ReZL_n, ImZL_n] for each n in 1..N.

    Model/Guide:
      - Unconstrained latents (R) for each component: L1_u, ReZF_u, ImZF_u, ReZL_u, ImZL_u
      - Priors: Uniform over parameter range
      - Transforms inside model to enforce SAME ranges as your MLE:
          L1    in [L1_lo, L1_hi]      via sigmoid -> affine
          ReZF  in [ReZF_lo, ReZF_hi]  via sigmoid -> affine
          ImZF  in [-ImZF_max, +ImZF_max] via tanh -> scale
          ReZL  in [ReZL_lo, ReZL_hi]  via sigmoid -> affine
          ImZL  in [-ImZL_max, +ImZL_max] via tanh -> scale
      - Likelihood:
          For each (n,f), observe y_{n,f} (complex) as 2-D real Normal
          with isotropic covariance sigma_{n,f}^2 I_2.

    Inputs:
      obs_tf:      [N,F] complex64 tensor (on device)
      noise_var_f: [N,F] float32 tensor (on device)
    """
    def __init__(self,
                 fm,
                 likelihood,
                 L: float = 1000.0,
                 device="cuda",
                 # VI hyperparams
                 svi_steps: int = 2000,
                 svi_lr: float = 1e-2,
                 first_stage_num_particles: int = 100,
                 second_stage_num_particles: int = 5,
                 # VI mu of L1 grid search hyperparams
                 L1_grid: torch.tensor = torch.linspace(100, 900, 400),
                 fixed_sigma: float = 0.02,
                 M: int = 200, # of Monte Carlo samples for each mu in mu_grid of size [G] from variational distribution q 
                 topK: int = 3, # Top K mus
                 inner_steps: int = 12,  #Inner SVI steps at each grid point
                 inner_lr: float = 5e-3 #Inner SVI LR
                 ):
        self.fm = fm
        self.device = device
        self.lik = likelihood
        self.L = float(L)
        #VI
        self.svi_steps = svi_steps
        self.svi_lr = svi_lr
        self.second_stage_num_particles = second_stage_num_particles
        #VI mu of L1 grid search
        self.L1_grid = L1_grid.to(device)
        self.first_stage_num_particles = first_stage_num_particles
        self.fixed_sigma = fixed_sigma
        self.M = M
        self.topK = topK
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        # parameter range
        self.L1_lo, self.L1_hi = 1.0, self.L
        self.ReZF_lo, self.ReZF_hi = 1.0, 4000.0
        self.ImZF_max = 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

    def model(self, y_ri, sig_f):
    # y_ri: [M, F, 2], sig_f: [M, F]
        M, F, _ = y_ri.shape
        device = self.device
        sig_nf2 = sig_f.unsqueeze(-1).expand(-1, -1, 2)  # [M, F, 2]

        with pyro.plate("obs", M):
            L1 = pyro.sample("L1", dist.Uniform(
                torch.tensor(self.L1_lo, device=device),
                torch.tensor(self.L1_hi, device=device)
            ))
            ZF_r = pyro.sample("ZF_re", dist.Uniform(
                torch.tensor(self.ReZF_lo, device=device),
                torch.tensor(self.ReZF_hi, device=device)
            ))
            ZF_i = pyro.sample("ZF_im", dist.Uniform(
                torch.tensor(-self.ImZF_max, device=device),
                torch.tensor(+self.ImZF_max, device=device)
            ))
            ZL_r = pyro.sample("ZL_re", dist.Uniform(
                torch.tensor(self.ReZL_lo, device=device),
                torch.tensor(self.ReZL_hi, device=device)
            ))
            ZL_i = pyro.sample("ZL_im", dist.Uniform(
                torch.tensor(-self.ImZL_max, device=device),
                torch.tensor(+self.ImZL_max, device=device)
            ))
            ZF = ZF_r + 1j * ZF_i
            ZL = ZL_r + 1j * ZL_i
            H = self.fm.compute_H_complex(L1, ZF, ZL)     # [M, F] complex
            H_ri = torch.view_as_real(H)                  # [M, F, 2]
            pyro.sample("y", dist.Normal(H_ri, sig_nf2).to_event(2), obs=y_ri) 
            #F and Re/Im are independent treat [F, 2] as event not batch shape

    # GUIDE: Normal on R with Sigmoid/Tanh transforms to match parameter ranges
    def guide(self, y_ri, sig_f):
        M, F, _ = y_ri.shape
        device = self.device

        # Variational parameters: one Normal per observation for each latent
        with pyro.plate("obs", M):
            # L1 in [L1_lo, L1_hi] via sigmoid -> affine
            L1_loc   = pyro.param("L1_loc",   torch.zeros(M, device=device))
            L1_scale = pyro.param("L1_scale", torch.full((M,), 0.02, device=device),
                                constraint=constraints.positive)
            q_L1 = dist.TransformedDistribution(
                dist.Normal(L1_loc, L1_scale),
                [SigmoidTransform(), AffineTransform(loc=self.L1_lo, scale=(self.L1_hi - self.L1_lo))]
            )
            pyro.sample("L1", q_L1)
            # ZF_re in [ReZF_lo, ReZF_hi]
            ZFr_loc   = pyro.param("ZF_re_loc", torch.zeros(M, device=device))
            ZFr_scale = pyro.param("ZF_re_scale", torch.full((M,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )
            pyro.sample("ZF_re", q_ZFr)

            # ZF_im in [-ImZF_max, +ImZF_max]
            ZFi_loc   = pyro.param("ZF_im_loc",   torch.zeros(M, device=device))
            ZFi_scale = pyro.param("ZF_im_scale", torch.full((M,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )
            pyro.sample("ZF_im", q_ZFi)

            # ZL_re in [ReZL_lo, ReZL_hi]
            ZLr_loc   = pyro.param("ZL_re_loc",   torch.zeros(M, device=device))
            ZLr_scale = pyro.param("ZL_re_scale", torch.full((M,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )
            pyro.sample("ZL_re", q_ZLr)

            # ZL_im in [-ImZL_max, +ImZL_max]
            ZLi_loc   = pyro.param("ZL_im_loc",   torch.zeros(M, device=device))
            ZLi_scale = pyro.param("ZL_im_scale", torch.full((M,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )
            pyro.sample("ZL_im", q_ZLi)

    def guide_L1_fixed_Z_frozen(self, y_ri, sig_f, mu_vec,
                            ZF_re_true, ZF_im_true,
                            ZL_re_true, ZL_im_true):
        M = y_ri.size(0)
        device = y_ri.device
        ZFr = torch.full((M,), ZF_re_true, device=device)
        ZFi = torch.full((M,), ZF_im_true, device=device)
        ZLr = torch.full((M,), ZL_re_true, device=device)
        ZLi = torch.full((M,), ZL_im_true, device=device)
        with pyro.plate("obs", M):
            L1_dist = dist.TransformedDistribution(
                dist.Normal(mu_vec, torch.full((M,), self.fixed_sigma, device=device)),
                [SigmoidTransform(), AffineTransform(self.L1_lo, self.L1_hi - self.L1_lo)]
            )
            pyro.sample("L1", L1_dist)
            # ZF, ZL fixed deterministically at the true values
            pyro.sample("ZF_re", Delta(ZFr).to_event(0))
            pyro.sample("ZF_im", Delta(ZFi).to_event(0))
            pyro.sample("ZL_re", Delta(ZLr).to_event(0))
            pyro.sample("ZL_im", Delta(ZLi).to_event(0))

    # Get per-run ELBO vector [M] in one pass (no training).
    def _per_run_elbo_once(self, guide_fn, y_ri, sig_f, num_particles):
        M = y_ri.size(0)
        device = self.device
        elbo_acc = torch.zeros(M, device=device)
        def under_obs(site):
            # is this site inside plate("obs") ?
            for fr in site["cond_indep_stack"]:
                if fr.name == "obs":
                    return True
            return False

        def keep_obs_only(t):
            # Reduce anything beyond the first dimension (keep dim0 = obs plate)
            # Right now all log_probs return [N] so not needed technically but good practice to do
            if t.dim() <= 1:
                return t
            return t.sum(dim=tuple(range(1, t.dim())))
        
        with torch.no_grad():
            for _ in range(num_particles):
                # guide first, sample z ~ q(z|mu) then replay model at those samples
                gt = poutine.trace(guide_fn).get_trace(y_ri, sig_f)
                mt = poutine.trace(poutine.replay(self.model, trace=gt)).get_trace(y_ri, sig_f)

                # one-particle contribution [M]
                vec = torch.zeros(M, device=device)

                # Model terms: log p(z,y|θ) = log p(y|z,θ) + log p(z) 
                for s in mt.nodes.values():
                    if s["type"] == "sample" and under_obs(s):
                        lp = s["fn"].log_prob(s["value"])
                        vec += keep_obs_only(lp)

                # Guide terms: - log q(z)
                for s in gt.nodes.values():
                    if s["type"] == "sample" and under_obs(s):
                        lq = s["fn"].log_prob(s["value"])
                        vec -= keep_obs_only(lq)

                elbo_acc += vec
        #average over particles
        return elbo_acc / float(num_particles)

    


    def _profile_L1_mu_grid(self, y_ri, sig_f):
        """
        Scan μ over self.mu_grid with Z frozen; return:
        - elbo_per_run: [G, M]  (per-run ELBO at each μ)
        - topk_mu:      [M, K]  (top-K μ per run, in μ-space)

        Note Z is NOT optimized here; use frozen guide.
        """
        pyro.clear_param_store()
        device = y_ri.device
        M, F, _ = y_ri.shape
        G = self.L1_grid.numel()

        elbo_per_run = torch.empty(G, M, device=device)

        # Convert L1_grid to corresponding mu_grid
        mu_grid = _range_to_sigmoid_x(self.L1_grid, self.L1_lo, self.L1_hi) #[G]

        # choose which guide to use during profiling (Z frozen)
        def make_guide(mu_vec):
            # L1 centered around mu_vec, Z held fixed at center of parameter range
            return (lambda y, s: self.guide_L1_fixed_Z_frozen(
                y, s, mu_vec,
                ZF_re_true=2000.0, ZF_im_true=0.0,
                ZL_re_true=100.0,  ZL_im_true=0.0
            ))

        for g, mu in enumerate(mu_grid):
            L1_center = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(mu)
            
            mu_vec = torch.full((M,), float(mu), device=device)  # [N]

            guide_mu = make_guide(mu_vec)

            # per-observation ELBO at this μ (still with frozen Z)
            elbo_per_run[g] = self._per_run_elbo_once(guide_mu, y_ri, sig_f, num_particles=self.first_stage_num_particles)
            if g % 50 == 0:
                print(f"[g={g}] mu={float(mu):+.4f}  L1≈{L1_center:.2f}  "
                  f"ELBO={torch.mean(elbo_per_run[g]).item():.3f}")

        _, topk_idx = torch.topk(elbo_per_run, k=self.topK, dim=0)
        # topk_mu: [M, K]
        topk_mu = mu_grid[topk_idx].transpose(0, 1).contiguous()

        return topk_mu, elbo_per_run    


    def predict(self, obs_tf, noise_var_f, snr):
        """
        Jointly estimate parameters for M runs of N observations with:
        (1) 1-D ELBO grid search over μ (L1) with frozen Z (profiling),
        (2) Full SVI with pyro starting from the best μ seeds.

        Returns posterior medians as the point estimate for each parameter.
        """
        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [M,F]."
        M, F = obs_tf.shape
        device = self.device

        # 0) Convert complex obs_tf [M, F] to real
        y_ri = torch.view_as_real(obs_tf)  # [M, F, 2]
        sig_f = torch.sqrt(noise_var_f / 2)  # [M, F]

        # 1) 1-D ELBO grid over the variational mean (mu) of L1 (Z frozen)
        topk_mu, elbo_per_run = self._profile_L1_mu_grid(y_ri, sig_f)  # [M,K]

        # For diagnostics: convert μ seeds to L1-space
        topk_L1 = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(topk_mu)  # [M,K]
        print("topk_mu in L1 space", topk_L1)

        # --- diagnostics: count + list where first seed lands ---
        true_L1  = 250.0
        wrong_L1 = 750.0
        tol = 5.0

        col0 = topk_L1[:, 0]  # [M]

        dev = col0.device
        true_t  = torch.tensor(true_L1,  device=dev)
        wrong_t = torch.tensor(wrong_L1, device=dev)

        near_true  = torch.isclose(col0, true_t,  atol=tol)
        near_wrong = torch.isclose(col0, wrong_t, atol=tol)

        first_at_750 = near_wrong.sum().item()
        first_at_250 = near_true.sum().item()

        print("First-column seeds at 750:", first_at_750)
        print("First-column seeds at 250:", first_at_250)
        print(f"First-column seeds at neither: {col0.numel() - first_at_750 - first_at_250}")

        # --- 2) Run SVI inference for the top-K L1 seeds for all M runs ---

        # Choose neutral Z seeds in unconstrained space: 0 -> mid-range after sigmoid/tanh
        zero = torch.zeros(M, device=device)

        pyro.clear_param_store()
        ps = pyro.get_param_store()

        def set_param(name, value, default_scale=0.25, fixed_sigma=None):
            value = value.detach().to(device)
            if name in ps:
                ps[name].data.copy_(value)
            else:
                pyro.param(name, value.clone())

            # ensure matching *_scale exists
            sname = name.replace("_loc", "_scale")
            if sname not in ps:
                init_scale = (fixed_sigma if (fixed_sigma is not None and name.endswith("L1_loc"))
                            else default_scale)
                pyro.param(sname, torch.full_like(value, init_scale),
                        constraint=constraints.positive)

        def seed_from(k):
            set_param("L1_loc", topk_mu[:, k], fixed_sigma=self.fixed_sigma)
            set_param("ZF_re_loc", zero, default_scale=0.25)
            set_param("ZF_im_loc", zero, default_scale=0.25)
            set_param("ZL_re_loc", zero, default_scale=0.25)
            set_param("ZL_im_loc", zero, default_scale=0.25)


        # Storage for per-(m,k) medians in physical space
        L1_med_all  = torch.empty(M, self.topK, device=device)
        ZFr_med_all = torch.empty(M, self.topK, device=device)
        ZFi_med_all = torch.empty(M, self.topK, device=device)
        ZLr_med_all = torch.empty(M, self.topK, device=device)
        ZLi_med_all = torch.empty(M, self.topK, device=device)
        # Storage for per-(m,k) ELBO values
        elbo_mk = torch.empty(M, self.topK, device=device)  # [M,K]
        
 
        # For each L1 seed, run full SVI
        for k in range(self.topK):
            pyro.clear_param_store()
            seed_from(k)
            optimizer = optim.Adam({"lr": self.svi_lr})
            svi = SVI(self.model, self.guide, optimizer,
                    loss=Trace_ELBO(num_particles=self.second_stage_num_particles))

            for step in range(self.svi_steps):
                loss = svi.step(y_ri, sig_f)
                if step % 250 == 0:
                    ps = pyro.get_param_store()

                    #All [M]
                    L1_loc  = ps["L1_loc"].detach()
                    ZFr_loc = ps["ZF_re_loc"].detach()
                    ZFi_loc = ps["ZF_im_loc"].detach()
                    ZLr_loc = ps["ZL_re_loc"].detach()
                    ZLi_loc = ps["ZL_im_loc"].detach()

                    L1_scale = ps["L1_scale"].detach()
                    ZFr_scale = ps["ZF_re_scale"].detach()
                    ZFi_scale = ps["ZF_im_scale"].detach()
                    ZLr_scale = ps["ZL_re_scale"].detach()
                    ZLi_scale = ps["ZL_im_scale"].detach()

                    L1_mean  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
                    ZFr_mean = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
                    ZFi_mean = self.ImZF_max * torch.tanh(ZFi_loc)
                    ZLr_mean = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
                    ZLi_mean = self.ImZL_max * torch.tanh(ZLi_loc)

                    def stats(x):
                        return float(x.mean()), float(x.std())

                    mL1, sL1   = stats(L1_mean)
                    mZFr, sZFr = stats(ZFr_mean); mZFi, sZFi = stats(ZFi_mean)
                    mZLr, sZLr = stats(ZLr_mean); mZLi, sZLi = stats(ZLi_mean)

                    print(f"SNR = {snr} | [seed {k} step {step}] ELBO={loss:.3f} | "
                        f"L1 {mL1:.2f}±{sL1:.2f} | "
                        f"ZF_re {mZFr:.1f}±{sZFr:.1f} | ZF_im {mZFi:.1f}±{sZFi:.1f} | "
                        f"ZL_re {mZLr:.1f}±{sZLr:.1f} | ZL_im {mZLi:.1f}±{sZLi:.1f} | ")
                       # f"L1_scale {L1_scale} | ZF_re_scale {ZFr_scale} | ZF_im_scale {ZFi_scale} | "
                        #f"ZL_re_scale {ZLr_scale} | ZL_im_scale {ZLi_scale}")

                    idx = torch.tensor([0, 1, 2], device=L1_mean.device)
                    print("   samples:",
                        "L1",     L1_mean[idx].tolist(),
                        "ZF_re",  ZFr_mean[idx].tolist(),
                        "ZF_im",  ZFi_mean[idx].tolist())

            # After SVI for this seed: compute medians
            with torch.no_grad():
                ps = pyro.get_param_store()

                L1_loc  = ps["L1_loc"].to(device)
                ZFr_loc = ps["ZF_re_loc"].to(device)
                ZFi_loc = ps["ZF_im_loc"].to(device)
                ZLr_loc = ps["ZL_re_loc"].to(device)
                ZLi_loc = ps["ZL_im_loc"].to(device)

                L1_med  = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
                ZFr_med = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
                ZFi_med = self.ImZF_max * torch.tanh(ZFi_loc)
                ZLr_med = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
                ZLi_med = self.ImZL_max * torch.tanh(ZLi_loc)
                print("L1 avg", torch.mean(L1_med).item())
                print("ZF_re avg", torch.mean(ZFr_med).item())
        # return { #[M] each
        #     "L1":    L1_med.float().cpu().numpy(), 
        #     "ZF_re": ZFr_med.float().cpu().numpy(),
        #     "ZF_im": ZFi_med.float().cpu().numpy(),
        #     "ZL_re": ZLr_med.float().cpu().numpy(),
        #     "ZL_im": ZLi_med.float().cpu().numpy(),
        #     }

                # Store medians for this seed k, per run m
                L1_med_all[:, k] = L1_med
                ZFr_med_all[:, k] = ZFr_med
                ZFi_med_all[:, k] = ZFi_med
                ZLr_med_all[:, k] = ZLr_med 
                ZLi_med_all[:, k] = ZLi_med

                # Compute per-run ELBO for this trained seed k
                # elbo_vec = self._per_run_elbo_once(
                #     self.guide,
                #     y_ri,
                #     sig_f,
                #     num_particles=100
                # )   # [M]

                # elbo_mk[:, k] = elbo_vec


        # Build complex ZF, ZL grids [M,K]
        ZF_all = ZFr_med_all + 1j * ZFi_med_all        # [M,K]
        ZL_all = ZLr_med_all + 1j * ZLi_med_all        # [M,K]

        M, K = L1_med_all.shape
        F = obs_tf.shape[1]
        dev = obs_tf.device

        # Flatten (M,K) -> (M*K,) for forward model
        L1_flat = L1_med_all.reshape(-1)               # [M*K]
        ZF_flat = ZF_all.reshape(-1)                   # [M*K]
        ZL_flat = ZL_all.reshape(-1)                   # [M*K]

        # Compute H for all (m,k) at once: [M*K,F]
        H_flat = self.fm.compute_H_complex(L1_flat, ZF_flat, ZL_flat)  # [M*K, F]

        # Reshape back to [M,K,F]
        H_mkf = H_flat.view(M, K, F)                   # [M,K,F]

        # Broadcast obs and noise var: [M,1,F]
        obs = obs_tf.to(dev).unsqueeze(1)              # [M,1,F]
        var = noise_var_f.to(dev).unsqueeze(1)         # [M,1,F]

        # NLL_{m,k} = sum_f |y_{m,f} - H_{m,k,f}|^2 / var_{m,f}
        diff    = obs - H_mkf                          # [M,K,F]
        nll_mk  = ((diff.abs() ** 2) / var).sum(-1)    # [M,K]

        i_best = torch.argmin(nll_mk, dim=1)   # [M]
        print("i_best (NLL-based):", i_best)
        print("L1 med all", L1_med_all)
        #print("ZFR med all", ZFr_med_all)

        # elbo_mk: [M,K] — larger is better
        #i_best = torch.argmax(elbo_mk, dim=1)  # [M], best seed index per observation
        #print("i_best", i_best)
        arM = torch.arange(M, device=device)

        L1_best  = L1_med_all[arM, i_best]
        ZFr_best = ZFr_med_all[arM, i_best]
        ZFi_best = ZFi_med_all[arM, i_best]
        ZLr_best = ZLr_med_all[arM, i_best]
        ZLi_best = ZLi_med_all[arM, i_best]


        true_L1   = 250.0
        true_ZFr  = 1000.0

        dev = L1_med_all.device
        true_L1_t  = torch.tensor(true_L1,  device=dev)
        true_ZFr_t = torch.tensor(true_ZFr, device=dev)

        # 1) RMSE using ONLY seed 0 (baseline single-seed behavior)
        L1_seed0  = L1_med_all[:, 0]     # [M]
        ZFr_seed0 = ZFr_med_all[:, 0]    # [M]

        rmse_L1_seed0  = torch.sqrt(((L1_seed0  - true_L1_t)**2).mean()).item()
        rmse_ZFr_seed0 = torch.sqrt(((ZFr_seed0 - true_ZFr_t)**2).mean()).item()

        print(f"RMSE L1 (seed 0):  {rmse_L1_seed0:.3f}")
        print(f"RMSE ZF_re (seed 0): {rmse_ZFr_seed0:.3f}")

        # 2) RMSE using ELBO-chosen best seed per m
        L1_best  = L1_med_all[arM, i_best]
        ZFr_best = ZFr_med_all[arM, i_best]

        rmse_L1_best  = torch.sqrt(((L1_best  - true_L1_t)**2).mean()).item()
        rmse_ZFr_best = torch.sqrt(((ZFr_best - true_ZFr_t)**2).mean()).item()

        print(f"RMSE L1 (ELBO best):  {rmse_L1_best:.3f}")
        print(f"RMSE ZF_re (ELBO best): {rmse_ZFr_best:.3f}")

        # 3) How often does ELBO actually improve over seed 0 for each m?
        improve_L1  = ((L1_best  - true_L1_t).abs() < (L1_seed0  - true_L1_t).abs()).float().mean().item()
        improve_ZFr = ((ZFr_best - true_ZFr_t).abs() < (ZFr_seed0 - true_ZFr_t).abs()).float().mean().item()

        print(f"Fraction of obs where ELBO-best improves |L1-error| over seed0:  {improve_L1:.3f}")
        print(f"Fraction of obs where ELBO-best improves |ZF_re-error| over seed0: {improve_ZFr:.3f}")


        true_L1 = 250.0
        dev = L1_med_all.device
        true_L1_t = torch.tensor(true_L1, device=dev)

        # Squared error per (m,k)
        err_L1_all = (L1_med_all - true_L1_t)**2              # [M,K]

        # Oracle best seed index per m (closest to 250)
        k_star_L1 = torch.argmin(err_L1_all, dim=1)           # [M]

        # NLL-best (what you're using now)
        i_best_nll = torch.argmin(nll_mk, dim=1)              # [M]

        # How often do they agree?
        match_frac = (i_best_nll == k_star_L1).float().mean().item()
        print(f"Fraction of obs where NLL-best seed == oracle best L1 seed: {match_frac:.3f}")


        # L1 error of NLL-best per obs
        L1_best_nll = L1_med_all[torch.arange(M, device=dev), i_best_nll]  # [M]
        err_best = (L1_best_nll - true_L1_t).abs()                         # [M]

        # Compare every (m,k) error to err_best[m]
        err_all_abs = (L1_med_all - true_L1_t).abs()                       # [M,K]
        better_exists = (err_all_abs < err_best.unsqueeze(1)).any(dim=1)  # [M]

        print("Fraction of obs where SOME seed is closer to 250 than NLL-best:",
            better_exists.float().mean().item())


        bad_idxs = (i_best_nll != k_star_L1).nonzero(as_tuple=True)[0]  # [num_bad]
        print("Number of obs where NLL-best != oracle-best:", bad_idxs.numel())

        # Show up to 5 examples
        for idx in bad_idxs[:5]:
            idx = idx.item()
            print(f"\n=== m = {idx} ===")
            print("L1_med_all[m]:", L1_med_all[idx].tolist())
            print("nll_mk[m]:    ", nll_mk[idx].tolist())
            print("oracle k* (closest to 250):", k_star_L1[idx].item(),
                "  L1 =", float(L1_med_all[idx, k_star_L1[idx]]))
            print("NLL best k:", i_best_nll[idx].item(),
                "  L1 =", float(L1_med_all[idx, i_best_nll[idx]]))
    
        return { #[M] each
            "L1":    L1_best.float().cpu().numpy(), 
            "ZF_re": ZFr_best.float().cpu().numpy(),
            "ZF_im": ZFi_best.float().cpu().numpy(),
            "ZL_re": ZLr_best.float().cpu().numpy(),
            "ZL_im": ZLi_best.float().cpu().numpy(),
        }

    