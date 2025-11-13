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

from estimators.base import Estimator  # your base
torch.set_float32_matmul_precision("high")
torch.set_printoptions(profile="full")   # show full tensors

def _sigmoid_to_range(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(u)

def _tanh_to_range(u: torch.Tensor, max_abs: float) -> torch.Tensor:
    return max_abs * torch.tanh(u)


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
                 svi_steps: int = 10000,
                 svi_lr: float = 1e-2,
                 num_particles: int = 1,
                 # VI mu of L1 grid search hyperparams
                 mu_grid: torch.tensor = torch.linspace(-2.0, 2.0, 450),
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
        self.num_particles = num_particles
        #VI mu of L1 grid search
        self.mu_grid = mu_grid.to(device)
        self.fixed_sigma = fixed_sigma
        self.M = M
        self.topK = topK
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        # parameter range
        self.L1_lo, self.L1_hi = 1.0, self.L
        self.ReZF_lo, self.ReZF_hi = 1.0, 2000.0
        self.ImZF_max = 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

    def model(self, y_ri, sig_f):
    # y_ri: [N, F, 2], sig_f: [N, F]
        N, F, _ = y_ri.shape
        device = y_ri.device
        sig_nf2 = sig_f.unsqueeze(-1).expand(-1, -1, 2)  # [N, F, 2]

        with pyro.plate("obs", N):
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
            H = self.fm.compute_H_complex(L1, ZF, ZL)     # [N, F] complex
            H_ri = torch.view_as_real(H)                  # [N, F, 2]
            pyro.sample("y", dist.Normal(H_ri, sig_nf2).to_event(2), obs=y_ri) 
            #F and Re/Im are independent treat [F, 2] as event not batch shape

    # GUIDE: Normal on R with Sigmoid/Tanh transforms to match model supports
    def guide(self, y_ri, sig_f):
        N, F, _ = y_ri.shape
        device = y_ri.device

        # Variational parameters: one Normal per observation for each latent
        with pyro.plate("obs", N):
            # L1 in [L1_lo, L1_hi] via sigmoid -> affine
            L1_loc   = pyro.param("L1_loc",   torch.zeros(N, device=device))
            L1_scale = pyro.param("L1_scale", torch.full((N,), 0.02, device=device),
                                constraint=constraints.positive)
            q_L1 = dist.TransformedDistribution(
                dist.Normal(L1_loc, L1_scale),
                [SigmoidTransform(), AffineTransform(loc=self.L1_lo, scale=(self.L1_hi - self.L1_lo))]
            )
            pyro.sample("L1", q_L1)
            # ZF_re in [ReZF_lo, ReZF_hi]
            ZFr_loc   = pyro.param("ZF_re_loc", torch.zeros(N, device=device))
            ZFr_scale = pyro.param("ZF_re_scale", torch.full((N,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )
            pyro.sample("ZF_re", q_ZFr)

            # ZF_im in [-ImZF_max, +ImZF_max]
            ZFi_loc   = pyro.param("ZF_im_loc",   torch.zeros(N, device=device))
            ZFi_scale = pyro.param("ZF_im_scale", torch.full((N,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )
            pyro.sample("ZF_im", q_ZFi)

            # ZL_re in [ReZL_lo, ReZL_hi]
            ZLr_loc   = pyro.param("ZL_re_loc",   torch.zeros(N, device=device))
            ZLr_scale = pyro.param("ZL_re_scale", torch.full((N,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )
            pyro.sample("ZL_re", q_ZLr)

            # ZL_im in [-ImZL_max, +ImZL_max]
            ZLi_loc   = pyro.param("ZL_im_loc",   torch.zeros(N, device=device))
            ZLi_scale = pyro.param("ZL_im_scale", torch.full((N,), 0.25, device=device),
                                constraint=constraints.positive)
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )
            pyro.sample("ZL_im", q_ZLi)

    def guide_L1_fixed_Z_frozen(self, y_ri, sig_f, mu_vec,
                            ZF_re_true=100.0, ZF_im_true=50.0,
                            ZL_re_true=100.0, ZL_im_true=-5.0):
        B = y_ri.size(0)
        device = y_ri.device
        ZFr = torch.full((B,), ZF_re_true, device=device)
        ZFi = torch.full((B,), ZF_im_true, device=device)
        ZLr = torch.full((B,), ZL_re_true, device=device)
        ZLi = torch.full((B,), ZL_im_true, device=device)
        with pyro.plate("obs", B):
            L1_dist = dist.TransformedDistribution(
                dist.Normal(mu_vec, torch.full((B,), self.fixed_sigma, device=device)),
                [SigmoidTransform(), AffineTransform(self.L1_lo, self.L1_hi - self.L1_lo)]
            )
            pyro.sample("L1", L1_dist)
            # ZF, ZL fixed deterministically at the true values
            pyro.sample("ZF_re", Delta(ZFr).to_event(0))
            pyro.sample("ZF_im", Delta(ZFi).to_event(0))
            pyro.sample("ZL_re", Delta(ZLr).to_event(0))
            pyro.sample("ZL_im", Delta(ZLi).to_event(0))

    # Get per-observation ELBO vector [N] in one pass (no training). Only handles 1 particle for now.
    def _per_obs_elbo_once(self, guide_fn, y_ri, sig_f):
        with torch.no_grad():
            # guide first, then replay model at those samples
            gt = poutine.trace(guide_fn).get_trace(y_ri, sig_f)
            mt = poutine.trace(poutine.replay(self.model, trace=gt)).get_trace(y_ri, sig_f)

            N = y_ri.size(0)
            vec = torch.zeros(N, device=y_ri.device)
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

            return vec  # shape [N]

    # GUIDE for mu of L1 grid search: mu of L1 is fixed, Z is not fixed
    def guide_L1_fixed(self, y_ri, sig_f, mu_vec):
        #mu_vec is candidate mu expanded to size [N] from mu_grid
        N, F, _ = y_ri.shape
        device = y_ri.device

        # Variational parameters: one Normal per observation for each latent
        with pyro.plate("obs", N):
            z_std = 0.25
            # L1: fixed around mu with sigma_fixed, NO pyro.param means not trainable
            q_L1 = dist.TransformedDistribution(
                dist.Normal(loc=mu_vec, scale=torch.full((N,), self.fixed_sigma, device=device)),
                [SigmoidTransform(), AffineTransform(self.L1_lo, self.L1_hi - self.L1_lo)]
            )
            pyro.sample("L1", q_L1)

            # ZF_re in [ReZF_lo, ReZF_hi]
            ZFr_loc   = pyro.param("ZF_re_loc", torch.zeros(N, device=device))
            ZFr_scale = pyro.param("ZF_re_scale", torch.full((N,), z_std, device=device),
                                constraint=constraints.positive)
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )
            pyro.sample("ZF_re", q_ZFr)

            # ZF_im in [-ImZF_max, +ImZF_max]
            ZFi_loc   = pyro.param("ZF_im_loc",   torch.zeros(N, device=device))
            ZFi_scale = pyro.param("ZF_im_scale", torch.full((N,), z_std, device=device),
                                constraint=constraints.positive)
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )
            pyro.sample("ZF_im", q_ZFi)

            # ZL_re in [ReZL_lo, ReZL_hi]
            ZLr_loc   = pyro.param("ZL_re_loc",   torch.zeros(N, device=device))
            ZLr_scale = pyro.param("ZL_re_scale", torch.full((N,), z_std, device=device),
                                constraint=constraints.positive)
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )
            pyro.sample("ZL_re", q_ZLr)

            # ZL_im in [-ImZL_max, +ImZL_max]
            ZLi_loc   = pyro.param("ZL_im_loc",   torch.zeros(N, device=device))
            ZLi_scale = pyro.param("ZL_im_scale", torch.full((N,), z_std, device=device),
                                constraint=constraints.positive)
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )
            pyro.sample("ZL_im", q_ZLi)
    
    def profile_L1_with_Z_frozen(self, obs_tf, noise_var_f, return_per_obs=True):
        """
        Sanity check: Fix ZF,ZL at true values and scan μ for L1.
        ELBO should peak near the true L1 for each observation.
        Returns:
            elbo_batch:   [G] tensor of ELBO over the batch at each μ
            elbo_per_obs: [G,N] tensor if return_per_obs=True else None
            topk_mu:      [N,K] best μ per observation (in μ-space)
        """
        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2
        device = obs_tf.device if hasattr(obs_tf, "device") else torch.device(self.device)
        N, F = obs_tf.shape

        y_ri = torch.view_as_real(obs_tf)                 # [N,F,2]
        sig_f = torch.sqrt(noise_var_f / 2.0)            # [N,F]

        G = self.mu_grid.numel()
        elbo_batch = torch.empty(G, device=device)
        elbo_per_obs = torch.empty(G, N, device=device) if return_per_obs else None

        loss_obj = Trace_ELBO(num_particles=self.num_particles)

        for g, mu in enumerate(self.mu_grid.to(device)):
            # map μ (R) through sigmoid->affine to an L1 candidate center (for logging only)
            L1_center = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(mu)
            mu_vec = torch.full((N,), float(mu), device=device)  # μ lives in Normal-space

            guide_mu = lambda y, s: self.guide_L1_fixed_Z_frozen(
                y, s, mu_vec,
                ZF_re_true=1000.0, ZF_im_true=50.0,
                ZL_re_true=100.0, ZL_im_true=-5.0
            )
            # small warmup (no params to learn; stabilizes MC)
            svi = SVI(self.model, guide_mu, pyro.optim.Adam({"lr": self.inner_lr}), loss=loss_obj)
            for _ in range(5):
                _ = svi.step(y_ri, sig_f)

            # batch ELBO
            loss = loss_obj.loss(self.model, guide_mu, y_ri, sig_f)  # scalar = -ELBO
            elbo_batch[g] = -loss

            # optional per-obs ELBO vector
            if return_per_obs:
                elbo_per_obs[g] = self._per_obs_elbo_once(guide_mu, y_ri, sig_f)
            # quick log
            print(f"[g={g:03d}] mu={float(mu):+.4f}  L1≈{float(L1_center):.2f}  "
                f"ELBO(batch)={elbo_batch[g].item():.3f}")

        # If you want topK per obs, do it on per-obs ELBO:
        if return_per_obs:
            _, topk_idx = torch.topk(elbo_per_obs, k=self.topK, dim=0)  # [K,N]
            topk_mu = self.mu_grid.to(device)[topk_idx].transpose(0, 1).contiguous()  # [N,K] in μ-space
        else:
            topk_mu = None

        return elbo_batch, elbo_per_obs, topk_mu



    def _profile_L1_mu_grid(self, y_ri, sig_f):
        """
        Scan μ over self.mu_grid with Z frozen; return:
        - elbo_batch:   [G]     (batch ELBO at each μ)
        - elbo_per_obs: [G, N]  (per-obs ELBO_n at each μ)
        - topk_mu:      [N, K]  (top-K μ per obs, in μ-space)

        NOTE: Z is NOT optimized here; we use a frozen-Z guide.
        """
        pyro.clear_param_store()
        device = y_ri.device
        N, F, _ = y_ri.shape
        G = self.mu_grid.numel()

        # storage
        elbo_batch   = torch.empty(G, device=device)
        elbo_per_obs = torch.empty(G, N, device=device)

        # choose which guide to use during profiling (Z frozen)
        def make_guide(mu_vec):
            # L1 centered around mu_vec, Z held fixed at some neutral mid-box values
            return (lambda y, s: self.guide_L1_fixed_Z_frozen(
                y, s, mu_vec,
                ZF_re_true=1000.0, ZF_im_true=0.0,
                ZL_re_true=100.0,  ZL_im_true=0.0
            ))

        loss_obj = Trace_ELBO(num_particles=self.num_particles)

        for g, mu in enumerate(self.mu_grid.to(device)):
            L1_center = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(mu)
            mu_vec = torch.full((N,), float(mu), device=device)  # [N]

            guide_mu = make_guide(mu_vec)

            # NO SVI / NO OPTIMIZATION HERE:
            # just evaluate the ELBO once at this μ with frozen Z
            batch_loss = loss_obj.loss(self.model, guide_mu, y_ri, sig_f)
            elbo_batch[g] = -float(batch_loss)  # loss is -ELBO

            # per-observation ELBO at this μ (still with frozen Z)
            elbo_per_obs[g] = self._per_obs_elbo_once(guide_mu, y_ri, sig_f)

            print(f"[g={g}] mu={float(mu):+.4f}  L1≈{float(L1_center):.2f}  "
                f"ELBO_batch={elbo_batch[g].item():.3f}  ELBO_obs0={elbo_per_obs[g,0].item():.3f}")

        # Top-K μ per observation by per-obs ELBO
        # elbo_per_obs: [G, N] → topk_idx: [K, N]
        _, topk_idx = torch.topk(elbo_per_obs, k=self.topK, dim=0)
        # topk_mu: [N, K]
        topk_mu = self.mu_grid.to(device)[topk_idx].transpose(0, 1).contiguous()

        return topk_mu, elbo_batch, elbo_per_obs

    def predict(self, obs_tf, noise_var_f):
        """
        Jointly estimate parameters for a batch of N observations with:
        (1) 1-D ELBO grid search over μ (L1) with frozen Z (profiling),
        (2) full VI with pyro starting from the best μ seeds.

        Returns per-observation posterior MEDIANS for each constrained parameter.
        """
        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [N,F]."
        N, F = obs_tf.shape
        device = self.device

        # 0) Convert complex obs_tf [N, F] to real
        y_ri = torch.view_as_real(obs_tf)  # [N, F, 2]
        sig_f = torch.sqrt(noise_var_f / 2)  # [N, F]

        # 1) 1-D ELBO grid over the variational mean (mu) of L1 (Z frozen)
        topk_mu, elbo_batch, elbo_per_obs = self._profile_L1_mu_grid(y_ri, sig_f)  # [N,K], [G], [G,N]

        # For diagnostics: convert μ seeds to L1-space
        topk_L1 = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(topk_mu)  # [N,K]
        print("topk_mu in L1 space", topk_L1)

        # --- diagnostics: count how many seeds land near true vs wrong basin ---
        true_L1  = 250.0
        wrong_L1 = 750.0
        tol = 20.0

        col0 = topk_L1[:, 0]  # [N]
        first_at_750 = torch.isclose(col0, torch.tensor(wrong_L1, device=topk_L1.device), atol=tol).sum().item()
        first_at_250 = torch.isclose(col0, torch.tensor(true_L1,  device=topk_L1.device), atol=tol).sum().item()
        print("First-column seeds at 750:", first_at_750)
        print("First-column seeds at 250:", first_at_250)
        print(f"First-column seeds at neither: {col0.numel() - first_at_750 - first_at_250}")

        mask_750_all = torch.isclose(
            topk_L1,
            torch.tensor(wrong_L1, device=topk_L1.device),
            atol=tol
        )  # [N,K]
        rows_all_750 = mask_750_all.all(dim=1).sum().item()
        print("Number of obs with all 3 seeds at 750:", rows_all_750)

        # --- 2) Run SVI inference from the top-K L1 seeds (Z NOT seeded from profiling) ---

        # Choose neutral Z seeds in unconstrained space: 0 → mid-range after sigmoid/tanh
        zero = torch.zeros(N, device=device)

        pyro.clear_param_store()
        ps = pyro.get_param_store()
        print(ps.keys())

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
            # L1_loc from μ seed k (in unconstrained space)
            set_param("L1_loc", topk_mu[:, k], fixed_sigma=self.fixed_sigma)

            # Z locs set to 0 → mid-range after sigmoid/tanh; no info from profiling
            set_param("ZF_re_loc", zero, default_scale=0.25)
            set_param("ZF_im_loc", zero, default_scale=0.25)
            set_param("ZL_re_loc", zero, default_scale=0.25)
            set_param("ZL_im_loc", zero, default_scale=0.25)

        best_loss = float("inf")
        best_state = None
        best_k = None
        # For each L1 seed, run a full VI pass (you can keep best seed later if you like)
        for k in range(self.topK):
            seed_from(k)
            optimizer = optim.Adam({"lr": self.svi_lr})
            svi = SVI(self.model, self.guide, optimizer,
                    loss=Trace_ELBO(num_particles=self.num_particles))

            print_every = 25
            last_loss = None

            for step in range(self.svi_steps):
                loss = svi.step(y_ri, sig_f)
                last_loss = loss
                if step % print_every == 0:
                    ps = pyro.get_param_store()

                    L1_loc  = ps["L1_loc"].detach()
                    ZFr_loc = ps["ZF_re_loc"].detach()
                    ZFi_loc = ps["ZF_im_loc"].detach()
                    ZLr_loc = ps["ZL_re_loc"].detach()
                    ZLi_loc = ps["ZL_im_loc"].detach()

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

                    print(f"[seed {k} step {step}] ELBO={loss:.3f} | "
                        f"L1 {mL1:.2f}±{sL1:.2f} | "
                        f"ZF_re {mZFr:.1f}±{sZFr:.1f} | ZF_im {mZFi:.1f}±{sZFi:.1f} | "
                        f"ZL_re {mZLr:.1f}±{sZLr:.1f} | ZL_im {mZLi:.1f}±{sZLi:.1f}")

                    idx = torch.tensor([0, 1, 2], device=L1_mean.device)
                    print("   samples:",
                        "L1",     L1_mean[idx].tolist(),
                        "ZF_re",  ZFr_mean[idx].tolist(),
                        "ZF_im",  ZFi_mean[idx].tolist())
            if last_loss is not None and last_loss < best_loss:
                best_loss = last_loss
                best_k = k
                # Deep copy all params
                ps = pyro.get_param_store()
                best_state = {name: p.detach().clone() for name, p in ps.items()}
        
        print(f"Best seed index: {best_k}, best loss: {best_loss:.3f}")

        # Restore best seed parameters into the param store
        if best_state is not None:
            ps = pyro.get_param_store()
            for name, best_val in best_state.items():
                if name in ps:
                    ps[name].data.copy_(best_val)
                else:
                    # In practice, everything should already exist, but just in case:
                    pyro.param(name, best_val.clone())

        # 3) Posterior summaries AFTER last seed’s SVI (you can optionally track best seed)
        with torch.no_grad():
            ps = pyro.get_param_store()

            def tparam(name):
                return ps[name].to(device)

            # L1
            L1_loc   = tparam("L1_loc")
            L1_scale = tparam("L1_scale")
            q_L1 = dist.TransformedDistribution(
                dist.Normal(L1_loc, L1_scale),
                [SigmoidTransform(), AffineTransform(loc=self.L1_lo, scale=(self.L1_hi - self.L1_lo))]
            )

            # ZF_re
            ZFr_loc   = tparam("ZF_re_loc")
            ZFr_scale = tparam("ZF_re_scale")
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )

            # ZF_im
            ZFi_loc   = tparam("ZF_im_loc")
            ZFi_scale = tparam("ZF_im_scale")
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )

            # ZL_re
            ZLr_loc   = tparam("ZL_re_loc")
            ZLr_scale = tparam("ZL_re_scale")
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )

            # ZL_im
            ZLi_loc   = tparam("ZL_im_loc")
            ZLi_scale = tparam("ZL_im_scale")
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )

            # MC posterior means
            S = getattr(self, "posterior_samples", 64)
            L1_s  = q_L1.rsample((S,))
            ZFr_s = q_ZFr.rsample((S,))
            ZFi_s = q_ZFi.rsample((S,))
            ZLr_s = q_ZLr.rsample((S,))
            ZLi_s = q_ZLi.rsample((S,))

            L1_mean  = L1_s.mean(0)
            ZFr_mean = ZFr_s.mean(0)
            ZFi_mean = ZFi_s.mean(0)
            ZLr_mean = ZLr_s.mean(0)
            ZLi_mean = ZLi_s.mean(0)

            # Medians via loc in latent space (monotone transforms)
            L1_med  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
            ZFr_med = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
            ZFi_med = self.ImZF_max * torch.tanh(ZFi_loc)
            ZLr_med = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
            ZLi_med = self.ImZL_max * torch.tanh(ZLi_loc)

            L1    = L1_med
            ZF_re = ZFr_med
            ZF_im = ZFi_med
            ZL_re = ZLr_med
            ZL_im = ZLi_med

        return {
            "L1":    L1.float().cpu().numpy(),
            "ZF_re": ZF_re.float().cpu().numpy(),
            "ZF_im": ZF_im.float().cpu().numpy(),
            "ZL_re": ZL_re.float().cpu().numpy(),
            "ZL_im": ZL_im.float().cpu().numpy(),
        }

    # def _profile_L1_mu_grid(self, y_ri, sig_f):
    #     """
    #     Scan μ over self.mu_grid; return:
    #     - elbo_batch: [G]   (batch ELBO at each μ)
    #     - elbo_per_obs: [G, N] (per-obs ELBO_n at each μ)
    #     Also returns top-K μ per obs (in μ-space): [N, K]
    #     """
    #     pyro.clear_param_store()
    #     device = y_ri.device
    #     N, F, _ = y_ri.shape
    #     G = self.mu_grid.numel()

    #     # storage
    #     elbo_batch = torch.empty(G, device=device)
    #     elbo_per_obs = torch.empty(G, N, device=device)
    #     Z_snap = torch.empty(G, N, 4, device=device)

    #     # choose which guide to use during profiling:
    #     def make_guide(mu_vec):
    #         # Normal VI on Z’s with warm-start, L1 fixed around mu
    #         # return (lambda y, s: self.guide_L1_fixed(y, s, mu_vec))
    #         # For frozen-Z sanity check, swap to:
    #         return (lambda y, s: self.guide_L1_fixed_Z_frozen(
    #         y, s, mu_vec, ZF_re_true=1000.0, ZF_im_true=0.0, ZL_re_true=100.0, ZL_im_true=0.0))
        
    #     loss_obj = Trace_ELBO(num_particles=self.num_particles)
    #     for g, mu in enumerate(self.mu_grid.to(device)):
            
    #         L1_center = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(mu)
    #         mu_vec = torch.full((N,), float(mu), device=device) 

    #         guide_mu = make_guide(mu_vec)
    #         svi = SVI(self.model, guide_mu, pyro.optim.Adam({"lr": self.inner_lr}),
    #                 loss=loss_obj)

    #         for _ in range(self.inner_steps):
    #             _ = svi.step(y_ri, sig_f)

    #         # Save Z params at this μ after SVI
    #         # ps = pyro.get_param_store()
    #         # # unconstrained loc/scale (shape [N])
    #         # ZFr_loc   = ps["ZF_re_loc"].detach() 
    #         # ZFr_scale = ps["ZF_re_scale"].detach()
    #         # ZFi_loc   = ps["ZF_im_loc"].detach()
    #         # ZFi_scale = ps["ZF_im_scale"].detach()
    #         # ZLr_loc   = ps["ZL_re_loc"].detach()
    #         # ZLr_scale = ps["ZL_re_scale"].detach()
    #         # ZLi_loc   = ps["ZL_im_loc"].detach()
    #         # ZLi_scale = ps["ZL_im_scale"].detach()

    #         # Z_snap[g, :, 0]   = ZFr_loc
    #         # Z_snap[g, :, 1]   = ZFi_loc
    #         # Z_snap[g, :, 2]   = ZLr_loc
    #         # Z_snap[g, :, 3]   = ZLi_loc

    #         #ELBO batch + per_obs loss
    #         batch_loss = loss_obj.loss(self.model, guide_mu, y_ri, sig_f)
    #         elbo_batch[g] = -float(batch_loss) #Negative because .loss returns neg ELBO because usually want to minimize

    #         elbo_per_obs[g] = self._per_obs_elbo_once(guide_mu, y_ri, sig_f)

    #         print(f"[g={g}] mu={float(mu):+.4f}  L1≈{float(L1_center):.2f}  "
    #             f"ELBO_batch={elbo_batch[g].item():.3f}  ELBO_obs0={elbo_per_obs[g,0].item():.3f}")

    #     # Top-K indices for μ per observation by per-obs ELBO
    #     _, topk_idx = torch.topk(elbo_per_obs, k=self.topK, dim=0)  # [G,N] -> [K,N]
    #     topk_mu = self.mu_grid.to(device)[topk_idx].transpose(0, 1).contiguous()  # [K,N] -> [N,K]

    #     Z_topk_u_loc = Z_snap[topk_idx, torch.arange(N, device=device)].transpose(0,1).contiguous()   # [N,K,4]
    #     return topk_mu, elbo_batch, elbo_per_obs, Z_topk_u_loc

    # def predict(self, obs_tf, noise_var_f):
    #     """
    #     Jointly estimate parameters for a batch of N observations with mu of L1 grid search + VI using pyro. 
    #     Returns per-observation posterior MEANS and MEDIANS for each constrained parameter. 
    #     Args
    #     ----
    #     obs_tf:      [N,F] complex tensor
    #     noise_var_f: [N,F] float tensor

    #     Returns
    #     -------
    #     dict of numpy arrays:
    #         {
    #           "L1":    [N],
    #           "ZF_re": [N], "ZF_im": [N],
    #           "ZL_re": [N], "ZL_im": [N]
    #         }
    #     """

    #     assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [N,F]."
    #     N, F = obs_tf.shape
    #     device = self.device

    #     # 0) Convert complex obs_tf [N, F] to real
    #     y_ri = torch.view_as_real(obs_tf) #[N, F, 2]
    #     sig_f = torch.sqrt(noise_var_f/2) #[N, F] Divide by 2 because var/2 for real/imag parts split for Complex Gaussian
    #     # 1) 1-D ELBO grid over the variational mean (mu) of L1 with warm restarts for ZF, ZL
    #     topk_mu, _, _, topk_Z = self._profile_L1_mu_grid(y_ri, sig_f) #[N,K], [N,K,4]

    #     topk_L1 = self.L1_lo + (self.L1_hi - self.L1_lo) * torch.sigmoid(topk_mu)  # [N,K]
    #     topk_ZFre = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(topk_Z[:, :, 0])
    #     topk_ZFim = self.ImZF_max * torch.tanh(topk_Z[:, :, 1])
    #     topk_ZLre = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(topk_Z[:, :, 2])
    #     topk_ZLim = self.ImZL_max * torch.tanh(topk_Z[:, :, 3])
    #     print("topk_mu in L1 space", topk_L1)

    #     true_L1  = 250.0
    #     wrong_L1 = 750.0
    #     tol = 20.0  # or whatever tolerance makes sense

    #     # --- 1) First column: how many are at the wrong value 750? ---
    #     col0 = topk_L1[:, 0]  # [N]

    #     first_at_750 = torch.isclose(
    #         col0,
    #         torch.tensor(wrong_L1, device=topk_L1.device),
    #         atol=tol
    #     ).sum().item()

    #     first_at_250 = torch.isclose(
    #         col0,
    #         torch.tensor(true_L1, device=topk_L1.device),
    #         atol=tol
    #     ).sum().item()

    #     print("First-column seeds at 750:", first_at_750)
    #     print("First-column seeds at 250:", first_at_250)
    #     print(f"First-column seeds at neither: {col0.numel() - first_at_750 - first_at_250}")

    #     mask_750_all = torch.isclose(
    #         topk_L1,
    #         torch.tensor(wrong_L1, device=topk_L1.device),
    #         atol=tol
    #     )  # [N, K] bool

    #     # Number of rows where *all* K seeds are at 750
    #     rows_all_750 = mask_750_all.all(dim=1).sum().item()
    #     print("Number of obs with all 3 seeds at 750:", rows_all_750)

    #     # print("topk ZFre", topk_ZFre)
    #     # print("topk ZFim", topk_ZFim)
    #     # print("topk ZLre", topk_ZLre)
    #     # print("topk ZLim", topk_ZLim)

    #     #Choose only top seed for now
    #     top_mu = topk_mu[:, 0] # [N]
    #     top_Z = topk_Z[:, 0, :] #[N,4]
        
    #     # 2) Run SVI inference from the top k seeds
    #     ps = pyro.get_param_store()
    #     print(ps.keys())
    #     def set_param(name, value, default_scale=0.25, fixed_sigma=None):
    #         value = value.detach().to(device)
    #         if name in ps:
    #             ps[name].data.copy_(value)
    #         else:
    #             pyro.param(name, value.clone())
            
    #         # ensure a matching _scale exists
    #         sname = name.replace("_loc", "_scale")
    #         if sname not in ps:
    #             init_scale = (fixed_sigma if (fixed_sigma is not None and name.endswith("L1_loc"))
    #                         else default_scale)
    #             pyro.param(sname, torch.full_like(value, init_scale),
    #                     constraint=constraints.positive)
    #     def seed_from(k):
    #         set_param("L1_loc",    topk_mu[:, k], fixed_sigma=self.fixed_sigma)
    #         set_param("ZF_re_loc", topk_Z[:, k, 0], default_scale=0.25)
    #         set_param("ZF_im_loc", topk_Z[:, k, 1], default_scale=0.25)
    #         set_param("ZL_re_loc", topk_Z[:, k, 2], default_scale=0.25)
    #         set_param("ZL_im_loc", topk_Z[:, k, 3], default_scale=0.25)

    #     for k in range(self.topK):
    #         # 1) reseed params for seed k
    #         seed_from(k)
    #         optimizer = optim.Adam({"lr": self.svi_lr})
    #         svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(num_particles=self.num_particles))
    #         print_every = 25
    #         for step in range(self.svi_steps):
    #         #print("this runs")
    #             loss = svi.step(y_ri, sig_f)
    #             if step % print_every == 0:
    #                 ps = pyro.get_param_store()
            
    #                 L1_loc  = ps["L1_loc"].detach() #[N]
    #                 ZFr_loc = ps["ZF_re_loc"].detach()
    #                 ZFi_loc = ps["ZF_im_loc"].detach()
    #                 ZLr_loc = ps["ZL_re_loc"].detach()
    #                 ZLi_loc = ps["ZL_im_loc"].detach()

    #                 L1_mean  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
    #                 ZFr_mean = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
    #                 ZFi_mean = self.ImZF_max * torch.tanh(ZFi_loc)
    #                 ZLr_mean = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
    #                 ZLi_mean = self.ImZL_max * torch.tanh(ZLi_loc)

    #                 # Print a compact summary (mean ± std across N) and a few examples
    #                 def stats(x): 
    #                     return float(x.mean()), float(x.std())

    #                 mL1,sL1   = stats(L1_mean)
    #                 mZFr,sZFr = stats(ZFr_mean); mZFi,sZFi = stats(ZFi_mean)
    #                 mZLr,sZLr = stats(ZLr_mean); mZLi,sZLi = stats(ZLi_mean)

    #                 print(f"[step {step}] ELBO={loss:.3f} | "
    #                     f"L1 {mL1:.2f}±{sL1:.2f} | "
    #                     f"ZF_re {mZFr:.1f}±{sZFr:.1f} | ZF_im {mZFi:.1f}±{sZFi:.1f} | "
    #                     f"ZL_re {mZLr:.1f}±{sZLr:.1f} | ZL_im {mZLi:.1f}±{sZLi:.1f}")

    #                 #Print a few specific rows (e.g., the first three obs)
    #                 idx = torch.tensor([0,1,2], device=L1_mean.device)
    #                 print("   samples:",
    #                     "L1",  L1_mean[idx].tolist(),
    #                     "ZF_re", ZFr_mean[idx].tolist(),
    #                     "ZF_im", ZFi_mean[idx].tolist())
                

    #     #3) Return posterior medians as our estimates
    #     with torch.no_grad():
    #         ps = pyro.get_param_store()

    #         def tparam(name):  # helper
    #             t = ps[name]
    #             return t.to(device)

    #         # L1 ~ Sigmoid -> Affine[L1_lo, L1_hi]
    #         L1_loc = tparam("L1_loc")
    #         L1_scale = tparam("L1_scale")
    #         q_L1 = dist.TransformedDistribution(
    #             dist.Normal(L1_loc, L1_scale),
    #             [SigmoidTransform(), AffineTransform(loc=self.L1_lo, scale=(self.L1_hi - self.L1_lo))]
    #         )

    #         # ZF_re ~ Sigmoid -> Affine[ReZF_lo, ReZF_hi]
    #         ZFr_loc = tparam("ZF_re_loc")
    #         ZFr_scale = tparam("ZF_re_scale")
    #         q_ZFr = dist.TransformedDistribution(
    #             dist.Normal(ZFr_loc, ZFr_scale),
    #             [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
    #         )

    #         # ZF_im ~ Tanh -> scale ImZF_max
    #         ZFi_loc   = tparam("ZF_im_loc")
    #         ZFi_scale = tparam("ZF_im_scale")
    #         q_ZFi = dist.TransformedDistribution(
    #             dist.Normal(ZFi_loc, ZFi_scale),
    #             [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
    #         )

    #         # ZL_re ~ Sigmoid -> Affine[ReZL_lo, ReZL_hi]
    #         ZLr_loc = tparam("ZL_re_loc")
    #         ZLr_scale = tparam("ZL_re_scale")
    #         q_ZLr = dist.TransformedDistribution(
    #             dist.Normal(ZLr_loc, ZLr_scale),
    #             [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
    #         )

    #         # ZL_im ~ Tanh -> scale ImZL_max
    #         ZLi_loc = tparam("ZL_im_loc")
    #         ZLi_scale = tparam("ZL_im_scale")
    #         q_ZLi = dist.TransformedDistribution(
    #             dist.Normal(ZLi_loc, ZLi_scale),
    #             [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
    #         )

    #         # --- Posterior summaries per observation 
    #         S = getattr(self, "posterior_samples", 64)  # MC samples for summaries
    #         # sample shapes: [S, N]
    #         L1_s = q_L1.rsample((S,))
    #         ZFr_s = q_ZFr.rsample((S,))
    #         ZFi_s = q_ZFi.rsample((S,))
    #         ZLr_s = q_ZLr.rsample((S,))
    #         ZLi_s = q_ZLi.rsample((S,))

    #         # Posterior means across samples (dim=0) - mean requires MC sampling
    #         L1_mean = L1_s.mean(0)
    #         ZFr_mean = ZFr_s.mean(0);  
    #         ZFi_mean = ZFi_s.mean(0);  
    #         ZLr_mean = ZLr_s.mean(0);  
    #         ZLi_mean = ZLi_s.mean(0);  

    #         # Posterior median is just transformation of loc in Normal space (monotone transforms preserve median but not mean)
    #         L1_med  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
    #         ZFr_med = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
    #         ZFi_med = self.ImZF_max * torch.tanh(ZFi_loc)
    #         ZLr_med = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
    #         ZLi_med = self.ImZL_max * torch.tanh(ZLi_loc)

    #         #Return medians for now
    #         ZF_re = ZFr_med
    #         ZF_im = ZFi_med
    #         ZL_re = ZLr_med
    #         ZL_im = ZLi_med
    #         L1 = L1_med

    #     # --- Return numpy arrays on CPU
    #     return {
    #         "L1":    L1.float().cpu().numpy(),
    #         "ZF_re": ZF_re.float().cpu().numpy(),
    #         "ZF_im": ZF_im.float().cpu().numpy(),
    #         "ZL_re": ZL_re.float().cpu().numpy(),
    #         "ZL_im": ZL_im.float().cpu().numpy(),
    #     }
