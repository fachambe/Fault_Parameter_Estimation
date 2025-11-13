import math
import time
import torch
import pyro
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints, SigmoidTransform, TanhTransform, AffineTransform
import pyro.distributions as dist

from estimators.base import Estimator  # your base
torch.set_float32_matmul_precision("high")

def _sigmoid_to_range(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(u)

def _tanh_to_range(u: torch.Tensor, max_abs: float) -> torch.Tensor:
    return max_abs * torch.tanh(u)


class VI(Estimator):
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
      obs_tf:      [N,F] complex64/complex32 tensor (on device)
      noise_var_f: [N,F] float32 tensor (on device)
    """
    def __init__(self,
                 fm,
                 likelihood,
                 L: float = 1000.0,
                 device="cuda",
                 # VI hyperparams
                 svi_steps: int = 1000,
                 svi_lr: float = 1e-2,
                 num_particles: int = 1):
        self.fm = fm
        self.device = device
        self.lik = likelihood
        self.L = float(L)

        #VI
        self.svi_steps = svi_steps
        self.svi_lr = svi_lr
        self.num_particles = num_particles
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
            L1 = pyro.sample("L1",   dist.Uniform(
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
        # Currently all parameters initialized to mu = 0 which corresponds to middle of parameter range since sigmoid(0) = 0.5
        # L1 scale matters a lot - if you set it to 0.2 it will just chase end of line L = 1000 (same as 1D case)
        # If you set it to 0.02 it will just sit in local optimum and not move
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
            ZFr_scale = pyro.param("ZF_re_scale", torch.full((N,), 0.5, device=device),
                                constraint=constraints.positive)
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )
            pyro.sample("ZF_re", q_ZFr)

            # ZF_im in [-ImZF_max, +ImZF_max]
            ZFi_loc   = pyro.param("ZF_im_loc",   torch.zeros(N, device=device))
            ZFi_scale = pyro.param("ZF_im_scale", torch.full((N,), 0.5, device=device),
                                constraint=constraints.positive)
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )
            pyro.sample("ZF_im", q_ZFi)

            # ZL_re in [ReZL_lo, ReZL_hi]
            ZLr_loc   = pyro.param("ZL_re_loc",   torch.zeros(N, device=device))
            ZLr_scale = pyro.param("ZL_re_scale", torch.full((N,), 0.5, device=device),
                                constraint=constraints.positive)
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )
            pyro.sample("ZL_re", q_ZLr)

            # ZL_im in [-ImZL_max, +ImZL_max]
            ZLi_loc   = pyro.param("ZL_im_loc",   torch.zeros(N, device=device))
            ZLi_scale = pyro.param("ZL_im_scale", torch.full((N,), 0.5, device=device),
                                constraint=constraints.positive)
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )
            pyro.sample("ZL_im", q_ZLi)

        
  

    def predict(self, obs_tf, noise_var_f):
        """
        Jointly estimate parameters for a batch of N observations with VI using pyro. 
        Returns per-observation posterior MEANS and MEDIANS for each constrained parameter. 
        Args
        ----
        obs_tf:      [N,F] complex tensor
        noise_var_f: [N,F] float tensor

        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [N],
              "ZF_re": [N], "ZF_im": [N],
              "ZL_re": [N], "ZL_im": [N]
            }
        """

        assert obs_tf.dim() == 2 and noise_var_f.dim() == 2, "Expect shapes [N,F]."
        N, F = obs_tf.shape
        device = self.device

        # 0) Convert complex obs_tf [N, F] to real
        y_ri = torch.view_as_real(obs_tf) #[N, F, 2]
        sig_f = torch.sqrt(noise_var_f/2) #[N, F] Divide by 2 because var/2 for real/imag parts split for Complex Gaussian
        # 1) Run SVI inference
        pyro.clear_param_store()
        optimizer = optim.Adam({"lr": self.svi_lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO(num_particles=self.num_particles))
        print_every = 25
        print("y_ri shape", y_ri.shape)
        print("sig_f shape", sig_f.shape)
        print("y_ri device", y_ri.device)
        print("sig f device", sig_f.device)
        for step in range(self.svi_steps):
            loss = svi.step(y_ri, sig_f)
            if step % print_every == 0:
                ps = pyro.get_param_store()
                # Grab unconstrained locs (shape [N])
                L1_loc  = ps["L1_loc"].detach()
                ZFr_loc = ps["ZF_re_loc"].detach()
                ZFi_loc = ps["ZF_im_loc"].detach()
                ZLr_loc = ps["ZL_re_loc"].detach()
                ZLi_loc = ps["ZL_im_loc"].detach()

                # Map to constrained space (approximate means by transforming loc)
                L1_mean  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
                ZFr_mean = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
                ZFi_mean = self.ImZF_max * torch.tanh(ZFi_loc)
                ZLr_mean = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
                ZLi_mean = self.ImZL_max * torch.tanh(ZLi_loc)

                # Print a compact summary (mean ± std across N) and a few examples
                def stats(x): 
                    return float(x.mean()), float(x.std())

                mL1,sL1   = stats(L1_mean)
                mZFr,sZFr = stats(ZFr_mean); mZFi,sZFi = stats(ZFi_mean)
                mZLr,sZLr = stats(ZLr_mean); mZLi,sZLi = stats(ZLi_mean)

                print(f"[step {step}] ELBO={loss:.3f} | "
                    f"L1 {mL1:.2f}±{sL1:.2f} | "
                    f"ZF_re {mZFr:.1f}±{sZFr:.1f} | ZF_im {mZFi:.1f}±{sZFi:.1f} | "
                    f"ZL_re {mZLr:.1f}±{sZLr:.1f} | ZL_im {mZLi:.1f}±{sZLi:.1f}")

                # Optionally print a few specific rows (e.g., the first three obs)
                idx = torch.tensor([0,1,2], device=L1_mean.device)
                print("   samples:",
                    "L1",  L1_mean[idx].tolist(),
                    "ZF_re", ZFr_mean[idx].tolist(),
                    "ZF_im", ZFi_mean[idx].tolist())
         # --- Build the learned guide distributions (same transforms as in guide)
        # Note: parameters live in the param store after SVI; shapes are [N]
        with torch.no_grad():
            ps = pyro.get_param_store()

            def tparam(name):  # helper
                t = ps[name]
                return t.to(device)

            # L1 ~ Sigmoid -> Affine[L1_lo, L1_hi]
            L1_loc   = tparam("L1_loc")
            L1_scale = tparam("L1_scale")
            q_L1 = dist.TransformedDistribution(
                dist.Normal(L1_loc, L1_scale),
                [SigmoidTransform(), AffineTransform(loc=self.L1_lo, scale=(self.L1_hi - self.L1_lo))]
            )

            # ZF_re ~ Sigmoid -> Affine[ReZF_lo, ReZF_hi]
            ZFr_loc   = tparam("ZF_re_loc")
            ZFr_scale = tparam("ZF_re_scale")
            q_ZFr = dist.TransformedDistribution(
                dist.Normal(ZFr_loc, ZFr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZF_lo, scale=(self.ReZF_hi - self.ReZF_lo))]
            )

            # ZF_im ~ Tanh -> scale ImZF_max
            ZFi_loc   = tparam("ZF_im_loc")
            ZFi_scale = tparam("ZF_im_scale")
            q_ZFi = dist.TransformedDistribution(
                dist.Normal(ZFi_loc, ZFi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZF_max)]
            )

            # ZL_re ~ Sigmoid -> Affine[ReZL_lo, ReZL_hi]
            ZLr_loc   = tparam("ZL_re_loc")
            ZLr_scale = tparam("ZL_re_scale")
            q_ZLr = dist.TransformedDistribution(
                dist.Normal(ZLr_loc, ZLr_scale),
                [SigmoidTransform(), AffineTransform(loc=self.ReZL_lo, scale=(self.ReZL_hi - self.ReZL_lo))]
            )

            # ZL_im ~ Tanh -> scale ImZL_max
            ZLi_loc   = tparam("ZL_im_loc")
            ZLi_scale = tparam("ZL_im_scale")
            q_ZLi = dist.TransformedDistribution(
                dist.Normal(ZLi_loc, ZLi_scale),
                [TanhTransform(), AffineTransform(loc=0.0, scale=self.ImZL_max)]
            )

            # --- Posterior summaries per observation 
            S = getattr(self, "posterior_samples", 64)  # MC samples for summaries
            # sample shapes: [S, N]
            L1_s = q_L1.rsample((S,))
            ZFr_s = q_ZFr.rsample((S,))
            ZFi_s = q_ZFi.rsample((S,))
            ZLr_s = q_ZLr.rsample((S,))
            ZLi_s = q_ZLi.rsample((S,))

            # Posterior means across samples (dim=0) - note mean requires MC sampling
            L1_mean = L1_s.mean(0)
            ZFr_mean = ZFr_s.mean(0);  
            ZFi_mean = ZFi_s.mean(0);  
            ZLr_mean = ZLr_s.mean(0);  
            ZLi_mean = ZLi_s.mean(0);  

            # Posterior median is just transformation of loc in Normal space (monotone transforms preserve median but not mean)
            L1_med  = self.L1_lo   + (self.L1_hi - self.L1_lo) * torch.sigmoid(L1_loc)
            ZFr_med = self.ReZF_lo + (self.ReZF_hi - self.ReZF_lo) * torch.sigmoid(ZFr_loc)
            ZFi_med = self.ImZF_max * torch.tanh(ZFi_loc)
            ZLr_med = self.ReZL_lo + (self.ReZL_hi - self.ReZL_lo) * torch.sigmoid(ZLr_loc)
            ZLi_med = self.ImZL_max * torch.tanh(ZLi_loc)

            #Return medians for now
            ZF_re = ZFr_med
            ZF_im = ZFi_med
            ZL_re = ZLr_med
            ZL_im = ZLi_med
            L1 = L1_med

        # --- Return numpy arrays on CPU
        return {
            "L1":    L1.float().cpu().numpy(),
            "ZF_re": ZF_re.float().cpu().numpy(),
            "ZF_im": ZF_im.float().cpu().numpy(),
            "ZL_re": ZL_re.float().cpu().numpy(),
            "ZL_im": ZL_im.float().cpu().numpy(),
            # (optionally also return medians or scales if you want uncertainty)
            # "L1_med":  L1_s.median(0).values.float().cpu().numpy(),
            # "ZF_re_sd": ZFr_s.std(0).float().cpu().numpy(), ...
        }
