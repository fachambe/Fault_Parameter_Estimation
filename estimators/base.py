# estimators/base.py
import torch


# ---------------------------------------------------------------------
# Parameterization helpers (used by all gradient-based estimators)
# ---------------------------------------------------------------------
def u_to_theta_sigmoid(u, lo, hi):
    """Map unconstrained u in R -> theta in (lo, hi) using sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(u)


def theta_to_u_sigmoid(x, lo, hi, eps: float = 1e-6):
    """Inverse of u_to_theta_sigmoid (logit transform)."""
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)
    return torch.log(z) - torch.log1p(-z)


def u_to_theta_tanh(u, max_abs):
    """Map unconstrained u in R -> theta in (-max_abs, +max_abs) using tanh."""
    return max_abs * torch.tanh(u)


def theta_to_u_tanh(x, max_abs, eps: float = 1e-6):
    """Inverse of u_to_theta_tanh (atanh transform)."""
    r = (x / max_abs).clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log1p(r) - 0.5 * torch.log1p(-r)


class Estimator:
    """
    Base class for all estimators.

    For single-parameter estimators:
      - Returns a single numpy array of shape [N]
    For multi-parameter estimators:
      - Returns dict of numpy arrays of shape [N] for each parameter
    """
    def __init__(self, fm, likelihood, target, fixed, true_range=None, mode=None, device="cuda"):
        self.fm = fm
        self.lik = likelihood
        self.target = target
        self.device = device
        self.mode = mode
        self.fixed = fixed
        self.true_range = true_range

        # Parse fixed params: convert dict to complex if needed
        fx = {k: (complex(fixed[k]["re"], fixed[k]["im"]) if isinstance(fixed[k], dict) else fixed[k])
              for k in fixed}
        self.L1_fix = torch.as_tensor(fx["L1"], device=self.device, dtype=torch.float32) #scalar
        self.ZF_fix = torch.as_tensor(fx["ZF"], device=self.device, dtype=torch.cfloat)
        self.ZL_fix = torch.as_tensor(fx["ZL"], device=self.device, dtype=torch.cfloat)

        # Parameter ranges from config (optional - not needed for grid search)
        if true_range is not None:
            self.L1_lo = float(true_range["L1"]["min"])
            self.L1_hi = float(true_range["L1"]["max"])
            self.ReZF_lo = float(true_range["ZF"]["re"]["min"])
            self.ReZF_hi = float(true_range["ZF"]["re"]["max"])
            self.ImZF_max = float(true_range["ZF"]["im"]["max"])  # symmetric: [-max, +max]
            self.ReZL_lo = float(true_range["ZL"]["re"]["min"])
            self.ReZL_hi = float(true_range["ZL"]["re"]["max"])
            self.ImZL_max = float(true_range["ZL"]["im"]["max"])  # symmetric: [-max, +max]

    def _u_to_theta(self, u):
        """
        Map unconstrained u[..., 5] -> (L1, ZF, ZL).
        Works for u of shape [..., 5].
        """
        assert u.shape[-1] == 5, f"Expected last dim=5, got {u.shape}"
        u0, u1, u2, u3, u4 = u.unbind(dim=-1)

        L1   = u_to_theta_sigmoid(u0, self.L1_lo, self.L1_hi)
        ReZF = u_to_theta_sigmoid(u1, self.ReZF_lo, self.ReZF_hi)
        ImZF = u_to_theta_tanh(u2, self.ImZF_max)
        ReZL = u_to_theta_sigmoid(u3, self.ReZL_lo, self.ReZL_hi)
        ImZL = u_to_theta_tanh(u4, self.ImZL_max)

        ZF = torch.complex(ReZF, ImZF)
        ZL = torch.complex(ReZL, ImZL)
        return L1, ZF, ZL

    def _theta_to_u(self, L1, ZF, ZL):
        """
        Inverse map: theta -> u[..., 5].
        Works for L1, ZF, ZL shaped as the same leading dims [...].
        """
        u0 = theta_to_u_sigmoid(L1, self.L1_lo, self.L1_hi)
        u1 = theta_to_u_sigmoid(ZF.real, self.ReZF_lo, self.ReZF_hi)
        u2 = theta_to_u_tanh(ZF.imag, self.ImZF_max)
        u3 = theta_to_u_sigmoid(ZL.real, self.ReZL_lo, self.ReZL_hi)
        u4 = theta_to_u_tanh(ZL.imag, self.ImZL_max)
        return torch.stack([u0, u1, u2, u3, u4], dim=-1)

    def fit(self, X_train, y_train):
        return self  # Only for ML models

    def predict(self, obs_tf, noise_var):
        raise NotImplementedError

    @staticmethod
    def _ensure_batch(x):
        return x.unsqueeze(0) if x.ndim == 1 else x