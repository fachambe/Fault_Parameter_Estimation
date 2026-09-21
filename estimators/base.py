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
    def __init__(self, fm, likelihood, network_params, device="cuda"):
        """
        Args:
            fm: Forward model instance
            likelihood: Likelihood function
            network_params: Dict with all parameter configs. Each param has: value, inferred, range
                   Example: {"L1": {"value": 250, "inferred": True, "range": (1, 999)}, ...}
            device: torch device
        """
        self.fm = fm
        self.lik = likelihood
        self.device = device
        self.network_params = network_params

        # Extract target parameter(s) from inferred flags
        self.targets = [k for k, v in network_params.items() if v["inferred"]]

        # For 1D estimators, ensure exactly one target
        if len(self.targets) == 1:
            self.target = self.targets[0]
        elif len(self.targets) == 0:
            raise ValueError("At least one parameter must have 'inferred': True")
        # Multi-parameter estimators can have multiple targets

        # Extract all parameter values (used as fixed values for non-target params)
        self.L1_val = network_params["L1"]["value"]
        self.ZF_re_val = network_params["ZF_re"]["value"]
        self.ZF_im_val = network_params["ZF_im"]["value"]
        self.ZL_re_val = network_params["ZL_re"]["value"]
        self.ZL_im_val = network_params["ZL_im"]["value"]

        # Store as torch tensors for forward model
        self.L1_fix = torch.as_tensor(self.L1_val, device=self.device, dtype=torch.float32)
        self.ZF_fix = torch.complex(
            torch.as_tensor(self.ZF_re_val, device=self.device, dtype=torch.float32),
            torch.as_tensor(self.ZF_im_val, device=self.device, dtype=torch.float32)
        )
        self.ZL_fix = torch.complex(
            torch.as_tensor(self.ZL_re_val, device=self.device, dtype=torch.float32),
            torch.as_tensor(self.ZL_im_val, device=self.device, dtype=torch.float32)
        )

        # Parameter ranges (for gradient-based estimators)
        self.L1_lo, self.L1_hi = network_params["L1"]["range"]
        self.ReZF_lo, self.ReZF_hi = network_params["ZF_re"]["range"]
        self.ImZF_lo, self.ImZF_hi = network_params["ZF_im"]["range"]
        self.ReZL_lo, self.ReZL_hi = network_params["ZL_re"]["range"]
        self.ImZL_lo, self.ImZL_hi = network_params["ZL_im"]["range"]

        # For tanh transforms (imaginary parts), store max absolute value
        self.ImZF_max = max(abs(self.ImZF_lo), abs(self.ImZF_hi))
        self.ImZL_max = max(abs(self.ImZL_lo), abs(self.ImZL_hi))

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