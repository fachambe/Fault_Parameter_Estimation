# core/likelihoods.py
import torch

class ComplexGaussianLik:
    """
    Vectorized Complex Gaussian log-likelihood over all theta candidates using PyTorch (GPU-compatible)
    
    Arguments: 
    - obs_tf: shape [N, F]
    - pred_tf: shape [K, F]
    - noise_var_f: shape [N, F] (noise var per freq)
    """
        
    def __call__(self, obs_tf, pred_tf, noise_var_f):
        """
        Returns [K] by summing over N and F.
        Use this for single-sample predict() (N == 1) or when you want one score over all obs.
        """
        diff = obs_tf.unsqueeze(0) - pred_tf.unsqueeze(1)        # [1, N, F] - [K, 1, F] = [K, N, F]
        ll = -(torch.abs(diff)**2 / noise_var_f.unsqueeze(0)).sum(dim=(1,2))  #[K, N, F] / [1, N, F] (noise var unsqueeze(0)) = [K, N, F] -> sum(dim=(1, 2)) = [K]
        return ll       
    
    def score_matrix(self, obs_tf, pred_tf, noise_var_f):
        """
        Returns [K, N] by summing over F only.
        Use this for batched predict_batch(): pick best candidate per observation.
        """
        diff = obs_tf.unsqueeze(0) - pred_tf.unsqueeze(1)              # [1,N,F] - [K,1,F] = [K,N,F]
        nv   = noise_var_f.unsqueeze(0)                                # [N,F] -> [1,N,F]
        return -((torch.abs(diff) ** 2) / nv).sum(dim=2)         # [K,N] sum over F 
    

    
class RiceanLikelihood:
    """
    Vectorized Rician (magnitude) log-likelihood over all L1 candidates.

    Arguments:
    - r_obs:        [N, F] observed magnitudes (or complex; will take abs)
    - pred_tf:      [K, F] complex predicted transfer functions (per candidate)
    - noise_var_f:  [N, F] noise variance per frequency (per observation)

    Returns:
    - __call__:         [K]   total log-likelihood per candidate (sum over N and F)
    - score_matrix():   [K,N] log-likelihood per (candidate, observation) (sum over F)
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def _rician_ll_terms(self, r_exp, A_exp, sigma2):
        """
        r_exp:   [K, N, F]
        A_exp:   [K, N, F]
        sigma2:  [K, N, F]
        returns: [K, N, F] elementwise log-likelihood contributions
        """
        eps = self.eps
        # Modified Bessel I0 with exponential scaling for stability
        bessel_arg = (2.0 * r_exp * A_exp) / (sigma2 + eps)           # [K,N,F]
        ll_bnf = (
            torch.log(r_exp + eps)                                    # log r
            - torch.log(sigma2 + eps)                                 # - log σ^2
            - (r_exp**2 + A_exp**2) / (sigma2 + eps)                  # - (r^2 + A^2)/σ^2
            + torch.log(torch.special.i0e(bessel_arg + eps))          # log I0e(·)
            + bessel_arg                                              # add back exp scaling
        )
        return ll_bnf

    def __call__(self, r_obs, pred_tf, noise_var_f):
        """
        Sum over N and F -> [K]
        """
        # Ensure magnitudes for observations; predictions stay complex but we use their magnitude A
        r = torch.abs(r_obs) if torch.is_complex(r_obs) else r_obs    # [N,F]
        A = torch.abs(pred_tf)                                        # [K,F]

        # Broadcast to [K,N,F]
        r_exp   = r.unsqueeze(0)                                      # [1,N,F] -> [K,N,F]
        A_exp   = A.unsqueeze(1)                                      # [K,1,F] -> [K,N,F]
        sigma2  = noise_var_f.unsqueeze(0)                            # [1,N,F] -> [K,N,F]

        ll_bnf = self._rician_ll_terms(r_exp, A_exp, sigma2)          # [K,N,F]
        return ll_bnf.sum(dim=(1, 2))                                 # [K]

    def score_matrix(self, r_obs, pred_tf, noise_var_f):
        """
        Sum over F only -> [K, N]
        Use this to pick the best candidate per observation in batched mode.
        """
        r = torch.abs(r_obs) if torch.is_complex(r_obs) else r_obs    # [N,F]
        A = torch.abs(pred_tf)                                        # [K,F]

        r_exp   = r.unsqueeze(0)                                      # [K,N,F]
        A_exp   = A.unsqueeze(1)                                      # [K,N,F]
        sigma2  = noise_var_f.unsqueeze(0)                            # [K,N,F]

        ll_bnf = self._rician_ll_terms(r_exp, A_exp, sigma2)          # [K,N,F]
        return ll_bnf.sum(dim=2)                                      # [K,N]

class WrappedPhaseGaussianLikelihood:
    """
    Vectorized wrapped-Gaussian (phase) log-likelihood over all L1 candidates.

    Arguments:
    - obs_tf:       [N, F] complex observed transfer functions
    - pred_tf:      [K, F] complex predicted transfer functions (per candidate)
    - noise_var_f:  [N, F] complex noise variance per frequency for obs
                     (i.e., variance of complex noise on H; used to approximate phase noise)
    Returns:
    - __call__:       [K]   total log-likelihood per candidate (sum over N and F)
    - score_matrix(): [K,N] log-likelihood per (candidate, observation) (sum over F)
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        # Wrap angles into (-pi, pi]
        return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi

    def _elementwise_ll(self, phase_obs, phase_pred, abs_H_pred, noise_var_f):
        """
        phase_obs:   [K, N, F]
        phase_pred:  [K, N, F]
        abs_H_pred:  [K, N, F]
        noise_var_f: [K, N, F]
        returns:     [K, N, F] elementwise log-likelihood contributions
        """
        eps = self.eps

        # Approx phase variance: Var[angle] ≈ Var[complex noise] / |H|^2
        phase_var = noise_var_f / (abs_H_pred ** 2 + eps)  # [K,N,F]

        # Wrapped difference
        delta = self._wrap_to_pi(phase_obs - phase_pred)   # [K,N,F]

        # Gaussian on circle (wrapped via minimal distance) approximation
        ll = -0.5 * (delta ** 2) / (phase_var + eps) - 0.5 * torch.log(2.0 * torch.pi * phase_var + eps)
        return ll

    def __call__(self, obs_tf, pred_tf, noise_var_f):
        """
        Sum over N and F -> [K]
        """
        # Angles
        phase_obs = torch.angle(obs_tf)                    # [N,F]
        phase_pred = torch.angle(pred_tf)                  # [K,F]
        abs_H_pred = torch.abs(pred_tf)                    # [K,F]

        # Broadcast to [K,N,F]
        phase_obs = phase_obs.unsqueeze(0)                 # [1,N,F] -> [K,N,F]
        phase_pred = phase_pred.unsqueeze(1)               # [K,1,F] -> [K,N,F]
        abs_H_pred = abs_H_pred.unsqueeze(1)               # [K,1,F] -> [K,N,F]
        noise_var  = noise_var_f.unsqueeze(0)              # [1,N,F] -> [K,N,F]

        ll_elem = self._elementwise_ll(phase_obs, phase_pred, abs_H_pred, noise_var)  # [K,N,F]
        return ll_elem.sum(dim=(1, 2))                     # [K]

    def score_matrix(self, obs_tf, pred_tf, noise_var_f):
        """
        Sum over F only -> [K, N]
        Use this to pick the best candidate per observation in batched mode.
        """
        phase_obs = torch.angle(obs_tf).unsqueeze(0)       # [1,N,F] -> [K,N,F]
        phase_pred = torch.angle(pred_tf).unsqueeze(1)     # [K,1,F] -> [K,N,F]
        abs_H_pred = torch.abs(pred_tf).unsqueeze(1)       # [K,1,F]
        noise_var  = noise_var_f.unsqueeze(0)              # [K,N,F]

        ll_elem = self._elementwise_ll(phase_obs, phase_pred, abs_H_pred, noise_var)  # [K,N,F]
        return ll_elem.sum(dim=2)                          # [K,N]
