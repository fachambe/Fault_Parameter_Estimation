# core/likelihoods.py
import torch

class ComplexGaussianLik:
    """
    Vectorized Complex Gaussian NLL over all theta candidates using PyTorch (GPU-compatible).

    Arguments:
    - obs_tf: shape [N, F]
    - pred_tf: shape [K, F]
    - noise_var_f: shape [N, F] or [N, 1]
    """

    def __call__(self, obs_tf, pred_tf, noise_var_f):
        """
        Returns NLL [K] by summing over N and F.
        Use for loss landscape plotting (N=1) or pooled MLE (one estimate for all N).
        """
        diff = obs_tf.unsqueeze(0) - pred_tf.unsqueeze(1)  # [1, N, F] - [K, 1, F] = [K, N, F]
        nll = (torch.abs(diff)**2 / noise_var_f.unsqueeze(0)).sum(dim=(1,2))  # [K]
        return nll

    def nll_elementwise(self, obs_tf, pred_tf, noise_var_f):
        """
        Returns NLL [N] by summing over F.
        Element-wise: obs_tf[n] vs pred_tf[n].
        Use for batched gradient optimization where each observation has its own parameter.
        """
        diff = obs_tf - pred_tf  # [N, F]
        return (torch.abs(diff)**2 / noise_var_f).sum(dim=1)  # [N]

    def nll_matrix(self, obs_tf, pred_tf, noise_var_f):
        """
        Returns NLL [K, N] by summing over F only.
        Use for grid search MLE: pick best (lowest NLL) grid candidate per observation.
        """
        diff = obs_tf.unsqueeze(0) - pred_tf.unsqueeze(1)  # [1,N,F] - [K,1,F] = [K,N,F]
        nv = noise_var_f.unsqueeze(0)                       # [N,F] -> [1,N,F]
        return ((torch.abs(diff) ** 2) / nv).sum(dim=2)     # [K,N] sum over F 
    