# estimators/base.py

class Estimator:
    """
    For single-parameter estimators:
      - Returns np.ndarray of shape [N]
        * dtype=float32 for real params (L1, ReZF, ImZF, ReZL, ImZL)
        * dtype=complex64 for full complex param (ZF or ZL with estimate=None)

    For multi-parameter estimators:
      - Returns np.ndarray of shape [N, D] (float32 or complex64) in a fixed column order [L1, ZF, ZL]. 
    """
    def __init__(self):
        raise NotImplementedError

    def fit(self, train_data):
        return self  # Only for ML models

    def predict(self, obs_tf, noise_var):
        raise NotImplementedError

    @staticmethod
    def _ensure_batch(x):
        return x.unsqueeze(0) if x.ndim == 1 else x