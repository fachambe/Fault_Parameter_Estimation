# estimators/base.py

class Estimator:
    """
    For single-parameter estimators:
      - Returns a single numpy array of shape [M]
    For multi-parameter estimators:
      - Returns dict of numpy arrays of shape [M] for each parameter 
    """
    def __init__(self):
        raise NotImplementedError

    def fit(self, X_train, y_train):
        return self  # Only for ML models

    def predict(self, obs_tf, noise_var):
        raise NotImplementedError

    @staticmethod
    def _ensure_batch(x):
        return x.unsqueeze(0) if x.ndim == 1 else x