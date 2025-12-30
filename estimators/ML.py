import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from .base import Estimator

from scipy.io import savemat
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

class MLRegressor(Estimator):

    def __init__(self,
                 fm,
                 likelihood,
                 estimator_type, # "SVR", "GBR", "RFR", "HGBR"
                 L: float = 1000.0,
                 device="cuda",
                 model_kwargs=None):
        self.fm = fm
        self.lik = likelihood
        self.estimator_type = estimator_type
        self.device = device
        self.L = float(L)

        #Parameter range
        self.L1_lo, self.L1_hi = 1.0, self.L
        self.ReZF_lo, self.ReZF_hi = 1.0, 4000.0
        self.ImZF_max = 100.0
        self.ReZL_lo, self.ReZL_hi = 1.0, 200.0
        self.ImZL_max = 100.0

        #Model
        self.model = self.build_model(estimator_type, model_kwargs)

    def build_model(self, estimator_type, kwargs):
        kwargs = kwargs or {}
        if estimator_type == "svr":
            return MultiOutputRegressor(SVR(**kwargs))
        elif estimator_type == "gbr":
            return MultiOutputRegressor(GradientBoostingRegressor(**kwargs))
        elif estimator_type == "rfr":
            return RandomForestRegressor(**kwargs)
        elif estimator_type == "hgbr":
            model = MultiOutputRegressor(
    HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=1e-3
    )
)              
            return model
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}")

    def fit(self, X_train, y_train):
        """
        Train ML Regressor. Here all N_train observations come from different parameter triplets (L1, ZF, ZL)
        X_train: [N_train, 2*F] numpy array
        y_train: [N_train, D] numpy array 
        """
        print("Fitting model...")
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, obs_tf, noise_var):
        """
        Jointly estimate parameters for M runs of N observations.
        Here M = 1000, N = 1. Will use M_test instead of M*N for simplicity. 
        Args
        ----
        obs_tf:      [M_test,2*F] numpy array
        noise_var_f: [M_test,2*F] numpy array

        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [M_test],
              "ZF_re": [M_test], "ZF_im": [M_test],
              "ZL_re": [M_test], "ZL_im": [M_test]
            }
        """
        print("Predicting model...")
        y_pred = self.model.predict(obs_tf)  # [N,5]
        return {
            "L1": y_pred[:, 0],
            "ZF_re": y_pred[:, 1],
            "ZF_im": y_pred[:, 2],
            "ZL_re": y_pred[:, 3],
            "ZL_im": y_pred[:, 4],
        }