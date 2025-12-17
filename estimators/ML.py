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
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
from sklearn import linear_model

class MLRegressor(Estimator):

    def __init__(self,
                 fm,
                 likelihood,
                 estimator_type, # "SVR", "GBR", "RFR", or custom
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
        if estimator_type == "svr":
            return SVR(**kwargs)
        elif estimator_type == "gbr":
            return GradientBoostingRegressor(**kwargs)
        elif estimator_type == "rfr":
            return RandomForestRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}")

    def fit(self, obs_tf_train, y_train):
        """
        Train the underlying ML model.

        obs_tf_train: [N_train, F] complex tensor
        y_train:      [N_train, D] numpy array (L1 or [L1,ZF,ZL...])
        """
        X = obs_tf_train
        self.model.fit(X, y_train)
        return self
    def predict(self, obs_tf, noise_var_f):
        """
        Jointly estimate parameters for M runs of N observations.
        Here M = 1000, N = 1. Will use M instead of M*N for simplicity. 
        Args
        ----
        obs_tf:      [M*N,F] complex tensor
        noise_var_f: [M*N,F] float tensor

        Returns
        -------
        dict of numpy arrays:
            {
              "L1":    [M],
              "ZF_re": [M], "ZF_im": [M],
              "ZL_re": [M], "ZL_im": [M]
            }
        """
        # 0) Convert complex obs_tf [M, F] to real
        X = torch.view_as_real(obs_tf)  # [M, F, 2]
        y_pred = self.model.predict(X)
        

        return self