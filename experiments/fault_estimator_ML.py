# ############ ML Models ##############

# # --- Supervised ML baselines: SVR / GradientBoosting / RandomForest ---
# import torch
# import pyro
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO
# import pyro.optim as optim
# from pyro.distributions import TransformedDistribution
# from pyro.distributions.transforms import SigmoidTransform
# from torch.distributions import constraints
# import numpy as np
# from torch.autograd.functional import jvp
# from torch.autograd.functional import jacobian
# from scipy.special import i0, i1
# from scipy.special import ive
# from scipy.signal import find_peaks

# #ML imports
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# rng = np.random.RandomState(42)
# num_runs = 1000
# snr_dbs = list(range(0, 41, 5))
# param_bounds = {"L1": (0.0, 1000.0)}

# def initial_network():
#     data = sio.loadmat("cable_parameter.mat")  # update path if needed
#     gamma_np = data['gamma'].squeeze()
#     Zc_np = data['Z_C'].squeeze()
#     pulFreq_np = data['pulFreq'].squeeze()
#     gamma = torch.tensor(gamma_np, dtype=torch.cfloat)
#     Zc = torch.tensor(Zc_np, dtype=torch.cfloat)
#     pulFreq = torch.tensor(pulFreq_np, dtype=torch.float32)
#     return [gamma, Zc, torch.tensor(1000.0), torch.tensor(10.0), pulFreq]

# def compute_H_complex_clean(L1_tensor, gamma_list, Zc_list):
#     ZL = 100 - 5j
#     L = 1000

#     tmp1= gamma_list*L
   
#     A = torch.cosh(tmp1)
#     B = Zc_list*torch.sinh(tmp1)
#     C = torch.sinh(tmp1)/ Zc_list
#     D = torch.cosh(tmp1)
#     # frequency response
#     h = ZL/(A*ZL+B)#+C*ZL*ZS+D*ZS)
#     # network impedance
#     #Z = (A*ZL+B)/(C*ZL+D) 
#     return h

# def compute_H_complex(L1_tensor, gamma_list, Zc_list):
#     ZL = 100 - 5j
#     ZF = 1000 + 5j
#     L = 1000

#     tmp1 = gamma_list * L
#     tmp2 = gamma_list * L1_tensor.unsqueeze(1)  #If L1_tensor is shape [B] (batch of scalar values) then it becomes [B, 1]. Then you can do tmp2 = [B, 1] * [F] -> [B, F]
#     tmp3 = gamma_list * (L - L1_tensor).unsqueeze(1)
#     tmp4 = Zc_list / ZF
#     tmp5 = Zc_list * Zc_list / ZF
#     A1 = torch.cosh(tmp1) + tmp4 * torch.sinh(tmp2) * torch.cosh(tmp3)
#     B1 = Zc_list * torch.sinh(tmp1) + tmp5 * torch.sinh(tmp2) * torch.sinh(tmp3)
#     h = ZL / (A1 * ZL + B1)
#     return h  # shape: [B, F]

# # --- Frequency setup ---
# gamma_full, Zc_full, _, _, list_of_freq = initial_network()
# fstart, fend, num_of_freq_points = 2e6, 10e6, 500
# desired_freqs = torch.linspace(fstart, fend, num_of_freq_points)
# vec_meas = torch.abs(list_of_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
# gamma_list = gamma_full[vec_meas]
# Zc_list = Zc_full[vec_meas]
# gamma_list = gamma_list.to(device)
# Zc_list = Zc_list.to(device)

# # -------- helpers: simulate one observation; build features ----------
# def simulate_one_obs(true_L1, snr_db, gamma_list, Zc_list, device):
#     """Return (features_x, target_L1) for one sample at a given SNR."""
#     F = gamma_list.shape[0]
#     true_L1_tensor = torch.tensor(true_L1, device=device)
#     true_tf_complex = compute_H_complex(true_L1_tensor.unsqueeze(0), gamma_list, Zc_list)  # [1, F]

#     snr_linear = 10 ** (snr_db / 10)
#     signal_power_mean = torch.mean(torch.abs(true_tf_complex[0]) ** 2)
#     noise_var = signal_power_mean / snr_linear
#     noise_var_f = torch.full_like(true_tf_complex[0].real, fill_value=noise_var)  # [F]
#     noise_std_f = torch.sqrt(noise_var_f / 2).unsqueeze(0)                         # [1, F]

#     # Simulate noisy observation
#     real_noise = noise_std_f * torch.randn_like(true_tf_complex.real)
#     imag_noise = noise_std_f * torch.randn_like(true_tf_complex.imag)
#     obs_tf_complex = true_tf_complex + real_noise + 1j * imag_noise                # [1, F]

#     # Build feature vector: [Re y_1, Im y_1, ..., Re y_F, Im y_F, SNR(optional)]
#     obs_tf_realimag = torch.view_as_real(obs_tf_complex).squeeze(0)                # [F, 2]
#     x = obs_tf_realimag.reshape(-1).detach().cpu().numpy()                         # [2F]
#     return x, float(true_L1)

# def simulate_dataset(N_per_snr, snr_dbs, gamma_list, Zc_list, device, include_snr_feature=True):
#     """
#     Simulate a pooled training set across SNRs.
#     Returns X (N x D), y (N,), and per-SNR index ranges for convenience.
#     """
#     F = gamma_list.shape[0]
#     D_base = 2 * F
#     total = N_per_snr * len(snr_dbs)

#     # Preallocate
#     X = np.zeros((total, D_base + (1 if include_snr_feature else 0)), dtype=np.float32)
#     y = np.zeros((total,), dtype=np.float32)
#     idx = 0
#     snr_ranges = {}  # map snr_db -> (start, end)

#     for snr_db in snr_dbs:
#         start = idx
#         for _ in range(N_per_snr):
#             true_L1 = rng.uniform(100.0, 900.0)
#             x_base, L1 = simulate_one_obs(true_L1, snr_db, gamma_list, Zc_list, device)
#             if include_snr_feature:
#                 X[idx, :-1] = x_base
#                 X[idx, -1]  = snr_db  # you can also use snr_linear
#             else:
#                 X[idx] = x_base
#             y[idx] = L1
#             idx += 1
#         snr_ranges[snr_db] = (start, idx)

#     return X, y, snr_ranges

# # --------------- Train ML models on simulated data -------------------
# # Choose your training budget (use a learning-curve to tune if needed)
# N_train_per_snr = 10000   # e.g., 10k per SNR → total 9*10k = 90k
# include_snr_feature = True  # fair if MLE/VI also know noise variance/SNR

# print("Simulating training set...")
# X_train, y_train, _ = simulate_dataset(
#     N_per_snr=N_train_per_snr,
#     snr_dbs=snr_dbs,
#     gamma_list=gamma_list,
#     Zc_list=Zc_list,
#     device=device,
#     include_snr_feature=include_snr_feature
# )
# print("Xtrain", X_train)

# # SVR prefers scaling; trees don't need it.
# svr_model = make_pipeline(
#     StandardScaler(with_mean=True, with_std=True),
#     SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=1.0)  # start here; tune via CV if desired
# )

# gbr_model = GradientBoostingRegressor(
#     n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.9,
#     random_state=42
# )

# rf_model = RandomForestRegressor(
#     n_estimators=500, max_depth=None, min_samples_leaf=1,
#     n_jobs=-1, random_state=42
# )

# print("Fitting SVR...")
# svr_model.fit(X_train, y_train)
# print("Fitting GradientBoostingRegressor...")
# gbr_model.fit(X_train, y_train)
# print("Fitting RandomForestRegressor...")
# rf_model.fit(X_train, y_train)

# # --------------- Evaluate on the same 1000-case test protocol --------
# def build_test_set_for_snr(snr_db, num_runs, gamma_list, Zc_list, device, include_snr_feature=True):
#     F = gamma_list.shape[0]
#     D_base = 2 * F
#     X = np.zeros((num_runs, D_base + (1 if include_snr_feature else 0)), dtype=np.float32)
#     y_true = np.zeros((num_runs,), dtype=np.float32)

#     snr_linear = 10 ** (snr_db / 10)
#     for i in range(num_runs):
#         true_L1 = rng.uniform(100.0, 900.0)
#         # replicate exactly your test-generation logic
#         true_L1_tensor = torch.tensor(true_L1, device=device)
#         true_tf_complex = compute_H_complex(true_L1_tensor.unsqueeze(0), gamma_list, Zc_list)  # [1, F]
#         signal_power_mean = torch.mean(torch.abs(true_tf_complex[0]) ** 2)
#         noise_var = signal_power_mean / snr_linear
#         noise_var_f = torch.full_like(true_tf_complex[0].real, fill_value=noise_var)  # [F]
#         noise_std_f = torch.sqrt(noise_var_f / 2).unsqueeze(0)                         # [1, F]
#         real_noise = noise_std_f * torch.randn_like(true_tf_complex.real)
#         imag_noise = noise_std_f * torch.randn_like(true_tf_complex.imag)
#         obs_tf_complex = true_tf_complex + real_noise + 1j * imag_noise                # [1, F]
#         obs_tf_realimag = torch.view_as_real(obs_tf_complex).squeeze(0)                # [F, 2]
#         x_base = obs_tf_realimag.reshape(-1).detach().cpu().numpy()

#         if include_snr_feature:
#             X[i, :-1] = x_base
#             X[i, -1]  = snr_db
#         else:
#             X[i] = x_base
#         y_true[i] = true_L1
#     return X, y_true

# rmse_svr, rmse_gbr, rmse_rf = [], [], []

# print("Evaluating ML models on test sets...")
# for snr_db in snr_dbs:
#     X_test, y_true = build_test_set_for_snr(
#         snr_db=snr_db, num_runs=1000,
#         gamma_list=gamma_list, Zc_list=Zc_list, device=device,
#         include_snr_feature=include_snr_feature
#     )
#     y_pred_svr = svr_model.predict(X_test)
#     y_pred_gbr = gbr_model.predict(X_test)
#     y_pred_rf  = rf_model.predict(X_test)

#     rmse_svr.append(np.sqrt(mean_squared_error(y_true, y_pred_svr)))
#     rmse_gbr.append(np.sqrt(mean_squared_error(y_true, y_pred_gbr)))
#     rmse_rf.append(np.sqrt(mean_squared_error(y_true, y_pred_rf)))

#     print(f"SNR {snr_db:2d} dB | SVR RMSE={rmse_svr[-1]:.2f} | GBR RMSE={rmse_gbr[-1]:.2f} | RF RMSE={rmse_rf[-1]:.2f}")

# # --------------- Plot alongside your existing curves -----------------
# plt.figure(figsize=(9,5))
# # If you already computed these earlier:
# # plt.plot(snr_dbs, rmse_per_snr,   marker='o', label='VI (median)')
# # plt.plot(snr_dbs, rmse_per_snr_2, marker='o', label='VI (mean)')
# # plt.plot(snr_dbs, np.sqrt(crlb_list), marker='x', linestyle='--', label='√CRLB')

# plt.plot(snr_dbs, rmse_svr, marker='s', label='SVR (ML)')
# plt.plot(snr_dbs, rmse_gbr, marker='^', label='GradientBoosting (ML)')
# plt.plot(snr_dbs, rmse_rf,  marker='d', label='RandomForest (ML)')

# plt.xlabel("SNR (dB)")
# plt.ylabel("RMSE (m)")
# plt.title("RMSE vs √CRLB — MLE / ELBO vs Supervised ML")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()










import jax
import os
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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

def initial_network():
     data = sio.loadmat('cable_parameter.mat')
     return [
        data['gamma'],  # gamma
        data['Z_C'],  # Zc,
        1000, #L
        10, # ZS
        data['pulFreq'], # frequency vector from 1.8e6 to 30e6
        100, # number of different impedances (frequency selectivity)
        200, # number of frequency points used between start and end frequency (if available), 
             # will be rounded to nearest integer multiple of number of different impedances
     ]
def initial_params(nbr, seed_num, isTest):
    seed = seed_num #Same seed to ensure reproducability
    key = jax.random.split(jax.random.PRNGKey(seed), nbr) 
    #ZF=jax.random.uniform(key[0],(nbr,1),float,1000,2000)+1j*jax.random.uniform(key[1],(nbr,1),float,-50,50)
    #ZL=jax.random.uniform(key[2],(nbr,1),float,50,100)+1j*jax.random.uniform(key[3],(nbr,1),float,-50,50)
    #Params are constants here, ZF is (100, 1) shape
    ZF=jax.random.uniform(key[0],(nbr,1),float,1000,1000)+1j*jax.random.uniform(key[1],(nbr,1),float,50,50) #1000 + 50j
    ZL=jax.random.uniform(key[2],(nbr,1),float,100,100)+1j*jax.random.uniform(key[3],(nbr,1),float,-5,-5) #100 - 5j
    #if(isTest == True):
    L1=jax.random.uniform(key[1],(nbr,1),float,200,200) #L1 = 200
    #else:
    #L1=jax.random.uniform(key[1],(nbr,1),float,120,120) #L1 = 100 - 900
    return [
        ZF, # ZF
        ZL, # ZL
        L1, # L1
        key,
    ]  
def u_comp(params, network):
    ZF, ZL, L1 = params
    gamma, Zc, L, ZS = network
    #WRONG tmp1= 2*gamma*L1-gamma*L
    tmp1= gamma*L
    tmp2= gamma*L1
    tmp3= gamma*L-gamma*L1
    tmp4= Zc/ZF
    tmp5= Zc*Zc/ZF
    #WRONG B = Zc*jnp.cosh(tmp1)+tmp5*jnp.sinh(tmp2)*jnp.sinh(tmp3)
    # A = jnp.cosh(gamma*L)
    # B = Zc * np.sinh(gamma * L)
    # C = (1 / Zc) * np.sinh(gamma * L)
    # D = np.cosh(gamma * L)
    
    # phi1 = np.array([
    #         [jnp.cosh(gamma*L1), Zc*jnp.sinh(gamma*L1)],
    #         [jnp.sinh(gamma*L1)/Zc, jnp.cosh(gamma*L1)]
    #     ])
    # phi2 = np.array([
    #         [jnp.cosh(tmp3), Zc*jnp.sinh(tmp3)],
    #         [jnp.sinh(tmp3)/Zc, jnp.cosh(tmp3)]
    #     ])
    # phif = np.array([
    #         [1, 0],
    #         [1/ZF, 1]
    #     ])
    # Phioverall = phi1
    # A = Phioverall[0, 0]
    # B = Phioverall[0, 1]
    # C = Phioverall[1, 0]
    # D = Phioverall[1, 1]
    # print("L", L)
    A = jnp.cosh(tmp1)+tmp4*jnp.sinh(tmp2)*jnp.cosh(tmp3)
    B = Zc*jnp.sinh(tmp1)+tmp5*jnp.sinh(tmp2)*jnp.sinh(tmp3)
    C = jnp.sinh(tmp1)/Zc+jnp.cosh(tmp2)*jnp.cosh(tmp3)/ZF
    D = jnp.cosh(tmp1)+tmp4*jnp.sinh(tmp3)*jnp.cosh(tmp2)
    # frequency response
    #h = 1/A
    h = ZL/(A*ZL+B)
    # h = ZL/(A*ZL+B+C*ZL*ZS+D*ZS)
    # print("h", h)
    # network impedance
    z = (A*ZL+B)/(C*ZL+D) 
    return h, z

def u_h_re_comp(params, network): 
    h,dum = u_comp(params, network)
    return jnp.real(h)

def u_h_im_comp(params, network):
    h,dum = u_comp(params, network)
    return jnp.imag(h)

def u_z_re_comp(params, network):
    dum,z = u_comp(params, network)
    return jnp.real(z)

def u_z_im_comp(params, network):
    dum,z = u_comp(params, network)
    return jnp.imag(z)
def get_fim(a,b):
    #a = dRE(U)/dtheta = (complex, complex, real)
    #b = dIm(U)/dtheta = (complex, complex, real)
    #We want du/dtheta
    #Recall dz/dtheta = 1/2 [dz/dtheta_r - j dz/theta_i] (definition of complex/writinger derivative)
    c = 0.5*(jnp.array(a[0:2]).reshape((2,1))+1j*jnp.array(b[0:2]).reshape((2,1)))# du/dtheta_1 a[0:2] real part of ZF, ZL + b[0:2] imag gart of ZF and reshape to column vector
    d = 0.5*(jnp.conj(jnp.array(a[0:2]).reshape((2,1)))+1j*jnp.conj(jnp.array(b[0:2]).reshape((2,1)))) # du/dtheta_1* form the derivative with respect to conjugate
    e = jnp.array(a[2]).reshape((1,1))+1j*jnp.array(b[2]).reshape((1,1)) # du/dtheta_2 L1 only which is real but u is still complex 
    f = jnp.vstack((c,d,e)) # du/dtheta = [du/dtheta_1, du/dtheta_1*, du/theta_2]

    g = 0.5*(jnp.array(a[0:2]).reshape((2,1))-1j*jnp.array(b[0:2]).reshape((2,1))) #du*/dtheta_1 
    h = 0.5*(jnp.conj(jnp.array(a[0:2]).reshape((2,1)))-1j*jnp.conj(jnp.array(b[0:2]).reshape((2,1)))) #du*/dtheta_1*
    i = jnp.vstack((g,h,jnp.conj(e))) # du*/dtheta = [du*/dtheta_1, du*/dtheta_1*, du*/theta_2]

    tmp=jnp.conj(f)@jnp.transpose(f)+jnp.conj(i)@jnp.transpose(i) #I(theta) = [ (du/dtheta)*(du/dtheta)T + (du*/dtheta)* (du*/dtheta)T] 
    return tmp
def get_CRLB(FIM_total):
    #CRLB(theta_real) = [I22 - 2 Re[PCP^H + P^* D P^H]]^-1 
    A = FIM_total[0:2,0:2]
    A_conj = FIM_total[2:4, 2:4]
    B = FIM_total[2:4,0:2]
    B_conj = FIM_total[0:2, 2:4]
    P = FIM_total[4, 0:2]
    P_conj = FIM_total[4, 2:4]
    P_herm = FIM_total[0:2, 4]
    P_trans = FIM_total[2:4, 4]
    Q = FIM_total[4,4]
    C = jnp.linalg.inv(A-B_conj@jnp.linalg.inv(A_conj)@B)
    D = -jnp.linalg.inv(A_conj)@B@C
    CRLB_r = 1/(Q-2*jnp.real(P@C@P_herm+P_conj@D@P_herm))
    return CRLB_r

def complex_to_real(data):
    real = data.real
    imag = data.imag
    # Initialize an empty list to collect the columns
    columns = []

    # Iterate through the columns of the array
    for j in range(data.shape[1]):
        # Append the j-th column of the real part
        columns.append(real[:, j:j+1])
        # Append the j-th column of the imaginary part
        columns.append(imag[:, j:j+1])
    
    # Horizontally stack all the collected columns
    result = np.hstack(columns)
    return result
def plot_feature_scatter_plots(X, y, file_name_prefix="scatter_plot"):
    """
    Plots scatter plots of each feature against the target variable y.
    
    Parameters:
    X (DataFrame or ndarray): Feature matrix
    y (Series or ndarray): Target variable
    file_name_prefix (str): Prefix for saved plot files
    """
    num_features = X.shape[1] if len(X.shape) > 1 else 1
    for i in range(num_features):
        plt.figure()
        plt.scatter(X[:, i], y, alpha=0.5)
        plt.xlabel(f"Feature {i+1}")
        plt.ylabel("Target Variable")
        plt.title(f"Scatter Plot of Feature {i+1} vs Target")
        plt.savefig(f"{file_name_prefix}_feature_{i+1}.png")
        plt.close()
def plot_feature_rows(X, start_row=100, end_row=300):
    """
    Plots the feature values for rows in the range [start_row, end_row).
    
    Parameters:
    X (DataFrame or ndarray): Feature matrix
    start_row (int): Starting row index
    end_row (int): Ending row index (exclusive)
    """
    rows_to_plot = X[start_row:end_row]  # Select the specified rows
    num_features = X.shape[1] if len(X.shape) > 1 else 1
    plt.figure(figsize=(15, 6))

    for i in range(num_features):
        plt.plot(range(start_row, end_row), rows_to_plot[:, i], label=f'Feature {i+1}')
    
    plt.xlabel("Row Index")
    plt.ylabel("Feature Values")
    plt.title(f"Feature Values from Row {start_row} to {end_row}")
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.show()

def plot_learning_curves(model, X, y, file_name):
    # Ensure the directory exists
    output_dir = "LearningCurves"
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for saving the file
    full_file_path = os.path.join(output_dir, file_name)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(train_errors, "r-+", linewidth=2, label="train")
    plt.plot(val_errors, "b-", linewidth=3, label="val")
    plt.xlabel("Training set size")
    plt.ylabel("MSE")
    plt.legend()

    # Annotate the last point on the graph with the corresponding error values
    plt.annotate(f"{train_errors[-1]:.2f}", 
                 xy=(len(train_errors)-1, train_errors[-1]), 
                 xytext=(-10, 10), 
                 textcoords="offset points", 
                 ha='center', 
                 color='red')
    
    plt.annotate(f"{val_errors[-1]:.2f}", 
                 xy=(len(val_errors)-1, val_errors[-1]), 
                 xytext=(-10, -20), 
                 textcoords="offset points", 
                 ha='center', 
                 color='blue')
    plt.savefig(full_file_path)
    plt.close()

def polynomial_regression_with_validation(X_train, y_train, X_val, y_val, degrees):
    validation_errors = []

    for degree in degrees:  
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_val_pred = model.predict(X_val_poly)
        
        mse_val = mean_squared_error(y_val, y_val_pred)
        validation_errors.append(mse_val)
        print(f"Degree: {degree}, Validation MSE: {mse_val}")

    best_degree = degrees[np.argmin(validation_errors)]
    print(f"Best degree: {best_degree}")

    final_poly = PolynomialFeatures(degree=best_degree)
    X_train_poly = final_poly.fit_transform(X_train)
    final_model = LinearRegression()
    final_model.fit(X_train_poly, y_train)

    return final_model, final_poly, validation_errors, best_degree

#MAIN STARTS HERE
if __name__ == '__main__':
    print("hello world")
    # define the gradient functions 
    u_h_re = jax.grad(u_h_re_comp)
    u_h_im = jax.grad(u_h_im_comp)
    u_z_re = jax.grad(u_z_re_comp)
    u_z_im = jax.grad(u_z_im_comp)

    #Parameters
    SNRdB = 20
    SNR = jnp.power(10,SNRdB/10)
    num_of_training_realizations = 2
    num_of_testing_realizations = 1
    num_of_freq_points = 500

    mse_list = []
    crlb_list = []
    fstart = 2e6
    fend = 10e6

    #Network initialization
    L = initial_network()[2]
    ZS = initial_network()[3]
    list_of_freq = initial_network()[4][0]

    #Frequency Index Calculations
    fstart_ind = (jnp.absolute(list_of_freq-fstart)).argmin() #(1.8 - 30) - (2) the argmin finds the index where this is 0 which is 8
    fend_ind = (jnp.absolute(list_of_freq-fend)).argmin() #172 is the index

    #Freq indices from 8 to 172 evenly spaced out at 100 values but rounded to nearest integer
    vec_meas = jnp.linspace(start=fstart_ind,stop=fend_ind,num=num_of_freq_points,dtype=jnp.int16)

    #Extract relevant pullfreq values, so 1.995MHz to 6.003 MHz
    fvec_sub = jnp.take(list_of_freq,vec_meas)
    print("fevec sub", fvec_sub)

    #Parameter Initialization for Training
    params_train = initial_params(num_of_training_realizations, 1515, False) #500 training realizations of H, L1 not constant
    params_test = initial_params(num_of_testing_realizations, 1505, True) #For Test L1 is constant

    Z_f = params_train[0]
    Z_L = params_train[1]
    L_1 = params_train[2]

    Z_f_test = params_test[0]
    Z_L_test = params_test[1]
    L_1_test = params_test[2]

    zabs = []
    H_f = [] # Initialize an empty list to store all row vectors
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    #Calculate H(f) and Z(f) for training - L1 is not constant 
    for i in range(num_of_training_realizations):
        H_f_train = []  # Initialize H_f_train for each realization
        habs = []
        sum_habssq=0
        sum_zabssq=0
        gamma_list = []
        for gamma, Zc in zip(jnp.take(initial_network()[0], vec_meas), jnp.take(initial_network()[1],vec_meas)): 
            gamma_list.append(gamma)
            network = [gamma.item(),Zc.item(), L, ZS] #Gamma, ZC change with freq, L, ZS do not
            param =[params_train[0][i].item(),params_train[1][i].item(),params_train[2][i].item()] #(ZF, ZL, L1) are constant for one realization
            #print("what is this", params_train[2][i].item())
            h_train, z_train = u_comp(param, network)
            # print("gamma", gamma)
            # print("Zc", Zc)
            # print("htrain", h_train)
            H_f_train.append(h_train.item())
            sum_habssq += jnp.abs(h_train)**2 #Calculating total power of transfer function
            sum_zabssq += jnp.abs(z_train)**2 
            habs.append(20*jnp.log10(jnp.abs(h_train))) #in dB
            #zabs.append(jnp.abs(z)) #ohms 
        #print("habs", habs)
        print(f"gamma", gamma_list)
        plt.title("Htrue")
        plt.plot(fvec_sub/1e6,habs)
        plt.show()
        sigma_n_sq_h = sum_habssq/num_of_freq_points/SNR  #Noise power = Signal Power/SNR
        sigma_n_sq_z = sum_zabssq/num_of_freq_points/SNR    
        noise_real_train = jax.random.normal(params_train[3][i], (num_of_freq_points, 1)) #params_train[3][3] = key[3]
        noise_imag_train = jax.random.normal(params_train[3][i], (num_of_freq_points, 1)) #params_train[3][4] = key[4]
        noise_real_train = jnp.sqrt(sigma_n_sq_h/2) * noise_real_train #Scale standard normal by actual variance which is sigma_n_sq_h
        noise_imag_train = jnp.sqrt(sigma_n_sq_h/2) * noise_imag_train #Factor of 2 because variance gets equally distributed in real and imaginary parts
        noise_train = np.array(noise_real_train + 1j*noise_imag_train) 

        # Ensure H_f_train is a row vector
        H_f_train = np.array(H_f_train).reshape(1, -1)  # Convert to 1x100 row vector
    
        # Ensure noise_train is a row vector
        noise_train = np.array(noise_train).reshape(1, -1)  # Convert to 1x100 row vector
    
        # Add noise to H_f_train
        H_f_train2 = H_f_train + noise_train
        plt.plot(fvec_sub/1e6, 20*jnp.log10(jnp.abs(H_f_train2)).T)
        plt.show()
        # Append the row vector to the final matrix
        H_f.append(20*jnp.log10(jnp.abs(H_f_train2)))
        
    # Stack vertically list of arrays
    #Training input data using H_f is X 
    X_train = np.vstack(H_f)
    y_train = L_1.reshape(-1)
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    std_y = scaler_y.scale_[0]

    #Calculate H(f) and Z(f) for testing - L1 is constant 
    # habs_test = []
    # zabs_test = []
    # sigma_list = []
    # H_f_t = [] # Initialize an empty list to store all row vectors
    # for i in range(num_of_testing_realizations):
    #     H_f_test = []  # Initialize H_f_train for each realization
    #     sum_habssq=0
    #     sum_zabssq=0
    #     for gamma, Zc in zip(jnp.take(initial_network()[0], vec_meas), jnp.take(initial_network()[1],vec_meas)):
    #         network = [gamma.item(),Zc.item(), L, ZS] #Gamma, ZC change with freq, L, ZS do not
    #         param =[params_test[0][i].item(),params_test[1][i].item(),params_test[2][i].item()] #(ZF, ZL, L1) are always constant
    #         h_test, z_test = u_comp(param, network)
    #         H_f_test.append(h_test.item())
    #         sum_habssq += jnp.abs(h_test)**2 #Calculating total power of transfer function signal
    #         sum_zabssq += jnp.abs(z_test)**2 
    #         #habs.append(20*jnp.log10(jnp.abs(h))) #in dB
    #         #zabs.append(jnp.abs(z)) #ohms 
    #     sigma_n_sq_h = sum_habssq/num_of_freq_points/SNR  #Noise power = Signal Power/SNR
    #     sigma_list.append(sigma_n_sq_h)
    #     sigma_n_sq_z = sum_zabssq/num_of_freq_points/SNR    
    #     noise_real_test = jax.random.normal(params_test[3][i], (num_of_freq_points, 1)) #params_train[3][3] = key[3]
    #     noise_imag_test = jax.random.normal(params_test[3][i], (num_of_freq_points, 1)) #params_train[3][4] = key[4]
    #     noise_real_test = jnp.sqrt(sigma_n_sq_h/2) * noise_real_test #Scale standard normal by actual variance which is sigma_n_sq_h
    #     noise_imag_test = jnp.sqrt(sigma_n_sq_h/2) * noise_imag_test #Factor of 2 because variance gets equally distributed in real and imaginary parts
    #     noise_test = np.array(noise_real_test + 1j*noise_imag_test) 

    #     # Ensure H_f_train is a row vector
    #     H_f_test = np.array(H_f_test).reshape(1, -1)  # Convert to 1x100 row vector
    
    #     # Ensure noise_train is a row vector
    #     noise_test = np.array(noise_test).reshape(1, -1)  # Convert to 1x100 row vector
    
    #     # Add noise to H_f_train
    #     H_f_test2 = H_f_test + noise_test
    #     # Append the row vector to the final matrix
    #     H_f_t.append(20*jnp.log10(jnp.abs(H_f_test2)))  
    
    # X_test = np.vstack(H_f_t)   
    # y_test = L_1_test.reshape(-1)
    # X_test = scaler_x.transform(X_test)
    #y_test = scaler_y.transform(y_test)
    #print(X_test)



############# Learning Curves ############

    #Linear Reg Model
# model = LinearRegression()
# plot_learning_curves(model, X_train, y_train, "LinearRegVal.pdf")
# poly_features = PolynomialFeatures(degree = 2)
# X_poly = poly_features.fit_transform(X_train)
# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X_poly, y_train, "PolyReg2.pdf")
#    sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3)
#    plot_learning_curves(model, X_train, y_train, "LinearRegSGD.pdf")
#    model.fit(X_train, y_train)

# #     # Polynomial Regression with Hyperparameter Tuning
# #     degrees = [1, 2, 3, 4, 5, 6, 7, 8]
# #     poly_model, poly_transform, validation_errors, best_degree = polynomial_regression_with_validation(X_train, y_train, X_val, y_val, degrees)

#     #SVR Model
# svr = SVR(kernel="rbf")
# plot_learning_curves(svr, X_train, y_train, "SVRRBF2.pdf")
# svr2 = SVR(kernel="sigmoid")
# plot_learning_curves(svr2, X_train, y_train, "SVRSigmoid2.pdf")
#    svr.fit(X_train, y_train)
    
#     svr2.fit(X_train, y_train)
#     #svr2 = SVR(kernel="poly", degree=8)
#     #multi_output_svr = MultiOutputRegressor(svr)
#     #multi_output_svr2 = MultiOutputRegressor(svr2)
#     #multi_output_svr.fit(X_train, y_train)
#     #multi_output_svr2.fit(X_train, y_train)

#     # Train a Decision Tree Regression model
# tree_reg = DecisionTreeRegressor(max_depth=3, max_features="log2", min_samples_leaf=1, min_samples_split=10, random_state=42)
# plot_learning_curves(tree_reg, X_train, y_train, "DecisionTree.pdf")
# forest_reg = RandomForestRegressor(n_estimators=100, max_depth=10, max_features="log2", min_samples_leaf=2, min_samples_split=2, random_state=42)
# plot_learning_curves(forest_reg, X_train, y_train, "RandomForest.pdf")
#    plot_learning_curves(tree_reg, X_train, y_train, "DecisionTree.pdf")
# #     # Gradient Boosting Regressor
# #     gbr = GradientBoostingRegressor()
# #     multi_output_gbr = MultiOutputRegressor(gbr)
# #     multi_output_gbr.fit(X_train, y_train)

    # Define the parameter grid
#     param_grid = {
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'sqrt', 'log2']
# }
#     param_grid_svr = {
#     'kernel': ['sigmoid'],  # Consider different kernel types
#     'C': [0.1, 1, 10, 100, 1000],          # Regularization parameter
#     'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
#     'epsilon': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],     # Epsilon in the loss function
#     'coef0': [0.0, 0.1, 0.5, 1.0]  # Independent term in kernel function (relevant for 'poly' and 'sigmoid')
# }
#     param_grid_forest = {
#     'n_estimators': [100, 200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'sqrt', 'log2'],
# }

#     # Perform grid search with cross-validation
#    grid_search = GridSearchCV(estimator=tree_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
#    grid_search_forest = GridSearchCV(estimator=forest_reg, param_grid=param_grid_forest, cv=5, scoring='neg_mean_squared_error')
#    grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, scoring='neg_mean_squared_error')
#    grid_search.fit(X_train, y_train)
#    grid_search_svr.fit(X_train, y_train)\
#    grid_search_forest.fit(X_train, y_train)

#     # Get the best parameters
#    best_params = grid_search.best_params_
#   print(f"Best parameters: {best_params}")
#    best_params_svr = grid_search_svr.best_params_
#    print(f"Best parameters: {best_params_svr}")
#    best_params_forest = grid_search_forest.best_params_
#    print(f"Best parameters: {best_params_forest}")

#     # Train the best model
#     best_tree_reg = grid_search.best_estimator_
#     #best_svr_reg = grid_search_svr.best_estimator_


##### Models ######

# model = LinearRegression()
# model.fit(X_train, y_train)

# poly_features = PolynomialFeatures(degree = 2)
# X_poly = poly_features.fit_transform(X_train)
# X_polytest = poly_features.transform(X_test)

# ridge_reg = Lasso(alpha=5.0)  # alpha is the regularization strength
# ridge_reg.fit(X_poly, y_train)

# svr = SVR(C=10, kernel="rbf")
# svr.fit(X_train, y_train)
# parameters = {
#     "kernel": ["rbf"],
#     "C": [1,10,100,1000],
#     "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#     }
# test = SVR()
# grid_searchsvr = GridSearchCV(estimator=test, param_grid=parameters, cv=5, scoring='neg_mean_squared_error')
# grid_searchsvr.fit(X_train, np.squeeze(y_train))
# best_modelsvr = grid_searchsvr.best_estimator_
# print("BEST PARAMS", grid_searchsvr.best_params_)
# y_test_svr = best_modelsvr.predict(X_test)
# mse_svr = mean_squared_error(y_test, y_test_svr)
# print("SVR MSE", mse_svr)
# print("y test pred", y_test_svr)
# print("y labels", y_test)


# # Define the parameter grid for GradientBoostingRegressor
# parameters_gbr = {
#     "n_estimators": [50, 100, 200],
#     "learning_rate": [0.01, 0.1, 0.2, 0.3],
#     "max_depth": [3, 4, 5],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "subsample": [0.8, 1.0]
# }

# # Initialize the GradientBoostingRegressor
# gbr = GradientBoostingRegressor()

# # Set up the GridSearchCV with the GradientBoostingRegressor
# grid_search_gbr = GridSearchCV(estimator=gbr, param_grid=parameters_gbr, cv=5, scoring='neg_mean_squared_error')

# # Fit the model to the training data
# grid_search_gbr.fit(X_train, np.squeeze(y_train))

# # Get the best model based on the grid search
# best_model_gbr = grid_search_gbr.best_estimator_

# # Print the best parameters found by GridSearchCV
# print("BEST PARAMS", grid_search_gbr.best_params_)

# # Use the best model to predict on the test data
# y_test_gbr = best_model_gbr.predict(X_test)

# # Calculate and print the Mean Squared Error for the predictions
# mse_gbr = mean_squared_error(y_test, y_test_gbr)
# print("GradientBoostingRegressor MSE", mse_gbr)

# # Print the predicted values and actual labels
# print("y test pred", y_test_gbr)
# print("y labels", y_test)

# gbr = GradientBoostingRegressor()
# gbr.fit(X_train, y_train)
# y_test_gbr = gbr.predict(X_test)
# mse_gbr = mean_squared_error(y_test, y_test_gbr)
# print("GBR MSE", mse_gbr)
# print("y test pred GBR", y_test_gbr)
# print("y labels", y_test)


# # svr2 = SVR(kernel="sigmoid")
# # svr2.fit(X_train, y_train)

# # #gradreg = GradientBoostingRegressor(n_estimators=100)
# # #gradreg.fit(X_train, y_train)

# # tree_reg = DecisionTreeRegressor(max_depth=3, max_features="log2", min_samples_leaf=1, min_samples_split=10, random_state=42)
# # tree_reg.fit(X_train, y_train)
# # forest_reg = RandomForestRegressor(n_estimators=100, max_depth=10, max_features="log2", min_samples_leaf=2, min_samples_split=2, random_state=42)
# # forest_reg.fit(X_train, y_train)

# # #     Predictions on Training Set
# # y_pred_train = model.predict(X_train)
# # y_pred_poly_train = ridge_reg.predict(X_poly)
# y_pred_svr_train = svr.predict(X_train)
# # y_pred_svr2_train = svr2.predict(X_train)
# # y_pred_tree_train = tree_reg.predict(X_train)
# # y_pred_forest_train = forest_reg.predict(X_train)
# # # #     Predictions on Test set
# # y_pred_test = model.predict(X_test)
# # y_pred_poly_test = ridge_reg.predict(X_polytest)
# y_pred_svr_test = svr.predict(X_test)
# # y_pred_svr2_test = svr2.predict(X_test)
# # y_pred_tree_test = tree_reg.predict(X_test)
# # y_pred_forest_test = forest_reg.predict(X_test)
# # print("y_pred_test", y_pred_test)
# # print("svr pred", y_pred_svr_test)

# # #     # Train Mean Squared Error
# # mse_train = mean_squared_error(y_train, y_pred_train)
# # mse_poly_train = mean_squared_error(y_train, y_pred_poly_train)
# mse_svr_train = mean_squared_error(y_train, y_pred_svr_train)
# # mse_svr2_train = mean_squared_error(y_train, y_pred_svr2_train)
# # mse_tree_train = mean_squared_error(y_train, y_pred_tree_train)
# # mse_forest_train = mean_squared_error(y_train, y_pred_forest_train)

# # #    # Test Mean Squared Error
# # mse_test = mean_squared_error(y_test, y_pred_test)
# # mse_poly_test = mean_squared_error(y_test, y_pred_poly_test)
# mse_svr_test = mean_squared_error(y_test, y_pred_svr_test)
# # mse_svr2_test = mean_squared_error(y_test, y_pred_svr2_test)
# # mse_tree_test = mean_squared_error(y_test, y_pred_tree_test)
# # mse_forest_test = mean_squared_error(y_test, y_pred_forest_test)

# # print(f"Original Train MSE: {mse_train}")
# # # print(f"Original Polynomial Regression Train MSE: {mse_poly_train}")
# print(f"Original Support Vector Regression Train MSE RBF: {mse_svr_train}")
# # # print(f"Original Support Vector Regression Train MSE Sigmoid: {mse_svr2_train}")
# # # print(f"Original Decision Tree Regression Train MSE: {mse_tree_train}")
# # # print(f"Original Random Forest Regression Train MSE: {mse_forest_train}")

# # # print("TEST MSE")
# # print(f"Original Test MSE: {mse_test}")
# # # print(f"Original Polynomial Regression Test MSE: {mse_poly_test}")
# print(f"Original Support Vector Regression Test MSE RBF: {mse_svr_test}")
# # # print(f"Original Support Vector Regression Test MSE Sigmoid: {mse_svr2_test}")
# # # print(f"Original Decision Tree Regression Test MSE: {mse_tree_test}")
# # # print(f"Original Random Forest Regression Test MSE: {mse_forest_test}")

# print(y_test)
# print(y_pred_svr_test)
# print(np.mean(y_pred_svr_test))
# # #   Variance
# errors = y_test - y_pred_test
# errors_poly = y_test - y_pred_poly_test
# errors_svr = y_test - y_pred_svr_test
# errors_svr2 = y_test - y_pred_svr2_test
# errors_tree = y_test - y_pred_tree_test
# errors_forest = y_test - y_pred_forest_test

# #   Compute the variance of the errors
# variance = np.var(errors)
# variance_poly = np.var(errors_poly)
# variance_svr = np.var(errors_svr)
# variance_svr2 = np.var(errors_svr2)
# variance_tree = np.var(errors_tree)
# variance_forest = np.var(errors_forest)

# print("VAR")
# print(f"Variance of Linear Reg prediction errors: {variance}")
# print(f"Variance of Linear Reg with Polynomial Basis prediction errors: {variance_poly}")
# print(f"Variance of SVR using RBF Basis prediction errors: {variance_svr}")
# print(f"Variance of SVR using Sigmoid Basis prediction errors: {variance_svr2}")
# print(f"Variance of Decision Tree prediction errors: {variance_tree}")
# print(f"Variance of Random Forest prediction errors: {variance_forest}")

### Extra ######

# mse_train_original = mse_train * (std_y ** 2)
# mse_poly_train_original = mse_poly_train * (std_y ** 2)
# mse_svr_train_original = mse_svr_train * (std_y ** 2)
# mse_svr2_train_original = mse_svr2_train * (std_y ** 2)
# mse_tree_train_original = mse_tree_train * (std_y ** 2)
# mse_forest_train_original = mse_forest_train * (std_y ** 2)

# mse_test_original = mse_test * (std_y ** 2)
# mse_poly_test_original = mse_poly_test * (std_y ** 2)
# mse_svr_test_original = mse_svr_test * (std_y ** 2)
# mse_svr2_test_original = mse_svr2_test * (std_y ** 2)
# mse_tree_test_original = mse_tree_test * (std_y ** 2)
# mse_forest_test_original = mse_forest_test * (std_y ** 2)

# # Convert the scaled variance of errors to the original variance
# variance_errors_original = variance * (std_y ** 2)
# variance_svr_original = variance_svr * (std_y ** 2)
# variance_svr2_original = variance_svr2 * (std_y ** 2)
# variance_tree_original = variance_tree * (std_y ** 2)
# variance_forest_original = variance_forest * (std_y ** 2)

# print(f"Original Train MSE: {mse_train_original}")
# print(f"Original Polynomial Regression Train MSE: {mse_poly_train_original}")
# print(f"Original Support Vector Regression Train MSE RBF: {mse_svr_train_original}")
# print(f"Original Support Vector Regression Train MSE Sigmoid: {mse_svr2_train_original}")
# print(f"Original Decision Tree Regression Train MSE: {mse_tree_train_original}")
# print(f"Original Random Forest Regression Train MSE: {mse_forest_train_original}")

# print("TEST MSE")
# print(f"Original Test MSE: {mse_test_original}")
# print(f"Original Polynomial Regression Test MSE: {mse_poly_test_original}")
# print(f"Original Support Vector Regression Test MSE RBF: {mse_svr_test_original}")
# print(f"Original Support Vector Regression Test MSE Sigmoid: {mse_svr2_test_original}")
# print(f"Original Decision Tree Regression Test MSE: {mse_tree_test_original}")
# print(f"Original Random Forest Regression Test MSE: {mse_forest_test_original}")

# print("VAR")
# print(f"Original Variance of Linear Reg prediction errors: {variance_errors_original}")
# print(f"Original Variance of SVR using RBF Basis prediction errors: {variance_svr_original}")
# print(f"Original Variance of SVR using Sigmoid Basis prediction errors: {variance_svr2_original}")
# print(f"Original Variance of Decision Tree prediction errors: {variance_tree_original}")
# print(f"Original Variance of Random Forest prediction errors: {variance_forest_original}")

# ########## CRLB Calculations ###############
# sigma_list = np.array(sigma_list)
# print("SIGMA LIST", sigma_list)
# list_of_crlbs = []
#     #CRLB changes for each testing realization
# for i in range(num_of_testing_realizations): 
#     FIM_total = jnp.zeros((5, 5))
#     for gamma, Zc in zip(jnp.take(initial_network()[0], vec_meas), jnp.take(initial_network()[1],vec_meas)):
#         network_perfreq = [gamma.item(),Zc.item(), L, ZS]
#         param_perfreq =[params_test[0][i].item(),params_test[1][i].item(),params_test[2][i].item()]
#         du_dtheta_r = u_h_re(param_perfreq, network_perfreq) 
#         du_dtheta_i = u_h_im(param_perfreq, network_perfreq)
#         FIM = get_fim(du_dtheta_r, du_dtheta_i)
#         FIM_total = FIM_total + FIM
        
#     #print("FIM total", FIM_total)
#     FIM_total = 1/sigma_list[i] * FIM_total
#     CRLB_real = get_CRLB(FIM_total)
#     print("CRLB REAL", CRLB_real)
#     list_of_crlbs.append(CRLB_real)
#     #CRLB = jnp.linalg.inv(FIM_total)
#     #print("CRLB", CRLB[4][4])
#     #print("CRLB from func", get_CRLB(FIM_total))
# crlbs_array = np.array(list_of_crlbs)
# average_crlb = np.mean(crlbs_array)
# print("Average CRLB", average_crlb)

    
