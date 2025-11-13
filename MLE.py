############### Code to generate RMSE vs CRLB for MLE and ELBO ###################

import torch
import pyro
import matplotlib.pyplot as plt
import scipy.io as sio
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import SigmoidTransform
from torch.distributions import constraints
import numpy as np
from torch.autograd.functional import jvp
from torch.autograd.functional import jacobian
from scipy.special import i0, i1
from scipy.special import ive
from scipy.signal import find_peaks


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
L = 1000
N = 1
num_of_peaks = 8
num_runs = 1000
L1_grid = torch.linspace(1, 1000, steps=1000, device=device)  # shape: (1000,)
mu_vals = torch.linspace(-3.0, 3.0, 500)
snr_dbs = list(range(0, 41, 5))
param_bounds = {"L1": (0.0, 1000.0)}

# --- helper: anneal sigma with SNR ---
def sigma_from_snr(snr_db, sigma_max=0.05, sigma_min=0.002, alpha=1.0, ref_db=0.0):
    """
    σ ∝ 10^{-alpha*(SNR_dB - ref_db)/20}; clipped to [sigma_min, sigma_max].
    Keeps ELBO smoothing roughly constant vs. likelihood curvature (∝ SNR_linear).
    """
    sigma = sigma_max * 10 ** (-alpha * (snr_db - ref_db) / 20.0)
    return float(np.clip(sigma, sigma_min, sigma_max))

def denormalize(norm_value, min_val, max_val):
    return norm_value * (max_val - min_val) + min_val
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def initial_network():
    data = sio.loadmat("cable_parameter.mat")  # update path if needed
    gamma_np = data['gamma'].squeeze()
    Zc_np = data['Z_C'].squeeze()
    pulFreq_np = data['pulFreq'].squeeze()
    gamma = torch.tensor(gamma_np, dtype=torch.cfloat)
    Zc = torch.tensor(Zc_np, dtype=torch.cfloat)
    pulFreq = torch.tensor(pulFreq_np, dtype=torch.float32)
    return [gamma, Zc, torch.tensor(1000.0), torch.tensor(10.0), pulFreq]

def compute_H_complex_clean(L1_tensor, gamma_list, Zc_list):
    ZL = 100 - 5j
    L = 1000

    tmp1= gamma_list*L
   
    A = torch.cosh(tmp1)
    B = Zc_list*torch.sinh(tmp1)
    C = torch.sinh(tmp1)/ Zc_list
    D = torch.cosh(tmp1)
    # frequency response
    h = ZL/(A*ZL+B)#+C*ZL*ZS+D*ZS)
    # network impedance
    #Z = (A*ZL+B)/(C*ZL+D) 
    return h

def compute_H_complex(L1_tensor, gamma_list, Zc_list):
    ZL = 100 - 5j
    ZF = 1000 + 5j
    tmp1 = gamma_list * L
    tmp2 = gamma_list * L1_tensor.unsqueeze(1)  #If L1_tensor is shape [B] (batch of scalar values) then it becomes [B, 1]. Then you can do tmp2 = [B, 1] * [F] -> [B, F]
    tmp3 = gamma_list * (L - L1_tensor).unsqueeze(1)
    tmp4 = Zc_list / ZF
    tmp5 = Zc_list * Zc_list / ZF
    A1 = torch.cosh(tmp1) + tmp4 * torch.sinh(tmp2) * torch.cosh(tmp3)
    B1 = Zc_list * torch.sinh(tmp1) + tmp5 * torch.sinh(tmp2) * torch.sinh(tmp3)
    h = ZL / (A1 * ZL + B1)
    return h  # shape: [B, F]
    
def get_fim(a, b):
    # a = dRe(U)/dtheta = (complex, complex, real)
    # b = dIm(U)/dtheta = (complex, complex, real)

    # Convert to torch tensors
    a = [torch.as_tensor(x) for x in a]
    b = [torch.as_tensor(x) for x in b]

    # du/dtheta_1 (complex part of ZF and ZL)
    c = 0.5 * (torch.stack(a[0:2]).reshape(2, 1) + 1j * torch.stack(b[0:2]).reshape(2, 1))
    # du/dtheta_1* (complex conjugate wrt theta_1*)
    d = 0.5 * (torch.conj(torch.stack(a[0:2]).reshape(2, 1)) + 1j * torch.conj(torch.stack(b[0:2]).reshape(2, 1)))

    # du/dtheta_2 (L1 is real, but u is complex)
    e = a[2].reshape(1, 1) + 1j * b[2].reshape(1, 1)

    # Combine into full du/dtheta vector
    f = torch.vstack((c, d, e))

    # Derivatives of u* (complex conjugate of u)
    g = 0.5 * (torch.stack(a[0:2]).reshape(2, 1) - 1j * torch.stack(b[0:2]).reshape(2, 1))  # du*/dtheta_1
    h = 0.5 * (torch.conj(torch.stack(a[0:2]).reshape(2, 1)) - 1j * torch.conj(torch.stack(b[0:2]).reshape(2, 1)))  # du*/dtheta_1*
    i = torch.vstack((g, h, torch.conj(e)))  # du*/dtheta

    # Compute Fisher Information Matrix
    fim = torch.conj(f).T @ f + torch.conj(i).T @ i
    return fim

def get_CRLB(fim_scalar):
    """
    fim_scalar: scalar Fisher Information value
    returns: scalar CRLB
    """
    return 1.0 / fim_scalar


def u_h_re_comp(L1_tensor, gamma_list, Zc_list):
    """
    Returns Re(H) given L1_tensor and network parameters
    L1_tensor: shape [B] or scalar tensor
    gamma_list: shape [F]
    Zc_list: shape [F]
    """
    h = compute_H_complex(L1_tensor, gamma_list, Zc_list)  # shape: [B, F] or [F]
    return torch.real(h)

def u_h_im_comp(L1_tensor, gamma_list, Zc_list):
    """
    Returns Im(H) given L1_tensor and network parameters
    """
    h = compute_H_complex(L1_tensor, gamma_list, Zc_list)
    return torch.imag(h)



def extract_peaks(h, k):
    """
    Returns k peak magnitudes and indices of input signal h
    """
    h_np = torch.abs(h).cpu().numpy()
    peaks, _ = find_peaks(h_np, distance=10)
    topk_idx = peaks[np.argsort(h_np[peaks])[-k:]]
    topk_mag = h_np[topk_idx]
    #return torch.tensor(topk_idx, dtype=torch.float32)
    #return torch.tensor(topk_mag, dtype=torch.float32)
    return torch.stack([torch.tensor(topk_idx, dtype=torch.float32), torch.tensor(topk_mag, dtype=torch.float32)])

def time_domain_mse_loss(L1_batch, obs_tf_complex, gamma_list, Zc_list):
    """
    Compute MSE between (h_obs) and (h_pred)

    Args:
        L1_batch: [B] tensor of candidate fault locations
        obs_tf_complex: [N, F] tensor of observed transfer functions
        gamma_list: [F] tensor of propagation constants
        Zc_list: [F] tensor of characteristic impedances

    Returns:
        loss: [B] tensor of MSE between time domain signals
        """

    # Predict transfer functions H(f) for each L1
    H_pred_freq = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]

    # Convert both predicted and observed TFs to time domain
    h_pred_td = torch.fft.ifft(H_pred_freq, dim=1)  # [B, F]
    h_obs_td = torch.fft.ifft(obs_tf_complex, dim=1)  # [N, F]

    # Reshape for broadcasting
    h_pred_td = h_pred_td.unsqueeze(1)  # [B, 1, T]
    h_obs_td = h_obs_td.unsqueeze(0)    # [1, N, T]

    # Compute element-wise squared error
    diff = torch.abs(h_pred_td - h_obs_td) ** 2  # [B, N, T]

    # Average over time and batch examples
    mse_loss = diff.mean(dim=(1, 2))  # [B]

    return mse_loss




def time_domain_residual_feature_mse_loss(L1_batch, obs_tf_complex, gamma_list, Zc_list, k):
    """
    Compute MSE between extracted time-domain features of (h_obs - h_clean) and (h_pred - h_clean).

    Args:
        L1_batch: [B] tensor of candidate fault locations
        obs_tf_complex: [N, F] tensor of observed transfer functions
        gamma_list: [F] tensor of propagation constants
        Zc_list: [F] tensor of characteristic impedances
        k: number of peaks to extract
        compute_H_clean_func: function to compute clean H(f)
        compute_H_complex_func: function to compute predicted H(f)

    Returns:
        loss: [B] tensor of MSE between residual peak features
    """

    B = L1_batch.shape[0]
    N, F = obs_tf_complex.shape
    assert N == 1, "Currently supports N=1 only"

    # Clean (no fault) transfer function
    h_clean_freq = compute_H_complex_clean(L1_batch, gamma_list, Zc_list)  # [F]
    h_clean_td = torch.fft.ifft(h_clean_freq).unsqueeze(0) #[1, T]

    obs_td = torch.fft.ifft(obs_tf_complex, dim=1)  # [N=1, T]
    
    residual_obs = obs_td - h_clean_td #[N, T]
    obs_feat = extract_peaks(residual_obs[0], k=k).to(L1_batch.device)

    losses = []

    H_pred_freq = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]
    h_pred_td = torch.fft.ifft(H_pred_freq, dim=1) #[B, T]
    residual_pred = h_pred_td - h_clean_td.expand(B, -1) #[B, T] - [B, T]
    
    for b in range(B):
        pred_feat = extract_peaks(residual_pred[b], k=k).to(L1_batch.device)
        mse = torch.nn.functional.mse_loss(obs_feat, pred_feat)
        losses.append(mse)

    return torch.stack(losses)  # [B]


def time_domain_feature_mse_loss(L1_batch, obs_tf_complex, gamma_list, Zc_list, k):
    """
    Compute MSE loss between extracted time-domain features (peaks) of predicted vs observed signals.

    Args:
        L1_batch: [B] tensor of candidate fault locations
        obs_tf_complex: [N, F] tensor of observed transfer functions
        gamma_list: [F] tensor of propagation constants
        Zc_list: [F] tensor of characteristic impedances
        k: number of peaks to compare
    Returns:
        loss: [B] tensor of MSE between extracted peak features
    """
    B = L1_batch.shape[0]
    N, F = obs_tf_complex.shape
    assert N == 1, "This function currently supports N=1 (single observation) only"

    # Convert observed to time domain
    obs_td = torch.fft.ifft(obs_tf_complex, dim=1).squeeze(0)  # [T]
    obs_feat = extract_peaks(obs_td, k=k).to(L1_batch.device)  # [k]

    H_pred_freq = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]    
    h_pred_td = torch.fft.ifft(H_pred_freq, dim=1)  # [B, T]

    # Extract peaks from each predicted signal and compute loss
    losses = []
    for i in range(B):
        pred_td = h_pred_td[i]
        pred_feat = extract_peaks(pred_td, k=k).to(L1_batch.device)
        mse = torch.nn.functional.mse_loss(obs_feat, pred_feat)
        losses.append(mse)

    return torch.stack(losses)  # [B]


def log_likelihood_time_domain_2(L1_batch, obs_td_complex, gamma_list, Zc_list, sigma_f_squared):
    """
    Time-domain log-likelihood using transformed frequency-domain noise.

    Arguments:
    - L1_batch: shape [B] (candidate L1 values)
    - obs_td_complex: shape [N, T] (time-domain noisy observations)
    - gamma_list: [F] frequency-dependent parameters
    - Zc_list: [F] frequency-dependent parameters
    - sigma_f_squared: [F] noise variance at each frequency

    Returns:
    - log-likelihoods: [B]
    """
    B = L1_batch.shape[0]
    N, T = obs_td_complex.shape
    F = len(gamma_list)

    # Frequency-domain noise covariance matrix
    Sigma_f = torch.diag(sigma_f_squared)  # [F, F]

    # Inverse FFT matrix (unitary up to scale)
    omega = torch.exp(2j * np.pi * torch.arange(F).view(-1, 1) * torch.arange(F).view(1, -1) / F)
    F_inv = omega / F  # [F, F]
    F_inv = F_inv.to(device)

    Sigma_f = Sigma_f.to(dtype=torch.complex64)
    # Time-domain noise covariance
    Sigma_t = F_inv @ Sigma_f @ F_inv.conj().T  # [F, F], dense Hermitian
    #print("Sigma_t", Sigma_t)
    # Invert and expand for batch
    Sigma_t_inv = torch.linalg.inv(Sigma_t)
    Sigma_t_inv_exp = Sigma_t_inv.unsqueeze(0).expand(N, -1, -1)  # [N, F, F]

    # Log-determinant
    log_det_term = torch.logdet(Sigma_t).real  # scalar

    # Predict H(f) and IFFT to get time-domain signals
    H_freq = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]
    h_time = torch.fft.ifft(H_freq, dim=1)  # [B, T]

    # Expand dims for broadcast
    h_time = h_time.unsqueeze(1).expand(-1, N, -1)      # [B, N, T]
    obs_td = obs_td_complex.unsqueeze(0).expand(B, -1, -1)  # [B, N, T]

    # Compute difference
    diff = obs_td - h_time  # [B, N, T]
    diff_complex = diff.view(B * N, T)

    # Mahalanobis distance: real part of x^H Σ⁻¹ x
    mahalanobis = torch.einsum("bi,ij,bj->b", diff_complex.conj(), Sigma_t_inv, diff_complex)
    mahalanobis = mahalanobis.view(B, N).real.sum(dim=1)  # [B]

    # Total log-likelihood
    log_likelihood = -N * log_det_term - mahalanobis

    return log_likelihood  # shape [B]

def log_likelihood_time_domain(L1_batch, obs_tf_complex, gamma_list, Zc_list, sigma_t_squared):
    """
    Compute the log-likelihood in the time domain assuming independent (but not identically distributed) noise.

    Args:
        L1_batch: [B] tensor of candidate fault locations
        obs_tf_complex: [N, F] tensor of observed transfer functions (frequency domain)
        gamma_list: [F] tensor of propagation constants
        Zc_list: [F] tensor of characteristic impedances
        sigma_t_squared: [F] tensor of time-domain noise variances

    Returns:
        log_likelihood: [B] tensor of log-likelihoods
    """

    B = L1_batch.shape[0]
    N, F = obs_tf_complex.shape
    device = obs_tf_complex.device

    # Constant noise variance
    #sigma_val = torch.tensor(0.1, dtype=torch.complex64, device=device)
    #Sigma_f = torch.eye(F, dtype=torch.complex64, device=device) * sigma_val  # [F, F]
    # Sigma_f = torch.diag(sigma_t_squared)
    # Sigma_f = Sigma_f.to(dtype=torch.complex64)
    # # Unitary DFT matrix U
    # k = torch.arange(F, device=device).view(-1, 1)
    # n = torch.arange(F, device=device).view(1, -1)
    # U = torch.exp(2j * np.pi * k * n / F) / torch.sqrt(torch.tensor(F, dtype=torch.float32))

    # # Convert Sigma_f to time domain
    # Sigma_t = U @ Sigma_f @ U.conj().T

    # # Convert back to frequency domain
    # Sigma_f_reconstructed = U.conj().T @ Sigma_t @ U

    # # Compare
    # diff = torch.abs(Sigma_f - Sigma_t)
    # print("Max abs difference between Sigma_t and sigma_F:", diff.max().item())

    # print("Sigma_f:\n", Sigma_f)
    # print("Sigma_t:\n", Sigma_t)
    #print("Sigma_f_reconstructed:\n", Sigma_f_reconstructed)

    # B = L1_batch.shape[0]
    # N, F = obs_tf_complex.shape  # F = T (time points)
    # device = obs_tf_complex.device

    # # Constant noise variance
    # sigma_val = torch.tensor(0.1, dtype=torch.complex64, device=device)  # must be complex and on same device
    # Sigma_f = torch.eye(F, dtype=torch.complex64, device=device) * sigma_val  # [F, F]

    # # DFT matrix U (unitary matrix)
    # U = torch.exp(2j * np.pi * torch.arange(F, device=device).view(-1, 1) * torch.arange(F, device=device).view(1, -1) / F)
    # U = U / torch.sqrt(torch.tensor(F))
    # Sigma_t = U @ Sigma_f @ U.conj().T

    # print("Sigma_f", Sigma_f)
    # print("Sigma_t", Sigma_t)

    # B = L1_batch.shape[0]
    # N, F = obs_tf_complex.shape  # F = T (time points)
    #  # Frequency-domain noise covariance matrix
    # sigma_val = torch.tensor(0.1)  # or whatever scalar variance you want
    # Sigma_f = torch.eye(torch.tensor(F)) * sigma_val
    # #Sigma_f = torch.diag(sigma_t_squared)  # [F, F]
    # #Sigma_f = Sigma_f.to(dtype=torch.complex64)
    # U = torch.exp(2j * np.pi * torch.arange(F).view(-1, 1) * torch.arange(F).view(1, -1) / torch.sqrt(torch.tensor(F)))
    # U = U.to(device)
    # Sigma_t = U @ Sigma_f @ U.conj().T
    # print("Sigma_f", Sigma_f)
    # print("Sigma_t", Sigma_t)

    # Inverse FFT matrix (unitary up to scale)
    

    # Compute predicted H(f) for each L1
    H_pred_freq = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]

    # IFFT to time domain
    h_pred_time = torch.fft.ifft(H_pred_freq, dim=1)  # [B, F]
    h_pred_time = h_pred_time.unsqueeze(1).expand(-1, N, -1)  # [B, N, F]

    # IFFT of observed signals
    obs_time = torch.fft.ifft(obs_tf_complex, dim=1)  # [N, F]
    obs_time = obs_time.unsqueeze(0).expand(B, -1, -1)  # [B, N, F]

    # Compute squared magnitude difference
    diff_sq = torch.abs(obs_time - h_pred_time) ** 2  # [B, N, F]
    #print("sigma_t_squared", sigma_t_squared)
    # Expand noise variance for broadcasting
    sigma_t_squared_exp = sigma_t_squared.view(1, 1, F)  # [1, 1, F]

    # Log-likelihood term (sum over N and T)
    log_likelihood = (-torch.log(np.pi * sigma_t_squared_exp) - diff_sq / sigma_t_squared_exp).sum(dim=2).sum(dim=1)  # [B]

    return log_likelihood

def log_likelihood_phase_batch(L1_batch, obs_tf_complex, noise_std_f, gamma_list, Zc_list):
    """
    Log-likelihood over phase observations using wrapped Gaussian approximation.

    Inputs:
    - L1_batch: shape [B]
    - obs_tf_complex: [N, F]
    - noise_std_f: [N, F] (standard deviation of complex Gaussian noise)
    - gamma_list, Zc_list: [F]

    Returns:
    - Log-likelihood over [B] L1 candidates
    """
    # Compute predicted H(f) for all L1 candidates
    H_pred_batch = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]
    phase_pred = torch.angle(H_pred_batch).unsqueeze(1).expand(-1, obs_tf_complex.shape[0], -1)  # [B, N, F]
    
    # Observed phase
    phase_obs = torch.angle(obs_tf_complex).unsqueeze(0).expand(L1_batch.shape[0], -1, -1)  # [B, N, F]

    # Predicted |H(f)| to compute phase noise variance (approximation)
    abs_H_pred = torch.abs(H_pred_batch).unsqueeze(1).expand(-1, obs_tf_complex.shape[0], -1)  # [B, N, F]
    noise_var_f = noise_std_f**2
    phase_var = noise_var_f.unsqueeze(0) / (abs_H_pred ** 2 + 1e-12)  # [B, N, F]

    # Wrap phase difference into [-pi, pi]
    delta = torch.remainder(phase_obs - phase_pred + np.pi, 2 * np.pi) - np.pi  # [B, N, F]

    # Negative log-likelihood (Gaussian on circle)
    log_prob = -0.5 * (delta ** 2) / (phase_var + 1e-12) - 0.5 * torch.log(2 * np.pi * phase_var + 1e-12)

    return log_prob.sum(dim=(1, 2))  # Sum over N and F → shape [B]


def log_likelihood_batch(L1_batch, obs_tf_complex, noise_var_f, gamma_list, Zc_list):
    """
    Vectorized Complex Gaussian log-likelihood over all L1 candidates using PyTorch (GPU-compatible)
    
    Arguments: 
    - L1_batch: shape [B]
    - obs_tf_complex: shape [N, F]
    - noise_var_f: shape [N, F] (noise var per freq)
    - gamma_list, Zc_list: shape [F]

    Returns:
    - Total log-likelihood for each L1 in batch, shape [B]
    """
    H_pred_batch = compute_H_complex(L1_batch, gamma_list, Zc_list)  # shape: [B, F]
    H_pred_batch = H_pred_batch.unsqueeze(1).expand(-1, N, -1)  # [B, N, F]
    obs_batch = obs_tf_complex.unsqueeze(0).expand(H_pred_batch.shape[0], -1, -1)  #[N, F] -> [1, N, F] -> [B, N, F]
    noise_var_batch = noise_var_f.unsqueeze(0).expand(H_pred_batch.shape[0], -1, -1) #[N, F] -> [1, N, F] -> [B, N, F]
    diff = obs_batch - H_pred_batch #[B, N, F]
    nll = (torch.abs(diff) ** 2) / noise_var_batch
    return -0.5 * nll.sum(dim=(1, 2))  # shape: [B]

def log_likelihood_magnitude_gaussian(L1_batch, r_obs, sigma_f_squared, gamma_list, Zc_list):
    """
    Log-likelihood assuming |H_obs| is Gaussian-distributed with mean |H(f)| and variance sigma_f_squared.
    This is a model mismatch approximation.
    
    Arguments:
    - L1_batch: [B]
    - r_obs: [N, F] observed magnitudes (|H_obs|)
    - sigma_f_squared: [F]
    
    Returns:
    - log-likelihoods: [B]
    """
    B = L1_batch.shape[0]
    N, F = r_obs.shape

    # Predict H and take magnitudes
    H_pred = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]
    A = torch.abs(H_pred)  # [B, F]

    # Broadcast for [B, N, F]
    r_exp = r_obs.unsqueeze(0)           # [1, N, F]
    A_exp = A.unsqueeze(1)               # [B, 1, F]
    sigma2 = sigma_f_squared.view(1, 1, F)  # [1, 1, F]

    # Gaussian log-likelihood
    ll_bnf = -0.5 * (
        torch.log(2 * torch.pi * sigma2 + 1e-12)
        + (r_exp - A_exp) ** 2 / (sigma2 + 1e-12)
    )  # [B, N, F]

    return ll_bnf.sum(dim=(1, 2))  # total log-likelihood per L1


def log_likelihood_ricean_batch(L1_batch, r_obs, sigma_f_squared, gamma_list, Zc_list):
    """
    Vectorized Ricean log-likelihood over all L1 candidates using PyTorch (GPU-compatible).

    Arguments:
    - L1_batch: [B] torch tensor of candidate L1 values (on device)
    - r_obs: [N, F] observed magnitudes (on device)
    - sigma_f_squared: [F] noise variance per frequency (on device)
    - gamma_list, Zc_list: [F] network params (on device)

    Returns:
    - log-likelihoods: [B] torch tensor of total log-likelihood per L1 (on device)
    """
    B = L1_batch.shape[0]
    N, F = r_obs.shape

    # Compute predicted H for each L1: [B, F]
    H_pred = compute_H_complex(L1_batch, gamma_list, Zc_list)  # [B, F]
    A = torch.abs(H_pred)  # [B, F]

    # Expand dims for broadcasting
    r_exp = r_obs.unsqueeze(0)              # [1, N, F]
    A_exp = A.unsqueeze(1)                  # [B, 1, F]
    sigma2 = sigma_f_squared.view(1, 1, F)  # [1, 1, F]

    # Compute the Ricean log-likelihood
    bessel_arg = 2 * r_exp * A_exp / sigma2  # [B, N, F]
    ll_bnf = (
        torch.log(r_exp + 1e-12)                           # log(r)
        - torch.log(sigma2 + 1e-12)                        # -log(σ²)
        - (r_exp ** 2 + A_exp ** 2) / sigma2               # - (r² + A²)/σ²
        + torch.log(torch.special.i0e(bessel_arg + 1e-12)) # log I₀e(...)
        + bessel_arg                                       # add back exp(⋅) removed from i0e
    )  # [B, N, F]

    # Total log-likelihood for each L1
    ll_total = ll_bnf.sum(dim=(1, 2))  # [B]

    return ll_total 

def ricean_fisher_info_numerical(L1_tensor, gamma_list, Zc_list, sigma_f_squared, num_samples=5000):
    """
    Numerically approximate Fisher Information for scalar parameter L1
    using the Ricean likelihood.

    Parameters:
        L1_tensor: scalar tensor with requires_grad=True
        gamma_list, Zc_list: torch tensors, shape [F]
        sigma_f_squared: torch tensor, shape [F]
        num_samples: number of Monte Carlo samples per frequency

    Returns:
        Scalar Fisher Information (float)
    """
    F = gamma_list.shape[0]

    # Ensure gradient tracking
    L1_tensor = L1_tensor.clone().detach().requires_grad_(True)

    # Compute complex transfer function
    H = compute_H_complex(L1_tensor.unsqueeze(0), gamma_list, Zc_list)[0]  # shape [F]
    A = torch.abs(H).detach().cpu().numpy()  # shape [F]
    
    # Compute dA/dL1 using autograd
    A_tensor = torch.abs(H)
    dA_list = []
    for i in range(F):
        dA_i, = torch.autograd.grad(A_tensor[i], L1_tensor, retain_graph=True)
        dA_list.append(dA_i)
    dA = torch.stack(dA_list).detach().cpu().numpy()  # shape [F]

    sigma2 = sigma_f_squared.detach().cpu().numpy()  # shape [F]

    fisher_info = 0.0
    for f in range(F):
        A_f = A[f]
        dA_f = dA[f]
        sigma2_f = sigma2[f]

        # Sample r from Rice(A_f, sigma2_f)
        s = np.sqrt(sigma2_f / 2)
        x = np.random.normal(loc=A_f, scale=s, size=num_samples)
        y = np.random.normal(loc=0.0, scale=s, size=num_samples)
        r_samples = np.sqrt(x**2 + y**2)  # shape: [num_samples]

        z = (2 * A_f * r_samples) / sigma2_f
        
        #I0 = np.exp(z) * ive(0, z)  # i0(z) = exp(z) * ive(0, z)
        #I1 = np.exp(z) * ive(1, z)  # i1(z) = exp(z) * ive(1, z)
        #I1_I0_ratio = I1 / (I0 + 1e-12)
        #I1_I0_ratio = i1(z) / (i0(z) + 1e-12)
        z_thresh = 50.0
        I1_I0_ratio = np.where(
            z < z_thresh,
            i1(z) / (i0(z) + 1e-12),
            1.0 - 0.5 / (z + 1e-12)
        )

        inside_term = (-A_f + r_samples * I1_I0_ratio) ** 2
        expectation = np.mean(inside_term)

        I_f = ((2 * dA_f) / sigma2_f) ** 2 * expectation
        fisher_info += I_f

    return fisher_info  # scalar float

def compute_fisher_info_jacobian(L1_tensor, gamma_list, Zc_list, sigma_f_squared):
    """
    Computes scalar Fisher Information for scalar parameter L1 using full Jacobian.
    
    Arguments:
    - L1_tensor: scalar torch tensor with .requires_grad=True
    - gamma_list, Zc_list: torch tensors of shape [F]
    - sigma_f_squared: noise variance per frequency, shape [F]
    
    Returns:
    - Scalar Fisher Information
    """

    # Ensure L1 has requires_grad
    L1_tensor = L1_tensor.clone().detach().requires_grad_(True)

    # Define real and imaginary parts of H(f) as vector-valued functions
    def H_real_fn(l1):
        return compute_H_complex(l1.unsqueeze(0), gamma_list, Zc_list)[0].real  # shape [F]

    def H_imag_fn(l1):
        return compute_H_complex(l1.unsqueeze(0), gamma_list, Zc_list)[0].imag  # shape [F]

    # Compute the Jacobians (shape [F]) w.r.t. scalar L1
    J_real = jacobian(H_real_fn, L1_tensor)  # shape: [F]
    J_imag = jacobian(H_imag_fn, L1_tensor)  # shape: [F]

    # Fisher Information
    fim = torch.sum(2*(J_real ** 2 + J_imag ** 2) / sigma_f_squared)

    return fim

def compute_fisher_info(L1_tensor, gamma_list, Zc_list, sigma_f_squared):
    """
    L1_tensor: scalar tensor with requires_grad=True
    gamma_list, Zc_list: shape [F]
    sigma_f_squared: shape [F] (per-frequency noise variance)
    """
    # Ensure gradient tracking
    L1_tensor = L1_tensor.clone().detach().requires_grad_(True)

    # Compute transfer function (shape [1, F]) and squeeze to [F]
    H = compute_H_complex(L1_tensor.unsqueeze(0), gamma_list, Zc_list)[0]
    H_real = torch.real(H)
    H_imag = torch.imag(H)

    # Compute per-frequency gradients manually
    dRe_list = []
    dIm_list = []
    #print("H shape", H.shape[0])
    for i in range(H.shape[0]):  # Loop over F frequencies
        dRe_i, = torch.autograd.grad(H_real[i], L1_tensor, retain_graph=True)
        dIm_i, = torch.autograd.grad(H_imag[i], L1_tensor, retain_graph=True)
        dRe_list.append(dRe_i)
        dIm_list.append(dIm_i)

    dRe = torch.stack(dRe_list)  # shape [F]
    dIm = torch.stack(dIm_list)  # shape [F]

    # Fisher Information: scalar
    fim = torch.sum((dRe**2 + dIm**2) / sigma_f_squared)
    print("fim", fim)
    return fim


# --- Setup ---
gamma_full, Zc_full, _, _, list_of_freq = initial_network()
fstart, fend, num_of_freq_points = 2e6, 10e6, 500
desired_freqs = torch.linspace(fstart, fend, num_of_freq_points)
vec_meas = torch.abs(list_of_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
gamma_list = gamma_full[vec_meas]
Zc_list = Zc_full[vec_meas]


# Ensure gamma_list and Zc_list are on GPU and have shape [F]
gamma_list = gamma_list.to(device)
Zc_list = Zc_list.to(device)
mse_list = []
mse_list2 = []
mse_list3 = []
mse_list4 = []
mse_list5 = []
mse_list6 = []
mse_list7 = []
mse_list8 = []
mse_list9 = []
crlb_list = []




@torch.inference_mode()
def build_test_set_for_snr(snr_db, num_runs, gamma_list, Zc_list, device, rng, include_snr_feature=True):
    """
    Vectorized: generates `num_runs` test samples at a fixed SNR in one shot.

    Returns:
        X: np.ndarray of shape [num_runs, 2*F (+1 if include_snr_feature)]
        y_true: np.ndarray of shape [num_runs]
    """

    F = gamma_list.shape[0]
    D_base = 2 * F

    # 1) Draw true L1 in batch
    y_true = rng.uniform(100.0, 900.0, size=num_runs).astype(np.float32)
    L1_tensor = torch.from_numpy(y_true).to(device)        # [B]

    # 2) Forward model in batch → [B, F] complex
    true_tf_complex = compute_H_complex(L1_tensor, gamma_list, Zc_list)  # [B, F]

    # 3) Per-sample noise level based on average signal power for that sample
    snr_linear = 10.0 ** (snr_db / 10.0)
    signal_power_mean = torch.mean(torch.abs(true_tf_complex) ** 2, dim=1)   # [B]
    noise_var  = signal_power_mean / snr_linear                               # [B]
    noise_std  = torch.sqrt(noise_var / 2.0)                                   # [B]

    # 4) Generate complex Gaussian noise with the same shape as the batch
    #    Expand per-sample std across frequency dimension
    noise_std_f = noise_std.unsqueeze(1).expand(-1, F)                        # [B, F]
    real_noise  = noise_std_f * torch.randn_like(true_tf_complex.real)
    imag_noise  = noise_std_f * torch.randn_like(true_tf_complex.imag)

    obs_tf_complex = true_tf_complex + real_noise + 1j * imag_noise           # [B, F]
    obs_tf_realimag = torch.view_as_real(obs_tf_complex)                       # [B, F, 2]

     # 5) Flatten to features per sample: [Re1, Im1, ..., ReF, ImF]
    X_base = obs_tf_realimag.reshape(num_runs, D_base).detach().cpu().numpy()  # [B, 2F]



def make_fixed_sets(snr_dbs, seed):
    rng = np.random.default_rng(seed)
    tgen = torch.Generator(device=device).manual_seed(seed)
    sets = {}
    for snr_db in snr_dbs:
        X_te, y_te = build_test_set_for_snr(
            snr_db=snr_db, num_runs=1000,
            gamma_list=gamma_list, 
            Zc_list=Zc_list, 
            device=device, 
            rng=rng, 
            include_snr_feature=True
        )
        sets[snr_db] = (X_te, y_te)
    return sets

dev_sets   = make_fixed_sets(snr_dbs, seed=1234)   # for learning curves / N*
print("devsets", dev_sets)
final_sets = make_fixed_sets(snr_dbs, seed=5678)   # for final one-shot report























# for snr_db in snr_dbs:
#     snr_linear = 10 ** (snr_db / 10)
#     sq_errors = []
#     sq_errors2 = []
#     sq_errors3 = []
#     sq_errors4 = []
#     sq_errors5 = []
#     sq_errors6 = []
#     sq_errors7 = []
#     sq_errors8 = []
#     sq_errors9 = []
#     crlbs = []
    
#     fixed_sigma = sigma_from_snr(snr_db, sigma_max=0.05, sigma_min=0.002, alpha=1.0, ref_db=0.0)

#     print(f"Running at SNR = {snr_db} dB...")


#     for runs in range(num_runs):
#         true_L1 = np.random.uniform(100.0, 900.0)
#         true_L1_tensor = torch.tensor(true_L1, device=device)
#         true_tf_complex = compute_H_complex(true_L1_tensor.unsqueeze(0), gamma_list, Zc_list) # [1, F]

#         signal_power_mean = torch.mean(torch.abs(true_tf_complex[0]) ** 2)
#         noise_var = signal_power_mean / snr_linear
#         noise_var_f = torch.full_like(true_tf_complex[0], fill_value=noise_var).real
#         #Freq dependent noise
#         # H_power = torch.abs(true_tf_complex[0]) ** 2 #[1, F]
#         # noise_var_f = H_power / snr_linear #[1, F]
#         noise_std_f = torch.sqrt(noise_var_f / 2).to(device)  #[1, F]
#         real_noise = noise_std_f * torch.randn(N, num_of_freq_points, device=device)
#         imag_noise = noise_std_f * torch.randn(N, num_of_freq_points, device=device)
#         noise_complex = real_noise + 1j * imag_noise
#         obs_tf_complex = true_tf_complex + noise_complex  #[1, F]
#         obs_tf_real = torch.view_as_real(obs_tf_complex)  # real + imag split [1, F, 2]
#         r_obs = torch.abs(obs_tf_complex)  # magnitude: [1, F]
#         obs_td_complex = torch.fft.ifft(obs_tf_complex) #time domain [1, F]

#         L1_candidates = L1_grid.to(device)
        
#         #Log likelihoods (MLE)
#         ll_vals = log_likelihood_batch(L1_candidates, obs_tf_complex, noise_var_f, gamma_list, Zc_list)
#         ll_vals_mag_ricean = log_likelihood_ricean_batch(L1_candidates, r_obs, noise_var_f, gamma_list, Zc_list)
#         ll_vals_mag_gaussian = log_likelihood_magnitude_gaussian(L1_candidates, r_obs, noise_var_f, gamma_list, Zc_list)
#         ll_vals_phase = log_likelihood_phase_batch(L1_candidates, obs_tf_complex, noise_std_f, gamma_list, Zc_list)
#         ll_vals_time = log_likelihood_time_domain_2(L1_candidates, obs_td_complex, gamma_list, Zc_list, noise_var_f) #Correct Log Likelihood in Time Domain
#         ll_vals_time_2 = log_likelihood_time_domain(L1_candidates, obs_tf_complex, gamma_list, Zc_list, noise_var_f) #Incorrect Log Likelihood in Time Domain
#         #Heuristic Loss function
#         ll_vals_clean = time_domain_mse_loss(L1_candidates, obs_tf_complex, gamma_list, Zc_list) #Time Domain MSE Loss ||h_obs - h_pred||^2 

#         # ELBO over μ (K=MC samples per μ)
#         elbo_vals = []
#         M = 500

#         for mu in mu_vals:
#             base = dist.Normal(mu, fixed_sigma)  # q_Z
#             q = TransformedDistribution(base, [SigmoidTransform()])  # q_X with Jacobian handled
#             samples = q.rsample((M,)).to(device)                     # x ~ q_X, shape [M]
#             L1_samples = denormalize(samples, 0.0, 1000.0).to(device)

#             H_pred = compute_H_complex(L1_samples, gamma_list, Zc_list)                 # [M, F]
#             H_pred_real = torch.view_as_real(H_pred)                                    # [M, F, 2]

#             noise_std = noise_std_f.squeeze(0)                                          # [F]
#             noise_std_expanded = noise_std.unsqueeze(-1).expand(-1, 2)                  # [F, 2]
#             dist_obs = dist.Independent(dist.Normal(H_pred_real, noise_std_expanded), 2)

#             log_lik = dist_obs.log_prob(obs_tf_real.squeeze(0))                         # [M]
#             log_q   = q.log_prob(samples).to(device)                                    # [M]
#             elbo    = (log_lik - log_q).mean()                                          # scalar
#             elbo_vals.append(elbo.item())

#         # argmax ELBO → μ*
#         max_elbo_idx = torch.tensor(elbo_vals).argmax()
#         mu_star = mu_vals[max_elbo_idx].item() #estimate for mu
#         K = 1000
#         z = mu_star + + fixed_sigma * torch.randn(K, device=device)
#         x = torch.sigmoid(z).clamp(1e-6, 1 - 1e-6)
#         L1_samples = 1000.0 * x
#         L1_hat_mean = L1_samples.mean().item()          # <-- posterior-mean estimate
#         mu_max_sigmoid =1.0 / (1.0 + np.exp(-mu_star))    
#         L1_hat = denormalize(mu_max_sigmoid, *param_bounds["L1"]) # <-- posterior median estimate
        
#         sq_errors8.append((L1_hat - true_L1) ** 2)
#         sq_errors9.append((L1_hat_mean - true_L1) ** 2)

#         #Feature Loss
#         #ll_vals_time_feature = time_domain_feature_mse_loss(L1_candidates, obs_tf_complex, gamma_list, Zc_list, num_of_peaks)
#         #ll_vals_clean_feature = time_domain_residual_feature_mse_loss(L1_candidates, obs_tf_complex, gamma_list, Zc_list, num_of_peaks)

#         #Argmax for MLE 
#         best_idx = torch.argmax(ll_vals).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors.append((L1_est - true_L1) ** 2)

#         best_idx = torch.argmax(ll_vals_mag_ricean).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors2.append((L1_est - true_L1) ** 2)

#         best_idx = torch.argmax(ll_vals_mag_gaussian).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors3.append((L1_est - true_L1) ** 2)

#         best_idx = torch.argmax(ll_vals_phase).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors4.append((L1_est - true_L1) ** 2)

#         best_idx = torch.argmax(ll_vals_time).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors5.append((L1_est - true_L1) ** 2)

#         best_idx = torch.argmax(ll_vals_time_2).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors6.append((L1_est - true_L1) ** 2)


#         #Argmin for loss functions

#         best_idx = torch.argmin(ll_vals_clean).item()
#         L1_est = L1_grid[best_idx].item()
#         sq_errors7.append((L1_est - true_L1) ** 2)


#         # best_idx = torch.argmin(ll_vals_time_feature).item()
#         # L1_est = L1_grid[best_idx].item()
#         # sq_errors8.append((L1_est - true_L1) ** 2)


#         # best_idx = torch.argmin(ll_vals_clean_feature).item()
#         # L1_est = L1_grid[best_idx].item()
#         # sq_errors9.append((L1_est - true_L1) ** 2)
        

        # --- CRLB Calculation ---

        # fim = compute_fisher_info_jacobian(true_L1_tensor, gamma_list, Zc_list, noise_var_f)
        # fim2 = ricean_fisher_info_numerical(true_L1_tensor, gamma_list, Zc_list, noise_var_f)

        # fim = compute_fisher_info(true_L1_tensor, gamma_list, Zc_list, noise_var_f) #returns tensor with one element 
        # print("fim", fim)
        # print("fim2", fim2)
        # crlb = 1.0 / fim.item()
        # crlb2 = 1.0 / fim2
 
        # print("MSE at this specific realization", (L1_est - true_L1) ** 2)
        # print("CRLB", crlb)
        # crlbs.append(crlb)
        # crlbs2.append(crlb2)
        # if runs % 10 == 0:
        #     print(f"  Processed {runs}/1000 at SNR = {snr_db} dB, Last MSE: {(L1_est - true_L1) ** 2:.3f}")
        #           , CRLB:{crlb:.3f}")
        # if (L1_est - true_L1) ** 2 > 100:
        #     print("L1 est", L1_est)
        #     print("true_L1", true_L1)
        # # Plot log-likelihood landscape when MSE > 100

        # plt.figure(figsize=(8, 4))
        # plt.plot(L1_grid.cpu().numpy(), ll_vals.cpu().numpy(), label="Log-Likelihood Phase")
        # plt.axvline(true_L1, color='r', linestyle='--', label='True $L_1$')
        # plt.axvline(L1_est, color='g', linestyle='--', label='Estimated $L_1$')
        # plt.xlabel("$L_1$ Candidate")
        # plt.ylabel("Log-Likelihood")
        # plt.title(f"Likelihood Landscape at SNR = {snr_db} dB (MSE = {(L1_est - true_L1) ** 2:.1f})")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

#     mse_list.append(np.mean(sq_errors))
#     mse_list2.append(np.mean(sq_errors2))
#     mse_list3.append(np.mean(sq_errors3))
#     mse_list4.append(np.mean(sq_errors4))
#     mse_list5.append(np.mean(sq_errors5))
#     mse_list6.append(np.mean(sq_errors6))
#     mse_list7.append(np.mean(sq_errors7))
#     mse_list8.append(np.mean(sq_errors8))
#     mse_list9.append(np.mean(sq_errors9))
    
#     # crlb_list.append(np.mean(crlbs))
#     # crlb_list2.append(np.mean(crlbs2))

#     print(f"Finished SNR = {snr_db} dB | MSE = {mse_list[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE2 = {mse_list2[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE3 = {mse_list3[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE4 = {mse_list4[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE5 = {mse_list5[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE6 = {mse_list6[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE7 = {mse_list7[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE8 = {mse_list8[-1]:.3f}")
#     print(f"Finished SNR = {snr_db} dB | MSE9 = {mse_list9[-1]:.3f}")
#     # print(f"Finished SNR = {snr_db} dB | CRLB = {crlb_list[-1]:.3f}")
#     # print(f"Finished SNR = {snr_db} dB | CRLB2 = {crlb_list2[-1]:.3f}")





# # Plot
# plt.figure()
# plt.plot(snr_dbs, np.sqrt(mse_list), marker='o', label='RMSE (H(f) (Gaussian)')
# plt.plot(snr_dbs, np.sqrt(mse_list2), marker='o', label='RMSE |H(f)| (Ricean)')
# plt.plot(snr_dbs, np.sqrt(mse_list3), marker='o', label='RMSE |H(f)| (Gaussian)')
# plt.plot(snr_dbs, np.sqrt(mse_list4), marker='o', label='RMSE arg(H(f)) (Von Mises)')
# plt.plot(snr_dbs, np.sqrt(mse_list5), marker='o', label='RMSE h(t) (Full Sigma_t)')
# plt.plot(snr_dbs, np.sqrt(mse_list6), marker='o', label='RMSE h(t) (Diagonal Sigma_t)')
# plt.plot(snr_dbs, np.sqrt(mse_list7), marker='o', label='RMSE h(t) (MSE Loss)')
# plt.plot(snr_dbs, np.sqrt(mse_list8), marker='o', label='label=RMSE (ELBO argmax μ → posterior median; σ scheduled)')
# plt.plot(snr_dbs, np.sqrt(mse_list9), marker='o', label='label=RMSE (ELBO argmax μ → posterior mean; σ scheduled)')
# #plt.plot(snr_dbs, np.sqrt(mse_list9), marker='o', label='RMSE phi(t) (Time Domain Features with Hclean subtracted)')



# plt.plot(snr_dbs, np.sqrt(crlb_list), marker='x', linestyle='--', label='sqrt(CRLB) (Full Complex Signal)')
# #plt.plot(snr_dbs, np.sqrt(crlb_list2), marker='x', linestyle='--', label='sqrt(CRLB) (Phase or Magnitude Only)')
# plt.xlabel("SNR (dB)")
# plt.ylabel("Error (RMSE / sqrt(CRLB))")
# plt.title("Grid Search RMSE vs sqrt(CRLB) for $L_1$")
# plt.grid(True)
# plt.legend(fontsize="x-small")
# plt.tight_layout()
# plt.show()


