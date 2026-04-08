import numpy as np
import random
import pyro
import torch
import copy
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pyro.distributions as dist

from pyro.distributions.transforms import SigmoidTransform, AffineTransform
from pyro.distributions import TransformedDistribution, constraints
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList, AutoNormal
from scipy.linalg import expm
from torch.func import jacfwd

from pyro.infer import SVI, Trace_ELBO
from collections import defaultdict


start_time = time.time()
torch.set_printoptions(precision=8)  # Show 8 decimal places

#Global configs
SCENARIO = "two_stage"
# Options:
#   - "no_fault": Network identification only (Stage 1) - infer load/cable params
#   - "with_fault": Fault localization only - infer fault params (assumes known network)
#   - "two_stage": Full workflow - Stage 1 (network ID) then Stage 2 (fault localization)
FORWARD_MODEL = "full"  # "full" or "admittance"
OPTIMIZER = "Adagrad"  # "Adam" or "Adagrad"
LR = 0.2  # Learning rate for optimizer

# 1 = Constant, 2 = Double RLC, 3 = Motor
FIXED_LOAD_TYPES = [
    3,  # load_0  (was load_21) R6-O3  Motor
    3,  # load_1  (was load_20) R6-O2  Motor
    1,  # load_2  (was load_19) R6-O1  Constant
    3,  # load_3  (was load_18) R5-O4  Motor
    2,  # load_4  (was load_17) R5-O3  Double RLC
    1,  # load_5  (was load_16) R5-O2  Constant
    3,  # load_6  (was load_15) R5-O1  Motor
    2,  # load_7  (was load_14) R4-O4  Double RLC
    3,  # load_8  (was load_13) R4-O3  Motor
    3,  # load_9  (was load_12) R4-O2  Motor
    2,  # load_10 (was load_11) R4-O1  Double RLC
    3,  # load_11 (was load_10) R3-O4  Motor
    1,  # load_12 (was load_9)  R3-O3  Constant
    1,  # load_13 (was load_8)  R3-O2  Constant
    1,  # load_14 (was load_7)  R3-O1  Constant
    3,  # load_15 (was load_6)  R2-O4  Motor
    3,  # load_16 (was load_5)  R2-O3  Motor
    3,  # load_17 (was load_4)  R2-O2  Motor
    3,  # load_18 (was load_3)  R2-O1  Motor
    1,  # load_19 (was load_2)  R1-O4  Constant
    3,  # load_20 (was load_1)  R1-O3  Motor
    1,  # load_21 (was load_0)  R1-O2  Constant
]

def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    Z_rec = torch.tensor([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec

# Network constants
num_loads = 22
num_of_conductors = 4
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

#frequencies = torch.logspace(torch.log10(torch.tensor(2e6)), torch.log10(torch.tensor(10e6)), 500) #2-10MHz
#frequencies = torch.logspace(torch.log10(torch.tensor(150e3)), torch.log10(torch.tensor(30e6)), 200) #150KHz - 30MHz
frequencies = torch.logspace(torch.log10(torch.tensor(150e3, device=device)),
                              torch.log10(torch.tensor(500e3, device=device)), 200, device=device) #150KHz - 500KHz
freq_range_mhz = frequencies / 1e6
omega = 2 * torch.pi * frequencies
num_freqs = len(omega)

# Filename prefix for saving figures (includes config info)
def format_freq(f_hz):
    """Format frequency as kHz or MHz string."""
    if f_hz >= 1e6:
        return f"{int(f_hz / 1e6)}mhz"
    else:
        return f"{int(f_hz / 1e3)}khz"

f_start_str = format_freq(frequencies[0].item())
f_end_str = format_freq(frequencies[-1].item())
FILENAME_PREFIX = f"{FORWARD_MODEL}_{f_start_str}-{f_end_str}_{OPTIMIZER}_lr{LR}"

#Transmitter/Receiver Constants
Z_RG = Z_R1 = Z_R2 = 50.0
Z_R3 = 50.0
ZT0 = ZTG1 = ZTG2 = 50.0
ZTG3 = 50.0
ZT12 = 100.0
ZT13 = ZT23 = 100.0
Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3).to(device)
Y_rec = torch.linalg.inv(Z_rec)
Z_rec = Z_rec.unsqueeze(0).repeat(num_freqs, 1, 1)
Y_rec = Y_rec.unsqueeze(0).repeat(num_freqs, 1, 1)

def calculate_H_nw(Phi_network, n, Z_rec):
    """
    Compute H_nw (transfer function) of ABCD matrix

    Parameters:
    - Phi_network: Overall network ABCD matrix (N, 2n, 2n)
    - n: number of conductors - 1
    - Z_rec: Receiver load matrix (N, n, n)

    Returns:
    - H_nw: Transfer function of ABCD Matrix (N, n, n)
    """
    Phi_networkinv = np.linalg.inv(Phi_network)
    # Split Phi_network into submatrices
    Phi_11 = Phi_networkinv[:, :n, :n]
    Phi_12 = Phi_networkinv[:, :n, n:]
    Phi_21 = Phi_networkinv[:, n:, :n]
    Phi_22 = Phi_networkinv[:, n:, n:]
    if torch.is_tensor(Z_rec):
        Z_rec = Z_rec.detach().cpu().numpy()
    H_nw = np.linalg.inv(Phi_11 + Phi_12 @ np.linalg.inv(Z_rec))
    return H_nw

def calculate_cable_transmission_matrix(R, L, C, G, length, omega):
    """
    Compute ABCD Matrix of MTL
    All inputs are converted to NumPy.
    """
    # ---- convert everything to numpy ----
    R = np.asarray(R)
    L = np.asarray(L)
    C = np.asarray(C)
    G = np.asarray(G)
    omega = np.asarray(omega)
    length = float(length)

    N, n, _ = R.shape
    omega = omega.reshape(-1, 1, 1)

    Z_T = R + 1j * omega * L
    Y_T = G + 1j * omega * C


    ZY = Z_T @ Y_T
    YZ = Y_T @ Z_T
    
    
    eigvals1 = np.zeros((N, n), dtype=complex)
    eigvecs1 = np.zeros((N, n, n), dtype=complex)
    eigvals2 = np.zeros((N, n), dtype=complex)
    eigvecs2 = np.zeros((N, n, n), dtype=complex)

    for i in range(N):
        #eigvals of YZ, ZY are the same and repeated basically
        eigvals1[i], eigvecs1[i] = np.linalg.eig(ZY[i])
        eigvals2[i], eigvecs2[i] = np.linalg.eig(YZ[i])
        
    gamma = np.zeros((N, n, n), dtype=complex)
    gamma[:, np.arange(n), np.arange(n)] = np.sqrt(eigvals1)

    Gamma = eigvecs1 @ gamma @ np.linalg.inv(eigvecs1)

    Zw = np.linalg.inv(Gamma) @ Z_T
    Yw = np.linalg.inv(Zw)

    Phi11 = matrix_cosh_numpy(Gamma * length)
    Phi12 = -matrix_sinh_numpy(Gamma * length) @ Zw
    Phi21 = -Yw @ matrix_sinh_numpy(Gamma * length)
    Phi22 = Yw @ matrix_cosh_numpy(Gamma * length) @ Zw

    Phi_cable = np.block([
        [Phi11, Phi12],
        [Phi21, Phi22]
    ])

    return Phi_cable


def calculate_room_admittance_matrix(Y_loads_room, cable_lengths_room, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                                     T_s, Tinv_s, ZC_s, YC_s, gamma_s):
    """
    Y_loads_room: list of 4 admittance matrices (torch.Tensor with shape (N, n, n))
    cable_lengths_room: list of 5 scalar lengths (floats or tensors)
    """

    # First branch (load_0 -> load_1)
    rho3 = reflection_coefficient(Y_loads_room[0], T_r, Tinv_r, ZC_r, YC_r)
    Y_3_carried = carry_back_load(rho3, T_r, YC_r, gamma_r, cable_lengths_room[0])
    Y_1new = Y_3_carried + Y_loads_room[1]
    rho1 = reflection_coefficient(Y_1new, T_r, Tinv_r, ZC_r, YC_r)
    Y_1new_carried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_room[1])

    # Second branch (load_2 -> load_3)
    rho4 = reflection_coefficient(Y_loads_room[2], T_r, Tinv_r, ZC_r, YC_r)
    Y_4_carried = carry_back_load(rho4, T_r, YC_r, gamma_r, cable_lengths_room[2])
    Y_2new = Y_4_carried + Y_loads_room[3]
    rho2 = reflection_coefficient(Y_2new, T_r, Tinv_r, ZC_r, YC_r)
    Y_2new_carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_room[3])

    # Combine branches and carry back through service panel wire
    Y_room = Y_1new_carried + Y_2new_carried
    rho5 = reflection_coefficient(Y_room, T_s, Tinv_s, ZC_s, YC_s)
    Y_room_carried = carry_back_load(rho5, T_s, YC_s, gamma_s, cable_lengths_room[4])

    return Y_room_carried
def get_mtl_matrices(R, L, C, G, n, omega):
    """
    Compute MTL matrices for multiple frequencies using PyTorch.

    Parameters:
    - R, L, C, G: (N, n, n) tensors
    - omega: (N,) tensor

    Returns:
    - T (N, n, n) - Eigenvectors
    - Tinv (N, n, n) - Inverse of Eigenvectors
    - gamma (N, n, n) - Propagation Constants
    - ZC (N, n, n) - Characteristic Impedance
    - YC (N, n, n) - Characteristic Admittance
    """
    N, n, _ = R.shape

    # Reshape omega to (N, 1, 1) for broadcasting
    omega = omega.view(-1, 1, 1)  

    # Compute impedance and admittance matrices
    Z_T = R + 1j * omega * L  # (N, n, n)
    Y_T = G + 1j * omega * C  # (N, n, n)

    # Compute ZY and YZ matrices
    ZY = torch.matmul(Z_T, Y_T)  # (N, n, n)
    YZ = torch.matmul(Y_T, Z_T)  # (N, n, n)
    
    # Compute eigenvalues and eigenvectors of YZ
    eigvals, eigvecs = torch.linalg.eig(YZ)  # eigvals: (N, n), eigvecs: (N, n, n)

    # Sort eigenvalues by magnitude to ensure consistent mode ordering across frequencies
    # This prevents "mode switching" where eigenvalue indices swap as frequency changes
    # sort_idx = torch.argsort(torch.abs(eigvals), dim=1)  # (N, n) indices sorted by |eigenvalue|
    # # Gather sorted eigenvalues
    # eigvals_sorted = torch.gather(eigvals, 1, sort_idx)  # (N, n)

    # # Gather sorted eigenvectors - need to expand sort_idx for the last dimension
    # # sort_idx_expanded = sort_idx.unsqueeze(-1).expand(-1, -1, n)  # (N, n, n)
    # # eigvecs_sorted = torch.gather(eigvecs, 1, sort_idx_expanded)  # (N, n, n)
    # sort_idx_expanded = sort_idx.unsqueeze(1).expand(-1, n, -1)  # (N, n, n) - note unsqueeze(1) not (-1)
    # eigvecs_sorted = torch.gather(eigvecs, 2, sort_idx_expanded)  # gather along dim=2 (COLUMNS)
    # # Use sorted values
    # eigvals = eigvals_sorted
    # eigvecs = eigvecs_sorted

    # Compute gamma as a diagonal matrix with sqrt(eigenvalues)
    gamma = torch.zeros((N, n, n), dtype=torch.complex64, device=R.device)  # Initialize with zeros
    gamma[:, torch.arange(n), torch.arange(n)] = torch.sqrt(eigvals)  # Assign square roots of eigenvalues

    # Compute batch-wise inverses
    inv_YT = torch.linalg.inv(Y_T)  # (N, n, n)
    inv_eigvecs = torch.linalg.inv(eigvecs)  # (N, n, n)

    # Compute characteristic impedance Zc
    Zc = inv_YT @ eigvecs @ gamma @ inv_eigvecs  # Batch matrix multiplication (N, n, n)

    # Compute characteristic admittance Yc
    Yc = torch.linalg.inv(Zc)  # (N, n, n)
    return eigvecs, inv_eigvecs, gamma, Zc, Yc

def calculate_cable_parameters(r_w, omega, n):
    """
    Compute R, L, C, G tensors for multiple frequencies.

    Parameters:
    - r_w: radius of MTL conductor (scalar)
    - omega: tensor of angular frequencies (num_freq,)
    - n: number of conductors - 1

    Returns:
    - R, L, C, G tensors (shape: (num_freq, n, n))
    """
    f = omega / (2 * torch.pi)  # Convert omega to frequency (Hz)
    num_freqs = len(f)
    # Constants (these are Python floats, not tensors)
    mu_0 = 4 * np.pi * 1e-7
    sigma = 5.8 * 1e7
    epsilon = 3.19 * 1e-11
    dc = 4 * 1e-4 + 3.02 * r_w
    dc2 = math.sqrt(2.0) * dc  
    tandelta = 1e-6
    delta = 1 / torch.sqrt(torch.pi * mu_0 * sigma * f)  # Skin depth (tensor of shape (num_freqs,))
    r = torch.where(
        r_w <= 2 * delta,
        1 / (sigma * torch.pi * r_w**2),  # Case where r_w <= 2*delta
        (1 / (2 * r_w)) * torch.sqrt((mu_0 * f) / (torch.pi * sigma))  # Case where r_w > 2*delta
    )
    R = torch.stack([
        torch.stack([2 * r, r, r], dim=-1),
        torch.stack([r, 2 * r, r], dim=-1),
        torch.stack([r, r, 2 * r], dim=-1)
    ], dim=-2).to(torch.complex64)
    # L matrix - dc, dc2, r_w are all Python floats, so np.log works fine
    L = (mu_0 / (2 * np.pi)) * torch.tensor([
        [2*np.log(dc / r_w), np.log((dc * dc2) / (dc * r_w)), np.log((dc * dc) / (dc2 * r_w))],
        [np.log((dc * dc2) / (dc * r_w)), 2*np.log(dc2 / r_w), np.log((dc2 * dc) / (dc * r_w))],
        [np.log( (dc * dc) / (dc2 * r_w)), np.log((dc2 * dc) / (dc * r_w)), 2*np.log(dc / r_w)]
    ], dtype=torch.complex64, device=device)

    L_new = L.unsqueeze(0).expand(num_freqs, -1, -1)
    C = mu_0 * epsilon * torch.linalg.inv(L)
    C_new = C.unsqueeze(0).expand(num_freqs, -1, -1)
    G_new = torch.zeros((num_freqs, n, n), dtype=torch.complex64, device=device)
    return R, L_new, C_new, G_new

# Function to compute constant impedance admittance matrix (type 1)
def constant_impedance(R_const, C_leak, omega):
    Z12 = Z13 = Z23 = R_const * torch.ones_like(omega, dtype=torch.complex64)
    ZG1 = ZG2 = ZG3 = 1 / (1j * omega * C_leak)
    return Z12, Z13, Z23, ZG1, ZG2, ZG3

# Function to compute double RLC admittance matrix (type 2)
def double_RLC(R_s, omega_0s, zeta_s, R_p, omega_0p, zeta_p, delta_1, delta_2, C_d_leak, omega):
    Z12 = (2j * (omega / omega_0p) * R_p * zeta_p) / (1 + 2j * (omega / omega_0p) * R_p * zeta_p - (omega**2 / omega_0p**2))
    Z12 += R_s + 2j * R_s * zeta_s * ((omega / omega_0s) - (omega_0s / omega))
    Z13 = Z12 * (1 + delta_1)
    Z23 = Z12 * (1 + delta_2)
    ZG1 = ZG2 = ZG3 = 1 / (1j * omega * C_d_leak)
    return Z12, Z13, Z23, ZG1, ZG2, ZG3

# Function to compute motor load admittance matrix (type 3)
def motor_load(C_m, L_m, R_m1, R_m2, C_m_leak, omega):
    Zprime = 1j * omega * L_m + R_m2
    Z12 = (1/3) * 1/(1j * omega * (C_m + (C_m_leak/2)) + 1 / R_m1 + 1 / Zprime)
    Z13 = Z23 = Z12
    ZG1 = (1/3) * 1/(1j * omega * C_m_leak + 1 / (1j * omega * C_m_leak + (R_m1 * Zprime)/(1j * omega * C_m * Zprime * R_m1 + Zprime + R_m1)))
    ZG2 = ZG3 = ZG1
    return Z12, Z13, Z23, ZG1, ZG2, ZG3

def compute_load_admittance_3d(load_params):
    """
    Compute 3x3 load admittance matrix from (Z12, Z13, Z23, ZG1, ZG2, ZG3)

    Parameters:
    - load_params (tuple of tensors): (Z12, Z13, Z23, ZG1, ZG2, ZG3)
      Each impedance tensor has shape (N,) with dtype torch.complex64.
    Returns:
    - Y_load (torch.Tensor): Load admittance matrix with shape (N, 3, 3)
    """
    Z12, Z13, Z23, ZG1, ZG2, ZG3 = load_params  # Unpack impedance values

    # Stack into a (N, 3, 3) tensor
    Y_load = torch.stack([
        torch.stack([1/ZG1 + 1/Z12 + 1/Z13, -1/Z12, -1/Z13], dim=-1),
        torch.stack([-1/Z12, 1/ZG2 + 1/Z12 + 1/Z23, -1/Z23], dim=-1),
        torch.stack([-1/Z13, -1/Z23, 1/ZG3 + 1/Z13 + 1/Z23], dim=-1)
    ], dim=-2)  # Shape: (N, 3, 3)
    return Y_load

def matrix_cosh(M):
    """
    Computes the matrix hyperbolic cosine using expm: cosh(M) = 0.5 * (exp(M) + exp(-M))
    M: (N, n, n) tensor
    Returns: (N, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) + torch.matrix_exp(-M))


def matrix_sinh(M):
    """
    Computes the matrix hyperbolic sine using expm: sinh(M) = 0.5 * (exp(M) - exp(-M))
    M: (N, n, n) tensor
    Returns: (N, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) - torch.matrix_exp(-M))

def matrix_cosh_numpy(M: np.ndarray) -> np.ndarray:
    """
    Matrix cosh for a stack of matrices.
    M: [N, n, n] complex
    Returns: [N, n, n] complex
    """
    M = np.asarray(M)
    out = np.empty_like(M, dtype=np.complex128)
    for i in range(M.shape[0]):
        out[i] = 0.5 * (expm(M[i]) + expm(-M[i]))
    return out

def matrix_sinh_numpy(M: np.ndarray) -> np.ndarray:
    """
    Matrix sinh for a stack of matrices.
    M: [N, n, n] complex
    Returns: [N, n, n] complex
    """
    M = np.asarray(M)
    out = np.empty_like(M, dtype=np.complex128)
    for i in range(M.shape[0]):
        out[i] = 0.5 * (expm(M[i]) - expm(-M[i]))
    return out

def reflection_coefficient(YL, T, T_inv, ZC, YC):
    """
    Implements eq. (12) of Tonello paper.
    Inputs: all (N, n, n) tensors
    Returns: rho (N, n, n)
    """
    inv_sum = torch.linalg.inv(YL + YC)
    rho = T_inv @ YC @ inv_sum @ (YL - YC) @ ZC @ T
    return rho

def carry_back_load(rhoL, T, YC, Gamma, length):
    """
    Implements eq. (13) of Tonello paper.
    Inputs: all (N, n, n) tensors, length is a scalar
    Returns: Y_R(x) (N, n, n)
    """
    # Compute matrix exponentials
    e_pos = torch.matrix_exp(Gamma * length)       # (N, n, n)
    e_neg = torch.matrix_exp(-Gamma * length)      # (N, n, n)

    # Compute numerator and denominator
    num = e_pos + torch.matmul(e_neg, rhoL)        # (N, n, n)
    den = e_pos - torch.matmul(e_neg, rhoL)        # (N, n, n)
    deninv = torch.linalg.inv(den)                 # (N, n, n)

    # Compute final YR
    YR = T @ num @ deninv @ torch.linalg.inv(T) @ YC
    return YR

def h_B(rhoL, ZC, T, T_inv, Gamma, length):
    """
    Implements eq. (14) of Tonello paper.
    Inputs: all (N, n, n) tensors, length is scalar
    Returns: h_B (N, n, n)
    """
    N, n, _ = ZC.shape
    device = ZC.device

    # Identity matrix expanded for batch
    U = torch.eye(n, dtype=torch.complex64, device=device).unsqueeze(0).expand(N, -1, -1)

    # Matrix exponentials
    e_pos = torch.matrix_exp(Gamma * length)
    e_neg = torch.matrix_exp(-Gamma * length)

    den = e_pos - e_neg @ rhoL
    deninv = torch.linalg.inv(den)

    hB = ZC @ T @ (U - rhoL) @ deninv @ T_inv @ torch.linalg.inv(ZC)
    return hB

def denormalize(norm_value, min_val, max_val):
    """ Convert normalized value (0 to 1) back to its original range. """
    return norm_value * (max_val - min_val) + min_val

def normalize(physical_value, min_val, max_val):
    """ Convert physical value to normalized [0, 1] range. Inverse of denormalize. """
    return (physical_value - min_val) / (max_val - min_val)

def loguniform(low, high, size=None):
    """ Generate samples from a log-uniform distribution. """
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))


# ---- Define Network Parameter Dictionary ----
BACKBONE_KEYS = ["l_w_0", "l_w_1", "l_w_4", "l_w_25", "l_w_28"]
network_params = {
    "cable_lengths": {  # 30 parameters, set all to 0.25
        f"l_w_{i}": {"value": denormalize(0.25, 2, 20), "inferred": True, "range": (2, 20), "infer_range": (6.0, 8.0)}
        for i in range(30)
    },
    "conductor_radii": {  # Fixed values, not inferred
        "r_w_servicepanel": {"value": denormalize(0.25, 1.03e-3, 2.06e-3), "inferred": False},
        "r_w_room": {"value": denormalize(0.25, 0.81e-3, 1.29e-3), "inferred": False}
    },
    "fault_parameters": {
        # Normalized position [0, 1], will be scaled to [0, L] in forward model
        "fault_position": {"value": 0.25, "inferred": True, "range": (0.0, 1.0)},
        # Complex fault impedance Z_fault = Z_fault_real + j*Z_fault_imag
        "Z_fault_real": {"value": 100.0, "inferred": True, "range": (0.0, 4000.0)},
        "Z_fault_imag": {"value": -50.0, "inferred": True, "range": (-100.0, 100.0)}
    },
    "admittance_parameters": {}, #Dynamically generate Y_eq GLC parameters based on load type
    "loads": {}  # Dynamically generated based on load type
}
def get_total_backbone_length(cable_lengths):
    """Calculate total backbone length L from cable_lengths dict of tensors."""
    total = torch.tensor(0.0, device=device)
    for key in BACKBONE_KEYS:
        val = cable_lengths[key]
        total = total + val  # Keep as tensor for gradient flow
    return total

def initialize_random_starts():
    """Generate random initial values for non-inferred params once."""
    for load_name, params in network_params["loads"].items():
        for param_name, param_info in params.items():
            if not param_info["inferred"] and "range" in param_info:
                param_info["random_init"] = torch.rand(1).item()  # 0-1
        for cable_name, cable_info in network_params["cable_lengths"].items():
            if not cable_info["inferred"]:
                cable_info["random_init"] = torch.rand(1).item()


# =============================================================================
# SIMPLIFIED MTL TRANSFER FUNCTION: Tx - Cable(L1) - Fault - Cable(L-L1) - Rx
# =============================================================================
def compute_H_simple_mtl_rho(L1, L_total, Y_fault, Z_load, R, L_mat, C, G, omega_vec, n=3):
    T, Tinv, gamma, ZC, YC = get_mtl_matrices(R, L_mat, C, G, n, omega) #MTL parameters of wires in service panel
    
    L2 = L_total - L1
    #node0 (Yrec)
    Y_rec = torch.linalg.inv(Z_load)
    rho1 = reflection_coefficient(Y_rec, T, Tinv, ZC, YC)
    h1 = h_B(rho1, ZC, T, Tinv, gamma, L2)
    Y_reccarried = carry_back_load(rho1, T, YC, gamma, L2)
    Y_node1 = Y_reccarried + Y_fault
    rho2 = reflection_coefficient(Y_node1, T, Tinv, ZC, YC)
    h2 = h_B(rho2, ZC, T, Tinv, gamma, L1)
    Y_reccarried2 = carry_back_load(rho2, T, YC, gamma, L1)
    hoverall = h1 @ h2
    return hoverall

def compute_H_simple_mtl(L1, L_total, Y_fault, Z_load, R, L_mat, C, G, omega_vec, n=3):
    """
    Compute simplified MTL transfer function: Tx - Cable(L1) - Fault - Cable(L-L1) - Load

    This mirrors the simple model structure but uses MTL matrices instead of scalars.

    Args:
        L1: Fault location (distance from Tx), scalar or tensor
        L_total: Total cable length, scalar
        Z_fault: Fault impedance matrix (N, n, n) or scalar (will be converted to diagonal)
        Z_load: Load impedance matrix (N, n, n)
        R, L_mat, C, G: PUL parameters (N, n, n) tensors
        omega_vec: Angular frequencies (N,) tensor
        n: Number of conductors - 1 (default 3)

    Returns:
        H: Transfer function matrix (N, n, n) complex
    """
    N = len(omega_vec)
    device = R.device
    L2 = L_total - L1

    # Reshape omega for broadcasting
    omega = omega_vec.view(-1, 1, 1)

    phinw_L2 = calculate_cable_transmission_matrix(R, L_mat, C, G, L2, omega)
    # Identity and zero blocks
    I = np.eye(n, dtype=complex)
    I = np.repeat(I[None, :, :], N, axis=0)   # (N, n, n)

    # Convert Y_fault to numpy if it's a torch tensor
    if torch.is_tensor(Y_fault):
        Y_fault_np = Y_fault.detach().cpu().numpy()
    else:
        Y_fault_np = Y_fault

    # Pre-allocate the full (N, 2n, 2n) matrix
    Phi_fault = np.zeros((N, 2*n, 2*n), dtype=complex)

    # Assign blocks
    Phi_fault[:, :n, :n] = I            # Top-left: A = I
    Phi_fault[:, :n, n:] = 0            # Top-right: B = 0  (already zeros)
    Phi_fault[:, n:, :n] = -Y_fault_np   # Bottom-left: C = -Y_fault
    Phi_fault[:, n:, n:] = I            # Bottom-right: D = I
    
    phinw_L1 = calculate_cable_transmission_matrix(R, L_mat, C, G, L1, omega)
    Phi_total = phinw_L2 @ Phi_fault @ phinw_L1  # Order: Tx → L1 → fault → L2 → Load
    H = calculate_H_nw(Phi_total, n, Z_load)
    return H


def plot_nll_vs_L1_simple_mtl(snr_db=40, L1_true=250.0, L_total=1000.0,
                               save_path="nll_vs_L1_simple_mtl_150-30_100m.pdf"):
    """
    Plot NLL vs L1 for the simplified MTL model.

    Uses the same Tx-cable-fault-cable-Rx structure as the simple model but with
    MTL matrices (3x3) instead of scalars.
    """
    # Use global omega and frequency range
    global omega, frequencies, freq_range_mhz
    N = len(omega)
    n = 3  # 3 conductors (4 wires - 1)
    device = omega.device

    # Cable parameters (using room wire radius)
    r_w = 0.81e-3  # Wire radius for room cables
    R, L_mat, C, G = calculate_cable_parameters(r_w, omega, n)

    # Load impedance matrix (100-5.0j on all diagonals)
    Z_load = (100.0 - 5.0j) * torch.eye(n, dtype=torch.complex64, device=device).unsqueeze(0).expand(N, -1, -1)

    # Fault impedance (scalar, will be diagonalized)
    Zf = torch.tensor(100.0 - 50.0j, dtype=torch.complex64, device=device)

    Y_fault = torch.zeros((n, n), dtype=torch.complex64, device=device)
    Y_fault[0, 0] = 1.0 / Zf   # fault on 0th conductor to reference
    Y_fault = Y_fault.unsqueeze(0).expand(N, -1, -1)
    
    H_true1 = compute_H_simple_mtl_rho(
        L1=L1_true, L_total=L_total, Y_fault=Y_fault,
        Z_load=Z_load, R=R, L_mat=L_mat, C=C, G=G, omega_vec=omega, n=n
    )

    # eps = 1e-12
    # H1_db = 20 * torch.log10(torch.abs(H_true1) + eps)
    # H2_db = 20 * torch.log10(torch.abs(H_true2) + eps)

    # # --- Extract (0,0) element ---
    # H1_00 = H1_db[:, 0, 0]
    # H2_00 = H2_db[:, 0, 0]

    # # --- Move to CPU ---
    # H1_00_np = H1_00.detach().cpu().numpy()
    # H2_00_np = H2_00.detach().cpu().numpy()

    # freq_np = freq_range_mhz  # already numpy

    # # --- Plot ---
    # plt.figure(figsize=(8, 5))

    # plt.plot(freq_np, H1_00_np, label="H_true1 (rho model)", linewidth=2)
    # plt.plot(freq_np, H2_00_np, '--', label="H_true2 (standard)", linewidth=2)

    # plt.xlabel("Frequency (MHz)")
    # plt.ylabel("|H₀₀| (dB)")
    # plt.title("MTL Transfer Function Comparison (Mode 0→0)")
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()
    
    # Use H[0,0] element (first mode to first mode transfer)
    H_true_00 = H_true1[:, 0, 0]  # (N,) complex

    # Add noise
    snr_lin = 10.0 ** (snr_db / 10.0)
    sig_pow = torch.mean(torch.abs(H_true_00) ** 2)
    var_f = sig_pow / snr_lin
    std_f = torch.sqrt(var_f / 2)
    noise = std_f * (torch.randn_like(H_true_00.real) + 1j * torch.randn_like(H_true_00.imag))
    H_obs = H_true_00 + noise

    # Sweep L1 and compute NLL
    L1_grid = torch.linspace(0.01, 0.99, 199)
    nll_values = []

    print(f"Computing NLL vs L1 for simplified MTL model...")
    print(f"  L_total = {L_total}m, L1_true = {L1_true}m, SNR = {snr_db}dB")

    with torch.no_grad():
        for L1 in L1_grid:
            L1_real = denormalize(L1, 0.0, L_total)
            #print("L1_real", L1_real)
            H_pred = compute_H_simple_mtl_rho(
                L1=L1_real.item(), L_total=L_total, Y_fault=Y_fault,
                Z_load=Z_load, R=R, L_mat=L_mat, C=C, G=G, omega_vec=omega, n=n
            )
            H_pred_00 = H_pred[:, 0, 0]

            # NLL = sum of |H_obs - H_pred|^2 / var
            residual = H_obs - H_pred_00
            nll = torch.sum(torch.abs(residual) ** 2 / var_f).item()
            nll_values.append(nll)
    
    # Find minimum
    plt.figure(figsize=(10, 6))
    plt.plot(L1_grid.cpu().numpy(), nll_values, 'b-', linewidth=2)
    plt.axvline(x=0.25, color='r', linestyle='--', linewidth=2, label='True fault_position=0.25')
    plt.xlabel('L1 (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Fault Location L1 (Simple Model with Complex Model Cable Params' \
    '+ MTL)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")
    


 
def generate_load_parameters_deterministic(num_loads):
    """
    Generate FIXED load parameters matching the paper (no randomness).
    """
    assert num_loads == len(FIXED_LOAD_TYPES), \
        f"Expected {len(FIXED_LOAD_TYPES)} loads, got {num_loads}"

    network_params["loads"].clear()
    total_parameters = 0
    load_types = []

    for i, load_type in enumerate(FIXED_LOAD_TYPES):
        load_types.append(load_type)

        if load_type == 1:  # Constant Impedance (2)
            network_params["loads"][f"load_{i}"] = {
                "R_const": {"value": denormalize(0.25, 10, 200), "inferred": True, "range": (10, 200)},
                "C_leak": {"value": denormalize(0.25, 0.1e-9, 2.0e-9), "inferred": True, "range": (0.1e-9, 2.0e-9)}
            }
            total_parameters += 2

        elif load_type == 2:  # Double RLC (9)
            network_params["loads"][f"load_{i}"] = {
                "R_s": {"value": denormalize(0.25, 10, 3000), "inferred": True, "range": (10, 3000)},
                "omega_0s": {"value": denormalize(0.25, 0.1e6, 30e6), "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_s": {"value": denormalize(0.25, 0.1, 2), "inferred": True, "range": (0.1, 2)},
                "R_p": {"value": denormalize(0.25, 10, 3000), "inferred": True, "range": (10, 3000)},
                "omega_0p": {"value": denormalize(0.25, 0.1e6, 30e6), "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_p": {"value": denormalize(0.25, 0.1, 2), "inferred": True, "range": (0.1, 2)},
                "delta_1": {"value": denormalize(0.25, -0.1, 0.1), "inferred": True, "range": (-0.1, 0.1)},
                "delta_2": {"value": denormalize(0.25, -0.1, 0.1), "inferred": True, "range": (-0.1, 0.1)},
                "C_d_leak": {"value": denormalize(0.25, 0.1e-9, 2e-9), "inferred": True, "range": (0.1e-9, 2e-9)}
            }
            total_parameters += 9

        elif load_type == 3:  # Motor (5)
            network_params["loads"][f"load_{i}"] = {
                "C_m": {"value": denormalize(0.25, 0.1e-9, 1e-9), "inferred": True, "range": (0.1e-9, 1e-9)},
                "L_m": {"value": denormalize(0.25, 5e-3, 20e-3), "inferred": True, "range": (5e-3, 20e-3)},
                "R_m1": {"value": denormalize(0.25, 2000, 15000), "inferred": True, "range": (2000, 15000)},
                "R_m2": {"value": 5.0, "inferred": False},
                "C_m_leak": {"value": denormalize(0.25, 0.2e-9, 5e-9), "inferred": True, "range": (0.2e-9, 5e-9)}
            }
            total_parameters += 5

    return total_parameters, load_types

# ---- Generate Load Parameters Based on Load Type ----
def generate_load_parameters(num_loads, omega):
    """ Generate parameters for loads (constant impedance, double RLC, or motor). """
    load_types = []  # Store the order of load types
    total_parameters = 0  # Track the total number of parameters

    for i in range(num_loads):
        load_type = np.random.choice([1, 2, 3])  # Randomly pick constant impedance, RLC, or motor
        load_types.append(load_type)

        if load_type == 1:  # Constant Impedance Load (2 params)
            network_params["loads"][f"load_{i}"] = {
                "R_const": {"value": denormalize(0.25, 10, 200), "inferred": True, "range": (10, 200)},
                "C_leak": {"value": denormalize(0.25, 0.1e-9, 2.0e-9), "inferred": True, "range": (0.1e-9, 2.0e-9)}
            }
            total_parameters += 2 

        elif load_type == 2:  # Double RLC Load (9 params)
            network_params["loads"][f"load_{i}"] = {
                "R_s": {"value": denormalize(0.25, 10, 3000), "inferred": True, "range": (10, 3000)},
                "omega_0s": {"value": denormalize(0.25, 0.1, 30e6), "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_s": {"value": denormalize(0.25, 0.1, 2), "inferred": True, "range": (0.1, 2)},
                "R_p": {"value": denormalize(0.25, 10, 3000), "inferred": True, "range": (10, 3000)},
                "omega_0p": {"value": denormalize(0.25, 0.1, 30e6), "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_p": {"value": denormalize(0.25, 0.1, 2), "inferred": True, "range": (0.1, 2)},
                "delta_1": {"value": denormalize(0.25, -0.1, 0.1), "inferred": True, "range": (-0.1, 0.1)},
                "delta_2": {"value": denormalize(0.25, -0.1, 0.1), "inferred": True, "range": (-0.1, 0.1)},
                "C_d_leak": {"value": denormalize(0.25, 0.1e-9, 2e-9), "inferred": True, "range": (0.1e-9, 2e-9)}
            }
            total_parameters += 9 

        elif load_type == 3:  # Motor Load (5 params)
            network_params["loads"][f"load_{i}"] = {
                "C_m": {"value": denormalize(0.25, 0.1e-9, 1e-9), "inferred": True, "range": (0.1e-9, 1e-9)},
                "L_m": {"value": denormalize(0.25, 5e-3, 20e-3), "inferred": True, "range": (5e-3, 20e-3)},
                "R_m1": {"value": denormalize(0.25, 2000, 15000), "inferred": True, "range": (2000, 15000)},
                "R_m2": {"value": 5, "inferred": False},  # Fixed, not inferred
                "C_m_leak": {"value": denormalize(0.25, 0.2e-9, 5e-9), "inferred": True, "range": (0.2e-9, 5e-9)}
            }
            total_parameters += 5 
    return total_parameters, load_types  # Return total parameters and load type distribution

# Convert network parameters into PyTorch tensors
def convert_to_tensors(network_params):
    tensor_params = {}
    for load, params in network_params["loads"].items():
        tensor_params[load] = {
            key: torch.tensor(val["value"], dtype=torch.complex64, device=device) for key, val in params.items()
        }
    return tensor_params

def calculate_Htrans(YTalpha, YTbeta, YTgamma, Ynw, ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32):
    """
    Compute Htrans from Ynw and transmitter impedance values

    Parameters:
    - YTalpha, YTbeta, YTgamma: Transmitter constants (scalar)
    - Ynw: Network input admittance matrix (N, n, n)
    - ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32: Transmitter constants (scalar)

    Returns:
    - Htrans: Transfer function of transmitter (N, n, n)
    """
    N = Ynw.shape[0]
    # Extract individual elements
    Ynw11 = Ynw[:, 0, 0].unsqueeze(1).unsqueeze(2)
    Ynw12 = Ynw[:, 0, 1].unsqueeze(1).unsqueeze(2)
    Ynw13 = Ynw[:, 0, 2].unsqueeze(1).unsqueeze(2)
    Ynw21 = Ynw[:, 1, 0].unsqueeze(1).unsqueeze(2)
    Ynw22 = Ynw[:, 1, 1].unsqueeze(1).unsqueeze(2)
    Ynw23 = Ynw[:, 1, 2].unsqueeze(1).unsqueeze(2)
    Ynw31 = Ynw[:, 2, 0].unsqueeze(1).unsqueeze(2)
    Ynw32 = Ynw[:, 2, 1].unsqueeze(1).unsqueeze(2)
    Ynw33 = Ynw[:, 2, 2].unsqueeze(1).unsqueeze(2)

    H11 = 1 + ZT0 * Ynw11 + ZT0 * YTalpha
    H12 = ZT0 * Ynw12 - ZT0 / ZT12
    H13 = ZT0 * Ynw13 - ZT0 / ZT13
    H21 = ZT0 * Ynw21 - ZT0 / ZT21
    H22 = 1 + ZT0 * Ynw22 + ZT0 * YTbeta
    H23 = ZT0 * Ynw23 - ZT0 / ZT23
    H31 = ZT0 * Ynw31 - ZT0 / ZT31
    H32 = ZT0 * Ynw32 - ZT0 / ZT32
    H33 = 1 + ZT0 * Ynw33 + ZT0 * YTgamma

    # Stack rows, then the full batch
    H_trans = torch.cat([
        torch.cat([H11, H12, H13], dim=2),
        torch.cat([H21, H22, H23], dim=2),
        torch.cat([H31, H32, H33], dim=2)
    ], dim=1)  # Resulting shape: (N, 3, 3)
    H_trans_inv = torch.linalg.inv(H_trans)  # Shape: (N, 3, 3)

    return H_trans_inv

def compute_fault_admittance_matrix(Z_fault_real, Z_fault_imag, N, n, k=0):
    """
    Compute shunt fault admittance matrix with conductor k to ground.

    Parameters:
    - Z_fault_real: Real part of fault impedance (scalar, Ohms)
    - Z_fault_imag: Imaginary part of fault impedance (scalar, Ohms)
    - N: number of frequency points
    - n: number of conductors - 1
    - k: which conductor has fault to ground (default 0)

    Returns:
    - Y_fault: (N, n, n) diagonal fault admittance matrix
    """
    # Construct complex impedance: Z = R + jX
    #print("Z_fault real dtype", type(Z_fault_real))
    #print("Z_fault real dtype", type(Z_fault_imag))
    # #] Ensure both have the same dtype for torch.complex()
    Z_fault_real = Z_fault_real.to(torch.float32) if isinstance(Z_fault_real, torch.Tensor) else torch.tensor(Z_fault_real, dtype=torch.float32, device=device)
    Z_fault_imag = Z_fault_imag.to(torch.float32) if isinstance(Z_fault_imag, torch.Tensor) else torch.tensor(Z_fault_imag, dtype=torch.float32, device=device)
    Z_fault = torch.complex(Z_fault_real, Z_fault_imag)
    Y_f = 1.0 / Z_fault  # Complex admittance
    Y_fault = torch.zeros(N, n, n, dtype=torch.complex64, device=device)
    Y_fault[:, k, k] = Y_f
    return Y_fault


def get_fault_segment_and_local_position(fault_position, cable_lengths):
    """
    Given normalized fault position [0,1], determine which backbone segment
    and local position within that segment.
    
    Parameters:
    - fault_position: tensor in [0, 1], 0 = Rx end, 1 = Tx end
    - cable_lengths: dict of cable length tensors (inferred)
    Returns:
    - segment_idx: which backbone segment (0-4)
    - local_position: position within that segment in meters (tensor)
    - segment_length: total length of that segment (tensor)
    """
    L = get_total_backbone_length(cable_lengths)
    fault_position_abs = fault_position * L  # Convert to meters
    cumulative = torch.tensor(0.0, device=device)
    for idx, key in enumerate(BACKBONE_KEYS):
        seg_len = cable_lengths[key] 
        if (cumulative + seg_len >= fault_position_abs).item() or idx == len(BACKBONE_KEYS) - 1:
            local_pos = fault_position_abs - cumulative
            return idx, local_pos, seg_len
        cumulative += seg_len
    # Fallback
    last_len = cable_lengths[BACKBONE_KEYS[-1]]
    return len(BACKBONE_KEYS) - 1, fault_position_abs - (L - last_len), last_len

def carry_back_with_fault(Y_load, T, Tinv, ZC, YC, gamma, cable_length,
                          local_fault_pos, Z_fault_real, Z_fault_imag):
    """
    Carry back admittance through a cable that has a shunt fault.

    The cable is split into two segments:
      [Y_load]---(len_1)---[FAULT node]---(len_2)---[output Y_carried]

    At the fault node, Y_fault is added in parallel (shunt to ground).

    Parameters:
    - Y_load: (N, n, n) admittance at the load end (Rx side)
    - T, Tinv, ZC, YC, gamma: MTL parameters
    - cable_length: total length of this segment
    - local_fault_pos: distance from load end to fault (meters)
    - Z_fault_real: real part of fault impedance (Ohms)
    - Z_fault_imag: imaginary part of fault impedance (Ohms)

    Returns:
    - Y_carried: (N, n, n) admittance seen from source end (Tx side)
    - h_total: (N, n, n) transfer function h_B through the faulted cable
    """
    n = Y_load.shape[1]
    N = Y_load.shape[0]

    # Segment lengths
    len_1 = local_fault_pos                    # Load to fault
    len_2 = cable_length - local_fault_pos     # Fault to source

    #Carry Yload through len1 to fault node
    Y_fault = compute_fault_admittance_matrix(Z_fault_real, Z_fault_imag, N, n)
    rho_1 = reflection_coefficient(Y_load, T, Tinv, ZC, YC)
    h_1 = h_B(rho_1, ZC, T, Tinv, gamma, len_1)
    Y_at_fault = carry_back_load(rho_1, T, YC, gamma, len_1)

    Y_after_fault = Y_at_fault + Y_fault

    #Carry combined admittance through len2 to source
    rho_2 = reflection_coefficient(Y_after_fault, T, Tinv, ZC, YC)
    h_2 = h_B(rho_2, ZC, T, Tinv, gamma, len_2)
    Y_carried = carry_back_load(rho_2, T, YC, gamma, len_2)

    h_total = h_1 @ h_2

    return Y_carried, h_total
def fit_CL_from_imaginary(Y_imag):
    """
    Y_imag: (N,) tensor of imaginary parts of Y at each frequency
    Fit C, L scalars with least squares from Im{Y(ω)} = ωC - 1/(ωL)
    """
    if torch.is_tensor(Y_imag):
        Y_imag = Y_imag.numpy()
    if torch.is_tensor(omega):
        omega_np = omega.numpy()
    A = np.column_stack([omega_np, -1/omega_np]) #[N, 2]
    b = Y_imag #[N, 1]
    #Ax = b -> [N, 2] x [2, 1] = [N, 1] -> x = [C, 1/L]^T
    # Least squares: A @ [C, 1/L]^T = b
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    C = result[0]       # First coefficient is C
    inv_L = result[1]   # Second coefficient is 1/L
    L = 1 / inv_L if inv_L != 0 else np.inf

    return C, L

    
def fit_GCL_params(Y_eq):
    """
    Fit Y = G + jωC + 1/(jωL) to the computed Y_eq.
    Y has structure: diagonal Y_d, off-diagonal Y_o (all same).
    Returns Gd, Cd, Ld, Go, Co, Lo for diagonal and off-diagonal elements.
    """
    Y_d = Y_eq[:, 0, 0] #(N, ) complex
    Y_o = Y_eq[:, 0, 1] #(N, ) complex
    #Fit diagonal
    G_d = Y_d.real.mean().item()
    C_d, L_d = fit_CL_from_imaginary(Y_d.imag)
    #Fit off diagonal
    G_o = Y_o.real.mean().item()
    C_o, L_o = fit_CL_from_imaginary(Y_o.imag)

    return {
        "G_d": G_d, "C_d": C_d, "L_d": L_d,
        "G_o": G_o, "C_o": C_o, "L_o": L_o,
    }
def fit_all_equivalent_admittances(Y_eqs):
    """
    Fit GCL parameters for all 5 branches.
    
    Args:
        Y_eqs: dict {"Y_1": tensor, "Y_2": tensor, ..., "Y_5": tensor}
               or list [Y_1, Y_2, Y_3, Y_4, Y_5]
    
    Returns:
        dict of dicts: {"Y_1": {G_d, C_d, ...}, "Y_2": {...}, ...}
    """
    if isinstance(Y_eqs, list):
        Y_eqs = {f"Y_{i+1}": Y for i, Y in enumerate(Y_eqs)}
    
    fitted_params = {}
    for name, Y_eq in Y_eqs.items():
        fitted_params[name] = fit_GCL_params(Y_eq)
    
    return fitted_params
def generate_admittance_parameters(fitted_params):
    """
    Generate admittance parameter values to be equal to fitted_params
    and ranges such that true value is always at normalized = 0.25.
    """
    for branch_name, params in fitted_params.items():
        network_params["admittance_parameters"][branch_name] = {}
        for param_name, value in params.items():
            if value > 0:
                lo = 0.5 * value
                hi = 2.5 * value
            elif value < 0:
                lo = 1.5 * value  # More negative
                hi = -0.5 * value  # Positive (crosses zero)
            else:
                lo, hi = -0.1, 0.1  # Handle zero case
            
            network_params["admittance_parameters"][branch_name][param_name] = {
                "value": value,
                "inferred": True,
                "range": (lo, hi)
            }
def reconstruct_Y(G_d, C_d, L_d, G_o, C_o, L_o):
    """Reconstruct Y from fitted GCL parameters."""
    N = len(omega)
    Y = torch.zeros(N, 3, 3, dtype=torch.complex64)
    
    # Y = G + jωC + 1/(jωL) = G + jωC - j/(ωL) = G + j(ωC - 1/(ωL))
    Y_d = G_d + 1j * (omega * C_d - 1/(omega * L_d))
    Y_o = G_o + 1j * (omega * C_o - 1/(omega * L_o))
    
    # Fill matrix
    Y[:, 0, 0] = Y[:, 1, 1] = Y[:, 2, 2] = Y_d
    Y[:, 0, 1] = Y[:, 1, 0] = Y_o
    Y[:, 0, 2] = Y[:, 2, 0] = Y_o
    Y[:, 1, 2] = Y[:, 2, 1] = Y_o
    
    return Y


def diagnose_per_branch_error(Y_true_list, fitted_params, save_path="per_branch_error.png"):
    """
    Compare true Y_eq vs reconstructed Y_eq for each branch.
    Identifies which branch contributes most error.

    Args:
        Y_true_list: list of 5 true Y_eq tensors [Y_1, ..., Y_5]
        fitted_params: dict from fit_all_equivalent_admittances()
        save_path: where to save the diagnostic plot
    """
    branch_names = [f"Y_{i+1}" for i in range(5)]

    print("\n" + "="*70)
    print("PER-BRANCH APPROXIMATION ERROR DIAGNOSIS")
    print("="*70)
    print(f"{'Branch':<10} | {'MAE (mag)':<12} | {'RMSE (mag)':<12} | {'Rel Error %':<12}")
    print("-"*70)

    errors = {}
    freq_mhz = (frequencies / 1e6).cpu().numpy()

    fig, axes = plt.subplots(5, 2, figsize=(14, 20))

    for i, (Y_true, branch_name) in enumerate(zip(Y_true_list, branch_names)):
        params = fitted_params[branch_name]

        # Reconstruct Y from fitted params
        Y_recon = reconstruct_Y(
            params["G_d"], params["C_d"], params["L_d"],
            params["G_o"], params["C_o"], params["L_o"]
        )

        # Compute errors (diagonal element [0,0] as representative)
        Y_true_diag = Y_true[:, 0, 0]
        Y_recon_diag = Y_recon[:, 0, 0]

        # Magnitude error
        mag_true = torch.abs(Y_true_diag).cpu().numpy()
        mag_recon = torch.abs(Y_recon_diag).cpu().numpy()
        mag_error = np.abs(mag_recon - mag_true)

        mae = np.mean(mag_error)
        rmse = np.sqrt(np.mean(mag_error**2))
        rel_error = 100 * np.mean(mag_error / mag_true)

        errors[branch_name] = {"mae": mae, "rmse": rmse, "rel_error": rel_error}
        print(f"{branch_name:<10} | {mae:<12.6f} | {rmse:<12.6f} | {rel_error:<12.1f}%")

        re_true = Y_true_diag.real.cpu().numpy()
        re_recon = Y_recon_diag.real.cpu().numpy()
        # Plot magnitude comparison
        ax1 = axes[i, 0]
        ax1.plot(freq_mhz, mag_true, 'b-', linewidth=2, label='True')
        ax1.plot(freq_mhz, mag_recon, 'r--', linewidth=2, label='Reconstructed')
        ax1.set_ylabel('|Y| (S)')
        ax1.set_title(f'{branch_name} Magnitude (Rel Error: {rel_error:.1f}%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot imaginary part comparison (shows capacitive/inductive behavior)
        ax2 = axes[i, 1]
        im_true = Y_true_diag.imag.cpu().numpy()
        im_recon = Y_recon_diag.imag.cpu().numpy()
        ax2.plot(freq_mhz, im_true, 'b-', linewidth=2, label='True Im{Y}')
        ax2.plot(freq_mhz, im_recon, 'r--', linewidth=2, label='Reconstructed Im{Y}')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Im{Y} (S)')
        ax2.set_title(f'{branch_name} Imaginary Part')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if i == 4:  # Last row
            ax1.set_xlabel('Frequency (MHz)')
            ax2.set_xlabel('Frequency (MHz)')

    print("-"*70)

    # Find worst branch
    worst_branch = max(errors, key=lambda x: errors[x]["rel_error"])
    print(f"\nWORST BRANCH: {worst_branch} with {errors[worst_branch]['rel_error']:.1f}% relative error")
    print("="*70)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved diagnostic plot to {save_path}")

    return errors


def calculate_Hnw_fault_admittance_version(cable_lengths, admittance_params, fault_params):
    """
    Calculate network Transfer Function Hnw with fault given admittance + cable parameters + fault parameters

    Parameters:
    - cable_lengths (dict): Dictionary of cable lengths (30 length)
    - admittance_params (dict): Dictionary of sampled admittance parameters (30 parameters)
    - fault_params (dict): Dictionary of fault parameters
    Returns:
    - (1,1)st entry of H_nw (N, n, n) where N = num of frequency points and n = num of conductors - 1
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    
    Y_branches = {}

    for branch_name, branch_params in admittance_params.items():
        G_d = branch_params["G_d"]
        C_d = branch_params["C_d"]
        L_d = branch_params["L_d"]
        G_o = branch_params["G_o"]
        C_o = branch_params["C_o"]
        L_o = branch_params["L_o"]
        Y_branches[branch_name] = reconstruct_Y(G_d, C_d, L_d, G_o, C_o, L_o)


    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room

    fault_location = fault_params["fault_position"]
    Z_fault_real = fault_params["Z_fault_real"]
    Z_fault_imag = fault_params["Z_fault_imag"]
    #From location + impedance find which cable has the fault 
    fault_seg_idx, local_fault_pos, _ = get_fault_segment_and_local_position(fault_location, cable_lengths)

    #node0 (Yrec)
    # ===== Backbone segment 0: l_w_0 (room wire) =====
    if fault_seg_idx == 0:
        Y_reccarried, h0 = carry_back_with_fault(
            Y_rec, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_0"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho0 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
        h0 = h_B(rho0, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{0}"])
        Y_reccarried = carry_back_load(rho0, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{0}"])

    #node1 (Y62)
    Y_node1 = Y_reccarried + Y_branches["Y_1"]
    # ===== Backbone segment 1: l_w_1 (room wire) =====
    if fault_seg_idx == 1:
        Y_node1carried, h1 = carry_back_with_fault(
            Y_node1, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_1"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho1 = reflection_coefficient(Y_node1, T_r, Tinv_r, ZC_r, YC_r)
        h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{1}"])
        Y_node1carried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{1}"])

    #node2 Junction box (Y61 || Y63)
    Y_node2 = Y_node1carried + Y_branches["Y_2"]
    # ===== Backbone segment 2: l_w_4 (service panel wire) =====
    if fault_seg_idx == 2:
        Y_node2carried, h2 = carry_back_with_fault(
            Y_node2, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths["l_w_4"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho2 = reflection_coefficient(Y_node2, T_s, Tinv_s, ZC_s, YC_s)
        h2 = h_B(rho2, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{4}"])
        Y_node2carried = carry_back_load(rho2, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{4}"])

    #node3 (4 rooms service panel)
    Y_node3 = Y_node2carried + Y_branches["Y_3"]
    # ===== Backbone segment 3: l_w_25 (service panel wire) =====
    if fault_seg_idx == 3:
        Y_node3carried, h3 = carry_back_with_fault(
            Y_node3, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths["l_w_25"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
        h3= h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{25}"])
        Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    Y_node4 = Y_node3carried + Y_branches["Y_4"]

    # ===== Backbone segment 4: l_w_28 (room wire) =====
    if fault_seg_idx == 4:
        Y_node4carried, h4 = carry_back_with_fault(
            Y_node4, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_28"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho4 = reflection_coefficient(Y_node4, T_r, Tinv_r, ZC_r, YC_r)
        h4 = h_B(rho4, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{28}"])
        Y_node4carried = carry_back_load(rho4, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{28}"])

    # node5 Transmitter Y13 connected in parallel
    Y_node6 = Y_node4carried + Y_branches["Y_5"]

    YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_node6, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    hoverall = h0 @ h1 @ h2 @ h3 @ h4
    #hoverall = h5 @ h4 @ h3 @ h2 @ h1
    H1 = hoverall @ H_trans 
    #H_nw_magnitude_db_1 = 20 * torch.log10(torch.abs(H1[:, 0, 0]))    
    H_nw = H1[:, 0, 0]
    return H_nw

def calculate_Hnw_nofault_admittance_version(cable_lengths, admittance_params):
    """
    Calculate network Transfer Function Hnw given admittance + cable parameters (instead of cable/load params)

    Parameters:
    - cable_lengths (dict): Dictionary of cable lengths tensors (30 length)
    - admittance_params (dict): Dictionary of sampled admittance parameters tensors (30 parameters)
    
    Returns:
    - (1,1)st entry of H_nw (N, n, n) where N = num of frequency points and n = num of conductors - 1
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    
    Y_branches = {}

    for branch_name, branch_params in admittance_params.items():
        G_d = branch_params["G_d"]
        C_d = branch_params["C_d"]
        L_d = branch_params["L_d"]
        G_o = branch_params["G_o"]
        C_o = branch_params["C_o"]
        L_o = branch_params["L_o"]
        Y_branches[branch_name] = reconstruct_Y(G_d, C_d, L_d, G_o, C_o, L_o)


    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room

    #node0 (Yrec)
    rho0 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h0 = h_B(rho0, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{0}"])
    Y_reccarried = carry_back_load(rho0, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{0}"])
    #node1 (Y62)
    Y_node1 = Y_reccarried + Y_branches["Y_1"]
    rho1 = reflection_coefficient(Y_node1, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{1}"])
    Y_node1carried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{1}"])
    #node2 Junction box (Y61 and  Y63)
    Y_node2 = Y_node1carried + Y_branches["Y_2"]
    rho2 = reflection_coefficient(Y_node2, T_s, Tinv_s, ZC_s, YC_s)
    h2 = h_B(rho2, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{4}"])
    Y_node3carried = carry_back_load(rho2, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{4}"])
    #node3 (4 rooms service panel)
    Y_node3 = Y_node3carried + Y_branches["Y_3"]
    rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
    h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{25}"])
    Y_node4carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    Y_node4 = Y_node4carried + Y_branches["Y_4"]
    rho4 = reflection_coefficient(Y_node4, T_r, Tinv_r, ZC_r, YC_r)
    h4 = h_B(rho4, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{28}"])
    Y_node5carried = carry_back_load(rho4, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{28}"])
    # node5 Transmitter Y13 connected in parallel
    Y_node5 = Y_node5carried + Y_branches["Y_5"]

    YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_node5, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    #hoverall = h1 @ h2
    hoverall = h0 @ h1 @ h2 @ h3 @ h4
    H1 = hoverall @ H_trans 
    #H_nw_magnitude_db_1 = 20 * torch.log10(torch.abs(H1[:, 0, 0]))    
    H_nw = H1[:, 0, 0]
    return H_nw
    

def compute_true_equivalent_admittance(cable_lengths, load_params):
    """
    Compute the 5 equivalent admittances from the complex model.
    Returns (Y_1, ... , Y_5) as tuple of (N, 3, 3) complex tensors.
    (Y_1, ..., Y_5) is ordered from receiver to transmitter. 
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    # Initialize dictionary to store load admittance matrices calculated from sampled_params
    Y_loads = {}

    for load, params in load_params.items():
        # Extract impedance-related parameters from the sampled parameters
        if 'R_const' in params and 'C_leak' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = constant_impedance(params['R_const'], params['C_leak'], omega)
        elif 'R_s' in params and 'omega_0s' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = double_RLC(
                params['R_s'], params['omega_0s'], params['zeta_s'], 
                params['R_p'], params['omega_0p'], params['zeta_p'], 
                params['delta_1'], params['delta_2'], params['C_d_leak'], omega)  
        elif 'C_m' in params and 'L_m' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = motor_load(
                params['C_m'], params['L_m'], params['R_m1'], 
                params['R_m2'], params['C_m_leak'], omega)

        else:
            raise ValueError(f"Unknown load model in {load}")
        
        # Compute the Nx3x3 admittance matrix for the given load
        Y_loads[load] = compute_load_admittance_3d((Z12, Z13, Z23, ZG1, ZG2, ZG3))
    
    
    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room
    #node0 (Yrec)
    rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{0}"])
    Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{0}"])
    #node1 (Y62)
    Y_node2 = Y_reccarried + Y_loads["load_0"]
    Y_eq1 = Y_loads["load_0"]
    
    rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
    h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{1}"])
    Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{1}"])
    #node2 Junction box (Y61 || Y63)
    rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{2}"])
    Y_61 = Y_63 + Y_loads["load_2"]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{3}"])
    Y_eq2 = Y_6
    Y_node3 = Y_node2carried + Y_6
    rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
    h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{4}"])
    Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{4}"])
    #node3 (4 rooms service panel)
    Y_5 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(3, 7)],  # load_3 to load_6
    [cable_lengths[f"l_w_{i}"] for i in range(5, 10)],  # l_w_5 to l_w_9
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_4 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(7, 11)],  # load_7 to load_10
    [cable_lengths[f"l_w_{i}"] for i in range(10, 15)],  # l_w_10 to l_w_14
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_3 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(11, 15)],  # load_11 to load_14
    [cable_lengths[f"l_w_{i}"] for i in range(15, 20)],  # l_w_15 to l_w_19
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_2 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(15, 19)],  # load_15 to load_18
    [cable_lengths[f"l_w_{i}"] for i in range(20, 25)],  # l_w_20 to l_w_24
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)   
    Y_eq3 = Y_5 + Y_4 + Y_3 + Y_2
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
    h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{25}"])
    Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{26}"])
    Y_12 = Y_14 + Y_loads["load_20"]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{27}"])
    Y_eq4 = Y_1
    Y_node5 = Y_node4carried + Y_1
    rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
    h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{28}"])
    Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{28}"])
    # node5 Transmitter Y13 connected in parallel
    rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{29}"])
    Y_eq5 = Y_13
    return Y_eq1, Y_eq2, Y_eq3, Y_eq4, Y_eq5
    
def calculate_Hnw_nofault(cable_lengths, sampled_params):
    """
    Calculate network Transfer Function Hnw given network parameters.
    
    Parameters:
    - cable_lengths (dict): Dictionary of cable lengths (30 lengths)
    - sampled_params (dict): Dictionary of sampled load parameters (22 loads of either type 1, 2, or 3)
    
    Returns:
    - (1,1)st entry of H_nw (N, n, n) where N = num of frequency points and n = num of conductors - 1
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    # Initialize dictionary to store load admittance matrices calculated from sampled_params
    Y_loads = {}
    num_loads = len(sampled_params)

    for load, params in sampled_params.items():
        # Extract impedance-related parameters from the sampled parameters
        if 'R_const' in params and 'C_leak' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = constant_impedance(params['R_const'], params['C_leak'], omega)
        elif 'R_s' in params and 'omega_0s' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = double_RLC(
                params['R_s'], params['omega_0s'], params['zeta_s'], 
                params['R_p'], params['omega_0p'], params['zeta_p'], 
                params['delta_1'], params['delta_2'], params['C_d_leak'], omega)  
        elif 'C_m' in params and 'L_m' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = motor_load(
                params['C_m'], params['L_m'], params['R_m1'], 
                params['R_m2'], params['C_m_leak'], omega)

        else:
            raise ValueError(f"Unknown load model in {load}")
        
        # Compute the Nx3x3 admittance matrix for the given load
        Y_loads[load] = compute_load_admittance_3d((Z12, Z13, Z23, ZG1, ZG2, ZG3))
    
    
    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room
    #node0 (Yrec)
    rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{0}"])
    Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{0}"])
    #node1 (Y62)
    Y_node2 = Y_reccarried + Y_loads["load_0"]
    #print("Y1", Y_loads["load_0"])
    
    rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
    h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{1}"])
    Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{1}"])
    #node2 Junction box (Y61 || Y63)
    rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{2}"])
    Y_61 = Y_63 + Y_loads["load_2"]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{3}"])
   # print("Y2", Y_6)
    
    Y_node3 = Y_node2carried + Y_6
    rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
    h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{4}"])
    Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{4}"])
    #node3 (4 rooms service panel)
    Y_5 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(3, 7)],  # load_3 to load_6
    [cable_lengths[f"l_w_{i}"] for i in range(5, 10)],  # l_w_5 to l_w_9
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_4 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(7, 11)],  # load_7 to load_10
    [cable_lengths[f"l_w_{i}"] for i in range(10, 15)],  # l_w_10 to l_w_14
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_3 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(11, 15)],  # load_11 to load_14
    [cable_lengths[f"l_w_{i}"] for i in range(15, 20)],  # l_w_15 to l_w_19
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_2 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(15, 19)],  # load_15 to load_18
    [cable_lengths[f"l_w_{i}"] for i in range(20, 25)],  # l_w_20 to l_w_24
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
   # print("Y3", Y_5 + Y_4 + Y_3 + Y_2)
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
    h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{25}"])
    Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{26}"])
    Y_12 = Y_14 + Y_loads["load_20"]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{27}"])
   # print("Y4", Y_1)
    Y_node5 = Y_node4carried + Y_1
    rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
    h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{28}"])
    Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{28}"])
    # node5 Transmitter Y13 connected in parallel
    rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{29}"])
   # print("Y5", Y_13)
    
    Y_node6 = Y_node5carried + Y_13

    YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_node6, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    #hoverall = h1 @ h2
    hoverall = h1 @ h2 @ h3 @ h4 @ h5
    #hoverall = h5 @ h4 @ h3 @ h2 @ h1
    H1 = hoverall @ H_trans 
    #H_nw_magnitude_db_1 = 20 * torch.log10(torch.abs(H1[:, 0, 0]))    
    H_nw = H1[:, 0, 0]
    return H_nw
    return H_nw_magnitude_db_1


def calculate_Hnw(cable_lengths, sampled_params, fault_params):
    """
    Calculate network Transfer Function Hnw given network parameters.
    
    Parameters:
    - cable_lengths (dict): Dictionary of cable lengths (30 lengths)
    - sampled_params (dict): Dictionary containing sampled load parameters (22 loads of either type 1, 2, or 3)
    - fault_params (dict): Dictionary containing the 2 fault parameters, length and impedance. 
    Returns:
    - (1,1)st entry of H_nw (N, n, n) where N = num of frequency points and n = num of conductors - 1
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    # Initialize dictionary to store load admittance matrices calculated from sampled_params
    Y_loads = {}
    

    fault_location = fault_params["fault_position"]
    Z_fault_real = fault_params["Z_fault_real"]
    Z_fault_imag = fault_params["Z_fault_imag"]
    #From location + impedance find which cable has the fault 
    fault_seg_idx, local_fault_pos, _ = get_fault_segment_and_local_position(fault_location, cable_lengths)
    for load, params in sampled_params.items():
        # Extract impedance-related parameters from the sampled parameters
        if 'R_const' in params and 'C_leak' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = constant_impedance(params['R_const'], params['C_leak'], omega)
        elif 'R_s' in params and 'omega_0s' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = double_RLC(
                params['R_s'], params['omega_0s'], params['zeta_s'], 
                params['R_p'], params['omega_0p'], params['zeta_p'], 
                params['delta_1'], params['delta_2'], params['C_d_leak'], omega)  
        elif 'C_m' in params and 'L_m' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = motor_load(
                params['C_m'], params['L_m'], params['R_m1'], 
                params['R_m2'], params['C_m_leak'], omega)

        else:
            raise ValueError(f"Unknown load model in {load}")
        
        # Compute the Nx3x3 admittance matrix for the given load
        Y_loads[load] = compute_load_admittance_3d((Z12, Z13, Z23, ZG1, ZG2, ZG3))
    
    
    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room

    #node0 (Yrec)
    # ===== Backbone segment 0: l_w_0 (room wire) =====
    if fault_seg_idx == 0:
        Y_reccarried, h1 = carry_back_with_fault(
            Y_rec, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_0"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
        h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{0}"])
        Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{0}"])

    #node1 (Y62)
    Y_node2 = Y_reccarried + Y_loads["load_0"]
    # ===== Backbone segment 1: l_w_1 (room wire) =====
    if fault_seg_idx == 1:
        Y_node2carried, h2 = carry_back_with_fault(
            Y_node2, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_1"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
        h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{1}"])
        Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{1}"])

    #node2 Junction box (Y61 || Y63)
    rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{2}"])
    Y_61 = Y_63 + Y_loads["load_2"]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{3}"])
    Y_node3 = Y_node2carried + Y_6
    # ===== Backbone segment 2: l_w_4 (service panel wire) =====
    if fault_seg_idx == 2:
        Y_node3carried, h3 = carry_back_with_fault(
            Y_node3, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths["l_w_4"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
        h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{4}"])
        Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{4}"])

    #node3 (4 rooms service panel)
    Y_5 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(3, 7)],  # load_3 to load_6
    [cable_lengths[f"l_w_{i}"] for i in range(5, 10)],  # l_w_5 to l_w_9
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_4 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(7, 11)],  # load_7 to load_10
    [cable_lengths[f"l_w_{i}"] for i in range(10, 15)],  # l_w_10 to l_w_14
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_3 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(11, 15)],  # load_11 to load_14
    [cable_lengths[f"l_w_{i}"] for i in range(15, 20)],  # l_w_15 to l_w_19
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_2 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(15, 19)],  # load_15 to load_18
    [cable_lengths[f"l_w_{i}"] for i in range(20, 25)],  # l_w_20 to l_w_24
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    # ===== Backbone segment 3: l_w_25 (service panel wire) =====
    if fault_seg_idx == 3:
        Y_node4carried, h4 = carry_back_with_fault(
            Y_node4, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths["l_w_25"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
        h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths[f"l_w_{25}"])
        Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{26}"])
    Y_12 = Y_14 + Y_loads["load_20"]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{27}"])
    Y_node5 = Y_node4carried + Y_1

    # ===== Backbone segment 4: l_w_28 (room wire) =====
    if fault_seg_idx == 4:
        Y_node5carried, h5 = carry_back_with_fault(
            Y_node5, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths["l_w_28"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
        h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths[f"l_w_{28}"])
        Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{28}"])

    # node5 Transmitter Y13 connected in parallel
    rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths[f"l_w_{29}"])
    Y_node6 = Y_node5carried + Y_13

    YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_node6, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    hoverall = h1 @ h2 @ h3 @ h4 @ h5
    #hoverall = h5 @ h4 @ h3 @ h2 @ h1
    H1 = hoverall @ H_trans 
    #H_nw_magnitude_db_1 = 20 * torch.log10(torch.abs(H1[:, 0, 0]))    
    H_nw = H1[:, 0, 0]
    return H_nw
    return H_nw_magnitude_db_1


def model_no_fault(H1_noisy):
    """
    Stage 1. Model with no fault
    """
    N, F, _ = H1_noisy.shape
    # Sample/fix load parameters
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
                load_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                load_dict[param_name] = torch.tensor(param_info["value"], device=device)
        load_params[load_name] = load_dict

    # Sample/fix admittance parameters
    admittance_params = {}
    for admittance_name, params in network_params["admittance_parameters"].items():
        admittance_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{admittance_name}_{param_name}", dist.Uniform(0.0, 1.0))
                admittance_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                admittance_dict[param_name] = torch.tensor(param_info["value"], device=device)
        admittance_params[admittance_name] = admittance_dict

    # Sample/fix cable parameters
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_info["inferred"]:
            infer_lo, infer_hi = cable_info.get("infer_range", cable_info["range"])
            norm_sample = pyro.sample(f"{cable_name}", dist.Uniform(0.0, 1.0))
            physical_value = denormalize(norm_sample, infer_lo, infer_hi)
            cable_lengths[cable_name] = physical_value
        else:
            cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

    if FORWARD_MODEL == "full":
        H1_pred_c = calculate_Hnw_nofault(cable_lengths, load_params).unsqueeze(0).expand(N, -1)
        H1_pred = torch.view_as_real(H1_pred_c)
    else:
        H1_pred_c = calculate_Hnw_nofault_admittance_version(cable_lengths, admittance_params).unsqueeze(0).expand(N, -1)
        H1_pred = torch.view_as_real(H1_pred_c)

    with pyro.plate("data", N):
        pyro.sample(
            "obs",
            dist.Independent(
                dist.Normal(loc=H1_pred, scale=0.00022655),
                reinterpreted_batch_ndims=2
            ),
            obs=H1_noisy
        )


def model_with_fault(H1_noisy):
    """
    Stage 2 - Network with fault - only infer fault parameters. 
    Load/cable parameters or admittance/cable parameters are fixed to their inferred values from Stage 1. 
    """
    N, F, _ = H1_noisy.shape

    # Sample/fix load parameters
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
                load_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                load_dict[param_name] = torch.tensor(param_info["value"], device=device)
        load_params[load_name] = load_dict

    # Sample/fix cable parameters
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_info["inferred"]:
            infer_lo, infer_hi = cable_info.get("infer_range", cable_info["range"])
            norm_sample = pyro.sample(f"{cable_name}", dist.Uniform(0.0, 1.0))
            physical_value = denormalize(norm_sample, infer_lo, infer_hi)
            cable_lengths[cable_name] = physical_value
        else:
            cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

    # Sample/fix fault parameters
    fault_params = {}
    for fault_name, fault_info in network_params["fault_parameters"].items():
        if fault_info["inferred"]:
            if fault_name == "fault_position":
                norm_sample = pyro.sample(f"{fault_name}", dist.Uniform(0.0, 1.0))
                fault_params[fault_name] = norm_sample  # Already normalized [0,1]
            else:
                min_val, max_val = fault_info["range"]
                norm_sample = pyro.sample(f"{fault_name}", dist.Uniform(0.0, 1.0))
                fault_params[fault_name] = denormalize(norm_sample, min_val, max_val)
        else:
            # Fixed fault parameters (use true values for testing)
            if fault_name == "Z_fault_real":
                fault_params[fault_name] = torch.tensor(100.0, device=device)
            elif fault_name == "Z_fault_imag":
                fault_params[fault_name] = torch.tensor(-50.0, device=device)
            else:
                fault_params[fault_name] = torch.tensor(0.25, device=device)
    # Sample/fix admittance parameters
    admittance_params = {}
    for admittance_name, params in network_params["admittance_parameters"].items():
        admittance_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{admittance_name}_{param_name}", dist.Uniform(0.0, 1.0))
                admittance_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                admittance_dict[param_name] = torch.tensor(param_info["value"], device=device)
        admittance_params[admittance_name] = admittance_dict

    # WITH FAULT forward model
    if FORWARD_MODEL == "full":
        H1_pred_c = calculate_Hnw(cable_lengths, load_params, fault_params).unsqueeze(0).expand(N, -1)
        H1_pred = torch.view_as_real(H1_pred_c)
    else:
        H1_pred_c = calculate_Hnw_fault_admittance_version(cable_lengths, admittance_params, fault_params).unsqueeze(0).expand(N, -1)
        H1_pred = torch.view_as_real(H1_pred_c)

    with pyro.plate("data", N):
        pyro.sample(
            "obs",
            dist.Independent(
                dist.Normal(loc=H1_pred, scale=0.00022655),
                reinterpreted_batch_ndims=2
            ),
            obs=H1_noisy
        )



def guide(H1_noisy):
    for load_name, params in network_params["loads"].items():
        for param_name, param_info in params.items():
            if not param_info["inferred"]:
                continue

            full_name = f"{load_name}_{param_name}"
            loc = pyro.param(f"{full_name}_loc", torch.tensor(0.0, device=device))  # std normal
            scale = pyro.param(f"{full_name}_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)

            q = TransformedDistribution(
                dist.Normal(loc, scale),
                [SigmoidTransform()]
            )
            
            pyro.sample(full_name, q)
            
    for admittance_name, params in network_params["admittance_parameters"].items():
        for param_name, param_info in params.items():
            if not param_info["inferred"]:
                continue

            full_name = f"{admittance_name}_{param_name}"
            loc = pyro.param(f"{full_name}_loc", torch.tensor(0.0, device=device))  # std normal
            scale = pyro.param(f"{full_name}_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)

            q = TransformedDistribution(
                dist.Normal(loc, scale),
                [SigmoidTransform()]
            )
            
            pyro.sample(full_name, q)

    for key, info in network_params["cable_lengths"].items():
        if not info["inferred"]:
            continue

        loc = pyro.param(f"{key}_loc", torch.tensor(0.0, device=device))
        scale = pyro.param(f"{key}_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)

        q = TransformedDistribution(dist.Normal(loc, scale), [SigmoidTransform()])
        pyro.sample(key, q)
    for key, info in network_params["fault_parameters"].items():
        if not info["inferred"]:
            continue
        if key == "fault_position":
            loc = pyro.param(f"{key}_loc", torch.logit(torch.tensor(0.5, device=device)))
            scale = pyro.param(f"{key}_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)
        else:
            loc = pyro.param(f"{key}_loc", torch.tensor(0.0, device=device))
            scale = pyro.param(f"{key}_scale", torch.tensor(0.1, device=device), constraint=constraints.positive)
        q = TransformedDistribution(dist.Normal(loc, scale), [SigmoidTransform()])
        pyro.sample(key, q)




def extract_posterior_means(param_history, num_samples=2048):
    """
    Extract posterior means (normalized [0,1]) from param_history after inference.
    Uses the final loc and scale values to construct the variational distribution,
    then computes the MC mean of the sigmoid-transformed Normal.

    Args:
        param_history: dict of parameter trajectories from SVI
        num_samples: number of MC samples for computing the mean

    Returns:
        dict: {param_name: normalized_mean} for all inferred parameters
    """
    posterior_means = {}

    for key, values in param_history.items():
        if "_loc" not in key:
            continue
        param_name = key.replace("_loc", "")
        loc = values[-1]  # Final iteration value

        # Look up corresponding scale
        scale_key = f"{param_name}_scale"
        scale = param_history[scale_key][-1]

        with torch.no_grad():
            q = TransformedDistribution(dist.Normal(loc, scale), [SigmoidTransform()])
            mc_mean = q.sample((num_samples,)).mean().item()

        posterior_means[param_name] = mc_mean

    return posterior_means


def calculate_rmse_per_parameter(param_history, sorted_keys, true_normalized_value=0.25, num_samples=2048):
    """
    Calculate RMSE (error) for each parameter comparing posterior mean to true value.
    Only includes parameters in sorted_keys, returned in that order.

    Args:
        param_history: dict of parameter trajectories from SVI
        sorted_keys: list of parameter names to include (in desired order)
        true_normalized_value: true value in normalized [0,1] space (default 0.25)
        num_samples: number of MC samples for computing posterior mean

    Returns:
        dict: {param_name: error} in sorted_keys order
    """
    posterior_means = extract_posterior_means(param_history, num_samples)

    rmse_dict = {}
    for key in sorted_keys:
        # Convert key format: sorted_keys uses dots, posterior_means uses underscores
        posterior_key = key.replace(".", "_")
        if posterior_key in posterior_means:
            rmse_dict[key] = abs(posterior_means[posterior_key] - true_normalized_value)

    return rmse_dict


def print_rmse_table(rmse_results, sorted_keys=None):
    """
    Print a formatted table of RMSE results.

    Args:
        rmse_results: dict from calculate_rmse_per_parameter()
        sorted_keys: optional list of parameter names to control print order
    """
    print("\n" + "=" * 80)
    print(f"{'Parameter':<30} {'Post. Mean':<12} {'True':<8} {'Error':<12} {'Sq. Error':<12}")
    print("-" * 80)

    per_param = rmse_results['per_parameter']

    if sorted_keys:
        # Print in specified order, only for params that exist
        keys_to_print = [k for k in sorted_keys if k in per_param]
    else:
        # Print all in alphabetical order
        keys_to_print = sorted(per_param.keys())

    for param_name in keys_to_print:
        info = per_param[param_name]
        print(f"{param_name:<30} {info['mean']:<12.4f} {0.25:<8.2f} {info['error']:<12.4f} {info['squared_error']:<12.6f}")

    print("-" * 80)
    print(f"{'Overall MSE:':<30} {rmse_results['overall_mse']:.6f}")
    print(f"{'Overall RMSE:':<30} {rmse_results['overall_rmse']:.6f}")
    print("=" * 80)




def update_network_params_from_posterior(posterior_means):
    """
    Update network_params with inferred posterior means from Stage 1.
    Denormalizes the posterior mean and stores the physical value in `value`,
    then marks those parameters as `inferred=False` for Stage 2.

    Args:
        posterior_means: dict from extract_posterior_means() - normalized [0,1] values
    """
    updated_count = 0

    # Update load parameters
    for load_name, params in network_params["loads"].items():
        for param_name, param_info in params.items():
            full_name = f"{load_name}_{param_name}"

            if full_name in posterior_means:
                norm_val = posterior_means[full_name]
                # Denormalize to physical value
                if "range" in param_info:
                    min_val, max_val = param_info["range"]
                    physical_val = denormalize(norm_val, min_val, max_val)
                    old_val = param_info["value"]
                    param_info["value"] = physical_val
                    print(f" Updated {full_name}: old value={normalize(old_val, min_val, max_val):.4f} -> new value={norm_val:.6g}")
                param_info["inferred"] = False
                updated_count += 1
    # Update admittance parameters
    for admittance_name, params in network_params["admittance_parameters"].items():
        for param_name, param_info in params.items():
            full_name = f"{admittance_name}_{param_name}"

            if full_name in posterior_means:
                norm_val = posterior_means[full_name]
                # Denormalize to physical value
                if "range" in param_info:
                    min_val, max_val = param_info["range"]
                    physical_val = denormalize(norm_val, min_val, max_val)
                    old_val = param_info["value"]
                    param_info["value"] = physical_val
                    print(f" Updated {full_name}: old value={normalize(old_val, min_val, max_val):.4f} -> new value={norm_val:.6g}")
                param_info["inferred"] = False
                updated_count += 1

    # Update cable parameters
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_name in posterior_means:
            norm_val = posterior_means[cable_name]
            min_val, max_val = cable_info.get("infer_range", cable_info["range"])
            physical_val = denormalize(norm_val, min_val, max_val)
            old_val = cable_info["value"]
            cable_info["value"] = physical_val
            cable_info["inferred"] = False
            updated_count += 1
            print(f" Updated {cable_name}: old value={normalize(old_val, min_val, max_val):.4f} -> new value={norm_val:.4f}")
    

    print(f"\nUpdated {updated_count} parameters from Stage 1 posterior means.")


def run_two_stage_inference(cable_lengths, load_params, admittance_params, fault_params,
                            snr_db, num_steps_stage1, num_steps_stage2,
                            threshold):
    """
    Run the complete two-stage inference workflow:

    Stage 1 (Network Identification):
        - No fault present
        - Infer cable/load or cable/admittance parameters
        - Use calculate_Hnw_nofault or calculate_Hnw_nofault_admittanceversion

    Stage 2 (Fault Localization):
        - Fault is present
        - Fix cable/load or cable/admittance parameters to Stage 1 posterior means
        - Only infer fault parameters
        - Use calculate_Hnw or calculate_Hnw_admittanceversion

    Returns:
        stage1_results: (losses, param_history, sorted_keys) from Stage 1
        stage2_results: (losses, param_history, sorted_keys) from Stage 2
    """

    # ========================================================================
    # STAGE 1: Network Identification (No Fault)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: Network Identification (No Fault)")
    print("="*70)

    # Generate Stage 1 observations with calculate_Hnw_nofault
    H1_clean_s1 = calculate_Hnw_nofault(cable_lengths, load_params)

    # Add noise
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigpow = torch.mean(torch.abs(H1_clean_s1)**2)
    var_f = sigpow / snr_lin
    std_f = torch.sqrt(var_f / 2)

    H1_noisy_s1 = H1_clean_s1 + std_f * torch.randn_like(H1_clean_s1.real) + \
                  1j * std_f * torch.randn_like(H1_clean_s1.imag)
    H1_noisy_s1_expanded = H1_noisy_s1.unsqueeze(0).expand(1, -1)
    H1_noisy_s1_real = torch.view_as_real(H1_noisy_s1_expanded)


    # Full parameter space
    if FORWARD_MODEL == "full":
        # Turn OFF fault parameter inference
        for fault_name in network_params["fault_parameters"]:
            network_params["fault_parameters"][fault_name]["inferred"] = False
        # Turn OFF admittance parameter inference 
        for admittance_name, params in network_params["admittance_parameters"].items():
            for param_name, param_info in params.items():
                param_info["inferred"] = False
    else:
        # Turn OFF load parameter inference 
        for load_name, params in network_params["loads"].items():
            for param_name, param_info in params.items():
                param_info["inferred"] = False
        # Turn OFF fault parameter inference
        for fault_name in network_params["fault_parameters"]:
            network_params["fault_parameters"][fault_name]["inferred"] = False
        # Turn OFF all cable inference
        for cable_name, cable_info in network_params["cable_lengths"].items():
            cable_info["inferred"] = False
        # Turn ON only backbone cables
        for key in BACKBONE_KEYS:
            network_params["cable_lengths"][key]["inferred"] = True
        

    
    # Run sensitivity analysis to select which load/cable params to infer
    print(f"\nRunning Stage 1 Sensitivity Analysis...")
    selected_s1, sorted_keys_s1 = perform_load_sensitivity_analysis(
        load_params, fault_params, cable_lengths, admittance_params,
        threshold=threshold, scenario="no_fault")

    # Run Stage 1 inference
    print(f"\nRunning Stage 1 SVI inference ({num_steps_stage1} steps)...")
    losses_s1, param_history_s1 = run_inference(
        H1_noisy_s1_real, model_no_fault, guide, sorted_keys_s1, num_steps=num_steps_stage1
    )

    # Extract posterior means and update network_params. All inferred parameters in Stage 1 
    # are turned off in update_network_params_from_posterior 
    print("\n--- Extracting, Updating, and Plotting ---")
    posterior_means_s1 = extract_posterior_means(param_history_s1)
    update_network_params_from_posterior(posterior_means_s1)
    plot_param_convergence(param_history_s1, losses_s1, sorted_keys_s1, "no_fault")
    plot_CI_and_pred_TF(param_history_s1, H1_clean_s1, "no_fault")

    #Reduced parameter space using admittance matrices

    # ========================================================================
    # STAGE 2: Fault Localization (With Fault)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Fault Localization (With Fault)")
    print("="*70)

    # Turn ON fault parameter inference
    network_params["fault_parameters"]["fault_position"]["inferred"] = True
    network_params["fault_parameters"]["Z_fault_real"]["inferred"] = True
    network_params["fault_parameters"]["Z_fault_imag"]["inferred"] = True

    # Build sorted_keys for Stage 2 (fault params only)
    sorted_keys_s2 = ["fault_position", "Z_fault_real", "Z_fault_imag"]
    print(f"\n[STAGE 2] Inferring fault params only: {sorted_keys_s2}")

    # Generate Stage 2 observations (with fault)
    H1_clean_s2 = calculate_Hnw(cable_lengths, load_params, fault_params)

    # Add noise (same SNR)
    sigpow_s2 = torch.mean(torch.abs(H1_clean_s2)**2)
    var_f_s2 = sigpow_s2 / snr_lin
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    H1_noisy_s2 = H1_clean_s2 + std_f_s2 * torch.randn_like(H1_clean_s2.real) + \
                  1j * std_f_s2 * torch.randn_like(H1_clean_s2.imag)
    H1_noisy_s2_expanded = H1_noisy_s2.unsqueeze(0).expand(1, -1)
    H1_noisy_s2_real = torch.view_as_real(H1_noisy_s2_expanded)

    # Run Stage 2 inference
    print(f"\nRunning Stage 2 SVI inference ({num_steps_stage2} steps)...")
    pyro.clear_param_store()  # Clear params from Stage 1
    losses_s2, param_history_s2 = run_inference(
        H1_noisy_s2_real, model_with_fault, guide, sorted_keys_s2, num_steps=num_steps_stage2
    )

    # Plot Stage 2 results
    plot_param_convergence(param_history_s2, losses_s2, sorted_keys_s2, "with_fault")
    plot_CI_and_pred_TF(param_history_s2, H1_clean_s2, "with_fault")

    return (losses_s1, param_history_s1, sorted_keys_s1), \
           (losses_s2, param_history_s2, sorted_keys_s2)



def run_inference(H1_noisy, model, guide, sorted_keys, num_steps):
    # H1_noisy is (N, F, 2), float NOT COMPLEX
    pyro.clear_param_store()

    if OPTIMIZER == "Adam":
        optimizer = pyro.optim.Adam({"lr": LR})
    elif OPTIMIZER == "Adagrad":
        optimizer = pyro.optim.Adagrad({"lr": LR})
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Use 'Adam' or 'Adagrad'.")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=20))
    top_20_most_sensitive = [
        key.replace(".", "_") + "_loc"
        for key in sorted_keys[:20]
    ]    
    losses = []
    param_history = defaultdict(list)  # Contains Python floats (more memory efficient)

    # Initialize parameters by running guide once
    guide(H1_noisy)

    # Save initial parameter values
    param_store = pyro.get_param_store()
    for name, value in param_store.items():
        param_history[name].append(value.detach().item())

    for step in range(num_steps):
        loss = svi.step(H1_noisy)
        losses.append(loss)

        param_store = pyro.get_param_store()
        for name, value in param_store.items():
            param_history[name].append(value.detach().item())
        
        if step % 25 == 0 or step == 0:
            print(f"\n===== Step {step} | ELBO: {loss:.6f} =====")
            print("\n Top 20 Most Sensitive Parameters")
            for key in top_20_most_sensitive:
                if key in param_store:
                    print(f"{key:40s} (sigmoid) = {torch.sigmoid(param_store[key])} | True value = 0.25")
        
    print("Inference complete.")
    return losses, param_history

def plot_param_convergence(param_history, losses, sorted_keys, scenario):
    # ELBO Plot
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("SVI ELBO Loss")
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.yscale("symlog")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FILENAME_PREFIX}_svi_elbo_loss_{scenario}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # Parameter Plots
    plt.figure(figsize=(14, 5))

    # (1) loc trajectories (normalized) - apply sigmoid since param_history stores raw loc
    plt.subplot(1, 2, 1)
    loc_keys = [k for k in param_history if "loc" in k]
    for key in loc_keys:
        vals = np.array(param_history[key])
        vals_sigmoid = 1 / (1 + np.exp(-vals))  # sigmoid in numpy
        plt.plot(vals_sigmoid, alpha=0.7)

    plt.title("Mean Convergence (normalized)")
    plt.xlabel("SVI step")
    plt.ylabel("Variational mean (sigmoid(loc))")
    plt.grid(True)

    # (2) scale trajectories
    plt.subplot(1, 2, 2)
    scale_keys = [k for k in param_history if "scale" in k]
    for key in scale_keys:
        plt.plot(param_history[key], alpha=0.7)  # matplotlib accepts lists directly
    plt.title("Scale Convergence")
    plt.xlabel("SVI step")
    plt.ylabel("Variational scale (std dev)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FILENAME_PREFIX}_svi_param_convergence_{scenario}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # --------- PRINT FINAL MEANS FOR TOP 20 MOST SENSITIVE ----------
    rows = []
    squared_errors = []
    top_20_keys = sorted_keys[:20]  # Only use top 20 most sensitive

    print("\n=== Final Posterior Means (Top 20 Most Sensitive, normalized) ===")
    print(f"{'Parameter':30s} | {'q-mean':>8s} | {'true':>8s} | {'error':>10s}")
    print("-" * 65)

    for name in top_20_keys:
        pyro_loc_key = name.replace(".", "_") + "_loc"
        if pyro_loc_key not in param_history:
            print(f"{name:30s} | q-mean = N/A")
            continue
        raw_loc = param_history[pyro_loc_key][-1]
        q_mean = 1 / (1 + np.exp(-raw_loc))  # sigmoid for Python float

        # Get true value (normalized) from network_params
        if pyro_loc_key == "Z_fault_real_loc":
            true_normalized = 0.025
        else:
            true_normalized = 0.25

        error = q_mean - true_normalized
        squared_errors.append(error ** 2)
        print(f"{name:30s} | {q_mean:8.4f} | {true_normalized:8.4f} | {error:+10.4f}")
        rows.append({
            "parameter": name,
            "q_mean_normalized": q_mean,
            "true_normalized": true_normalized,
            "error": error,
        })

    # Compute and print RMSE for top 20
    if squared_errors:
        rmse = np.sqrt(np.mean(squared_errors))
        print("-" * 65)
        print(f"{'RMSE (top 20, normalized)':30s} | {rmse:.6f}")
        print(f"Number of parameters: {len(squared_errors)}")

    # Save to CSV with RMSE as separate column (scalar in first row only)
    if rows:
        df = pd.DataFrame(rows)
        df["rmse_top20"] = np.nan
        if squared_errors:
            df.loc[0, "rmse_top20"] = rmse
        df.to_csv(f"{FILENAME_PREFIX}_posterior_means_{scenario}.csv", index=False)
        print(f"\nSaved to {FILENAME_PREFIX}_posterior_means_{scenario}.csv")

def perform_load_sensitivity_analysis(load_params, fault_params, cable_lengths, admittance_params, threshold, scenario):
    """
    Perform sensitivity analysis on network parameters.
    Supports both full model (load + all cables) and admittance model (admittance params + backbone cables).

    Returns:
    Selected_keys: List of selected params like ['load_0.C_m_leak', 'load_1.C_m_leak', 'load_3.C_m_leak'm ...] not sorted
    Sorted_keys: List of all params sorted from most sensitive to least sensitive
    Sensitivities: List of corresponding sensitivities to Sorted_keys
    """

    variations = {}

    if FORWARD_MODEL == "full":
        # Full model: use load params and all cable lengths
        if scenario == "with_fault":
            nominal_H = calculate_Hnw(cable_lengths, load_params, fault_params)
        else:
            nominal_H = calculate_Hnw_nofault(cable_lengths, load_params)

        # Analyze load parameters
        for load_name, param_dict in network_params["loads"].items():
            for param_name, param_info in param_dict.items():
                if not param_info["inferred"]:
                    continue

                lo, hi = param_info["range"]
                # Use log scale for resistance parameters (wide range)
                if param_name in ["R_const", "R_s", "R_p", "R_m1"]:
                    values = np.logspace(np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-6)), 10)
                else:
                    values = np.linspace(lo, hi, 10)
                param_variations = []

                for val in values:
                    perturbed_loads = copy.deepcopy(load_params)
                    perturbed_loads[load_name][param_name] = torch.tensor(val, device=device)
                    if scenario == "with_fault":
                        H_var = calculate_Hnw(cable_lengths, perturbed_loads, fault_params)
                    else:
                        H_var = calculate_Hnw_nofault(cable_lengths, perturbed_loads)
                    diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                    param_variations.append(diff)

                total_var = sum(param_variations)
                key = f"{load_name}.{param_name}"
                variations[key] = total_var

        # Analyze all cable length parameters for full model
        for cable_name, cable_info in network_params["cable_lengths"].items():
            if not cable_info["inferred"]:
                continue

            lo, hi = cable_info["range"]
            values = np.linspace(lo, hi, 10)
            param_variations = []

            for val in values:
                perturbed_cables = copy.deepcopy(cable_lengths)
                perturbed_cables[cable_name] = torch.tensor(val)
                if scenario == "with_fault":
                    H_var = calculate_Hnw(perturbed_cables, load_params, fault_params)
                else:
                    H_var = calculate_Hnw_nofault(perturbed_cables, load_params)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

            total_var = sum(param_variations)
            variations[cable_name] = total_var

    else:  # FORWARD_MODEL == "admittance"
        # Admittance model: use admittance params and backbone cables only
        if scenario == "with_fault":
            nominal_H = calculate_Hnw_fault_admittance_version(cable_lengths, admittance_params, fault_params)
        else:
            nominal_H = calculate_Hnw_nofault_admittance_version(cable_lengths, admittance_params)

        # Analyze admittance parameters
        for branch_name, params in network_params["admittance_parameters"].items():
            for param_name, param_info in params.items():
                if not param_info["inferred"]:
                    continue

                lo, hi = param_info["range"]
                values = np.linspace(lo, hi, 10)
                param_variations = []

                for val in values:
                    perturbed_admittance = copy.deepcopy(admittance_params)
                    perturbed_admittance[branch_name][param_name] = torch.tensor(val, device=device)
                    if scenario == "with_fault":
                        H_var = calculate_Hnw_fault_admittance_version(cable_lengths, perturbed_admittance, fault_params)
                    else:
                        H_var = calculate_Hnw_nofault_admittance_version(cable_lengths, perturbed_admittance)
                    diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                    param_variations.append(diff)

                total_var = sum(param_variations)
                key = f"{branch_name}.{param_name}"
                variations[key] = total_var

        # Analyze only backbone cable length parameters for admittance model
        for cable_name in BACKBONE_KEYS:
            cable_info = network_params["cable_lengths"].get(cable_name)
            if cable_info is None or not cable_info["inferred"]:
                continue

            lo, hi = cable_info["range"]
            values = np.linspace(lo, hi, 10)
            param_variations = []

            for val in values:
                perturbed_cables = copy.deepcopy(cable_lengths)
                perturbed_cables[cable_name] = torch.tensor(val, device=device)
                if scenario == "with_fault":
                    H_var = calculate_Hnw_fault_admittance_version(perturbed_cables, admittance_params, fault_params)
                else:
                    H_var = calculate_Hnw_nofault_admittance_version(perturbed_cables, admittance_params)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

            total_var = sum(param_variations)
            variations[cable_name] = total_var

    # Analyze fault parameters only if scenario = "with_fault"
    if scenario == "with_fault":
        for fault_name, fault_info in network_params["fault_parameters"].items():
            if not fault_info["inferred"]:
                continue

            lo, hi = fault_info["range"]

            # For Z_fault_real, use log scale since range is wide (1 to 4000)
            if fault_name == "Z_fault_real":
                values = np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), 10)
                print("Z_fault_real values (log scale):", values)
            else:
                values = np.linspace(lo, hi, 10)

            param_variations = []

            for val in values:
                perturbed_fault = copy.deepcopy(fault_params)
                perturbed_fault[fault_name] = torch.tensor(val, device=device)

                if FORWARD_MODEL == "full":
                    H_var = calculate_Hnw(cable_lengths, load_params, perturbed_fault)
                else:
                    H_var = calculate_Hnw_fault_admittance_version(cable_lengths, admittance_params, perturbed_fault)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

            total_var = sum(param_variations)
            variations[fault_name] = total_var

    # Normalize
    total_sum = sum(variations.values())
    normalized = {k: v / total_sum for k, v in variations.items()}
    sensitivities = []
    # Select parameters above threshold
    selected = [k for k, v in normalized.items() if v > threshold]
    flag = True
    print("\n--- Network Parameter Sensitivity Analysis ---")
    print(f"Forward model: {FORWARD_MODEL}")
    for k in sorted(normalized, key=normalized.get, reverse=True):
        if normalized[k] <= threshold and flag:
            print("Parameters after this are below the threshold")
            flag = False
        print(f"{k}: {normalized[k]*100:.5f}%")
        sensitivities.append(f"{normalized[k]*100:.5f}%")


    print(f"\nSelected parameters (>{threshold*100:.2f}%): {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    # Set inferred=False for non-sensitive parameters (only if threshold > 0)
    if threshold > 0:
        disabled_count = 0
        for param_key, sensitivity in normalized.items():
            if sensitivity <= threshold:
                # Parse the param_key to find where it belongs
                if "." in param_key:
                    # Load or admittance parameter: "load_0.C_m" or "branch_0.param"
                    parts = param_key.split(".")
                    entity_name = parts[0]
                    param_name = parts[1]

                    # Check if it's a load parameter
                    if entity_name in network_params["loads"]:
                        if param_name in network_params["loads"][entity_name]:
                            network_params["loads"][entity_name][param_name]["inferred"] = False
                            disabled_count += 1
                    # Check if it's an admittance parameter
                    elif "admittance_parameters" in network_params and entity_name in network_params["admittance_parameters"]:
                        if param_name in network_params["admittance_parameters"][entity_name]:
                            network_params["admittance_parameters"][entity_name][param_name]["inferred"] = False
                            disabled_count += 1
                else:
                    # Cable length or fault parameter
                    if param_key in network_params["cable_lengths"]:
                        network_params["cable_lengths"][param_key]["inferred"] = False
                        disabled_count += 1
                    elif "fault_parameters" in network_params and param_key in network_params["fault_parameters"]:
                        network_params["fault_parameters"][param_key]["inferred"] = False
                        disabled_count += 1

        print(f"\nDisabled inference for {disabled_count} non-sensitive parameters (sensitivity <= {threshold*100:.2f}%)")

    sorted_keys = sorted(normalized, key=normalized.get, reverse=True)
    #sort selected keys too
    selected = sorted(selected, key=normalized.get, reverse=True)
    return selected, sorted_keys, sensitivities

def plot_CI_and_pred_TF(param_history, true_tf, scenario, num_samples=200):
    true_tf_db = 20*torch.log10(torch.abs(true_tf))
    tf_samples = []

    for i in range(num_samples):
        # Build sampled parameters for both full and reduced models
        sampled_cable_lengths = {}
        sampled_load_params = {}
        sampled_fault_params = {}
        sampled_admittance_params = {}

        # Sample load params (used by full model)
        for load_name, params in network_params["loads"].items():
            sampled_load_params[load_name] = {}
            for param_name, param_info in params.items():
                pyro_key = f"{load_name}_{param_name}_loc"
                if pyro_key in param_history:  # Check if was inferred (exists in param_history)
                    lo, hi = param_info["range"]
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{load_name}_{param_name}_scale"][-1]
                    z = np.random.normal(loc, scale)
                    sigmoid_z = 1 / (1 + np.exp(-z))
                    sampled_load_params[load_name][param_name] = torch.tensor(denormalize(sigmoid_z, lo, hi), device=device)
                else:
                    sampled_load_params[load_name][param_name] = torch.tensor(param_info["value"], device=device)

        # Sample cable params (used by both models)
        for cable_name, cable_info in network_params["cable_lengths"].items():
            pyro_key = f"{cable_name}_loc"
            if pyro_key in param_history:  # Check if was inferred
                lo, hi = cable_info.get("infer_range", cable_info["range"])
                loc = param_history[pyro_key][-1]
                scale = param_history[f"{cable_name}_scale"][-1]
                z = np.random.normal(loc, scale)
                sigmoid_z = 1 / (1 + np.exp(-z))
                sampled_cable_lengths[cable_name] = torch.tensor(denormalize(sigmoid_z, lo, hi), device=device)
            else:
                sampled_cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

        # Sample admittance params (used by reduced model)
        if "admittance_parameters" in network_params:
            for branch_name, params in network_params["admittance_parameters"].items():
                sampled_admittance_params[branch_name] = {}
                for param_name, param_info in params.items():
                    pyro_key = f"{branch_name}_{param_name}_loc"
                    if pyro_key in param_history:  # Check if was inferred
                        lo, hi = param_info["range"]
                        loc = param_history[pyro_key][-1]
                        scale = param_history[f"{branch_name}_{param_name}_scale"][-1]
                        z = np.random.normal(loc, scale)
                        sigmoid_z = 1 / (1 + np.exp(-z))
                        sampled_admittance_params[branch_name][param_name] = denormalize(sigmoid_z, lo, hi)
                    else:
                        sampled_admittance_params[branch_name][param_name] = param_info["value"]

        # Sample fault params (if with_fault scenario)
        if scenario == "with_fault":
            for fault_name, fault_info in network_params["fault_parameters"].items():
                pyro_key = f"{fault_name}_loc"
                if pyro_key in param_history:  # Check if was inferred
                    lo, hi = fault_info["range"]
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{fault_name}_scale"][-1]
                    z = np.random.normal(loc, scale)
                    sigmoid_z = 1 / (1 + np.exp(-z))
                    sampled_fault_params[fault_name] = torch.tensor(denormalize(sigmoid_z, lo, hi), device=device)
                else:
                    sampled_fault_params[fault_name] = torch.tensor(fault_info["value"], device=device)

        # Compute TF with sampled parameters based on FORWARD_MODEL
        if FORWARD_MODEL == "full":
            if scenario == "with_fault":
                H_sample = calculate_Hnw(sampled_cable_lengths, sampled_load_params, sampled_fault_params)
            else:
                H_sample = calculate_Hnw_nofault(sampled_cable_lengths, sampled_load_params)
        else:  # FORWARD_MODEL == "reduced"
            if scenario == "with_fault":
                H_sample = calculate_Hnw_fault_admittance_version(sampled_cable_lengths, sampled_admittance_params, sampled_fault_params)
            else:
                H_sample = calculate_Hnw_nofault_admittance_version(sampled_cable_lengths, sampled_admittance_params)
        tf_samples.append(20*torch.log10(torch.abs(H_sample)).detach().numpy())

    # Stack samples: (num_samples, num_freqs)
    tf_samples = np.stack(tf_samples, axis=0)
    
    # Compute mean and percentiles
    tf_mean = np.mean(tf_samples, axis=0)
    tf_lower = np.percentile(tf_samples, 2.5, axis=0)
    tf_upper = np.percentile(tf_samples, 97.5, axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_range_mhz.numpy(), tf_mean, 'k-', linewidth=1.5, label='Model')
    plt.plot(freq_range_mhz.numpy(), true_tf_db.detach().numpy(), 'r--', linewidth=1.5, label='Truth')
    plt.fill_between(freq_range_mhz.numpy(), tf_lower, tf_upper, 
                     alpha=0.3, color='steelblue', label='95% CI')
    
    plt.xscale('log')
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel(r'$H_{1,1}$ (dB)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{FILENAME_PREFIX}_tf_posterior_CI_{scenario}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {FILENAME_PREFIX}_tf_posterior_CI_{scenario}.pdf")
    return tf_mean, tf_lower, tf_upper


def plot_gamma_vs_frequency_complex():
    """
    Plot propagation constant gamma vs frequency for the complex MTL model.

    In MTL theory with modal decomposition, gamma is a diagonal matrix where each
    diagonal entry represents the propagation constant for that mode.
    For a 3-conductor system (n=3), we have 3 modes.

    Creates 6 subplots: alpha (attenuation) and beta (phase constant) for each mode.
    """
    # Get cable parameters
    r_w = 0.8128e-3  # Wire radius (same as used in calculate_cable_parameters)
    n = num_of_conductors - 1  # n = 3

    R, L, C, G = calculate_cable_parameters(r_w, omega, n)

    # Get MTL matrices including gamma
    _, _, gamma, _, _ = get_mtl_matrices(R, L, C, G, n, omega)

    # Extract diagonal entries for each mode
    # gamma has shape (num_freqs, n, n), diagonal entries are the modal propagation constants
    gamma_mode0 = gamma[:, 0, 0]  # Mode 0
    gamma_mode1 = gamma[:, 1, 1]  # Mode 1
    gamma_mode2 = gamma[:, 2, 2]  # Mode 2

    # Extract alpha (real) and beta (imag) for each mode
    alpha0 = gamma_mode0.real.cpu().numpy()
    beta0 = gamma_mode0.imag.cpu().numpy()
    alpha1 = gamma_mode1.real.cpu().numpy()
    beta1 = gamma_mode1.imag.cpu().numpy()
    alpha2 = gamma_mode2.real.cpu().numpy()
    beta2 = gamma_mode2.imag.cpu().numpy()

    freq_mhz = freq_range_mhz.cpu().numpy()

    # Create figure with 6 subplots (3 rows x 2 columns)
    _, axes = plt.subplots(3, 2, figsize=(14, 12))

    mode_data = [
        (alpha0, beta0, 'Mode 0', 'b'),
        (alpha1, beta1, 'Mode 1', 'r'),
        (alpha2, beta2, 'Mode 2', 'g'),
    ]

    for i, (alpha, beta, mode_name, color) in enumerate(mode_data):
        # Plot alpha (attenuation) - left column
        ax_alpha = axes[i, 0]
        ax_alpha.plot(freq_mhz, alpha, f'{color}-', linewidth=1.5)
        ax_alpha.set_xscale('log')
        ax_alpha.set_xlabel('Frequency (MHz)', fontsize=11)
        ax_alpha.set_ylabel(r'$\alpha$ (Np/m)', fontsize=11)
        ax_alpha.set_title(fr'{mode_name}: Attenuation $\alpha = \Re\{{\gamma_{{{i},{i}}}\}}$', fontsize=12)
        ax_alpha.grid(True, which='both', linestyle='--', alpha=0.5)
        # Add min/max text box
        alpha_text = f'Min: {alpha.min():.6f}\nMax: {alpha.max():.6f}'
        ax_alpha.text(0.95, 0.05, alpha_text, transform=ax_alpha.transAxes, fontsize=9,
                      verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot beta (phase constant) - right column
        ax_beta = axes[i, 1]
        ax_beta.plot(freq_mhz, beta, f'{color}-', linewidth=1.5)
        ax_beta.set_xscale('log')
        ax_beta.set_xlabel('Frequency (MHz)', fontsize=11)
        ax_beta.set_ylabel(r'$\beta$ (rad/m)', fontsize=11)
        ax_beta.set_title(fr'{mode_name}: Phase Constant $\beta = \Im\{{\gamma_{{{i},{i}}}\}}$', fontsize=12)
        ax_beta.grid(True, which='both', linestyle='--', alpha=0.5)
        # Add min/max text box
        beta_text = f'Min: {beta.min():.4f}\nMax: {beta.max():.4f}'
        ax_beta.text(0.95, 0.05, beta_text, transform=ax_beta.transAxes, fontsize=9,
                     verticalalignment='bottom', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(r'MTL Propagation Constants $\gamma$ vs Frequency (Modal Decomposition)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('gamma_vs_frequency_mtl_oldfreqrange.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nFrequency range: {freq_mhz[0]:.4f} - {freq_mhz[-1]:.2f} MHz")
    for i, (alpha, beta, mode_name, _) in enumerate(mode_data):
        print(f"{mode_name} - Alpha: [{alpha.min():.6f}, {alpha.max():.6f}] Np/m, "
              f"Beta: [{beta.min():.4f}, {beta.max():.4f}] rad/m")


def plot_nll_vs_ReZF_complex_mtl(cable_lengths, load_params, fault_params, snr_db, save_path="nll_vs_ReZF_complex_mtl_150-10.pdf"):
    """
    Plot Z_fault_real (ReZF) vs NLL.
    """
    print("snr db", snr_db)
    obs_tf_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigpow_s2 = torch.mean(torch.abs(obs_tf_clean)**2)
    var_f_s2 = sigpow_s2 / snr_lin
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    obs_tf_noisy = obs_tf_clean + std_f_s2 * torch.randn_like(obs_tf_clean.real) + \
                  1j * std_f_s2 * torch.randn_like(obs_tf_clean.imag)

    # Sweep ReZF normalized from 0 to 1, then denormalize to physical range
    ReZF_min, ReZF_max = 1.0, 4000.0
    ReZF_normalized = torch.linspace(0.01, 0.99, 200, dtype=torch.float32)
    losses = []
    original_ReZF = fault_params["Z_fault_real"]  # Save original value

    with torch.no_grad():
        for ReZF_norm in ReZF_normalized:
            # Denormalize to physical value
            ReZF_physical = ReZF_min + ReZF_norm * (ReZF_max - ReZF_min)
            fault_params["Z_fault_real"] = ReZF_physical
            pred_tf = calculate_Hnw(cable_lengths, load_params, fault_params)
            diff = obs_tf_noisy - pred_tf
            nll = (diff.abs().pow(2) / var_f_s2).sum()
            losses.append(nll.item())
    fault_params["Z_fault_real"] = original_ReZF  # Restore original value

    # Get true normalized value for plotting
    true_ReZF = original_ReZF.item() if torch.is_tensor(original_ReZF) else original_ReZF
    true_ReZF_normalized = (true_ReZF - ReZF_min) / (ReZF_max - ReZF_min)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ReZF_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
    plt.axvline(x=true_ReZF_normalized, color='r', linestyle='--', linewidth=2,
                label=f'True ReZF={true_ReZF:.1f}Ω (norm={true_ReZF_normalized:.3f})')
    plt.xlabel('Re{Z_fault} (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Fault Impedance Real Part (Complex Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss landscape plot to {save_path}")


def plot_nll_vs_L1_complex_mtl(cable_lengths, load_params, fault_params, snr_db, save_path="nll_vs_L1_complex_mtl_150-2df"):
    """
    plot fault_position vs NLL.
    """
    print("snr db", snr_db)
    obs_tf_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigpow_s2 = torch.mean(torch.abs(obs_tf_clean)**2)
    var_f_s2 = sigpow_s2 / snr_lin #scalar
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    obs_tf_noisy = obs_tf_clean + std_f_s2 * torch.randn_like(obs_tf_clean.real) + \
                  1j * std_f_s2 * torch.randn_like(obs_tf_clean.imag)

    # Sweep L1 from 0 to 1 (normalized)
    L1_normalized = torch.linspace(0.01, 0.99, 199, dtype=torch.float32)
    losses = []
    original_fault_position = fault_params["fault_position"]  # Save original value
    with torch.no_grad():
        for L1 in L1_normalized:
            #fault_params["fault_position"] = denormalize(L1, min_val, max_val)
            fault_params["fault_position"] = L1
            pred_tf = calculate_Hnw(cable_lengths, load_params, fault_params)
            diff = obs_tf_noisy - pred_tf
            nll = (diff.abs().pow(2) / var_f_s2).sum()   # correct
            losses.append(nll.item())
    fault_params["fault_position"] = original_fault_position  # Restore original value

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(L1_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
    plt.axvline(x=0.25, color='r', linestyle='--', linewidth=2, label='True fault_position=0.25')
    plt.xlabel('L1 (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Fault Location L1 (Complex Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss landscape plot to {save_path}")

def validate_reduced_model(cable_lengths, load_params, admittance_params, save_path="reduced_model_validation2.png"):
    """
    Compare full model (calculate_Hnw_nofault) vs reduced model (calculate_Hnw_nofault_admittance_version).
    Plots both transfer functions and shows the approximation error.
    
    Args:
        cable_lengths: dict of cable lengths
        load_params: dict of load parameters (for full model)
        admittance_params: dict of fitted GCL parameters (for reduced model)
        save_path: where to save the plot
    """
    # Compute transfer functions from both models
    with torch.no_grad():
        H_full = calculate_Hnw_nofault(cable_lengths, load_params)  # Ground truth
        H_reduced = calculate_Hnw_nofault_admittance_version(cable_lengths, admittance_params)
    
    # Convert to dB
    H_full_db = 20 * torch.log10(torch.abs(H_full)).cpu().numpy()
    H_reduced_db = 20 * torch.log10(torch.abs(H_reduced)).cpu().numpy()
    
    # Compute error
    error_db = H_reduced_db - H_full_db
    freq_mhz = (frequencies / 1e6).cpu().numpy()
    
    # Compute error statistics
    mae = np.mean(np.abs(error_db))
    rmse = np.sqrt(np.mean(error_db**2))
    max_error = np.max(np.abs(error_db))
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top plot: Transfer functions overlaid
    ax1 = axes[0]
    ax1.plot(freq_mhz, H_full_db, 'b-', linewidth=2, label='Full Model (Ground Truth)')
    ax1.plot(freq_mhz, H_reduced_db, 'r--', linewidth=2, label='Reduced Model (G+jωC+1/jωL)')
    ax1.fill_between(freq_mhz, H_full_db - 1, H_full_db + 1, alpha=0.2, color='blue', label='±1 dB band')
    ax1.set_ylabel('|H(f)| (dB)', fontsize=12)
    ax1.set_title('Transfer Function Comparison: Full vs Reduced Model', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Error
    ax2 = axes[1]
    ax2.plot(freq_mhz, error_db, 'g-', linewidth=1.5, label='Error (Reduced - Full)')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(y=mae, color='r', linestyle='--', linewidth=1, label=f'MAE = {mae:.3f} dB')
    ax2.axhline(y=-mae, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(freq_mhz, -rmse, rmse, alpha=0.2, color='orange', label=f'±RMSE = {rmse:.3f} dB')
    ax2.set_xlabel('Frequency (MHz)', fontsize=12)
    ax2.set_ylabel('Error (dB)', fontsize=12)
    ax2.set_title(f'Approximation Error (MAE={mae:.3f} dB, RMSE={rmse:.3f} dB, Max={max_error:.3f} dB)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("REDUCED MODEL VALIDATION SUMMARY")
    print("="*60)
    print(f"Mean Absolute Error (MAE):  {mae:.4f} dB")
    print(f"Root Mean Square Error:     {rmse:.4f} dB")
    print(f"Maximum Absolute Error:     {max_error:.4f} dB")
    print(f"Error within ±1 dB:         {100*np.mean(np.abs(error_db) < 1):.1f}% of frequencies")
    print(f"Error within ±3 dB:         {100*np.mean(np.abs(error_db) < 3):.1f}% of frequencies")
    print("="*60)
    print(f"Saved validation plot to {save_path}")
    
    return {"mae": mae, "rmse": rmse, "max_error": max_error, "error_db": error_db}

#CRLB helper functions
def get_inferred_param_order():
    """
    Get ordered list of inferred parameters for consistent flat tensor indexing.

    Returns:
        param_order: List of tuples describing each parameter:
            - ("cable", cable_name, None) for cable lengths
            - ("load", load_name, param_name) for load parameters
        num_params: Total number of inferred parameters
    """
    param_order = []

    # Cable lengths first (in sorted order for consistency)
    for cable_name in sorted(network_params["cable_lengths"].keys(), key=lambda x: int(x.split("_")[-1])):
        if network_params["cable_lengths"][cable_name]["inferred"]:
            param_order.append(("cable", cable_name, None))

    # Load parameters (sorted by load number, then by param name)
    for load_name in sorted(network_params["loads"].keys(), key=lambda x: int(x.split("_")[-1])):
        for param_name in sorted(network_params["loads"][load_name].keys()):
            if network_params["loads"][load_name][param_name]["inferred"]:
                param_order.append(("load", load_name, param_name))

    return param_order, len(param_order)


def get_true_param_flat():
    """
    Get flat tensor of true parameter values in the order defined by get_inferred_param_order().

    Returns:
        params_flat: [P] tensor of true parameter values in physical units
    """
    param_order, num_params = get_inferred_param_order()
    params_flat = torch.zeros(num_params, dtype=torch.float32, device=device)

    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            params_flat[i] = network_params["cable_lengths"][name]["value"]
        elif ptype == "load":
            params_flat[i] = network_params["loads"][name][subname]["value"]

    return params_flat


def build_params_from_flat(params_flat, param_order):
    """
    Unpack flat parameter tensor into cable_lengths and load_params dictionaries.

    Args:
        params_flat: [P] tensor of parameter values
        param_order: List from get_inferred_param_order()

    Returns:
        cable_lengths: Dict of cable length tensors
        load_params: Dict of load parameter dicts
    """
    # Initialize with non-inferred (fixed) values
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        cable_lengths[cable_name] = torch.tensor(cable_info["value"], dtype=torch.float32, device=device)

    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_params[load_name] = {}
        for param_name, param_info in params.items():
            load_params[load_name][param_name] = torch.tensor(param_info["value"], dtype=torch.float32, device=device)

    # Override with values from flat tensor
    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            cable_lengths[name] = params_flat[i]
        elif ptype == "load":
            load_params[name][subname] = params_flat[i]

    return cable_lengths, load_params


def H_nofault_wrapper(params_flat):
    """
    Wrapper for calculate_Hnw_nofault that takes a flat parameter tensor,
    converts it to dict format that serves as acceptable input for calculate_Hnw_nofault. 

    Returns [F, 2] real tensor (Re, Im stacked) for Jacobian computation.

    Args:
        params_flat: [P] tensor of inferred parameter values

    Returns:
        H_real_imag: [F, 2] tensor where [:, 0] = Re(H), [:, 1] = Im(H)
    """
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params = build_params_from_flat(params_flat, param_order)

    # Compute transfer function
    H = calculate_Hnw_nofault(cable_lengths, load_params)  # [F] complex

    # Stack real and imaginary parts
    return torch.stack([H.real, H.imag], dim=-1)  # [F, 2]

def H_fault_wrapper(params_complex, params_real):
    """
    Wrapper for calculate_Hnw_fault. Complex params 
    """
    H = calculate_Hnw(cable_lengths, load_params, fault_params)  # [F] complex

def build_S2_from_nullspace(U2, tol=1e-8):
    """
    Build selector matrix S2 consisting only of original parameters theta_i
    whose standard basis vectors e_i are orthogonal to Null(J).

    Parameters
    ----------
    U2 : torch.Tensor, shape [p, p-r]
        Columns of U2 form an orthonormal basis for Null(J).
    tol : float
        Numerical tolerance for deciding whether a row is zero.

    Returns
    -------
    S2 : torch.Tensor, shape [m, p]
        Selector matrix picking only coordinates fully orthogonal to Null(J).
    keep_indices : list[int]
        Indices i such that e_i ⟂ Null(J).
    """
    p = U2.shape[0]
    row_norms = torch.linalg.norm(U2, dim=1)   # norm of each row
    keep_mask = row_norms < tol
    keep_indices = torch.nonzero(keep_mask, as_tuple=True)[0].tolist()

    S2 = torch.eye(p, dtype=U2.dtype, device=U2.device)[keep_mask]
    return S2, keep_indices


def compute_real_FIM_and_CRLB(var_f, sorted_keys_s1, sensitivities):
    """
    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately)
    P = number of parameters (inputs)
    F = number of frequencies (outputs)
    Compute Real Fisher Information Matrix and CRLB for P inferred real parameters.
    Note FIM and CRLB are normalized here. 
    Uses float64 precision for numerical stability.

    Args:
        var_f: Noise variance (determined by SNR) [] if white noise (constant)
        or [F] if frequency dependent
    Returns:
        CRLB_U1U1T: [P] Dict of sqrt Cramér-Rao Lower Bounds for alpha = U1U1^T theta
        CRLB_S2: Dict of sqrt Cramér-Rao Lower Bounds for alpha = S2 theta
    """
    param_order_list, _ = get_inferred_param_order()

    # Use float64 for CRLB computation (better numerical precision)
    params_flat = get_true_param_flat().double()
    var_f_64 = var_f.double() if isinstance(var_f, torch.Tensor) else torch.tensor(var_f, dtype=torch.float64)

    # Wrapper that ensures float64 output
    def H_nofault_wrapper_f64(params):
        param_order, _ = get_inferred_param_order()
        cable_lengths, load_params = build_params_from_flat(params, param_order)
        H = calculate_Hnw_nofault(cable_lengths, load_params)
        # Convert to complex128 and stack
        H = H.to(torch.complex128)
        return torch.stack([H.real, H.imag], dim=-1)

    # Compute Jacobian dH/dtheta in float64
    J = jacfwd(H_nofault_wrapper_f64)(params_flat)  # [F, 2, P]

    # Build scaling vector s_p = d theta_p / d phi_p
    scale_list = [] #[P]

    # Cable lengths
    for cable_name in sorted(network_params["cable_lengths"].keys(), key=lambda x: int(x.split("_")[-1])):
        if network_params["cable_lengths"][cable_name]["inferred"]:
            min_val, max_val = network_params["cable_lengths"][cable_name]["range"]
            s = max_val - min_val
            scale_list.append(s)
    # Load parameters
    for load_name in sorted(network_params["loads"].keys(), key=lambda x: int(x.split("_")[-1])):
        for param_name in sorted(network_params["loads"][load_name].keys()):
            if network_params["loads"][load_name][param_name]["inferred"]:
                min_val, max_val = network_params["loads"][load_name][param_name]["range"]
                true_val = network_params["loads"][load_name][param_name]["value"]
                if param_name in ["R_const", "R_s", "R_p", "R_m1"]:
                    s = true_val * (math.log(max_val) - math.log(min_val))
                    scale_list.append(s)
                else:
                    s = max_val - min_val
                    scale_list.append(s)
    scale = torch.tensor(scale_list, dtype=torch.float64, device=J.device)   # [P]
    D = torch.diag(scale) #[P, P] 

    Delta = J[:, 0, :] + 1j * J[:, 1, :]  # ∂g/∂θ [F, P] complex
    Delta_tilde = J[:, 0, :] - 1j * J[:, 1, :]  # ∂g*/∂θ = (∂g/∂θ)^* for real θ [F, P] complex

    Delta = Delta.unsqueeze(-1)         # [F, P, 1]
    Delta_tilde = Delta_tilde.unsqueeze(-1)  # [F, P, 1]
    # Δ_f ⊗ Δ̃_f^T + Δ̃_f ⊗ Δ_f^T = 2*Re(Δ⊗Δᴴ) which is real
    I_f = (Delta @ Delta_tilde.transpose(-1,-2)) + (Delta_tilde @ Delta.transpose(-1,-2))  # [F, P, P]
    # FIM should be real - take real part (imag should be ~0 due to numerics)
    I = ((1 / var_f_64) * I_f.sum(dim=0)).real  # [P, P] real and should be symmetric + PSD
    I2 = D.T @ I @ D

    # #Unnormalized FIM
    # eigvals, eigvecs = torch.linalg.eigh(I)
    # rank = torch.linalg.matrix_rank(I)
    # print("Rank of FIM", rank)
    # dimension = I.shape[0]
    # singular = rank < dimension
    # print("FIM is singular?", singular)
    # print("Eigvals of FIM (descending)", torch.sort(eigvals, descending=True).values)
    # U, S, Vh = torch.linalg.svd(I)
    # print("Singular values of FIM(descending)", torch.sort(S, descending=True).values)

    # eps = torch.finfo(I2.dtype).eps
    # rtol = max(I2.shape[-2], I2.shape[-1]) * eps
    # tol = rtol * S.max()           # since default atol = 0

    # num_zeroed = (S <= tol).sum().item()
    # num_kept = (S > tol).sum().item()

    # print(f"eps       = {eps:.3e}")
    # print(f"rtol      = {rtol:.3e}")
    # print(f"sigma_max = {S.max().item():.3e}")
    # print(f"tol       = {tol.item():.3e}")
    # print(f"zeroed    = {num_zeroed}")
    # print(f"kept      = {num_kept}")

    #Normalized FIM
    eigvals2, eigvecs2 = torch.linalg.eigh(I2) 
    dimension2 = I2.shape[0]
    rank2 = torch.linalg.matrix_rank(I2)
    print("Rank of Normalized FIM", rank2)
    singular2 = rank2 < dimension2
    print("Normalized FIM is singular?", singular2)
    print("Eigvals of Normalized FIM (descending)", torch.sort(eigvals2, descending=True).values)

    U, S, Vh = torch.linalg.svd(I2)
    
    print("Singular values of Normalized FIM (descending)", torch.sort(S, descending=True).values)

    eps = torch.finfo(I2.dtype).eps
    rtol = max(I2.shape[-2], I2.shape[-1]) * eps
    tol = rtol * S.max()           # since default atol = 0

    num_zeroed = (S <= tol).sum().item()
    num_kept = (S > tol).sum().item()

    print(f"eps       = {eps:.3e}")
    print(f"rtol      = {rtol:.3e}")
    print(f"sigma_max = {S.max().item():.3e}")
    print(f"tol       = {tol.item():.3e}")
    print(f"zeroed    = {num_zeroed}")
    print(f"kept      = {num_kept}")
    
    # Diagnostic: parameter sensitivities
    J_flat = J.reshape(-1, J.shape[-1])  # [F*2, P]
    param_sensitivity = torch.abs(J_flat).max(dim=0).values

    print(f"Jacobian range: [{param_sensitivity.min():.2e}, {param_sensitivity.max():.2e}]")
    print(f"Ratio: {param_sensitivity.max()/param_sensitivity.min():.2e}")

    tol = 1e-8
    mask = eigvals2 > tol

    Lambda1 = eigvals2[mask]          # shape [r]
    U1 = eigvecs2[:, mask]            # shape [p, r]
    U2 = eigvecs2[:, ~mask]            #shape[p, p-r] 

    J_pinv = U1 @ torch.diag(1.0 / Lambda1) @ U1.T
    J_pinv_torch = torch.linalg.pinv(I2)
    #Sanity check they should be equal
    print("max |manual pinv - torch pinv| =", torch.max(torch.abs(J_pinv - J_pinv_torch)).item())

    
    S2, keep_indices = build_S2_from_nullspace(U2, tol=1e-8)

    print("keep_indices =", keep_indices)
    print("param order", param_order_list)
    print("S2 shape =", S2.shape)
    print("S2 =\n", S2)
    
    SU2 = S2 @ U2
    print("S2 @ U2 =\n", SU2)
    print("||S2 @ U2||_F =", torch.linalg.norm(SU2).item())
    print("max |S2 @ U2| =", torch.max(torch.abs(SU2)).item())


    CRLB_U1T = 1.0 / Lambda1
    CRLB_U1U1T = torch.diag(J_pinv)
    CRLB_S2 = torch.diag(S2 @ J_pinv @ S2.T)

    # CRLB for $\alpha = U_1^T theta$ should be just Lambda_1^{-1} 
    # CRLB for $\alpha = U_1 U_1^T theta$ is J^{\dagger}
    # CRLB_normalized = torch.diag(torch.linalg.pinv(I2))
    # Cov_physical = D @ torch.linalg.pinv(I2) @ D   # [P, P] matrix
    # CRLB_reconstructed = torch.diag(Cov_physical)  # [P] vector

    # Build mapping from param_order_list index to sorted_keys_s1 key
    def param_order_to_key(entry):
        """Convert param_order_list entry to sorted_keys_s1 format."""
        param_type, name1, name2 = entry
        if param_type == "cable":
            return name1  # e.g., "l_w_4"
        else:
            return f"{name1}.{name2}"  # e.g., "load_1.C_s"

    # Create mapping: key -> index in param_order_list.
    key_to_idx = {param_order_to_key(param_order_list[i]): i for i in range(len(param_order_list))}


    print("="*220)
    print(f"{'Idx':<5} {'Parameter':<22} {'Sens':<10} {'CRLB S2':<14} {'Unc S2':<10} {'CRLB U1U1T':<14} {'Unc U1U1T':<10}")
    print("-"*220)

    # Build dicts sorted by sorted_keys_s1 order
    crlb_u1u1t_dict = {}  # key -> CRLB value
    crlb_s2_dict = {}     # key -> CRLB value (only for params in keep_indices)

    # Print in sorted_keys_s1 order
    for index, key in enumerate(sorted_keys_s1):
        if key not in key_to_idx:
            continue
        i = key_to_idx[key]
        sens = sensitivities[index]
        crlb_u1u1t = CRLB_U1U1T[i].item()
        crlb_u1u1t_dict[key] = math.sqrt(crlb_u1u1t)
        uncert_u1u1t_pct = math.sqrt(crlb_u1u1t) * 100
        if i in keep_indices:
            s2_idx = keep_indices.index(i)
            crlb_s2 = CRLB_S2[s2_idx].item()
            crlb_s2_dict[key] = math.sqrt(crlb_s2)
            uncert_s2_pct = math.sqrt(crlb_s2) * 100
            crlb_s2_str = f"{crlb_s2:.2e}"
            uncert_s2_str = f"{uncert_s2_pct:.2f}%"
        else:
            crlb_s2_str = "N/A"
            uncert_s2_str = "N/A"

        print(f"{i:<5} {key:<22} {sens:<10}  {crlb_s2_str:<14} {uncert_s2_str:<10} {crlb_u1u1t:<14.2e} {uncert_u1u1t_pct:>5.2f}%")

    print("="*220)
    return crlb_u1u1t_dict, crlb_s2_dict

    
@torch.no_grad()
def fim_from_complex_jac_general(du_aug, var_NF, p1, p2):
    """
    Build the augmented complex FIM from Wirtinger Jacobians.
    Generalized for p1 complex parameters and p2 real parameters.

    The augmented parameter vector is ordered as:
        θ_aug = [z_1, ..., z_p1, z_1*, ..., z_p1*, r_1, ..., r_p2]

    Total augmented dimension: 2*p1 + p2

    Inputs
    ------
    du_aug : [F, 2*p1 + p2] complex
        Columns in GROUPED order:
            [∂u/∂z_1, ..., ∂u/∂z_p1, ∂u/∂z_1*, ..., ∂u/∂z_p1*, ∂u/∂r_1, ..., ∂u/∂r_p2]

    var_NF : [F] or scalar, real
        Noise variance σ² per frequency.

    p1 : int
        Number of complex parameters

    p2 : int
        Number of real parameters

    Returns
    -------
    I : [2*p1 + p2, 2*p1 + p2] complex
        Augmented complex Fisher information matrix.

    Notes
    -----
    Key Wirtinger identities for conjugated output u*:
        ∂u*/∂z   = (∂u/∂z*)*
        ∂u*/∂z*  = (∂u/∂z)*
        ∂u*/∂r   = (∂u/∂r)*

    The permutation to get du*/dθ_aug from du/dθ_aug:
        - Swap z_i with z_i* for each complex parameter
        - Keep real parameters in place
        - Conjugate everything
    """
    p_aug = 2 * p1 + p2  # Total augmented dimension

    # Build permutation: [p1:2*p1, 0:p1, 2*p1:2*p1+p2]
    # When p1=0, this reduces to identity permutation [0, 1, ..., p2-1]
    # This swaps z with z* and keeps r in place
    perm = list(range(p1, 2*p1)) + list(range(0, p1)) + list(range(2*p1, p_aug))
    perm_tensor = torch.tensor(perm, device=du_aug.device)

    # Get du*/dθ_aug by permuting and conjugating
    du_star = du_aug.index_select(-1, perm_tensor).conj()  # [F, p_aug]

    # print("du_aug", du_aug)
    # print("du_star", du_star)
    # print("du aug shape", du_aug.shape)
    # print("du star shape", du_star.shape)
    
    # Build FIM: I = Σ_f (1/σ²) * [∂u/∂θ^H @ ∂u/∂θ + ∂u*/∂θ^H @ ∂u*/∂θ]
    f = du_aug.unsqueeze(-1)      # [F, p_aug, 1]
    i = du_star.unsqueeze(-1)     # [F, p_aug, 1]

    # [F, p_aug, p_aug]
    I_f = (f.conj() @ f.transpose(-1, -2)) + (i.conj() @ i.transpose(-1, -2))
    # Weight by 1/variance and sum over frequencies
    if isinstance(var_NF, (int, float)):
        w = 1.0 / var_NF
    else:
        w = (1.0 / var_NF).unsqueeze(-1).unsqueeze(-1)  # [F, 1, 1]

    I = (I_f * w).sum(dim=0)  # [p_aug, p_aug]

    return I



def compute_complex_partials_general(forward_fn, params_complex, params_real):
    """
    Compute Wirtinger partial derivatives for general parameter sets.

    Args:
        forward_fn: Function that takes (z_re, z_im, r) tensors and returns [F] complex output
        params_complex: [p1] complex tensor of complex parameter values (or None)
        params_real: [p2] real tensor of real parameter values (or None)

    Returns:
        du_aug: [F, 2*p1 + p2] complex - augmented Jacobian
            Columns: [∂u/∂z_1, ..., ∂u/∂z_p1, ∂u/∂z_1*, ..., ∂u/∂z_p1*, ∂u/∂r_1, ..., ∂u/∂r_p2]
        p1: Number of complex parameters
        p2: Number of real parameters
    """
    p1 = len(params_complex) if params_complex is not None and len(params_complex) > 0 else 0
    p2 = len(params_real) if params_real is not None and len(params_real) > 0 else 0

    if p1 == 0 and p2 == 0:
        raise ValueError("Must have at least one parameter")

    # Wrapper that returns [F, 2] real output for autodiff
    def forward_real_output(z_re, z_im, r):
        u = forward_fn(z_re, z_im, r)  # [F] complex
        return torch.stack([u.real, u.imag], dim=-1)  # [F, 2]

    # Setup inputs
    if p1 > 0:
        z_re = params_complex.real.clone()
        z_im = params_complex.imag.clone()
    else:
        z_re = torch.tensor([], device=device, dtype=torch.float32)
        z_im = torch.tensor([], device=device, dtype=torch.float32)

    if p2 > 0:
        r = params_real.clone()
    else:
        r = torch.tensor([], device=device, dtype=torch.float32)

    partials = []

    if p1 > 0:
        # Jacobian w.r.t. z_re: [F, 2, p1]
        J_z_re = jacfwd(forward_real_output, argnums=0)(z_re, z_im, r)
        # Jacobian w.r.t. z_im: [F, 2, p1]
        J_z_im = jacfwd(forward_real_output, argnums=1)(z_re, z_im, r)

        # Convert to complex: ∂u/∂θ = ∂Re(u)/∂θ + j*∂Im(u)/∂θ
        du_dz_re = J_z_re[..., 0, :] + 1j * J_z_re[..., 1, :]  # [F, p1]
        du_dz_im = J_z_im[..., 0, :] + 1j * J_z_im[..., 1, :]  # [F, p1]

        # Wirtinger derivatives
        du_dz = 0.5 * (du_dz_re - 1j * du_dz_im)       # ∂u/∂z
        du_dz_conj = 0.5 * (du_dz_re + 1j * du_dz_im)  # ∂u/∂z*

        partials.append(du_dz)        # [F, p1]
        partials.append(du_dz_conj)   # [F, p1]

    if p2 > 0:
        # Jacobian w.r.t. r: [F, 2, p2]
        J_r = jacfwd(forward_real_output, argnums=2)(z_re, z_im, r)
        du_dr = J_r[..., 0, :] + 1j * J_r[..., 1, :]  # [F, p2]
        partials.append(du_dr)

    # Stack all partials: [F, 2*p1 + p2]
    du_aug = torch.cat(partials, dim=-1)
    return du_aug, p1, p2


@torch.no_grad()
def get_crlb_from_augmented_fim(FIM, p1, p2):
    """
    Compute CRLB from augmented complex FIM.

    For complex parameters, returns variance of the complex estimate.
    For real parameters, returns variance of the real estimate.

    Args:
        FIM: [2*p1+p2, 2*p1+p2] complex augmented FIM
        p1: Number of complex parameters
        p2: Number of real parameters

    Returns:
        crlb_complex: [p1] real - CRLB for each complex parameter (or None if p1=0)
        crlb_real: [p2] real - CRLB for each real parameter (or None if p2=0)
    """
    p_aug = 2 * p1 + p2

    if p1 == 0:
        # All real parameters - direct inversion
        FIM_inv = torch.linalg.pinv(FIM.real + 1e-10 * torch.eye(p2, device=FIM.device))
        return None, torch.diag(FIM_inv)

    if p2 == 0:
        # All complex parameters
        # Use Schur complement: S = A - B @ A*^{-1} @ B*
        A = FIM[0:p1, 0:p1]
        B = FIM[0:p1, p1:2*p1]
        A_conj_inv = torch.linalg.pinv(A.conj() + 1e-10 * torch.eye(p1, device=FIM.device))
        S = A - B @ A_conj_inv @ B.conj()
        S_inv = torch.linalg.pinv(S + 1e-10 * torch.eye(p1, device=FIM.device))
        crlb_complex = torch.diag(S_inv).real
        return crlb_complex, None

    # Mixed case: full block inversion
    FIM_inv = torch.linalg.pinv(FIM + 1e-10 * torch.eye(p_aug, device=FIM.device, dtype=FIM.dtype))

    # CRLB for complex params: diagonal of top-left p1×p1 block
    crlb_complex = torch.diag(FIM_inv[0:p1, 0:p1]).real

    # CRLB for real params: diagonal of bottom-right p2×p2 block
    crlb_real = torch.diag(FIM_inv[2*p1:p_aug, 2*p1:p_aug]).real

    return crlb_complex, crlb_real


if __name__ == '__main__':
    start_time = time.time()
    # Assume network_params is already initialized
    #total_params, load_types = generate_load_parameters(num_loads, omega)
    total_params, load_types = generate_load_parameters_deterministic(num_loads)
    num_cable_params = len(network_params["cable_lengths"])
    num_fault_params = len(network_params["fault_parameters"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of fault parameters: {num_fault_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params + num_fault_params}")
    print(f"Load type distribution: {load_types}")
    # Turn off inference for ALL parameters except fault_position
    # for load_name, params in network_params["loads"].items():
    #     for param_name in params:
    #         network_params["loads"][load_name][param_name]["inferred"] = False
    # for cable_name in network_params["cable_lengths"]:
    #     network_params["cable_lengths"][cable_name]["inferred"] = False
    # network_params["fault_parameters"]["fault_position"]["inferred"] = True
    # network_params["fault_parameters"]["Z_fault_real"]["inferred"] = False
    # network_params["fault_parameters"]["Z_fault_imag"]["inferred"] = False
    # print("\n[CONFIG] Only inferring fault_position (all other params fixed)")

    #plot_gamma_vs_frequency_complex()
    
    #plot_nll_vs_L1_simple_mtl(snr_db=40, L1_true=25.0, L_total=100.0)
    cable_lengths = {
        key: torch.tensor(val["value"], device=device) for key, val in network_params["cable_lengths"].items()
    }

    load_params = {
        load_name: {
            param_name: torch.tensor(param_info["value"], device=device)
            for param_name, param_info in params.items()
        }
        for load_name, params in network_params["loads"].items()
    }
    fault_params = {
        key: torch.tensor(val["value"], device=device) for key, val in network_params["fault_parameters"].items()
    }
    # Compute true equivalent admittances from detailed model
    Y_1, Y_2, Y_3, Y_4, Y_5 = compute_true_equivalent_admittance(cable_lengths, load_params)
    # Fit all 5 branches
    fitted = fit_all_equivalent_admittances([Y_1, Y_2, Y_3, Y_4, Y_5])
    # Update network_params with true values
    generate_admittance_parameters(fitted)
    admittance_params = {
        admittance_name: {
            param_name: torch.tensor(param_info["value"], device=device)
            for param_name, param_info in params.items()
        }
        for admittance_name, params in network_params["admittance_parameters"].items()
    }

    # Full parameter space
    if FORWARD_MODEL == "full":
        # Turn OFF fault parameter inference
        for fault_name in network_params["fault_parameters"]:
            network_params["fault_parameters"][fault_name]["inferred"] = False
        # Turn OFF admittance parameter inference 
        for admittance_name, params in network_params["admittance_parameters"].items():
            for param_name, param_info in params.items():
                param_info["inferred"] = False
    
    selected_s1, sorted_keys_s1, sensitivities = perform_load_sensitivity_analysis(
        load_params, fault_params, cable_lengths, admittance_params,
        threshold=0.025, scenario="no_fault"
    )

    # Manually disable redundant parameters
    # params_to_disable = ["load_15.C_m_leak", "load_17.C_m_leak"]
    # for param_key in params_to_disable:
    #     # Set inferred=False in network_params
    #     parts = param_key.split(".")
    #     load_name, param_name = parts[0], parts[1]
    #     network_params["loads"][load_name][param_name]["inferred"] = False

    #     # Remove from selected_s1 and sorted_keys_s1
    #     if param_key in selected_s1:
    #         selected_s1.remove(param_key)
    #     if param_key in sorted_keys_s1:
    #         idx = sorted_keys_s1.index(param_key)
    #         sorted_keys_s1.remove(param_key)
    #         sensitivities.pop(idx)  # Also remove corresponding sensitivity

    # network_params["cable_lengths"]["l_w_23"]["inferred"] = False
    # network_params["loads"]["load_17"]["C_m_leak"]["inferred"] = False
    # network_params["loads"]["load_18"]["C_m_leak"]["inferred"] = False

    num_obs = 1
    params_flat = get_true_param_flat()
    param_order_list, _ = get_inferred_param_order()
    cable_lengths, load_params = build_params_from_flat(params_flat, param_order_list)
    H_true = calculate_Hnw_nofault(cable_lengths, load_params)  # [F] complex
    sigpow = torch.mean(torch.abs(H_true)**2)

    current_model = model_no_fault

    # SNR sweep
    snr_dbs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    rmse_results = {key: [] for key in selected_s1}
    crlb_results = {key: [] for key in selected_s1}

    for snr_db in snr_dbs:
        print(f"\n{'='*50}")
        print(f"SNR = {snr_db} dB")
        print('='*50)

        snr_lin = 10.0 ** (snr_db / 10.0)
        var_f = sigpow / snr_lin
        std_f = torch.sqrt(var_f / 2)

        H1_noisy_c = H_true + std_f * torch.randn_like(H_true.real) + \
                        1j * std_f * torch.randn_like(H_true.imag)
        H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1)
        H1_noisy = torch.view_as_real(H1_noisy_c_expanded)

        # CRLB at this SNR
        crlb_u1u1t_dict, _ = compute_real_FIM_and_CRLB(var_f, sorted_keys_s1, sensitivities)

        # Run SVI inference
        pyro.clear_param_store()
        losses, param_history = run_inference(H1_noisy, current_model, guide, sorted_keys_s1, num_steps=500)

        # Calculate RMSE
        rmse_dict = calculate_rmse_per_parameter(param_history, selected_s1, true_normalized_value=0.25)
        print("rmse_dict", rmse_dict)
        print("crlb dict", crlb_u1u1t_dict)
        # Store results
        for key in selected_s1:
            if key in rmse_dict and key in crlb_u1u1t_dict:
                rmse_results[key].append(rmse_dict[key])
                crlb_results[key].append(crlb_u1u1t_dict[key])  

    # Plot RMSE vs sqrt(CRLB) across SNR for each parameter
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, key in enumerate(selected_s1):
        print("idx", idx)
        print("key", key)
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.plot(snr_dbs, rmse_results[key], 'bo-', label='RMSE', markersize=6)
        ax.plot(snr_dbs, crlb_results[key], 'r--', label='sqrt(CRLB)', linewidth=2)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error')
        ax.set_title(key)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(sorted_keys_s1), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig("rmse_vs_crlb_snr_sweep.pdf", dpi=150, bbox_inches='tight')
    #plt.show()

    print("\nSaved plot to rmse_vs_crlb_snr_sweep.pdf")


    # adad

    # # Validate the reduced model
    # # validation_results = validate_reduced_model(
    # #     cable_lengths, 
    # #     load_params, 
    # #     network_params["admittance_parameters"],
    # #     save_path="reduced_model_validation3.pdf"
    # # )
    
    # #plot_nll_vs_ReZF_complex_mtl(cable_lengths, load_params, fault_params, 40)
    
    # #plot_nll_vs_L1_complex_mtl(cable_lengths, load_params, fault_params, 40)

    # # sorted_keys = ["fault_position"]

    


    
    # # ========================================================================
    # # RUN INFERENCE BASED ON SCENARIO
    # # ========================================================================
    # print(f"\n[SCENARIO: {SCENARIO}]")

    # if SCENARIO == "two_stage":
    #     # ----------------------------------------------------------------
    #     # TWO-STAGE WORKFLOW
    #     # Stage 1: Network identification (no fault) -> infer cable/load params
    #     # Stage 2: Fault localization -> use Stage 1 posteriors, infer fault params
    #     # ----------------------------------------------------------------
    #     stage1_results, stage2_results = run_two_stage_inference(
    #         cable_lengths=cable_lengths,
    #         load_params = load_params,
    #         admittance_params=admittance_params,
    #         fault_params=fault_params,
    #         snr_db=40,
    #         num_steps_stage1=500,
    #         num_steps_stage2=500,
    #         threshold=0.0,
    #     )
    #     losses_s1, param_history_s1, sorted_keys_s1 = stage1_results
    #     losses_s2, param_history_s2, sorted_keys_s2 = stage2_results

    # else:
    #     # ----------------------------------------------------------------
    #     # SINGLE-STAGE WORKFLOW (no_fault or with_fault)
    #     # ----------------------------------------------------------------
    #     # Run sensitivity analysis
    #     selected, sorted_keys = perform_load_sensitivity_analysis(
    #         load_params, fault_params, cable_lengths,
    #         threshold=0.0, scenario=SCENARIO,
    #         admittance_params=admittance_params
    #     )

    #     num_obs = 1

    #     # Generate observations based on scenario
    #     if SCENARIO == "no_fault":
    #         H1_clean = calculate_Hnw_nofault(cable_lengths, load_params)
    #     else:  # with_fault
    #         H1_clean = calculate_Hnw(cable_lengths, load_params, fault_params)

    #     # Add noise
    #     snr_db = 40
    #     snr_lin = 10.0 ** (snr_db / 10.0)
    #     sigpow = torch.mean(torch.abs(H1_clean)**2)
    #     var_f = sigpow / snr_lin
    #     std_f = torch.sqrt(var_f / 2)

    #     H1_noisy_c = H1_clean + std_f * torch.randn_like(H1_clean.real) + \
    #                  1j * std_f * torch.randn_like(H1_clean.imag)
    #     H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1)
    #     H1_noisy = torch.view_as_real(H1_noisy_c_expanded)

    #     # Select model based on scenario
    #     if SCENARIO == "no_fault":
    #         current_model = model_no_fault
    #     else:
    #         current_model = model_with_fault

    #     # Run SVI inference
    #     losses, param_history = run_inference(H1_noisy, current_model, guide, sorted_keys)

    #     # Plot results
    #     plot_param_convergence(param_history, losses, sorted_keys)
    #     plot_CI_and_pred_TF(param_history, sorted_keys, H1_clean)

    print("My program took", time.time() - start_time, "to run")
