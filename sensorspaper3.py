import numpy as np
import random
import pyro
import copy
import math
import pandas as pd
from pyro.distributions.transforms import SigmoidTransform, AffineTransform
from pyro.distributions import TransformedDistribution, constraints
from torch.distributions import Normal
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.infer.autoguide import AutoMultivariateNormal, AutoGuideList, AutoNormal

import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

import scipy as sp 
import numpy as np
import time
start_time = time.time()
torch.set_printoptions(precision=8)  # Show 8 decimal places

# ============================================================================
# SCENARIO CONFIGURATION
# ============================================================================
# Options:
#   - "no_fault": Network identification only (Stage 1) - infer load/cable params
#   - "with_fault": Fault localization only - infer fault params (assumes known network)
#   - "two_stage": Full workflow - Stage 1 (network ID) then Stage 2 (fault localization)
SCENARIO = "two_stage"
# ============================================================================

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

# ---- Define Network Constants ----
num_loads = 22
num_of_conductors = 4

# Set device FIRST before creating any tensors
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

#frequencies = torch.logspace(torch.log10(torch.tensor(2e6)), torch.log10(torch.tensor(10e6)), 500) #2-10MHz
#frequencies = torch.logspace(torch.log10(torch.tensor(150e3)), torch.log10(torch.tensor(30e6)), 200) #150KHz - 30MHz
frequencies = torch.logspace(torch.log10(torch.tensor(150e3, device=device)),
                              torch.log10(torch.tensor(500e3, device=device)), 200, device=device) #150KHz - 2MHz
freq_range_mhz = frequencies / 1e6
omega = 2 * torch.pi * frequencies
num_freqs = len(omega)

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
from scipy.linalg import expm
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
    


 
def generate_load_parameters_deterministic(num_loads, omega):
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
def calculate_Hnw_nofault(cable_lengths, sampled_params):
    """
    Calculate network Transfer Function Hnw given network parameters.
    
    Parameters:
    - cable_lengths (dict): Dictionary of cable lengths (30 lengths)
    - sampled_params (dict): Dictionary containing sampled load parameters (22 loads of either type 1, 2, or 3)
    
    Returns:
    - (1,1)st entry of H_nw (N, n, n) where N = num of frequency points and n = num of conductors - 1
    """
    r_w_servicepanel = network_params["conductor_radii"]["r_w_servicepanel"]["value"]
    r_w_room = network_params["conductor_radii"]["r_w_room"]["value"]
    # Initialize dictionary to store load admittance matrices calculated from sampled_params
    Y_loads = {}
    num_loads = len(sampled_params)

    # sampled_params_reindexed = {
    #     f"load_{i}": sampled_params[f"load_{num_loads - 1 - i}"]
    #     for i in range(num_loads)
    # }
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
    Stage 1. Healthy network (no fault) - only infers load and cable parameters.
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

    H1_pred_c = calculate_Hnw_nofault(cable_lengths, load_params).unsqueeze(0).expand(N, -1)
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
    Load/cable parameters are fixed to their inferred values from Stage 1. 
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

    # WITH FAULT forward model
    H1_pred_c = calculate_Hnw(cable_lengths, load_params, fault_params).unsqueeze(0).expand(N, -1)
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


def model(H1_noisy):
    """
    Wrapper that selects the appropriate model based on SCENARIO config.
    """
    if SCENARIO == "no_fault":
        return model_no_fault(H1_noisy)
    else:
        return model_with_fault(H1_noisy)



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



def estimate_sigmoid_normal_means(loc, scale, num_samples=2048):
    """
    Returns:
      sigmoid_loc: sigmoid(E[z]) = sigmoid(loc)
      mc_mean:     E[sigmoid(z)] estimated by Monte Carlo sampling
    """
    with torch.no_grad():
        sigmoid_loc = torch.sigmoid(loc).item()
        q = TransformedDistribution(dist.Normal(loc, scale), [SigmoidTransform()])
        mc_mean = q.sample((num_samples,)).mean().item()

    return sigmoid_loc, mc_mean


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


def run_two_stage_inference(cable_lengths, load_params, fault_params,
                            omega, snr_db=40, num_steps_stage1=20000, num_steps_stage2=20000,
                            threshold=0.0):
    """
    Run the complete two-stage inference workflow:

    Stage 1 (Network Identification):
        - No fault present
        - Infer cable/load parameters
        - Use calculate_Hnw_nofault

    Stage 2 (Fault Localization):
        - Fault is present
        - Fix cable/load params to Stage 1 posterior means
        - Only infer fault parameters
        - Use calculate_Hnw

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

    # Configure for no-fault scenario
    # Turn OFF fault parameter inference
    for fault_name in network_params["fault_parameters"]:
        network_params["fault_parameters"][fault_name]["inferred"] = False

    # Run sensitivity analysis to select which load/cable params to infer
    print(f"\nRunning Stage 1 Sensitivity Analysis...")
    selected_s1, sorted_keys_s1 = perform_load_sensitivity_analysis(
        load_params, fault_params, cable_lengths,
        threshold=threshold, scenario="no_fault"
    )
    # Generate Stage 1 observations (no fault)
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

    # Run Stage 1 inference
    print(f"\nRunning Stage 1 SVI inference ({num_steps_stage1} steps)...")
    losses_s1, param_history_s1 = run_inference(
        H1_noisy_s1_real, model_no_fault, guide, sorted_keys_s1, num_steps=num_steps_stage1
    )

    # Extract posterior means and update network_params
    print("\n--- Extracting, Updating, and Plotting ---")
    posterior_means_s1 = extract_posterior_means(param_history_s1)
    update_network_params_from_posterior(posterior_means_s1)
    plot_param_convergence(param_history_s1, losses_s1, sorted_keys_s1, "no_fault")
    plot_CI_and_pred_TF(param_history_s1, H1_clean_s1, "no_fault")

    
    # ========================================================================
    # STAGE 2: Fault Localization (With Fault)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: Fault Localization (With Fault)")
    print("="*70)

    # Configure for with-fault scenario
    # Network params are now fixed (inferred=False from Stage 1 update)
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
    #optimizer = pyro.optim.ClippedAdam({"lr": 0.01, "clip_norm": 5.0})
    optimizer = optim.Adagrad({"lr": 0.2})
    optimizer2 = pyro.optim.Adam({"lr": 0.2})
    auto_guide = AutoMultivariateNormal(model)
    auto_guide2 = AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=20))
    top_20_most_sensitive = [
        key.replace(".", "_") + "_loc"
        for key in sorted_keys[:20]
    ]    
    losses = []
    param_history = defaultdict(list) #Contains tensors

    # Initialize parameters by running guide once
    guide(H1_noisy)

    # Save initial parameter values 
    param_store = pyro.get_param_store()
    for name, value in param_store.items():
        param_history[name].append(value.detach().clone())
    
    for step in range(num_steps):
        loss = svi.step(H1_noisy)
        losses.append(loss)

        param_store = pyro.get_param_store()
        for name, value in param_store.items():
            param_history[name].append(value.detach().clone())
        
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
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"svi_elbo_loss_{scenario}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Parameter Plots
    plt.figure(figsize=(14, 5))

    # (1) loc trajectories (normalized) - apply sigmoid since param_history stores raw loc
    plt.subplot(1, 2, 1)
    loc_keys = [k for k in param_history if "loc" in k]
    for key in loc_keys:
        vals = torch.stack(param_history[key])
        vals_sigmoid = torch.sigmoid(vals)
        plt.plot(vals_sigmoid.numpy(), alpha=0.7)

    plt.title("Mean Convergence (normalized)")
    plt.xlabel("SVI step")
    plt.ylabel("Variational mean (sigmoid(loc))")
    plt.grid(True)

    # (2) scale trajectories
    plt.subplot(1, 2, 2)
    scale_keys = [k for k in param_history if "scale" in k]
    for key in scale_keys:
        vals = torch.stack(param_history[key])
        plt.plot(vals.numpy(), alpha=0.7)
    plt.title("Scale Convergence")
    plt.xlabel("SVI step")
    plt.ylabel("Variational scale (std dev)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"svi_param_convergence_{scenario}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --------- PRINT FINAL MEANS FOR TOP 20 MOST SENSITIVE ----------
    rows = []
    squared_errors = []
    top_20_keys = sorted_keys[:20]  # Only use top 20 most sensitive

    print("\n=== Final Posterior Means (Top 20 Most Sensitive, normalized) ===")
    print(f"{'Parameter':30s} | {'q-mean':>8s} | {'true':>8s} | {'error':>10s}")
    print("-" * 65)

    # Custom guide: name_loc
    for name in top_20_keys:
        pyro_loc_key = name.replace(".", "_") + "_loc"
        if pyro_loc_key not in param_history:
            print(f"{name:30s} | q-mean = N/A")
            continue
        raw_loc = param_history[pyro_loc_key][-1]
        q_mean = torch.sigmoid(raw_loc).item()

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
        df.to_csv(f"posterior_means_{scenario}.csv", index=False)
        print(f"\nSaved to posterior_means_{scenario}.csv")

def perform_load_sensitivity_analysis(load_params, fault_params, cable_lengths, threshold, scenario):
    if scenario == "with_fault":
        nominal_H = calculate_Hnw(cable_lengths, load_params, fault_params) #[200] complex
    else:
        nominal_H = calculate_Hnw_nofault(cable_lengths, load_params)
    variations = {}

    # Analyze load parameters
    for load_name, param_dict in network_params["loads"].items():
        for param_name, param_info in param_dict.items():
            if not param_info["inferred"]:
                continue

            lo, hi = param_info["range"]
            values = np.linspace(lo, hi, 10)
            param_variations = []

            for val in values:
                perturbed_loads = copy.deepcopy(load_params)
                perturbed_loads[load_name][param_name] = torch.tensor(val, device=device)
                if scenario == "with_fault":
                    H_var = calculate_Hnw(cable_lengths, perturbed_loads, fault_params)
                else:
                    H_var = calculate_Hnw_nofault(cable_lengths, perturbed_loads)
                #L2 norm diff between transfer functions for each perturbed param
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)
            #Each param is perturbed 10 times in parameter range, sum total variation
            total_var = sum(param_variations)
            key = f"{load_name}.{param_name}"
            variations[key] = total_var

    # Analyze cable length parameters
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
                perturbed_fault[fault_name] = torch.tensor(val)

                H_var = calculate_Hnw(cable_lengths, load_params, perturbed_fault)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

            total_var = sum(param_variations)
            variations[fault_name] = total_var

    # Normalize
    total_sum = sum(variations.values())
    normalized = {k: v / total_sum for k, v in variations.items()}
    #normalized contains sensitivites = v / total_sum that sum up to 1 for every parameter, lar
    # larger this number the more sensitive

    # Select parameters above threshold
    selected = [k for k, v in normalized.items() if v > threshold]
    flag = True
    print("\n--- Network Parameter Sensitivity Analysis ---")
    for k in sorted(normalized, key=normalized.get, reverse=True):
        if normalized[k] <= threshold and flag:
            print("Parameters after this are below the threshold")
            flag = False
        print(f"{k}: {normalized[k]*100:.2f}%")

    print(f"\nSelected parameters (>{threshold*100:.2f}%): {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    # Update network_params in-place to only infer selected ones
    for load_name, param_dict in network_params["loads"].items():
        for param_name in param_dict:
            key = f"{load_name}.{param_name}"
            network_params["loads"][load_name][param_name]["inferred"] = key in selected #bool

    for cable_name in network_params["cable_lengths"]:
        network_params["cable_lengths"][cable_name]["inferred"] = cable_name in selected
    if scenario == "with_fault":
        for fault_name in network_params["fault_parameters"]:
            network_params["fault_parameters"][fault_name]["inferred"] = fault_name in selected

    sorted_keys = sorted(normalized, key=normalized.get, reverse=True)
    #key=normalized.get means sort keys by their values  and reverse=True means sort from largest to smallest sensitivity (value)
    return selected, sorted_keys

def plot_CI_and_pred_TF(param_history, true_tf, scenario, num_samples=200):
    true_tf_db = 20*torch.log10(torch.abs(true_tf))
    tf_samples = []

    for i in range(num_samples):
        # Build sampled cable_lengths and load_params
        sampled_cable_lengths = {}
        sampled_load_params = {}
        sampled_fault_params = {}

        for load_name, params in network_params["loads"].items():
            sampled_load_params[load_name] = {}
            for param_name, param_info in params.items():
                pyro_key = f"{load_name}_{param_name}_loc"
                if pyro_key in param_history:  # Check if was inferred (exists in param_history)
                    lo, hi = param_info["range"]
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{load_name}_{param_name}_scale"][-1]
                    z = torch.normal(mean=loc, std=scale)
                    sampled_load_params[load_name][param_name] = torch.tensor(denormalize(torch.sigmoid(z).item(), lo, hi), device=device)
                else:
                    sampled_load_params[load_name][param_name] = torch.tensor(param_info["value"], device=device)

        for cable_name, cable_info in network_params["cable_lengths"].items():
            pyro_key = f"{cable_name}_loc"
            if pyro_key in param_history:  # Check if was inferred
                lo, hi = cable_info.get("infer_range", cable_info["range"])
                loc = param_history[pyro_key][-1]
                scale = param_history[f"{cable_name}_scale"][-1]
                z = torch.normal(mean=loc, std=scale)
                sampled_cable_lengths[cable_name] = torch.tensor(denormalize(torch.sigmoid(z).item(), lo, hi), device=device)
            else:
                sampled_cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

        if scenario == "with_fault":
            for fault_name, fault_info in network_params["fault_parameters"].items():
                pyro_key = f"{fault_name}_loc"
                if pyro_key in param_history:  # Check if was inferred
                    lo, hi = fault_info["range"]
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{fault_name}_scale"][-1]
                    z = torch.normal(mean=loc, std=scale)
                    sampled_fault_params[fault_name] = torch.tensor(denormalize(torch.sigmoid(z).item(), lo, hi), device=device)
                else:
                    sampled_fault_params[fault_name] = torch.tensor(fault_info["value"], device=device)
                    
        # Compute TF with sampled parameters
        if scenario == "with_fault":
            H_sample = calculate_Hnw(sampled_cable_lengths, sampled_load_params, sampled_fault_params)
        else:
            H_sample = calculate_Hnw_nofault(sampled_cable_lengths, sampled_load_params)
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
    plt.savefig(f'tf_posterior_CI_{scenario}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to tf_posterior_CI_{scenario}.png")
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

if __name__ == '__main__':
    start_time = time.time()
    # Assume network_params is already initialized
    #total_params, load_types = generate_load_parameters(num_loads, omega)
    total_params, load_types = generate_load_parameters_deterministic(num_loads, omega)
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
    # Convert from network_params to dict of tensors (on correct device)
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
    #plot_nll_vs_ReZF_complex_mtl(cable_lengths, load_params, fault_params, 40)
    
    #plot_nll_vs_L1_complex_mtl(cable_lengths, load_params, fault_params, 40)
    #rwrwr

    # sorted_keys = ["fault_position"]
    # num_obs = 1


    # H1_clean = calculate_Hnw(cable_lengths, load_params, fault_params)

    # # Add noise
    # snr_db = 40
    # snr_lin = 10.0 ** (snr_db / 10.0)
    # sigpow = torch.mean(torch.abs(H1_clean)**2)
    # var_f = sigpow / snr_lin
    # std_f = torch.sqrt(var_f / 2)

    # H1_noisy_c = H1_clean + std_f * torch.randn_like(H1_clean.real) + \
    #                 1j * std_f * torch.randn_like(H1_clean.imag)
    # H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1)
    # H1_noisy = torch.view_as_real(H1_noisy_c_expanded)

    # current_model = model_with_fault

    # # Run SVI inference
    # losses, param_history = run_inference(H1_noisy, current_model, guide, sorted_keys, num_steps=1000)


    # selected_s1, sorted_keys_s1 = perform_load_sensitivity_analysis(
    #     load_params, fault_params, cable_lengths,
    #     threshold=0.0, scenario="no_fault"
    # )
    







    
    # ========================================================================
    # RUN INFERENCE BASED ON SCENARIO
    # ========================================================================
    print(f"\n[SCENARIO: {SCENARIO}]")

    if SCENARIO == "two_stage":
        # ----------------------------------------------------------------
        # TWO-STAGE WORKFLOW
        # Stage 1: Network identification (no fault) -> infer cable/load params
        # Stage 2: Fault localization -> use Stage 1 posteriors, infer fault params
        # ----------------------------------------------------------------
        stage1_results, stage2_results = run_two_stage_inference(
            cable_lengths=cable_lengths,
            load_params=load_params,
            fault_params=fault_params,
            omega=omega,
            snr_db=40,
            num_steps_stage1=100,
            num_steps_stage2=100,
            threshold=0.0
        )
        losses_s1, param_history_s1, sorted_keys_s1 = stage1_results
        losses_s2, param_history_s2, sorted_keys_s2 = stage2_results

    else:
        # ----------------------------------------------------------------
        # SINGLE-STAGE WORKFLOW (no_fault or with_fault)
        # ----------------------------------------------------------------
        # Run sensitivity analysis
        selected, sorted_keys = perform_load_sensitivity_analysis(
            load_params, fault_params, cable_lengths,
            threshold=0.0, scenario=SCENARIO
        )

        num_obs = 1

        # Generate observations based on scenario
        if SCENARIO == "no_fault":
            H1_clean = calculate_Hnw_nofault(cable_lengths, load_params)
        else:  # with_fault
            H1_clean = calculate_Hnw(cable_lengths, load_params, fault_params)

        # Add noise
        snr_db = 40
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigpow = torch.mean(torch.abs(H1_clean)**2)
        var_f = sigpow / snr_lin
        std_f = torch.sqrt(var_f / 2)

        H1_noisy_c = H1_clean + std_f * torch.randn_like(H1_clean.real) + \
                     1j * std_f * torch.randn_like(H1_clean.imag)
        H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1)
        H1_noisy = torch.view_as_real(H1_noisy_c_expanded)

        # Select model based on scenario
        if SCENARIO == "no_fault":
            current_model = model_no_fault
        else:
            current_model = model_with_fault

        # Run SVI inference
        losses, param_history = run_inference(H1_noisy, current_model, guide, sorted_keys)

        # Plot results
        plot_param_convergence(param_history, losses, sorted_keys)
        plot_CI_and_pred_TF(param_history, sorted_keys, H1_clean)

    print("My program took", time.time() - start_time, "to run")
