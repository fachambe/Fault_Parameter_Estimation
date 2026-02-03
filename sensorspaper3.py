import numpy as np
import random
import pyro
from pyro.distributions.transforms import SigmoidTransform, AffineTransform
from pyro.distributions import TransformedDistribution, constraints
from torch.distributions import Normal
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution

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

# FIXED_LOAD_TYPES = [
#     1,  # load_0  R1-O2  Constant
#     3,  # load_1  R1-O3  Motor
#     1,  # load_2  R1-O4  Constant
#     3,  # load_3  R2-O1  Motor
#     3,  # load_4  R2-O2  Motor
#     3,  # load_5  R2-O3  Motor
#     3,  # load_6  R2-O4  Motor
#     1,  # load_7  R3-O1  Constant
#     1,  # load_8  R3-O2  Constant
#     1,  # load_9  R3-O3  Constant
#     3,  # load_10 R3-O4  Motor
#     2,  # load_11 R4-O1  Double RLC
#     3,  # load_12 R4-O2  Motor
#     3,  # load_13 R4-O3  Motor
#     2,  # load_14 R4-O4  Double RLC
#     3,  # load_15 R5-O1  Motor
#     1,  # load_16 R5-O2  Constant
#     2,  # load_17 R5-O3  Double RLC
#     3,  # load_18 R5-O4  Motor
#     1,  # load_19 R6-O1  Constant
#     3,  # load_20 R6-O2  Motor
#     3,  # load_21 R6-O3  Motor
# ]

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
    # Constants
    mu_0 = 4 * torch.pi * 1e-7
    sigma = 5.8 * 1e7
    epsilon = 3.19 * 1e-11
    dc = 4 * 1e-4 + 3.02 * r_w
    dc2 = torch.sqrt(torch.tensor(2.0)) * dc
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
    L = (mu_0 / (2 * torch.pi)) * torch.tensor([
        [2*np.log(dc / r_w), np.log((dc * dc2) / (dc * r_w)), np.log((dc * dc) / (dc2 * r_w))],
        [np.log((dc * dc2) / (dc * r_w)), 2*np.log(dc2 / r_w), np.log((dc2 * dc) / (dc * r_w))],
        [np.log( (dc * dc) / (dc2 * r_w)), np.log((dc2 * dc) / (dc * r_w)), 2*np.log(dc / r_w)]
    ], dtype=torch.complex64)

    L_new = L.unsqueeze(0).expand(num_freqs, -1, -1)
    C = mu_0 * epsilon * torch.linalg.inv(L) 
    C_new = C.unsqueeze(0).expand(num_freqs, -1, -1) 
    G_new = torch.zeros((num_freqs, n, n), dtype=torch.complex64)
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
    - Y_load (torch.Tensor): Load admittance matrix with shape (N, n, n)
    """
    Z12, Z13, Z23, ZG1, ZG2, ZG3 = load_params  # Unpack impedance values

    # Stack into a (N, 3, 3) tensor
    Y_load = torch.stack([
        torch.stack([1/ZG1 + 1/Z12 + 1/Z13, -1/Z12, -1/Z13], dim=-1),
        torch.stack([-1/Z12, 1/ZG2 + 1/Z12 + 1/Z23, -1/Z23], dim=-1),
        torch.stack([-1/Z13, -1/Z23, 1/ZG3 + 1/Z13 + 1/Z23], dim=-1)
    ], dim=-2)  # Shape: (N, 3, 3)
    return Y_load
def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    Z_rec = torch.tensor([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec
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

def loguniform(low, high, size=None):
    """ Generate samples from a log-uniform distribution. """
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))

# ---- Define Network Parameter Dictionary ----
network_params = {
    "cable_lengths": {  # 30 parameters, set all to 0.25
        f"l_w_{i}": {"value": denormalize(0.25, 2, 20), "inferred": True, "range": (2, 20), "infer_range": (6.0, 8.0)}
        for i in range(30)
    },
    "conductor_radii": {  # Fixed values, not inferred
        "r_w_servicepanel": {"value": denormalize(0.25, 1.03e-3, 2.06e-3), "inferred": False},
        "r_w_room": {"value": denormalize(0.25, 0.81e-3, 1.29e-3), "inferred": False}
    },
    "loads": {}  # Dynamically generated based on load type
}
# ---- Define Network Constants ----
num_loads = 22
num_of_conductors = 4
frequencies = torch.logspace(torch.log10(torch.tensor(150e3)), torch.log10(torch.tensor(30e6)), 200) #150KHz - 30MHz
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
Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3)
Y_rec = torch.linalg.inv(Z_rec)
Z_rec = Z_rec.unsqueeze(0).repeat(num_freqs, 1, 1)
Y_rec = Y_rec.unsqueeze(0).repeat(num_freqs, 1, 1)
 
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
            key: torch.tensor(val["value"], dtype=torch.complex64) for key, val in params.items()
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

def calculate_Hnw(cable_lengths, sampled_params):
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

def model(H1_noisy):
    # H1_noisy is (N, F, 2), float NOT COMPLEX
    N, F, _ = H1_noisy.shape  
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
                load_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                # Use middle of range (0.5) instead of true value for non-inferred params
                # Some params (like R_m2) have no range - use stored value for those
                if "range" in param_info:
                    min_val, max_val = param_info["range"]
                    load_dict[param_name] = torch.tensor(denormalize(0.25, min_val, max_val))
                else:
                    load_dict[param_name] = torch.tensor(param_info["value"])
        load_params[load_name] = load_dict

    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_info["inferred"]:
            # Use infer_range to make sure only +/-1M from true value
            infer_lo, infer_hi = cable_info.get("infer_range", cable_info["range"])
            norm_sample = pyro.sample(f"{cable_name}", dist.Uniform(0.0, 1.0))
            # Convert from [0,1] to physical value using infer_range
            physical_value = denormalize(norm_sample, infer_lo, infer_hi)
            cable_lengths[cable_name] = physical_value
        else:
            # Use middle of range (0.5) instead of true value for non-inferred cables
            min_val, max_val = cable_info["infer_range"]
            cable_lengths[cable_name] = torch.tensor(denormalize(0.25, min_val, max_val))

        
    H1_pred_c = calculate_Hnw(cable_lengths, load_params).unsqueeze(0).expand(N, -1) #[N, F] complex 
    H1_pred = torch.view_as_real(H1_pred_c) #[N, F, 2] real 
    with pyro.plate("data", N):  # Outer plate, tells pyro all N observations are independent
        pyro.sample(
        "obs",
        # dist.Independent(
        #     dist.StudentT(df = 1.0, loc=H1_pred_expanded, scale=0.4),
        #     reinterpreted_batch_ndims=1
        # ),
        # obs=H1_noisy
        #0.00022655
        dist.Independent(
            dist.Normal(loc=H1_pred, scale=0.00022655),
            reinterpreted_batch_ndims=2 #event dims are (F, 2)
        ),
        obs=H1_noisy
    )

# --------------------------
# Variational Guide
# --------------------------

def guide(H1_noisy):
    for load_name, params in network_params["loads"].items():
        for param_name, param_info in params.items():
            if not param_info["inferred"]:
                continue

            full_name = f"{load_name}_{param_name}"
            loc = pyro.param(f"{full_name}_loc", torch.tensor(0.0))  # std normal
            scale = pyro.param(f"{full_name}_scale", torch.tensor(1.0), constraint=constraints.positive)

            q = TransformedDistribution(
                dist.Normal(loc, scale),
                [SigmoidTransform()]
            )
            
            pyro.sample(full_name, q)
    for key, info in network_params["cable_lengths"].items():
        if not info["inferred"]:
            continue

        loc = pyro.param(f"{key}_loc", torch.tensor(0.0))
        scale = pyro.param(f"{key}_scale", torch.tensor(1.0), constraint=constraints.positive)

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

# --------------------------
# Inference Function
# --------------------------
def run_inference(H1_noisy, model, guide, sorted_keys, num_steps=50):
    # H1_noisy is (N, F, 2), float NOT COMPLEX
    pyro.clear_param_store()
    #optimizer = pyro.optim.ClippedAdam({"lr": 0.01, "clip_norm": 5.0})
    optimizer = optim.Adagrad({"lr": 0.2})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=20))

    losses = []
    param_history = defaultdict(list) #Contains tensors

    # Initialize parameters by running guide once
    guide(H1_noisy)

    # Save initial parameter values (should all be 0.5)
    param_store = pyro.get_param_store()
    for name, value in param_store.items():
        if name.endswith("_loc"):
            param_history[name].append(torch.sigmoid(value).detach().clone())
        else:
            param_history[name].append(value.detach().clone())

    for step in range(num_steps):
        loss = svi.step(H1_noisy)
        losses.append(loss)

        param_store = pyro.get_param_store()
        #Save param values every step but need to -0.25 because technically should converge to 0.25 for cables
        for name, value in param_store.items():
            if name.endswith("_loc"):
                if name.replace("_loc", "") in network_params["cable_lengths"]:

                # scale_name = name.replace("_loc", "_scale")
                # loc = value
                # scale = param_store[scale_name]
                # mc_mean = mc_theta_mean_from_loc_scale(loc, scale, num_samples=256)
                #param_history[name].append(mc_mean.detach().clone())
                    param_history[name].append(torch.sigmoid(value).detach().clone())
                else:
                    param_history[name].append(torch.sigmoid(value).detach().clone())
            else:
                param_history[name].append(value.detach().clone())
        if step % 25 == 0 or step == 0:
            print(f"\n=== Top 20 Sensitive Parameters (Variational Means) at step {step} | ELBO is {loss} ===")
            for name in sorted_keys[:20]:
                #load_0.C_m_leak -> load_0_C_m_leak_loc
                pyro_loc = name.replace(".", "_") + "_loc"
                pyro_scale = name.replace(".", "_") + "_scale"

                if pyro_loc in param_store and pyro_scale in param_store:
                    loc = param_store[pyro_loc]
                    scale = param_store[pyro_scale]
                    sigmoid_loc, mc_mean = estimate_sigmoid_normal_means(loc, scale, num_samples=2048)
                    if(name in network_params["cable_lengths"]):
                        print(
                            f"{name:30s} | sigmoid(loc) = {sigmoid_loc:.6f} | E_q[sigmoid(z)] ≈ {mc_mean:.6f} | True value = {0.25}"
                        )
                    else:
                        print(
                            f"{name:30s} | sigmoid(loc) = {sigmoid_loc:.6f} | E_q[sigmoid(z)] ≈ {mc_mean:.6f} | True value = {0.25}"
                        )
                else:
                    print(f"{name:30s} | N/A")


    print("Inference complete.")
    return losses, param_history


def plot_param_convergence(param_history, losses, sorted_keys):
    #Add ELBO plot too
    loc_keys = [k for k in param_history if k.endswith("_loc")]
    scale_keys = [k for k in param_history if k.endswith("_scale")]

    # --------- ELBO Plot (separate figure) ----------
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("SVI ELBO Loss")
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("svi_elbo_loss_allparamstest.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --------- Parameter Plots (2 panels) ----------
    plt.figure(figsize=(14, 5))

    # (1) loc trajectories (normalized)
    plt.subplot(1, 2, 1)
    for key in loc_keys:
        vals = torch.stack(param_history[key]) #Convert list of scalar tensor to a single vector tensor
        plt.plot(vals.numpy(), alpha=0.7)
    plt.title("Mean Convergence (normalized)")
    plt.xlabel("SVI step")
    plt.ylabel("Variational mean (sigmoid(loc))")
    plt.grid(True)

    # (2) scale trajectories
    plt.subplot(1, 2, 2)
    for key in scale_keys:
        vals = torch.stack(param_history[key])
        plt.plot(vals.numpy(), alpha=0.7)
    plt.title("Scale Convergence")
    plt.xlabel("SVI step")
    plt.ylabel("Variational scale (std dev)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("svi_param_convergence_allparamtest.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --------- PRINT FINAL MEANS RANKED BY SENSITIVITY ----------
    # Use last recorded loc value for each parameter
    names_to_print = sorted_keys

    print("\n=== Final Posterior Means (normalized), ranked by sensitivity ===")

    for name in names_to_print:
        pyro_loc_key = name.replace(".", "_") + "_loc"   # e.g. load_0.C_m_leak-> load_0_C_m_leak_loc 

        if pyro_loc_key not in param_history:
            print(f"{name:30s} | q-mean = N/A (not in param_history)")
            continue

        q_mean = float(param_history[pyro_loc_key][-1])     #  pull last entry from param history
        print(f"{name:30s} | q-mean = {q_mean:.4f}")

def perform_load_sensitivity_analysis(load_params, network_params, cable_lengths, omega, threshold=0.01):
    import copy

    nominal_H = calculate_Hnw(cable_lengths, load_params) #[200] complex
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
                perturbed_loads[load_name][param_name] = torch.tensor(val)

                H_var = calculate_Hnw(cable_lengths, perturbed_loads)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

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

            H_var = calculate_Hnw(perturbed_cables, load_params)
            diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
            param_variations.append(diff)

        total_var = sum(param_variations)
        variations[cable_name] = total_var

    # Normalize
    total_sum = sum(variations.values())
    normalized = {k: v / total_sum for k, v in variations.items()}
    #contains sensitivites = v / total_sum now that sum up to 1, larger this number the more sensitive

    # Select parameters above threshold
    selected = [k for k, v in normalized.items() if v > threshold]

    print("\n--- Load & Cable Length Sensitivity Analysis ---")
    for k in sorted(normalized, key=normalized.get, reverse=True):
        print(f"{k}: {normalized[k]*100:.2f}%")

    print(f"\nSelected parameters (>{threshold*100:.2f}%): {selected}")
    print(f"Number of selected parameters: {len(selected)}")


    # Update network_params in-place to only infer selected ones
    for load_name, param_dict in network_params["loads"].items():
        for param_name in param_dict:
            key = f"{load_name}.{param_name}"
            network_params["loads"][load_name][param_name]["inferred"] = key in selected

    for cable_name in network_params["cable_lengths"]:
        network_params["cable_lengths"][cable_name]["inferred"] = cable_name in selected

    sorted_keys = sorted(normalized, key=normalized.get, reverse=True)
    #key=normalized.get means sort keys by their values  and reverse=True means sort from largest to smallest sensitivity (value)
    return selected, sorted_keys

def plot_CI_and_pred_TF(param_history, sorted_keys, true_tf, num_samples=200):
    # Convert true_tf to magnitude in dB if it's complex
    if true_tf.is_complex():
        true_tf_db = 20 * torch.log10(torch.abs(true_tf))
    else:
        true_tf_db = true_tf  # Assume already in dB

    # Collect final posterior loc and scale for each inferred parameter
    posterior_params = {}
    for name in sorted_keys:
        pyro_loc_key = name.replace(".", "_") + "_loc"
        pyro_scale_key = name.replace(".", "_") + "_scale"
        if pyro_loc_key not in param_history:
            continue
        # Get final loc and scale values
        final_loc = float(param_history[pyro_loc_key][-1])
        final_scale = float(param_history[pyro_scale_key][-1])

        #Determine the range for denormalization
        if name in network_params["cable_lengths"]:
            info = network_params["cable_lengths"][name]
            lo, hi = info.get("infer_range", info["range"])
        else:
            load_name, param_name = name.split(".")
            info = network_params["loads"][load_name][param_name]
            lo, hi = info["range"]
        
        posterior_params[name] = {
            "loc": final_loc,
            "scale": final_scale,
            "range": (lo, hi)
        }

    # Sample from posterior and compute TF for each sample
    tf_samples = []

    for i in range(num_samples):
        # Build sampled cable_lengths and load_params
        sampled_cable_lengths = {}
        sampled_load_params = {}
        
        # Initialize load_params structure from network_params
        for load_name, params in network_params["loads"].items():
            sampled_load_params[load_name] = {}
            for param_name, param_info in params.items():
                # Use middle of range (0.5) for non-inferred params (not true value)
                # Some params (like R_m2) have no range - use stored value for those
                if "range" in param_info:
                    min_val, max_val = param_info["range"]
                    sampled_load_params[load_name][param_name] = torch.tensor(denormalize(0.5, min_val, max_val))
                else:
                    sampled_load_params[load_name][param_name] = torch.tensor(param_info["value"])
            # Initialize cable_lengths from network_params
        for cable_name, cable_info in network_params["cable_lengths"].items():
            # Use middle of range (0.5) for non-inferred cables (not true value)
            min_val, max_val = cable_info["range"]
            sampled_cable_lengths[cable_name] = torch.tensor(denormalize(0.5, min_val, max_val))

        # Now override with samples from posterior for inferred params
        for name, pinfo in posterior_params.items():
            loc, scale = pinfo["loc"], pinfo["scale"]
            lo, hi = pinfo["range"]
            
            # Sample from Normal(loc, scale) then apply sigmoid
            z = torch.normal(mean=torch.tensor(loc), std=torch.tensor(scale))
            norm_sample = torch.sigmoid(z).item()
            physical_value = denormalize(norm_sample, lo, hi)
            
            if name in network_params["cable_lengths"]:
                sampled_cable_lengths[name] = torch.tensor(physical_value)
            else:
                load_name, param_name = name.split(".")
                sampled_load_params[load_name][param_name] = torch.tensor(physical_value)
        
        # Compute TF with sampled parameters
        H_sample = calculate_Hnw(sampled_cable_lengths, sampled_load_params)
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
    plt.savefig('tf_posterior_CI2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to tf_posterior_CItest.png")
    return tf_mean, tf_lower, tf_upper
# --------------------------
# Entry Point
# --------------------------
if __name__ == '__main__':
    start_time = time.time()
    # random.seed(45)
    # np.random.seed(45)
    # torch.manual_seed(45)

    # Assume network_params is already initialized
    #total_params, load_types = generate_load_parameters(num_loads, omega)
    total_params, load_types = generate_load_parameters_deterministic(num_loads, omega)
    num_cable_params = len(network_params["cable_lengths"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params}")
    print(f"Load type distribution: {load_types}")

    # Fixed cable lengths and load parameters at 0.25
    cable_lengths = {
        key: torch.tensor(val["value"]) for key, val in network_params["cable_lengths"].items()
    }
    load_params = {
        load_name: {
            param_name: torch.tensor(param_info["value"])
            for param_name, param_info in params.items()
        }
        for load_name, params in network_params["loads"].items()
    }
    selected, sorted_keys = perform_load_sensitivity_analysis(load_params, network_params, cable_lengths, omega)
    
    # Force only ONE inferred parameter
    # selected = ["l_w_4", "l_w_25", "load_1.C_m_leak", "l_w_28", "l_w_24"]
    # sorted_keys = ["l_w_4", "l_w_25", "load_1.C_m_leak", "l_w_28", "l_w_24"]
    # # selected = ["load_15.C_m_leak"]
    # # sorted_keys = ["load_15.C_m_leak"]
    # # selected = ["load_1.C_m"]
    # # sorted_keys = ["load_1.C_m"]
    

    # #Turn OFF inference for everything
    # for load_name, params in network_params["loads"].items():
    #     for p_name, p_info in params.items():
    #         p_info["inferred"] = False
    # for cable_name, cable_info in network_params["cable_lengths"].items():
    #         print("cable name", cable_info)
    #         cable_info["inferred"] = False
    # print("asdsad", network_params["cable_lengths"].items())
    
    
    # # Turn ON inference only
    # network_params["loads"]["load_1"]["C_m_leak"]["inferred"] = True
    # network_params["cable_lengths"]["l_w_4"]["inferred"] = True
    # network_params["cable_lengths"]["l_w_25"]["inferred"] = True
    # network_params["cable_lengths"]["l_w_28"]["inferred"] = True
    # network_params["cable_lengths"]["l_w_24"]["inferred"] = True

    # network_params["loads"]["load_15"]["C_m_leak"]["inferred"] = True
    # network_params["loads"]["load_1"]["C_m"]["inferred"] = True
    
    num_obs = 1
    H1_clean = calculate_Hnw(cable_lengths, load_params) #complex [200]

    snr_db = 40
    snr_lin = 10.0 ** (snr_db / 10.0)

    sigpow  = torch.mean(torch.abs(H1_clean)**2)          
    var_f   = (sigpow / snr_lin)                     
    std_f   = torch.sqrt(var_f / 2) 

    H1_noisy_c = H1_clean + std_f*torch.randn_like(H1_clean.real) + 1j*std_f*torch.randn_like(H1_clean.imag) #[F] complex
    H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1) #[N, F] complex
    H1_noisy = torch.view_as_real(H1_noisy_c_expanded) #[N, F, 2] real
    
    # print("H1_clean", H1_clean)
    # print("shape", H1_clean.shape)
    
    # #print("H1_clean", H1_clean)
    # #print("H2_clean", H2_clean)
    # plt.figure(figsize=(8,6))
    # plt.plot(freq_range_mhz, 20*torch.log10(H1_clean), label=r"$H_{1,1}$ (Model)", color='b')
    # plt.plot(freq_range_mhz, 20*torch.log10(obs), label=r"$H_{1,1}$ (Model)", color='r')
    # plt.xscale("log")
    # plt.xlabel("Frequency (MHz)", fontsize=12)
    # plt.ylabel(r"Magnitude (dB)", fontsize=12)
    # plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.show()
    

    # Run SVI inference
    losses, param_history = run_inference(H1_noisy, model, guide, sorted_keys)
    # Plot the parameter convergence
    plot_param_convergence(param_history, losses, sorted_keys)
    #Confidence Interval + Predicted TF
    plot_CI_and_pred_TF(param_history, sorted_keys, H1_clean)
    print("My program took", time.time() - start_time, "to run")
