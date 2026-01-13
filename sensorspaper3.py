import numpy as np
import random
import matplotlib.pyplot as plt
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
    Z12 = Z13 = Z23 = torch.full_like(omega, R_const.item(), dtype=torch.complex64)  # Convert to array
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
        f"l_w_{i}": {"value": denormalize(0.25, 2, 20), "inferred": True, "range": (2, 20)}
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
    H_nw_magnitude_db_1 = 20 * torch.log10(torch.abs(H1[:, 0, 0]))    


    ABCD1 = calculate_cable_transmission_matrix(R_r, L_r, C_r, G_r, cable_lengths[f"l_w_{0}"], omega)

    I3 = np.eye(3, dtype=complex)
    I3 = np.repeat(I3[np.newaxis, :, :], num_freqs, axis=0)
    Zeros3 = np.zeros(( num_freqs, 3, 3), dtype=complex)

    Y0 = Y_loads["load_0"].detach().cpu().numpy()

    # Ensure complex dtype (important if it might be real sometimes)
    Y0 = Y0.astype(np.complex128, copy=False)
    ABCD2 = np.block([
        [I3, Zeros3],
        [-Y0, I3]
    ])
    ABCD3 = calculate_cable_transmission_matrix(R_r, L_r, C_r, G_r, cable_lengths[f"l_w_{1}"], omega)
    #ABCD4 = calculate_cable_transmission_matrix(R_s, L_s, C_s, G_s, cable_lengths[f"l_w_{1}"], omega)
    h2 = calculate_H_nw(ABCD1 @ ABCD2 @ ABCD3, num_of_conductors-1, Z_rec)
    H_nw_magnitude_db_2 = 20 * np.log10(np.abs(h2[:, 0, 0]))  # Convert magnitude to dB
 
    
    return H_nw_magnitude_db_1

def model(H1_noisy):
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                min_val, max_val = param_info["range"]
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
                load_dict[param_name] = denormalize(norm_sample, min_val, max_val)
            else:
                load_dict[param_name] = torch.tensor(param_info["value"])
        load_params[load_name] = load_dict

    # cable_lengths = {
    #     key: torch.tensor(param["value"]) for key, param in network_params["cable_lengths"].items()
    # }
    cable_lengths = {
        key: param["value"] for key, param in network_params["cable_lengths"].items()
    }
    H1_pred = calculate_Hnw(cable_lengths, load_params) 
    N, D = H1_noisy.shape  # e.g., (1000, 200)

    H1_pred_expanded = H1_pred.unsqueeze(0).expand(N, -1)
    #print("what is this shape", H1_pred_expanded.shape)
    with pyro.plate("data", N):  # Outer plate, tells pyro the 1000 observations are independent
        pyro.sample(
        "obs",
        dist.Independent(
            dist.StudentT(df = 1.0, loc=H1_pred_expanded, scale=0.4),
            reinterpreted_batch_ndims=1
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
# --------------------------
# Inference Function
# --------------------------
def run_inference(H1_noisy, model, guide):
    pyro.clear_param_store()
    optimizer = optim.Adagrad({"lr": 0.05})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))

    num_steps = 10000
    losses = []
    param_history = defaultdict(list)

    for step in range(num_steps):
        loss = svi.step(H1_noisy)
        losses.append(loss)

        # Track all param values
        for name, value in pyro.get_param_store().items():
            param_history[name].append(value.item())

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"\n[Step {step}] ELBO Loss = {loss:.4f}")
            print("Variational parameter means (normalized via sigmoid):")
            for name, value in pyro.get_param_store().items():
                if name.endswith("_loc"):
                    normalized_mean = torch.sigmoid(value).item()
                    print(f"  {name}: {normalized_mean:.4f}")
            
        # # Print progress every 10 steps
        # if step % 10 == 0:
        #     print(f"\n[Step {step}] ELBO Loss = {loss:.4f}")
        #     print("Current parameter means (normalized):")
        #     print("Parameter means sorted by sensitivity:")
        #     for key in sorted_keys:
        #         # Convert "load_0.C_m_leak" → "load_0_C_m_leak"
        #         pyro_key = key.replace(".", "_") + "_loc"
        #         if pyro_key in pyro.get_param_store():
        #             val = pyro.get_param_store()[pyro_key]
        #             normalized_val = val.sigmoid().item()
        #             print(f"  {pyro_key}: {normalized_val:.4f}")

    print("\nInference complete.")
    return losses, param_history


def plot_param_convergence(param_history):
    loc_keys = [k for k in param_history if k.endswith("_loc")]
    scale_keys = [k for k in param_history if k.endswith("_scale")]

    plt.figure(figsize=(12, 5))

    # --- Plot loc ---
    plt.subplot(1, 2, 1)
    for key in loc_keys:
        norm_vals = [torch.sigmoid(torch.tensor(x)).item() for x in param_history[key]]
        plt.plot(norm_vals, label=key)
    plt.title("Mean Convergence")
    plt.xlabel("SVI step")
    plt.ylabel("Mean (normalized)")
    plt.grid(True)

    # --- Plot scale ---
    plt.subplot(1, 2, 2)
    for key in scale_keys:
        plt.plot(param_history[key], label=key)
    plt.title("Variance Convergence")
    plt.xlabel("SVI step")
    plt.ylabel("Scale (std dev)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def print_variational_means_by_sensitivity(sorted_keys, param_history):
    print("\n--- Variational Means Ordered by Sensitivity ---")

    for key in sorted_keys:
        loc_key = f"{key}_loc"

        if loc_key not in param_history:
            print(f"{key}: NOT INFERRED")
            continue

        # Final unconstrained mean
        final_loc = param_history[loc_key][-1]

        # Convert to tensor if needed
        if not torch.is_tensor(final_loc):
            final_loc = torch.tensor(final_loc)

        # Map to normalized space (same as plotting)
        normalized_mean = torch.sigmoid(final_loc).item()

        print(f"{key:<30s}  mean = {normalized_mean:.6f}")

def perform_load_sensitivity_analysis(load_params, network_params, cable_lengths, omega, threshold=0.005):
    import copy

    nominal_H = calculate_Hnw(cable_lengths, load_params)
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
    return selected, sorted_keys

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
    
    # Only infer top 3 from sensitivity
    # selected_keys = {"load_0.C_m_leak", "load_20.C_m_leak", "load_2.C_leak", "load_0.C_m", "load_1.C_m_leak", "load_21.C_leak", "load_20.C_m", "load_19.C_leak", "load_10.C_d_leak", "load_4.C_d_leak"}
    # for load_name, params in network_params["loads"].items():
    #     for param_name, param_info in params.items():
    #         full_key = f"{load_name}.{param_name}"
    #         param_info["inferred"] = full_key in selected_keys


    
    
    num_obs = 1000
    H1_clean = calculate_Hnw(cable_lengths, load_params)
    # print("H1_clean", H1_clean)
    # #print("H2_clean", H2_clean)
    # plt.figure(figsize=(8,6))
    # plt.plot(freq_range_mhz, H1_clean, label=r"$H_{1,1}$ (Model)", color='b')
    # #plt.plot(freq_range_mhz, H2_clean, label=r"$H_{1,1}$ (Model)", color='r')
    # plt.xscale("log")
    # plt.xlabel("Frequency (MHz)", fontsize=12)
    # plt.ylabel(r"Magnitude (dB)", fontsize=12)
    # plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.show()
    
    H1_noisy = torch.stack([
        H1_clean + torch.tensor(np.random.normal(0.0, 1.0, size=H1_clean.shape), dtype=torch.float32)
        for _ in range(num_obs)
    ])
    # Run SVI inference
    losses, param_history = run_inference(H1_noisy, model, guide)
    # Plot the parameter convergence
    plot_param_convergence(param_history)

    print("My program took", time.time() - start_time, "to run")


# def model(H1_noisy):
#     """ Pyro model defining priors and likelihood for H1_noisy """
#     cable_lengths = {}
#     load_params = {}

#     for key, param in network_params["cable_lengths"].items():
#         min_phys, max_phys = param["range"]
#         # norm_center = 0.25
#         # delta_norm = 1.0 / (max_phys - min_phys)
#         # norm_lower = max(0.0, norm_center - delta_norm)
#         # norm_upper = min(1.0, norm_center + delta_norm)
#         norm_sample = pyro.sample(f"{key}", dist.Uniform(0.0, 1.0))
#         length_meters = denormalize(norm_sample, min_phys, max_phys)
#         cable_lengths[key] = length_meters

    
#     # Sample loads based on their type
#     for load_name, params in network_params["loads"].items():
#         load_dict = {}
#         for param_name, param_info in params.items():
#             if param_info["inferred"]:
#                 min_val, max_val = param_info["range"]
#                 # Normalize prior as Uniform in [0,1]
#                 norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
#                 load_dict[param_name] = denormalize(norm_sample, min_val, max_val)
#             else:
#                 load_dict[param_name] = torch.tensor(param_info["value"])
#         load_params[load_name] = load_dict
#     # --- Forward model --- Returns (N,) tensor of dtype float32 (Transfer Function Magnitude) given network parameters as input
#     H1_pred = calculate_Hnw(cable_lengths, load_params) 
#     # --- Noise and likelihood using Student-t ---
#     pyro.sample("obs", dist.StudentT(df=1.0, loc=H1_pred, scale=0.4).to_event(1), obs=H1_noisy)

# def guide(H1_noisy):
#     """ Mean-field variational guide for all inferred parameters """
    
#     # --- Cable Lengths ---
#     for key, param in network_params["cable_lengths"].items():
#         if not param["inferred"]:
#             continue
#             # Learn variational parameters (mean and std) for normalized value
#         loc = pyro.param(f"{key}_loc", torch.tensor(0.5), constraint=dist.constraints.unit_interval)
#         scale = pyro.param(f"{key}_scale", torch.tensor(0.05), constraint=dist.constraints.positive)
#         # Truncated Gaussian between [0, 1]
#         q = TransformedDistribution(
#             dist.Normal(loc, scale),
#             [SigmoidTransform()]  # Truncate to (0,1)
#         )
#         pyro.sample(key, q)

#     # --- Load Parameters ---
#     for load_name, params in network_params["loads"].items():
#         for param_name, param_info in params.items():
#             if not param_info["inferred"]:
#                 continue

#             full_name = f"{load_name}_{param_name}"

#             loc = pyro.param(f"{full_name}_loc", torch.tensor(0.5), constraint=dist.constraints.unit_interval)
#             scale = pyro.param(f"{full_name}_scale", torch.tensor(0.05), constraint=dist.constraints.positive)

#             q = TransformedDistribution(
#                 dist.Normal(loc, scale),
#                 [SigmoidTransform()]  # Truncate to (0,1)
#             )

#             pyro.sample(full_name, q)
# def run_inference(H1_noisy, model, guide):
#     pyro.clear_param_store()

#     # AdaGrad optimizer with high learning rate as per the paper
#     optimizer = optim.Adagrad({"lr": 0.6})

#     # ELBO with 14 samples (particles)
#     svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=14))

#     num_steps = 3000
#     losses = []

#     for step in range(num_steps):
#         loss = svi.step(H1_noisy)
#         losses.append(loss)
#         if step % 100 == 0:
#             print(f"[Step {step}] ELBO Loss = {loss:.4f}")

#     print("Inference complete.")
#     return losses
# # ---- Initialize Network Parameters and Run Model ----
# if __name__ == '__main__':
#     # Generate load parameters and update global network_params
#     total_params, load_types = generate_load_parameters(num_loads, omega)
#     print(f"Total number of network parameters: {total_params}")
#     print(f"Load type distribution: {load_types}")

#     # Build cable_lengths and load_params from network_params
#     cable_lengths = {
#         key: torch.tensor(val["value"]) if not isinstance(val["value"], torch.Tensor) else val["value"]
#         for key, val in network_params["cable_lengths"].items()
#     }
#     load_params = {}
#     for load_name, params in network_params["loads"].items():
#         load_dict = {
#             param_name: torch.tensor(param_info["value"]) if not isinstance(param_info["value"], torch.Tensor) else param_info["value"]
#             for param_name, param_info in params.items()
#         }
#         load_params[load_name] = load_dict
#     # 1) Calculate H1
#     H1 = calculate_Hnw(cable_lengths, load_params)
#     # # 2) Add random noise with std dev of 1 dB
#     noise_dB = np.random.normal(loc=0.0, scale=1.0, size=H1.shape)
#     H1_noisy = H1 + noise_dB
#     # # 3) Perform Inference
#     losses = run_inference(H1_noisy, model, guide)
#     #print("Zrec", Z_rec)
#     # Run and trace the model
#     # trace = poutine.trace(model).get_trace(123123)  # Pass in your observed data
#     # trace.compute_log_prob()  # Optional: useful if you want to access log probs

#     # Print all sample sites
#     # for name, site in trace.iter_stochastic_nodes():
#     #     print(f"{name}: {site['value']}")
#     print("My program took", time.time() - start_time, "to run")

    

    # print("H1", H1.shape)
    # plt.figure(figsize=(8,6))
    # plt.plot(freq_range_mhz, H1, label=r"$H_{1,1}$ (Model)", color='b')
    # #plt.xscale("log")
    # #plt.plot(freq_range_mhz, H_nw2_noisy, label=r"$H_{1,1}$ (Model)(ABCD)", color='r')
    # plt.xlabel("Frequency (MHz)", fontsize=12)
    # plt.ylabel(r"Magnitude (dB)", fontsize=12)
    # plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.tight_layout()

    # plt.figure(figsize=(8,6))
    # plt.plot(freq_range_mhz, H1, label=r"$H_{1,1}$ (Model)", color='b')
    # plt.xscale("log")
    # #plt.plot(freq_range_mhz, H_nw_magnitude_db_2, label=r"$H_{1,1}$ (Model)(ABCD)", color='r')
    # plt.xlabel("Frequency (MHz)", fontsize=12)
    # plt.ylabel(r"Magnitude (dB)", fontsize=12)
    # plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.tight_layout()
    # plt.show()