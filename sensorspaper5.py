import os
import numpy as np
import random
import pyro
import pyro.poutine as poutine
import torch
import copy
import math
import json
import pandas as pd
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pyro.distributions as dist
import shutil

from pyro.distributions.transforms import SigmoidTransform, AffineTransform
from pyro.distributions import TransformedDistribution, constraints
from torch.distributions import constraints
from pyro.distributions.torch_distribution import TorchDistribution
from scipy.linalg import expm
from torch.func import jacfwd
from pyro.infer import SVI, Trace_ELBO


start_time = time.time()
torch.set_printoptions(precision=8)  # Show 8 decimal places

OPTIMIZER = "Adam"  # "Adam" or "Adagrad"
LR = 0.02 #Learning rate for optimizer
NUM_STEPS = 500 #Num of SVI steps
NUM_PARTICLES = 12  # Number of particles for SVI
VECTORIZE_PARTICLES = False  # Whether to vectorize particles (faster but uses more memory)
p = 3 #Number of inferred network parameters in Stage 1. For stage 2 just set to 3 since always inferring the 3 fault parameters. 
seed = 95
M = 25 #Number of Monte Carlo trials per SNR to calculate RMSE (number of SVI runs)
M2 = 100 #Number of Monte Carlo samples for expectation of FIM and expectation of prior
alpha = 3.0 #Hyperparameter of beta prior
IS_BAYESIAN = False #Return frequentist RMSE vs CRLB or Bayesian RMSE vs BCRLB
SCENARIO = "with_fault" #Forward model contains fault or not (stage 2 vs stage 1 respectively)

if IS_BAYESIAN:
    OUTPUT_DIR = f"bayesian_p{p}_M{M}_seed{seed}"
else:
    OUTPUT_DIR = f"frequentist_p{p}_M{M}_seed{seed}"

# 1 = Constant, 2 = Double RLC, 3 = Motor
FIXED_LOAD_TYPES = [
    3,  # load_0 R6-O3  Motor
    3,  # load_1 R6-O2  Motor
    1,  # load_2 R6-O1  Constant
    3,  # load_3 R5-O4  Motor
    2,  # load_4 R5-O3  Double RLC
    1,  # load_5 R5-O2  Constant
    3,  # load_6 R5-O1  Motor
    2,  # load_7 R4-O4  Double RLC
    3,  # load_8 R4-O3  Motor
    3,  # load_9 R4-O2  Motor
    2,  # load_10 R4-O1  Double RLC
    3,  # load_11 R3-O4  Motor
    1,  # load_12 R3-O3  Constant
    1,  # load_13 R3-O2  Constant
    1,  # load_14 R3-O1  Constant
    3,  # load_15 R2-O4  Motor
    2,  # load_16 R2-O3  Double RLC
    1,  # load_17 R2-O2  Constant
    3,  # load_18 R2-O1  Motor
    1,  # load_19 R1-O4  Constant
    3,  # load_20 R1-O3  Motor
    1,  # load_21 R1-O2  Constant
]

def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    Z_rec = torch.tensor([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec

def denormalize(norm_value, min_val, max_val):
    """ Convert normalized value (0 to 1) back to its original range. """
    return norm_value * (max_val - min_val) + min_val

def normalize(physical_value, min_val, max_val):
    """ Convert physical value to normalized [0, 1] range. Inverse of denormalize. """
    return (physical_value - min_val) / (max_val - min_val)

def loguniform(low, high, size=None):
    """ Generate samples from a log-uniform distribution. """
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))

def matrix_cosh(M):
    """
    Computes the matrix hyperbolic cosine using expm: cosh(M) = 0.5 * (exp(M) + exp(-M))
    M: (F, n, n) tensor
    Returns: (F, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) + torch.matrix_exp(-M))

def matrix_sinh(M):
    """
    Computes the matrix hyperbolic sine using expm: sinh(M) = 0.5 * (exp(M) - exp(-M))
    M: (F, n, n) tensor
    Returns: (F, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) - torch.matrix_exp(-M))

# Wrapper functions for Jacobian computation
def H_nofault_wrapper(params):
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params, _ = build_params_from_flat(params, param_order)
    H = calculate_Hnw_nofault(cable_lengths, load_params)
    return torch.stack([H.real, H.imag], dim=-1)

def H_fault_wrapper(params):
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params, fault_params = build_params_from_flat(params, param_order)
    H = calculate_Hnw(cable_lengths, load_params, fault_params)
    return torch.stack([H.real, H.imag], dim=-1)

# Network constants
num_loads = 22
num_of_conductors = 4
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#frequencies = torch.logspace(torch.log10(torch.tensor(2e6)), torch.log10(torch.tensor(10e6)), 500) #2-10MHz
#frequencies = torch.logspace(torch.log10(torch.tensor(150e3)), torch.log10(torch.tensor(30e6)), 200) #150KHz - 30MHz
frequencies = torch.logspace(torch.log10(torch.tensor(150e3, device=device)),
                              torch.log10(torch.tensor(10e6, device=device)), 200, device=device) #150KHz - 500KHz
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
FILENAME_PREFIX = f"{f_start_str}-{f_end_str}_{OPTIMIZER}_lr{LR}"

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

BACKBONE_KEYS = ["l_w_0", "l_w_1", "l_w_4", "l_w_25", "l_w_28"]

# ---- Define Network Parameter Dictionary ----
network_params = {
    "cable_lengths": {  # 30 parameters, set all to 0.25
        f"l_w_{i}": {"value": 0.25, "inferred": True, "range": (6, 8)}
        for i in range(30)
    },
    "conductor_radii": {  # Fixed values, not inferred
        "r_w_servicepanel": {"value": 0.25, "inferred": False, "range": (1.03e-3, 2.06e-3)},
        "r_w_room": {"value": 0.25, "inferred": False, "range": (0.81e-3, 1.29e-3)}
    },
    "fault_parameters": {
        # Normalized position [0, 1], will be scaled to [0, L] in forward model
        "fault_position": {"value": 0.25, "inferred": True, "range": (0.0, 1.0)},
        # Complex fault impedance Z_fault = Z_fault_real + j*Z_fault_imag
        "Z_fault_real": {"value": 0.1, "inferred": True, "range": (0.0, 1000.0)},
        "Z_fault_imag": {"value": 0.25, "inferred": True, "range": (-100.0, 100.0)}
    },
    "loads": {}  # Dynamically generated based on load type
}


def calculate_cable_parameters(r_w, omega, n):
    """
    Compute R, L, C, G tensors for multiple frequencies.

    Parameters:
    - r_w: radius of MTL conductor (scalar)
    - omega: tensor of angular frequencies (F,)
    - n: number of conductors - 1 (scalar)

    Returns:
    - R, L, C, G tensors (shape: (F, n, n))
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
    delta = 1 / torch.sqrt(torch.pi * mu_0 * sigma * f)  # Skin depth
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

def get_mtl_matrices(R, L, C, G, n, omega):
    """
    Compute MTL matrices for multiple frequencies using PyTorch.

    Parameters:
    - R, L, C, G: (F, n, n) tensors
    - omega: (F,) tensor

    Returns:
    - T (F, n, n) - Eigenvectors
    - Tinv (F, n, n) - Inverse of Eigenvectors
    - gamma (F, n, n) - Propagation Constants
    - ZC (F, n, n) - Characteristic Impedance
    - YC (F, n, n) - Characteristic Admittance
    """
    F, n, _ = R.shape

    # Reshape omega to (F, 1, 1) for broadcasting
    omega = omega.view(-1, 1, 1)  

    # Compute impedance and admittance matrices
    Z_T = R + 1j * omega * L  # (F, n, n)
    Y_T = G + 1j * omega * C  # (F, n, n)

    # Compute ZY and YZ matrices
    ZY = torch.matmul(Z_T, Y_T)  # (F, n, n)
    YZ = torch.matmul(Y_T, Z_T)  # (F, n, n)
    
    # Compute eigenvalues and eigenvectors of YZ
    eigvals, eigvecs = torch.linalg.eig(YZ)  # eigvals: (F, n), eigvecs: (F, n, n)

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
    gamma = torch.zeros((F, n, n), dtype=torch.complex64, device=R.device)  # Initialize with zeros
    gamma[:, torch.arange(n), torch.arange(n)] = torch.sqrt(eigvals)  # Assign square roots of eigenvalues

    # Compute batch-wise inverses
    inv_YT = torch.linalg.inv(Y_T)  # (F, n, n)
    inv_eigvecs = torch.linalg.inv(eigvecs)  # (F, n, n)

    # Compute characteristic impedance Zc
    Zc = inv_YT @ eigvecs @ gamma @ inv_eigvecs  # Batch matrix multiplication (F, n, n)

    # Compute characteristic admittance Yc
    Yc = torch.linalg.inv(Zc)  # (F, n, n)
    return eigvecs, inv_eigvecs, gamma, Zc, Yc

def calculate_room_admittance_matrix(Y_loads_room, cable_lengths_room, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                                     T_s, Tinv_s, ZC_s, YC_s, gamma_s):
    """
    Returns equivalent admittance matrix of a room in the network. 
    
    Args:
        Y_loads_room: list of 4 admittance matrices (torch.Tensor with shape [P,F,n,n] or [F,n,n])
        cable_lengths_room: list of 5 cable lengths (torch.tensor with shape [P,1] or [] scalar)

    Returns:
        Y_room_carried: Equiv. room admittance matrix (torch.tensor with shape [P,F,n,n] or [F,n,n])
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

# Function to compute constant impedance admittance matrix (type 1)
def constant_impedance(R_const, C_leak, omega):
    # When vectorize_particles=True, R_const and C_leak have shape [P,1]
    # omega has shape [F]. Result should be [P, F] or [F] depending on input.
    ZG1 = 1 / (1j * omega * C_leak)
    ZG2 = ZG3 = ZG1
    # Ensure Z12 has same shape as ZG1 by using ones_like(ZG1) instead of ones_like(omega)
    Z12 = R_const.to(torch.complex64) * torch.ones_like(ZG1)
    Z13 = Z23 = Z12
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

    Args:
        load_params (tuple of tensors): (Z12, Z13, Z23, ZG1, ZG2, ZG3)
        Each impedance tensor can have shape (..., F) where ... are batch dims (e.g., particles).
        Note: Different tensors may have different batch dims if only some params are sampled.

    Returns:
        Y_load (torch.Tensor): Load admittance matrix with shape (..., F, 3, 3)
    """
    Z12, Z13, Z23, ZG1, ZG2, ZG3 = load_params  # Unpack impedance values

    # Compute admittance elements
    Y11 = 1/ZG1 + 1/Z12 + 1/Z13
    Y12 = -1/Z12
    Y13 = -1/Z13
    Y22 = 1/ZG2 + 1/Z12 + 1/Z23
    Y23 = -1/Z23
    Y33 = 1/ZG3 + 1/Z13 + 1/Z23

    # Broadcast all elements to common shape before stacking
    # This handles cases where only some parameters have particle batch dims
    elements = [Y11, Y12, Y13, Y22, Y23, Y33]
    target_shape = torch.broadcast_shapes(*[e.shape for e in elements])
    Y11, Y12, Y13, Y22, Y23, Y33 = [e.expand(target_shape) for e in elements]

    # Stack into (..., F, 3, 3) tensor using dim=-1 for columns, dim=-2 for rows
    row1 = torch.stack([Y11, Y12, Y13], dim=-1)  # (..., F, 3)
    row2 = torch.stack([Y12, Y22, Y23], dim=-1)  # (..., F, 3)
    row3 = torch.stack([Y13, Y23, Y33], dim=-1)  # (..., F, 3)
    Y_load = torch.stack([row1, row2, row3], dim=-2)  # (..., F, 3, 3)
    return Y_load


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
                "R_const": {"value": 0.25, "inferred": True, "range": (10, 200)},
                "C_leak": {"value": 0.25, "inferred": True, "range": (0.1e-9, 2.0e-9)}
            }
            total_parameters += 2

        elif load_type == 2:  # Double RLC (9)
            network_params["loads"][f"load_{i}"] = {
                "R_s": {"value": 0.25, "inferred": True, "range": (10, 3000)},
                "omega_0s": {"value": 0.25, "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_s": {"value": 0.25, "inferred": True, "range": (0.1, 2)},
                "R_p": {"value": 0.25, "inferred": True, "range": (10, 3000)},
                "omega_0p": {"value": 0.25, "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_p": {"value": 0.25, "inferred": True, "range": (0.1, 2)},
                "delta_1": {"value": 0.25, "inferred": True, "range": (-0.1, 0.1)},
                "delta_2": {"value": 0.25, "inferred": True, "range": (-0.1, 0.1)},
                "C_d_leak": {"value": 0.25, "inferred": True, "range": (0.1e-9, 2e-9)}
            }
            total_parameters += 9

        elif load_type == 3:  # Motor (5)
            network_params["loads"][f"load_{i}"] = {
                "C_m": {"value": 0.25, "inferred": True, "range": (0.1e-9, 1e-9)},
                "L_m": {"value": 0.25, "inferred": True, "range": (5e-3, 20e-3)},
                "R_m1": {"value": 0.25, "inferred": True, "range": (2000, 15000)},
                "R_m2": {"value": 5.0, "inferred": False},
                "C_m_leak": {"value": 0.25, "inferred": True, "range": (0.2e-9, 5e-9)}
            }
            total_parameters += 5

    return total_parameters, load_types


def reflection_coefficient(YL, T, T_inv, ZC, YC):
    """
    Implements eq. (12) of Tonello paper.
    Inputs: all (F, n, n) tensors except YL which can be [P, F, n, n]
    Returns: rho (F, n, n) or [P, F, n, n]
    """
    inv_sum = torch.linalg.inv(YL + YC)
    rho = T_inv @ YC @ inv_sum @ (YL - YC) @ ZC @ T
    return rho

def carry_back_load(rhoL, T, YC, Gamma, length):
    """Implements eq. (13) of Tonello paper.

    Args:
        rhoL: Reflection coefficient. Shape: (P, F, n, n), may have batch dims (particles).
        T, YC, Gamma: MTL matrices. Shape: (F, n, n).
        length: Cable length. Scalar or shape [P, 1].

    Returns:
        Y_R(x) with shape (P, F, n, n).
    """
    # Reshape length for broadcasting with Gamma [F, n, n]
    if isinstance(length, torch.Tensor) and length.dim() > 0:
        # Add 2 dims for F, n, n: [P, 1] -> [P, 1, 1, 1]
        length_bc = length.view(*length.shape, 1, 1)
    else:
        length_bc = length

    # Compute matrix exponentials
    e_pos = torch.matrix_exp(Gamma * length_bc)       # (P, F, n, n)
    e_neg = torch.matrix_exp(-Gamma * length_bc)      # (P, F, n, n)

    # Compute numerator and denominator
    num = e_pos + torch.matmul(e_neg, rhoL)        # (P, F, n, n)
    den = e_pos - torch.matmul(e_neg, rhoL)        # (P, F, n, n)
    deninv = torch.linalg.inv(den)                 # (P, F, n, n)

    # Compute final YR - broadcasting handles batch dims
    YR = T @ num @ deninv @ torch.linalg.inv(T) @ YC
    return YR

def h_B(rhoL, ZC, T, T_inv, Gamma, length):
    """
    Implements eq. (14) of Tonello paper.

    Args:
        rhoL: (P, F, n, n) - reflection coefficient, may have batch dims (particles)
        ZC, T, T_inv, Gamma: (F, n, n) - MTL matrices
        length: scalar or tensor with batch dims [P, 1]

    Returns:
        h_B (P, F, n, n)
    """
    n = ZC.shape[-1]
    device = ZC.device

    # Identity matrix - will broadcast with rhoL
    U = torch.eye(n, dtype=torch.complex64, device=device)

    # Reshape length for broadcasting with Gamma [F, n, n]
    if isinstance(length, torch.Tensor) and length.dim() > 0:
        length_bc = length.view(*length.shape, 1, 1)
    else:
        length_bc = length

    # Matrix exponentials
    e_pos = torch.matrix_exp(Gamma * length_bc)
    e_neg = torch.matrix_exp(-Gamma * length_bc)

    den = e_pos - e_neg @ rhoL
    deninv = torch.linalg.inv(den)

    hB = ZC @ T @ (U - rhoL) @ deninv @ T_inv @ torch.linalg.inv(ZC)
    return hB

def calculate_Htrans(YTalpha, YTbeta, YTgamma, Ynw, ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32):
    """
    Compute Htrans from Ynw and transmitter impedance values

    Parameters:
    - YTalpha, YTbeta, YTgamma: Transmitter constants (scalar)
    - Ynw: Network input admittance matrix (..., F, n, n) - may have batch dims
    - ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32: Transmitter constants (scalar)

    Returns:
    - Htrans: Transfer function of transmitter (..., F, n, n)
    """
    # Extract individual elements using ... for arbitrary batch dims
    # Ynw[..., i, j] gives shape (..., F), add two dims for matrix construction
    Ynw11 = Ynw[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
    Ynw12 = Ynw[..., 0, 1].unsqueeze(-1).unsqueeze(-1)
    Ynw13 = Ynw[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
    Ynw21 = Ynw[..., 1, 0].unsqueeze(-1).unsqueeze(-1)
    Ynw22 = Ynw[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
    Ynw23 = Ynw[..., 1, 2].unsqueeze(-1).unsqueeze(-1)
    Ynw31 = Ynw[..., 2, 0].unsqueeze(-1).unsqueeze(-1)
    Ynw32 = Ynw[..., 2, 1].unsqueeze(-1).unsqueeze(-1)
    Ynw33 = Ynw[..., 2, 2].unsqueeze(-1).unsqueeze(-1)

    H11 = 1 + ZT0 * Ynw11 + ZT0 * YTalpha
    H12 = ZT0 * Ynw12 - ZT0 / ZT12
    H13 = ZT0 * Ynw13 - ZT0 / ZT13
    H21 = ZT0 * Ynw21 - ZT0 / ZT21
    H22 = 1 + ZT0 * Ynw22 + ZT0 * YTbeta
    H23 = ZT0 * Ynw23 - ZT0 / ZT23
    H31 = ZT0 * Ynw31 - ZT0 / ZT31
    H32 = ZT0 * Ynw32 - ZT0 / ZT32
    H33 = 1 + ZT0 * Ynw33 + ZT0 * YTgamma

    # Stack rows using dim=-1 for columns, dim=-2 for rows
    H_trans = torch.cat([
        torch.cat([H11, H12, H13], dim=-1),
        torch.cat([H21, H22, H23], dim=-1),
        torch.cat([H31, H32, H33], dim=-1)
    ], dim=-2)  # Resulting shape: (..., F, 3, 3)
    H_trans_inv = torch.linalg.inv(H_trans)

    return H_trans_inv
def compute_fault_admittance_matrix(Z_fault_real, Z_fault_imag, F, n, k=0):
    """
    Compute shunt fault admittance matrix with conductor k to ground.

    Parameters:
    - Z_fault_real: Real part of fault impedance of shape [] or [P, 1]
    - Z_fault_imag: Imaginary part of fault impedance of shape [] or [P, 1]
    - F: number of frequency points
    - n: number of conductors - 1
    - k: which conductor has fault to ground (default 0)

    Returns:
    - Y_fault: (F, n, n) diagonal fault admittance matrix
    """
    # Ensure tensors, preserving dtype for gradient flow
    if not isinstance(Z_fault_real, torch.Tensor):
        Z_fault_real = torch.tensor(Z_fault_real, dtype=torch.float32, device=device)
    if not isinstance(Z_fault_imag, torch.Tensor):
        Z_fault_imag = torch.tensor(Z_fault_imag, dtype=torch.float32, device=device)

    # Compute complex admittance Y = 1/Z = 1/(R + jX)
    # Using Y = (R - jX) / (R^2 + X^2) to avoid torch.complex issues with jacfwd
    denom = Z_fault_real**2 + Z_fault_imag**2
    Y_real = Z_fault_real / denom
    Y_imag = -Z_fault_imag / denom

    # Build Y_fault without in-place ops (for forward-mode AD compatibility)
    # Create one-hot mask for position (k, k) without in-place ops
    mask = torch.eye(n, dtype=torch.float32, device=device)[k:k+1, :].T @ torch.eye(n, dtype=torch.float32, device=device)[k:k+1, :]

    # Create complex Y_fault: expand to (N, n, n)
    Y_fault_real = Y_real * mask.unsqueeze(0).expand(F, -1, -1)
    Y_fault_imag = Y_imag * mask.unsqueeze(0).expand(F, -1, -1)
    Y_fault = torch.complex(Y_fault_real, Y_fault_imag)
    return Y_fault


def get_total_backbone_length(cable_lengths):
    """Calculate total backbone length L from cable_lengths dict of tensors."""
    total = torch.tensor(0.0, device=device)
    for key in BACKBONE_KEYS:
        val = cable_lengths[key]
        total = total + val  # Keep as tensor for gradient flow
    return total


def get_fault_segment_and_local_position(fault_position, cable_lengths):
    """
    Given normalized fault position [0,1], determine which backbone segment
    and local position within that segment.

    Parameters:
    - fault_position: tensor in [0, 1] of shape [] or [P, 1]
    - cable_lengths: dict of cable length tensors (inferred)
    Returns:
    - segment_idx: which backbone segment (0-4) - discrete, non-differentiable
    - local_position: position within that segment in meters (tensor, differentiable)
    - segment_length: total length of that segment (tensor, differentiable)
    """
    L = get_total_backbone_length(cable_lengths)
    fault_position_abs = fault_position * L  # Convert to meters - keeps gradient

    # First pass: determine segment index using detached values (discrete decision)
    cumulative_detached = 0.0
    fault_pos_val = fault_position_abs.detach().item() if isinstance(fault_position_abs, torch.Tensor) else fault_position_abs
    segment_idx = len(BACKBONE_KEYS) - 1  # Default to last segment
    for idx, key in enumerate(BACKBONE_KEYS):
        seg_len_val = cable_lengths[key].detach().item() if isinstance(cable_lengths[key], torch.Tensor) else cable_lengths[key]
        if cumulative_detached + seg_len_val >= fault_pos_val or idx == len(BACKBONE_KEYS) - 1:
            segment_idx = idx
            break
        cumulative_detached += seg_len_val

    # Second pass: compute local_pos with gradient flow using original tensors
    cumulative = torch.tensor(0.0, device=device)
    for idx, key in enumerate(BACKBONE_KEYS):
        if idx == segment_idx:
            local_pos = fault_position_abs - cumulative
            seg_len = cable_lengths[key]
            return segment_idx, local_pos, seg_len
        cumulative = cumulative + cable_lengths[key]

    # Fallback (shouldn't reach here)
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
    - Y_load: (F, n, n) admittance at the load end (Rx side)
    - T, Tinv, ZC, YC, gamma: MTL parameters
    - cable_length: total length of this segment
    - local_fault_pos: distance from load end to fault (meters)
    - Z_fault_real: real part of fault impedance (Ohms)
    - Z_fault_imag: imaginary part of fault impedance (Ohms)

    Returns:
    - Y_carried: (F, n, n) admittance seen from source end (Tx side)
    - h_total: (F, n, n) transfer function h_B through the faulted cable
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


def calculate_Hnw_nofault(cable_lengths, load_params):
    """
    Takes parameters in [0,1] range, converts to physical inside.
    Autograd then computes gradients w.r.t. normalized params automatically.

    Returns TF of shape [F] or [P,F] 
    """
    
    # Convert cable lengths from [0,1] to physical
    cable_lengths_physical = {}
    for cable_name, norm_val in cable_lengths.items():
        lo, hi = network_params["cable_lengths"][cable_name]["range"]
        cable_lengths_physical[cable_name] = lo + norm_val * (hi - lo)
    
    # Convert load params from [0,1] to physical
    load_params_physical = {}
    for load_name, params in load_params.items():
        load_params_physical[load_name] = {}
        for param_name, norm_val in params.items():
            if param_name == 'R_m2': #R_m2 is just kept as 5 since it's not inferred
                load_params_physical[load_name][param_name] = norm_val
            else:
                lo, hi = network_params["loads"][load_name][param_name]["range"]
                load_params_physical[load_name][param_name] = lo + norm_val * (hi - lo)
    
    lo, hi = network_params["conductor_radii"]["r_w_servicepanel"]["range"]
    r_w_servicepanel = denormalize(network_params["conductor_radii"]["r_w_servicepanel"]["value"], lo, hi)
    lo, hi = network_params["conductor_radii"]["r_w_room"]["range"]
    r_w_room = denormalize(network_params["conductor_radii"]["r_w_room"]["value"], lo, hi)
    #Calculate load admittance matrices from load params
    Y_loads = {}
    for load, params in load_params_physical.items():
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
        
        Y_loads[load] = compute_load_admittance_3d((Z12, Z13, Z23, ZG1, ZG2, ZG3)) #[P, F, n, n] or [F, n, n]
    
    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1) #[F, n, n]
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room
    #node0 (Yrec)
    rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{0}"])
    Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{0}"])
    #node1 (Y62)
    Y_node2 = Y_reccarried + Y_loads["load_0"]
    #print("Y1", Y_loads["load_0"])
    
    rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
    h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{1}"])
    Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{1}"])
    #node2 Junction box (Y61 || Y63)
    rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{2}"])
    Y_61 = Y_63 + Y_loads["load_2"]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{3}"])
    # print("Y2", Y_6)

    Y_node3 = Y_node2carried + Y_6
    rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
    h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical[f"l_w_{4}"])
    Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths_physical[f"l_w_{4}"])
    #node3 (4 rooms service panel)
    Y_5 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(3, 7)],  # load_3 to load_6
    [cable_lengths_physical[f"l_w_{i}"] for i in range(5, 10)],  # l_w_5 to l_w_9
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_4 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(7, 11)],  # load_7 to load_10
    [cable_lengths_physical[f"l_w_{i}"] for i in range(10, 15)],  # l_w_10 to l_w_14
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_3 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(11, 15)],  # load_11 to load_14
    [cable_lengths_physical[f"l_w_{i}"] for i in range(15, 20)],  # l_w_15 to l_w_19
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_2 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(15, 19)],  # load_15 to load_18
    [cable_lengths_physical[f"l_w_{i}"] for i in range(20, 25)],  # l_w_20 to l_w_24
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
   # print("Y3", Y_5 + Y_4 + Y_3 + Y_2)
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
    h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical[f"l_w_{25}"])
    Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths_physical[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{26}"])
    Y_12 = Y_14 + Y_loads["load_20"]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{27}"])
   # print("Y4", Y_1)
    Y_node5 = Y_node4carried + Y_1
    rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
    h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{28}"])
    Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{28}"])
    # node5 Transmitter Y13 connected in parallel
    rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{29}"])
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
    # Use ... to handle arbitrary batch dimensions (e.g., particle dimension)
    H_nw = H1[..., 0, 0]
    return H_nw #[P, F] or [F]

def calculate_Hnw(cable_lengths, load_params, fault_params):
    """
    Calculate network Transfer Function Hnw including fault.
    """

    # Convert cable lengths from [0,1] to physical
    cable_lengths_physical = {}
    for cable_name, norm_val in cable_lengths.items():
        lo, hi = network_params["cable_lengths"][cable_name]["range"]
        cable_lengths_physical[cable_name] = lo + norm_val * (hi - lo)
    
    # Convert load params from [0,1] to physical
    load_params_physical = {}
    for load_name, params in load_params.items():
        load_params_physical[load_name] = {}
        for param_name, norm_val in params.items():
            if param_name == 'R_m2': #R_m2 is just kept as 5 since it's not inferred
                load_params_physical[load_name][param_name] = norm_val
            else:
                lo, hi = network_params["loads"][load_name][param_name]["range"]
                load_params_physical[load_name][param_name] = lo + norm_val * (hi - lo)
    
    lo, hi = network_params["conductor_radii"]["r_w_servicepanel"]["range"]
    r_w_servicepanel = denormalize(network_params["conductor_radii"]["r_w_servicepanel"]["value"], lo, hi)
    lo, hi = network_params["conductor_radii"]["r_w_room"]["range"]
    r_w_room = denormalize(network_params["conductor_radii"]["r_w_room"]["value"], lo, hi)

    # Initialize dictionary to store load admittance matrices calculated from sampled_params
    Y_loads = {}
    
    # Convert fault params from [0,1] to physical
    lo, hi = network_params["fault_parameters"]["fault_position"]["range"]
    fault_location = lo + fault_params["fault_position"] * (hi - lo)
    lo, hi = network_params["fault_parameters"]["Z_fault_real"]["range"]
    Z_fault_real = lo + fault_params["Z_fault_real"] * (hi - lo)
    lo, hi = network_params["fault_parameters"]["Z_fault_imag"]["range"]
    Z_fault_imag = lo + fault_params["Z_fault_imag"] * (hi - lo)

    #From location + impedance find which cable has the fault
    fault_seg_idx, local_fault_pos, _ = get_fault_segment_and_local_position(fault_location, cable_lengths_physical)

    for load, params in load_params_physical.items():
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
            cable_lengths_physical["l_w_0"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
        h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{0}"])
        Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{0}"])

    #node1 (Y62)
    Y_node2 = Y_reccarried + Y_loads["load_0"]
    # ===== Backbone segment 1: l_w_1 (room wire) =====
    if fault_seg_idx == 1:
        Y_node2carried, h2 = carry_back_with_fault(
            Y_node2, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths_physical["l_w_1"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
        h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{1}"])
        Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{1}"])

    #node2 Junction box (Y61 || Y63)
    rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{2}"])
    Y_61 = Y_63 + Y_loads["load_2"]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{3}"])
    Y_node3 = Y_node2carried + Y_6
    # ===== Backbone segment 2: l_w_4 (service panel wire) =====
    if fault_seg_idx == 2:
        Y_node3carried, h3 = carry_back_with_fault(
            Y_node3, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths_physical["l_w_4"], local_fault_pos, Z_fault_real, Z_fault_imag
        )
    else:
        rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
        h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical[f"l_w_{4}"])
        Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths_physical[f"l_w_{4}"])

    #node3 (4 rooms service panel)
    Y_5 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(3, 7)],  # load_3 to load_6
    [cable_lengths_physical[f"l_w_{i}"] for i in range(5, 10)],  # l_w_5 to l_w_9
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_4 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(7, 11)],  # load_7 to load_10
    [cable_lengths_physical[f"l_w_{i}"] for i in range(10, 15)],  # l_w_10 to l_w_14
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_3 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(11, 15)],  # load_11 to load_14
    [cable_lengths_physical[f"l_w_{i}"] for i in range(15, 20)],  # l_w_15 to l_w_19
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_2 = calculate_room_admittance_matrix(
    [Y_loads[f"load_{i}"] for i in range(15, 19)],  # load_15 to load_18
    [cable_lengths_physical[f"l_w_{i}"] for i in range(20, 25)],  # l_w_20 to l_w_24
    T_r, Tinv_r, ZC_r, YC_r, gamma_r,
    T_s, Tinv_s, ZC_s, YC_s, gamma_s
)
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    # ===== Backbone segment 3: l_w_25 (service panel wire) =====
    if fault_seg_idx == 3:
        Y_node4carried, h4 = carry_back_with_fault(
            Y_node4, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
            cable_lengths_physical["l_w_25"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
        h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical[f"l_w_{25}"])
        Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths_physical[f"l_w_{25}"])
    #node4 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{26}"])
    Y_12 = Y_14 + Y_loads["load_20"]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{27}"])
    Y_node5 = Y_node4carried + Y_1

    # ===== Backbone segment 4: l_w_28 (room wire) =====
    if fault_seg_idx == 4:
        Y_node5carried, h5 = carry_back_with_fault(
            Y_node5, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            cable_lengths_physical["l_w_28"], local_fault_pos, Z_fault_real, Z_fault_imag)
    else:
        rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
        h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical[f"l_w_{28}"])
        Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{28}"])

    # node5 Transmitter Y13 connected in parallel
    rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths_physical[f"l_w_{29}"])
    Y_node6 = Y_node5carried + Y_13

    YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_node6, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    hoverall = h1 @ h2 @ h3 @ h4 @ h5
    #hoverall = h5 @ h4 @ h3 @ h2 @ h1
    H1 = hoverall @ H_trans 
    H_nw = H1[:, 0, 0]
    return H_nw

def model_no_fault(H1_noisy, std_f):
    """
    Stage 1. Model with no fault
    H1_noisy has shape [N, F, 2]
    When vectorize_particles=True, samples have shape [P, 1] where P = num_particles.
    The forward model output shape is [P, F] with particles or [F] without.
    """
    N, F, _ = H1_noisy.shape
    # Beta distribution parameters on correct device
    beta_conc = torch.tensor(5.0, device=device)

    # Sample/fix load parameters - when vectorize_particles = True sample all particles at once
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Beta(beta_conc, beta_conc))
                load_dict[param_name] = norm_sample
            else:
                load_dict[param_name] = torch.tensor(param_info["value"], device=device)
        load_params[load_name] = load_dict


    # Sample/fix cable parameters
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_info["inferred"]:
            norm_sample = pyro.sample(f"{cable_name}", dist.Beta(beta_conc, beta_conc))
            cable_lengths[cable_name] = norm_sample
        else:
            cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

    # calculate_Hnw_nofault should return [P, F] where P = num of particles or [F] if scalar
    H1_pred_c = calculate_Hnw_nofault(cable_lengths, load_params)

    # Handle both vectorized [P, F] and non-vectorized [F] cases
    if H1_pred_c.dim() == 1:
        # Non-vectorized: [F] -> [N, F]
        H1_pred_c = H1_pred_c.unsqueeze(0).expand(N, -1)
    else:
        # Vectorized: [P, F] -> [P, N, F] (insert N before F)
        H1_pred_c = H1_pred_c.unsqueeze(-2).expand(*H1_pred_c.shape[:-1], N, H1_pred_c.shape[-1])

    H1_pred = torch.view_as_real(H1_pred_c) #[P, N, F, 2]

    with pyro.plate("data", N):
        pyro.sample(
            "obs",
            dist.Independent(
                dist.Normal(loc=H1_pred, scale=std_f),
                reinterpreted_batch_ndims=2
            ),
            obs=H1_noisy
        )

def model_with_fault(H1_noisy, std_f):
    """
    Stage 2 - Network with fault - only infer fault parameters. 
    Load/cable parameters are fixed to their inferred values from Stage 1. 
    """
    N, F, _ = H1_noisy.shape
    # Beta distribution parameters on correct device
    beta_conc = torch.tensor(5.0, device=device)

    # Sample/fix load parameters 
    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_dict = {}
        for param_name, param_info in params.items():
            if param_info["inferred"]:
                norm_sample = pyro.sample(f"{load_name}_{param_name}", dist.Uniform(0.0, 1.0))
                load_dict[param_name] = norm_sample
            else:
                load_dict[param_name] = torch.tensor(param_info["value"], device=device)
        load_params[load_name] = load_dict


    # Sample/fix cable parameters
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if cable_info["inferred"]:
            norm_sample = pyro.sample(f"{cable_name}", dist.Uniform(0.0, 1.0))
            cable_lengths[cable_name] = norm_sample
        else:
            cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

    # Sample/fix fault parameters
    fault_params = {}
    for fault_name, fault_info in network_params["fault_parameters"].items():
        if fault_info["inferred"]:
            norm_sample = pyro.sample(f"{fault_name}", dist.Uniform(0.0, 1.0))
            fault_params[fault_name] = norm_sample
        else:
            fault_params[fault_name] = torch.tensor(fault_info["value"], device=device)


    #[F]
    H1_pred_c = calculate_Hnw(cable_lengths, load_params, fault_params)

    # Handle both vectorized [P, F] and non-vectorized [F] cases
    if H1_pred_c.dim() == 1:
        # Non-vectorized: [F] -> [N, F]
        H1_pred_c = H1_pred_c.unsqueeze(0).expand(N, -1)
    else:
        # Vectorized: [P, F] -> [P, N, F] (insert N before F)
        H1_pred_c = H1_pred_c.unsqueeze(-2).expand(*H1_pred_c.shape[:-1], N, H1_pred_c.shape[-1])

    H1_pred = torch.view_as_real(H1_pred_c) #[P, N, F, 2]

    with pyro.plate("data", N):
        pyro.sample(
            "obs",
            dist.Independent(
                dist.Normal(loc=H1_pred, scale=std_f),
                reinterpreted_batch_ndims=2
            ),
            obs=H1_noisy
        )

def guide(H1_noisy, std_f):
    for load_name, params in network_params["loads"].items():
        for param_name, param_info in params.items():
            if not param_info["inferred"]:
                continue

            full_name = f"{load_name}_{param_name}"

            #loc = pyro.param(f"{full_name}_loc", torch.tensor(-1.1, device=device))

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
        
        #loc = pyro.param(f"{key}_loc", torch.tensor(-1.1, device=device))

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


def set_network_params_from_normalized(sampled_theta, param_order_list):
    """
    Update global network_params dict with sampled theta only for inferred keys in param_order_list.
    Args:
        sampled_theta: Sampled tensor of theta of shape [p] where theta in [0, 1]
    """
    counter = 0
    for params in param_order_list:
        if params[0] == "load":
            entity_name = params[1]
            param_name = params[2]
            network_params["loads"][entity_name][param_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1
        elif params[0] == "cable":
            cable_name = params[1]
            network_params["cable_lengths"][cable_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1
        elif params[0] == "fault_param":
            fault_name = params[1]
            network_params["fault_parameters"][fault_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1    

def perform_global_sensitivity_analysis(cable_lengths_norm, load_params_norm):
    """
    Perform global sensitivity analysis on network parameters.
    
    All parameters are in NORMALIZED [0,1] range.
    Sweeps each parameter over [0,1] and measures ||H(perturbed) - H(nominal)||.

    Args:
        cable_lengths_norm: Dict of {cable_name: normalized_value} for cables with inferred=True
        load_params_norm: Dict of {load_name: {param_name: normalized_value}} for inferred params

    Returns:
        selected_keys: List of top p most sensitive params sorted by sensitivity
        sorted_keys: List of all params sorted from most sensitive to least sensitive
        sensitivities: List of sensitivity values (%) corresponding to sorted_keys
    """
    variations = {}

    # Compute nominal H with current normalized params
    nominal_H = calculate_Hnw_nofault(cable_lengths_norm, load_params_norm)

    # === Analyze load parameters ===
    for load_name, param_dict in network_params["loads"].items():
        for param_name, param_info in param_dict.items():
            if not param_info["inferred"]:
                continue

            # Sweep in normalized [0,1] space
            # Use log scale for resistance parameters (wide range)
            if param_name in ["R_const", "R_s", "R_p", "R_m1"]:
                values = np.logspace(0, 1, 10)
            else:
                values = np.linspace(0, 1, 10)
            param_variations = []

            for norm_val in values:
                # Deep copy and perturb this one parameter
                perturbed_loads = copy.deepcopy(load_params_norm)
                
                # Ensure load exists in dict (might not if only some params inferred)
                if load_name not in perturbed_loads:
                    perturbed_loads[load_name] = {}
                
                perturbed_loads[load_name][param_name] = torch.tensor(norm_val, device=device)
                
                H_var = calculate_Hnw_nofault(cable_lengths_norm, perturbed_loads)
                diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
                param_variations.append(diff)

            total_var = sum(param_variations)
            key = f"{load_name}.{param_name}"
            variations[key] = total_var

    # === Analyze cable length parameters ===
    for cable_name, cable_info in network_params["cable_lengths"].items():
        if not cable_info["inferred"]:
            continue

        # Sweep in normalized [0,1] space
        values = np.linspace(0, 1, 10)
        param_variations = []

        for norm_val in values:
            perturbed_cables = copy.deepcopy(cable_lengths_norm)
            perturbed_cables[cable_name] = torch.tensor(norm_val, device=device)
            
            H_var = calculate_Hnw_nofault(perturbed_cables, load_params_norm)
            diff = torch.linalg.norm(H_var - nominal_H, ord=2).item()
            param_variations.append(diff)

        total_var = sum(param_variations)
        variations[cable_name] = total_var

    # === Normalize sensitivities ===
    total_sum = sum(variations.values())
    if total_sum > 0:
        normalized = {k: v / total_sum for k, v in variations.items()}
    else:
        normalized = {k: 0.0 for k in variations.keys()}
    
    sensitivities = []

    # Sort all parameters by sensitivity (most to least)
    sorted_params = sorted(normalized, key=normalized.get, reverse=True)

    # Select top p most sensitive parameters
    selected = sorted_params[:p]

    print("\n--- Global Sensitivity Analysis (Normalized Params) ---")
    for idx, k in enumerate(sorted_params):
        if idx == p:
            print(f"--- Top {p} selected above this line ---")
        print(f"{k}: {normalized[k]*100:.5f}%")
        sensitivities.append(normalized[k] * 100)

    print(f"\nSelected top {p} most sensitive parameters: {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    # === Set inferred=False for parameters NOT in top p ===
    if p > 0:
        disabled_count = 0
        for param_key in sorted_params[p:]:
            if "." in param_key:
                # Load parameter: "load_0.C_m_leak"
                parts = param_key.split(".")
                entity_name = parts[0]
                param_name = parts[1]

                if entity_name in network_params["loads"]:
                    if param_name in network_params["loads"][entity_name]:
                        network_params["loads"][entity_name][param_name]["inferred"] = False
                        disabled_count += 1
            else:
                # Cable length parameter
                if param_key in network_params["cable_lengths"]:
                    network_params["cable_lengths"][param_key]["inferred"] = False
                    disabled_count += 1

        print(f"\nDisabled inference for {disabled_count} parameters (not in top {p})")

    sorted_keys = sorted_params
    return selected, sorted_keys, sensitivities

#CRLB helper functions
def get_inferred_param_order():
    """
    Get ordered list of inferred parameters for consistent flat tensor indexing.

    Returns:
        param_order: List of tuples describing each parameter:
            - ("cable", cable_name, None) for cable lengths
            - ("load", load_name, param_name) for load parameters
        
        num_params: Total number of inferred parameters (p)
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

    for fault_name in network_params["fault_parameters"]:
        if network_params["fault_parameters"][fault_name]["inferred"]:
            param_order.append(("fault_param", fault_name, None))

    return param_order, len(param_order)


def get_true_param_flat():
    """
    Get flat tensor of true inferred parameter values in the order defined by get_inferred_param_order().

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
        elif ptype == "fault_param":
            params_flat[i] = network_params["fault_parameters"][name]["value"]

    return params_flat


def build_params_from_flat(params_flat, param_order):
    """
    Unpack flat parameter tensor into cable_lengths, load_params, and fault_params dictionaries.

    Args:
        params_flat: [P] tensor of parameter values
        param_order: List from get_inferred_param_order()

    Returns:
        cable_lengths: Dict of cable length tensors
        load_params: Dict of load parameter dicts
        fault_params: Dict of fault parameter tensors
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

    fault_params = {}
    for fault_name, fault_info in network_params["fault_parameters"].items():
        fault_params[fault_name] = torch.tensor(fault_info["value"], dtype=torch.float32, device=device)

    # Override with values from flat tensor
    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            cable_lengths[name] = params_flat[i]
        elif ptype == "load":
            load_params[name][subname] = params_flat[i]
        elif ptype == "fault_param":
            fault_params[name] = params_flat[i]

    return cable_lengths, load_params, fault_params
def compute_real_FIM(var_f):
    """
    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately) \\
    p = number of parameters (inputs) \\
    f = number of frequencies (outputs) \\
    Compute Real Fisher Information Matrix for normalized theta in [0, 1]
    Args:
        var_f: Noise variance (determined by SNR) [] if white noise (constant)
        or [F] if frequency dependent
    Returns:
        I: FIM [p, p] in param_order_list order
    """
    params_flat = get_true_param_flat()
    if SCENARIO == 'no_fault':
    # Compute Jacobian dH/dtheta
        J = jacfwd(H_nofault_wrapper)(params_flat)  # [F, 2, P]
    else:
        J = jacfwd(H_fault_wrapper)(params_flat) #[F, 2, P]
    Delta = J[:, 0, :] + 1j * J[:, 1, :]  # ∂g/∂θ [F, P] complex
    Delta_tilde = J[:, 0, :] - 1j * J[:, 1, :]  # ∂g*/∂θ = (∂g/∂θ)^* for real θ [F, P] complex

    Delta = Delta.unsqueeze(-1)         # [F, P, 1]
    Delta_tilde = Delta_tilde.unsqueeze(-1)  # [F, P, 1]
    # Δ_f ⊗ Δ̃_f^T + Δ̃_f ⊗ Δ_f^T = 2*Re(Δ⊗Δᴴ) which is real
    I_f = (Delta @ Delta_tilde.transpose(-1,-2)) + (Delta_tilde @ Delta.transpose(-1,-2))  # [F, P, P]
    # FIM should be real - take real part (imag should be ~0 due to numerics)
    I = ((1 / var_f) * I_f.sum(dim=0)).real  # [P, P] real and should be symmetric + PSD
    #I2 = D.T @ I @ D
    return I


def compute_real_CRLB(var_f, sorted_keys_s1, sensitivities):
    """
    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately) \\
    P = number of parameters (inputs) \\
    F = number of frequencies (outputs) \\
    Compute Real Fisher Information Matrix and CRLB for P inferred real parameters. \\
    Note FIM and CRLB are normalized here. \\
    Uses float64 precision for numerical stability. 

    Returns:
        CRLB_U1U1T: [P] Dict of Cramér-Rao Lower Bounds for alpha = U1U1^T theta
        CRLB_S2: [] depends on span of null space but will be < P. Dict of Cramér-Rao Lower Bounds for alpha = S2 theta
    """
    param_order_list, _ = get_inferred_param_order()
    I = compute_real_FIM(var_f) #[p, p] Normalized FIM in [0, 1] space for all parameters
    eigvals, eigvecs = torch.linalg.eigh(I)
    print("Eigvals of FIM (descending)", torch.sort(eigvals, descending=True).values)
    eigvals = torch.sort(eigvals, descending=True).values
    v_min = eigvecs[:, 0]
    for name, coeff in zip(param_order_list, v_min):
        print(f"{str(name):30s} {coeff.item(): .4f}")
    lambda_max = eigvals[0]
    lambda_min = eigvals[-1]
    condition_number = lambda_max / lambda_min
    print("lambda_min", lambda_min)
    print("lambda_max", lambda_max)
    print("condition number", condition_number)
    #J_pinv = U1 @ torch.diag(1.0 / Lambda1) @ U1.T
    J_pinv_torch = torch.linalg.pinv(I)
    #Sanity check they should be equal
    #print("max |manual pinv - torch pinv| =", torch.max(torch.abs(J_pinv - J_pinv_torch)).item())

    CRLB_U1U1T = torch.diag(J_pinv_torch)

    # Build mapping from param_order_list index to sorted_keys_s1 key
    def param_order_to_key(entry):
        """Convert param_order_list entry to sorted_keys_s1 format."""
        param_type, name1, name2 = entry
        if param_type == "cable":
            return name1  # e.g., "l_w_4"
        elif param_type == "load":
            return f"{name1}.{name2}"  # e.g., "load_1.C_s"
        elif param_type == "fault_param":
            return name1
    # Create mapping: key -> index in param_order_list.
    key_to_idx = {param_order_to_key(param_order_list[i]): i for i in range(len(param_order_list))}


    print("="*220)
    print(f"{'Idx':<5} {'Parameter':<22} {'Sens':<10} {'CRLB U1U1T':<14} {'Unc U1U1T':<10}")
    print("-"*220)

    # Build dicts sorted by sorted_keys_s1 order
    crlb_u1u1t_dict = {}  # key -> CRLB value
    # Print in sorted_keys_s1 order
    for index, key in enumerate(sorted_keys_s1):
        if key not in key_to_idx:
            continue
        i = key_to_idx[key]
        sens = sensitivities[index]
        crlb_u1u1t = CRLB_U1U1T[i].item()
        crlb_u1u1t_dict[key] = crlb_u1u1t
        uncert_u1u1t_pct = math.sqrt(crlb_u1u1t) * 100

        print(f"{i:<5} {key:<22} {sens:<10} {crlb_u1u1t:<14.2e} {uncert_u1u1t_pct:>5.2f}%")

    print("="*220)
    return crlb_u1u1t_dict

def get_true_param_value(key):
    """Get the true normalized value for a parameter key."""
    if "." in key:
        # Load parameter: "load_name.param_name"
        load_name, param_name = key.split(".")
        return network_params["loads"][load_name][param_name]["value"]
    elif key in network_params.get("fault_parameters", {}):
        # Fault parameter: "fault_position", "Z_fault_real", "Z_fault_imag"
        return network_params["fault_parameters"][key]["value"]
    else:
        # Cable length
        return network_params["cable_lengths"][key]["value"]

def run_inference(H1_noisy, model, guide, sorted_keys, std_f, snr_db, m, M):
    # H1_noisy is (N, F, 2), float NOT COMPLEX
    # print("H1 noise shape", H1_noisy.shape)
    pyro.clear_param_store()

    if OPTIMIZER == "Adam":
        optimizer = pyro.optim.Adam({"lr": LR})
    elif OPTIMIZER == "Adagrad":
        optimizer = pyro.optim.Adagrad({"lr": LR})
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Use 'Adam' or 'Adagrad'.")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=NUM_PARTICLES))
    #svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=NUM_PARTICLES, vectorize_particles=VECTORIZE_PARTICLES, max_plate_nesting=1))
    #svi = SVI(model, guide, optimizer, loss =TraceMeanField_ELBO(num_particles=50, vectorize_particles=True))
    #call guide to initialize params
    guide(H1_noisy, std_f)

    param_store = pyro.get_param_store()
    print(f"\n===== STEP 0 (INITIALIZATION) =====")
    for key in sorted_keys[:10]:
        store_key = key.replace(".", "_") + "_loc"

        if store_key in param_store:
            true_norm = get_true_param_value(key)
            loc = param_store[store_key].detach()
            sig = torch.sigmoid(loc)
            print(f"{key:40s} (sigmoid) = {sig.item():.4f} | True = {true_norm:.4f}")

    losses = []
    best_loss = float("inf")
    best_params = None

    for step in range(NUM_STEPS):
        loss = svi.step(H1_noisy, std_f)
        losses.append(loss)


        if loss < best_loss:
            best_loss = loss
            best_params = {
                name: value.detach().clone()
                for name, value in pyro.get_param_store().items()
        }

        if step % 25 == 0:
            print(f"\n===== SNR {snr_db} | m = {m+1}/{M} | Step {step} | loss = -ELBO: {loss:.6f} =====")
            print("\n Top 10 Most Sensitive Parameters")
            param_store = pyro.get_param_store()

            for key in sorted_keys[:10]:
                store_key = key.replace(".", "_") + "_loc"
                if store_key in param_store:
                    true_norm = get_true_param_value(key)
                    loc = param_store[store_key].detach()
                    sig = torch.sigmoid(loc)
                    print(f"{key:40s} (sigmoid) = {sig.item():.4f} | True = {true_norm:.4f}")



    # Restore best params into Pyro param store
    param_store = pyro.get_param_store()
    for name, value in best_params.items():
        param_store[name] = value.clone()
        
    # Convert best_params into old param_history format:
    # key -> [single best value]
    best_param_history = {
        name: [value.detach().cpu().item()]
        for name, value in best_params.items()
    }
    print("Inference complete.")
    return losses, best_param_history

def plot_CI_and_pred_TF(param_history, seed, snr_db, num_samples=200):
    param_order_list, p = get_inferred_param_order()
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    if SCENARIO == 'with_fault':
        H_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    else:
        H_clean = calculate_Hnw_nofault(cable_lengths, load_params)
    H_clean_db = 20*torch.log10(torch.abs(H_clean))
    
    tf_samples = []

    for i in range(num_samples):
        # Build sampled parameters for both full and reduced models
        sampled_cable_lengths = {}
        sampled_load_params = {}
        sampled_fault_params = {}

        # Sample load params
        for load_name, params in network_params["loads"].items():
            sampled_load_params[load_name] = {}
            for param_name, param_info in params.items():
                pyro_key = f"{load_name}_{param_name}_loc"
                if pyro_key in param_history:
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{load_name}_{param_name}_scale"][-1]
                    z = np.random.normal(loc, scale)
                    sigmoid_z = 1 / (1 + np.exp(-z))
                    sampled_load_params[load_name][param_name] = torch.tensor(sigmoid_z, device=device)
                else:
                    sampled_load_params[load_name][param_name] = torch.tensor(param_info["value"], device=device)

        # Sample cable params
        for cable_name, cable_info in network_params["cable_lengths"].items():
            pyro_key = f"{cable_name}_loc"
            if pyro_key in param_history:  # Check if was inferred
                loc = param_history[pyro_key][-1]
                scale = param_history[f"{cable_name}_scale"][-1]
                z = np.random.normal(loc, scale)
                sigmoid_z = 1 / (1 + np.exp(-z))
                sampled_cable_lengths[cable_name] = torch.tensor(sigmoid_z, device=device)
            else:
                sampled_cable_lengths[cable_name] = torch.tensor(cable_info["value"], device=device)

        # # Sample fault params (if with_fault scenario)
        if SCENARIO == "with_fault":
            for fault_name, fault_info in network_params["fault_parameters"].items():
                pyro_key = f"{fault_name}_loc"
                if pyro_key in param_history:  # Check if was inferred
                    loc = param_history[pyro_key][-1]
                    scale = param_history[f"{fault_name}_scale"][-1]
                    z = np.random.normal(loc, scale)
                    sigmoid_z = 1 / (1 + np.exp(-z))
                    sampled_fault_params[fault_name] = torch.tensor(sigmoid_z, device=device)
                else:
                    sampled_fault_params[fault_name] = torch.tensor(fault_info["value"], device=device)

        # Compute TF with sampled parameters based on FORWARD_MODEL
        if SCENARIO == "with_fault":
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
    plt.plot(freq_range_mhz.numpy(), H_clean_db.detach().numpy(), 'r--', linewidth=1.5, label='Truth')
    plt.fill_between(freq_range_mhz.numpy(), tf_lower, tf_upper, 
                     alpha=0.3, color='steelblue', label='95% CI')
    
    plt.xscale('log')
    plt.xlabel('Frequency (MHz)', fontsize=12)
    plt.ylabel(r'$H_{1,1}$ (dB)', fontsize=12)
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if IS_BAYESIAN:
        filename = f'bayesian_{snr_db}dBSNR_{FILENAME_PREFIX}_tf_posterior_CI_{SCENARIO}_p{p}.pdf'
    else:
        filename = f'{snr_db}dBSNR_{FILENAME_PREFIX}_tf_posterior_CI_{SCENARIO}_p{p}_seed{seed}.pdf'

    if OUTPUT_DIR:
        filename = os.path.join(OUTPUT_DIR, filename)

    # Save plot data
    plot_data = {
        'tf_mean': tf_mean,
        'tf_lower': tf_lower,
        'tf_upper': tf_upper,
        'H_clean_db': H_clean_db.detach().cpu().numpy(),
        'freq_range_mhz': freq_range_mhz.cpu().numpy(),
    }
    np.savez(filename.replace('.pdf', '.npz'), **plot_data)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}, {filename.replace('.pdf', '.npz')}, {filename.replace('.pdf', '.json')}")
    #return tf_mean, tf_lower, tf_upper

def calculate_mse_monte_carlo(var_f, selected_s1, snr_db, M, seed):
    """
    Compute Frequentist MSE via Monte Carlo at specific SNR.
    For each trial:
      1. Generate data y ~ p(y|θ_true) at theta_true
      2. Run SVI to get estimate θ̂
      3. Compute (θ̂ - θ_true)²

    Returns:
        mse_dict: {param_name: MSE} in selected keys order
    """
    param_order_list, _ = get_inferred_param_order()
    squared_errors = {key: [] for key in selected_s1}
    std_f = torch.sqrt(var_f / 2)

    # Compute H_clean from current network_params (which should be set to theta_true in main)
    cable_lengths, load_params, fault_params = build_params_from_flat(get_true_param_flat(), param_order_list)
    if SCENARIO == 'with_fault':
        H_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    else:
        H_clean = calculate_Hnw_nofault(cable_lengths, load_params)

    for m in range(M):
        print(f"Run {m+1}/{M}")

        # 1. Generate noisy observation (different observation each run because of noise)
        H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                     1j * std_f * torch.randn_like(H_clean.imag)
        H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0)) #[1, F, 2]

        # 2. Run SVI inference 
        #auto_guide = AutoMultivariateNormal(model_no_fault)  # Full covariance guide
        if SCENARIO == 'with_fault':
            losses, param_history = run_inference(H1_noisy, model_with_fault, guide, selected_s1, std_f, snr_db, m, M)
        else:
            losses, param_history = run_inference(H1_noisy, model_no_fault, guide, selected_s1, std_f, snr_db, m, M)
        
        # 3. Extract posterior means for this run
        posterior_means = extract_posterior_means(param_history)

        # Plot TF vs reconstructed TF from these posterior means - just plot for last one
        if m == M-1:
            plot_CI_and_pred_TF(param_history, seed, snr_db)

        # 4. Compute squared errors vs true theta
        for key in selected_s1:
            # Get true value directly from network_params
            true_val = get_true_param_value(key)

            posterior_key = key.replace(".", "_")
            if posterior_key in posterior_means:
                estimate = posterior_means[posterior_key]
                squared_errors[key].append((estimate - true_val)**2)

    # Average squared errors
    mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
    return mse_dict

def print_cosine_similarity_matrix(param_keys, cosine_matrix, threshold=0.94):
    """Print cosine similarity matrix in a readable format."""
    
    print("\n" + "="*70)
    print("JACOBIAN COSINE SIMILARITY ANALYSIS")
    print("="*70)
    
    n_params = len(param_keys)
    
    # Shorten parameter names for display
    def shorten_name(name, max_len=12):
        if "." in name:
            parts = name.split(".")
            # e.g., "load_0.C_m_leak" -> "L0.Cm"
            load_part = parts[0].replace("load_", "L")
            param_part = parts[1].replace("_leak", "").replace("_const", "")
            return f"{load_part}.{param_part}"[:max_len]
        return name[:max_len]
    
    short_keys = [shorten_name(k) for k in param_keys]
    
    # Find high correlations first
    high_corrs = []
    for i in range(n_params):
        for j in range(i+1, n_params):
            val = cosine_matrix[i, j].item() if hasattr(cosine_matrix[i, j], 'item') else cosine_matrix[i, j]
            if abs(val) > threshold:
                high_corrs.append((param_keys[i], param_keys[j], val))
    
    # Print high correlations summary first
    if high_corrs:
        print(f"\n⚠️  HIGH CORRELATIONS (|r| > {threshold}):")
        print("-" * 50)
        for k1, k2, val in sorted(high_corrs, key=lambda x: -abs(x[2])):
            print(f"  {k1:<25} ↔ {k2:<25} r={val:+.4f}")
        print("-" * 50)
    else:
        print(f"\n✓ No correlations above {threshold}")
    
    # For large matrices, print in blocks
    block_size = 8  # Number of columns per block
    n_blocks = (n_params + block_size - 1) // block_size
    
    print(f"\nFull correlation matrix ({n_params}x{n_params}):")
    
    for block in range(n_blocks):
        col_start = block * block_size
        col_end = min((block + 1) * block_size, n_params)
        
        if n_blocks > 1:
            print(f"\n--- Columns {col_start+1}-{col_end} of {n_params} ---")
        
        # Header row
        print(f"{'':>18}", end="")
        for j in range(col_start, col_end):
            print(f"{short_keys[j]:>10}", end="")
        print()
        
        # Separator
        print(f"{'':>18}" + "-" * (10 * (col_end - col_start)))
        
        # Data rows
        for i in range(n_params):
            print(f"{short_keys[i]:>16} |", end="")
            for j in range(col_start, col_end):
                val = cosine_matrix[i, j].item() if hasattr(cosine_matrix[i, j], 'item') else cosine_matrix[i, j]
                if i == j:
                    print(f"{'---':>10}", end="")  # Diagonal
                elif abs(val) > threshold:
                    print(f"{val:>9.3f}*", end="")  # High correlation
                elif abs(val) > 0.7:
                    print(f"{val:>9.3f}°", end="")  # Moderate
                else:
                    print(f"{val:>10.3f}", end="")
            print()
    
    print("="*70)

def compute_jacobian_cosine_similarity(param_keys, scenario="no_fault"):
    """
    Compute cosine similarity between Jacobian columns for specified parameters.

    Each Jacobian column J_i = ∂H/∂θ_i is a vector over frequencies.
    Cosine similarity measures if two parameters affect H in the same "direction"
    (same frequency pattern) regardless of magnitude.

    Uses jacfwd (automatic differentiation) for exact gradients.

    Args:
        param_keys: List of sensitive parameter keys from most to least to analyze (e.g., ["load_11.C_m_leak", "load_15.C_m_leak"])
        scenario: "no_fault" or "with_fault"

    Returns:
        jacobians: Dict mapping param_key -> Jacobian column (complex vector over frequencies)
        cosine_matrix: 2D numpy array of cosine similarities between all pairs
    """
    # Get true parameter values as flat tensor
    params_flat = get_true_param_flat()
    param_order, _ = get_inferred_param_order()

    # Build mapping from param_key to param_order
    # (param_key always most sensitive -> least sensitive)
    # (param_order always cable first then loads parameters)
    key_to_idx = {}
    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            key_to_idx[name] = i
        elif ptype == "load":
            key_to_idx[f"{name}.{subname}"] = i

    # Compute full real Jacobian using automatic differentiation in param_order
    J_full = jacfwd(H_nofault_wrapper)(params_flat)  # [F, 2, P]

    # Create dict of complex Jacobian columns [F] in order of param_keys
    jacobians = {}
    for key in param_keys:
        idx = key_to_idx[key]
        # J_full[:, :, idx] is [F, 2] - real and imag parts
        J_col = torch.complex(J_full[:, 0, idx], J_full[:, 1, idx])  # [F] complex
        jacobians[key] = J_col

    # Compute cosine similarity matrix betwen param_keys
    n_params = len(param_keys)
    cosine_matrix = np.zeros((n_params, n_params))

    for i, key_i in enumerate(param_keys):
        J_i = jacobians[key_i]
        # Flatten complex to real: [Re(J), Im(J)]
        J_i_real = torch.cat([J_i.real.flatten(), J_i.imag.flatten()])
        norm_i = torch.linalg.norm(J_i_real)

        for j, key_j in enumerate(param_keys):
            J_j = jacobians[key_j]
            J_j_real = torch.cat([J_j.real.flatten(), J_j.imag.flatten()])
            norm_j = torch.linalg.norm(J_j_real)

            # Cosine similarity
            if norm_i > 0 and norm_j > 0:
                cosine_matrix[i, j] = (torch.dot(J_i_real, J_j_real) / (norm_i * norm_j)).item()
            else:
                cosine_matrix[i, j] = 0.0
    
    print_cosine_similarity_matrix(param_keys, cosine_matrix)
    return jacobians, cosine_matrix


def perform_local_prior_averaged_sensitivity_analysis(alpha, M, scenario):
    """
    Sample M θ_nominal values from the prior Beta(alpha, alpha).
    Compute local sensitivity around each sampled θ.
    Average the sensitivity scores across all M samples.
    Select a fixed top-p set based on averaged sensitivities.

    This is useful for Bayesian estimation where θ_true varies,
    giving a more robust parameter selection than single-point local sensitivity.

    Args:
        alpha: Beta distribution parameter (Beta(alpha, alpha))
        M: Number of Monte Carlo samples from prior
        scenario: "no_fault" or "with_fault"

    Returns:
        selected_keys: List of top p most sensitive params sorted by sensitivity
        sorted_keys: List of all params sorted from most sensitive to least sensitive
        sensitivities: List of averaged sensitivity values (%) corresponding to sorted_keys
    """
    param_order_list, n_params = get_inferred_param_order()

    # Build parameter key list from param_order
    param_keys = []
    for item in param_order_list:
        if item[0] == 'cable':
            param_keys.append(item[1])
        elif item[0] == 'load':
            param_keys.append(f"{item[1]}.{item[2]}")
        elif item[0] == 'fault':
            param_keys.append(item[1])

    # Accumulate sensitivities across M samples
    sensitivity_accumulator = torch.zeros(n_params, dtype=torch.float64)

    print(f"\n--- Prior-Averaged Sensitivity Analysis (M={M}, α={alpha}) ---")

    for m in range(M):
        # 1. Sample theta from prior Beta(alpha, alpha)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        theta_true_normalized = beta_dist.sample((n_params,))

        # 2. Set network_params to this sampled theta
        set_network_params_from_normalized(theta_true_normalized, param_order_list)

        # 3. Get flat params and compute Jacobian
        params_flat = get_true_param_flat()
        J = jacfwd(H_nofault_wrapper)(params_flat)  # [F, 2, P]
        n_freq = J.shape[0]
        J_flat = J.reshape(n_freq * 2, n_params)  # [F*2, P]

        # 4. Sensitivity = norm of each column (each parameter's Jacobian)
        sensitivities_raw = torch.norm(J_flat, dim=0)  # [P]

        # 5. Normalize and accumulate
        sensitivities_normalized = sensitivities_raw / sensitivities_raw.sum()
        sensitivity_accumulator += sensitivities_normalized

        if (m + 1) % 10 == 0:
            print(f"  Processed {m + 1}/{M} samples...")

    # Average sensitivities
    avg_sensitivities = sensitivity_accumulator / M

    # Create dict mapping param_key -> averaged sensitivity
    sensitivity_dict = {param_keys[i]: avg_sensitivities[i].item()
                        for i in range(n_params)}

    # Sort by sensitivity (descending)
    sorted_params = sorted(sensitivity_dict.keys(),
                          key=lambda k: sensitivity_dict[k],
                          reverse=True)

    # Select top p
    selected = sorted_params[:p]

    # Build sensitivities list in sorted order (as percentages)
    sensitivities = [sensitivity_dict[k] * 100 for k in sorted_params]

    # Print results
    print(f"\n--- Prior-Averaged Sensitivity Results ---")
    for idx, k in enumerate(sorted_params):
        if idx == p:
            print(f"--- Top {p} selected above this line ---")
        print(f"{k}: {sensitivity_dict[k]*100:.5f}%")

    print(f"\nSelected top {p} most sensitive parameters: {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    # Set inferred=False for parameters NOT in top p
    if p > 0:
        disabled_count = 0
        for param_key in sorted_params[p:]:
            if "." in param_key:
                parts = param_key.split(".")
                entity_name = parts[0]
                param_name = parts[1]
                if entity_name in network_params["loads"]:
                    if param_name in network_params["loads"][entity_name]:
                        network_params["loads"][entity_name][param_name]["inferred"] = False
                        disabled_count += 1
            else:
                if param_key in network_params["cable_lengths"]:
                    network_params["cable_lengths"][param_key]["inferred"] = False
                    disabled_count += 1
                elif "fault_parameters" in network_params and param_key in network_params["fault_parameters"]:
                    network_params["fault_parameters"][param_key]["inferred"] = False
                    disabled_count += 1

        print(f"\nDisabled inference for {disabled_count} parameters (not in top {p})")

    sorted_keys = sorted_params
    return selected, sorted_keys, sensitivities

def perform_local_sensitivity_analysis():
    """
    Perform LOCAL sensitivity analysis at the specific θ_true point using Jacobians.
    
    Unlike global sensitivity which sweeps the entire range, this computes
    the gradient ∂H/∂θ at the current parameter values.

    Returns:
        selected_keys: List of top p most sensitive params sorted by sensitivity
        sorted_keys: List of all params sorted from most sensitive to least sensitive
        sensitivities: List of sensitivity values (%) corresponding to sorted_keys
    """
    params_flat = get_true_param_flat()
    param_order, _ = get_inferred_param_order()

    # Select wrapper based on scenario
    if SCENARIO == "no_fault":
        wrapper = H_nofault_wrapper
    elif SCENARIO == "with_fault":
        wrapper = H_fault_wrapper
    else:
        raise ValueError(f"Unknown scenario: {SCENARIO}. Use 'no_fault' or 'with_fault'.")

    # Compute Jacobian dH/dtheta
    J = jacfwd(wrapper)(params_flat)  # [F, 2, P]
    n_freq = J.shape[0]
    n_params = J.shape[2]
    J_flat = J.reshape(n_freq * 2, n_params)  # [F*2, P]
    
    # Sensitivity = norm of each column (each parameter's Jacobian)
    sensitivities_raw = torch.norm(J_flat, dim=0)  # [P]
    # Normalize to percentages
    sensitivities_normalized = sensitivities_raw / sensitivities_raw.sum()

    # Build parameter key list from param_order
    param_keys = []
    for item in param_order:
        if item[0] == 'cable':
            # ('cable', 'l_w_0', None)
            param_keys.append(item[1])
        elif item[0] == 'load':
            # ('load', 'load_0', 'C_m_leak')
            param_keys.append(f"{item[1]}.{item[2]}")
        elif item[0] == 'fault_param':
            # ('fault_param', 'Z_fault_real', None)
            param_keys.append(item[1])
    
    # Create dict mapping param_key -> sensitivity
    sensitivity_dict = {param_keys[i]: sensitivities_normalized[i].item() 
                        for i in range(n_params)}
    
    # Sort by sensitivity (descending)
    sorted_params = sorted(sensitivity_dict.keys(), 
                          key=lambda k: sensitivity_dict[k], 
                          reverse=True)
    
    # Select top p
    selected = sorted_params[:p]

    # Build sensitivities list in sorted order (as percentages)
    sensitivities = [sensitivity_dict[k] * 100 for k in sorted_params]
    
    # Print results
    print("\n--- LOCAL Sensitivity Analysis (at θ_true) ---")
    for idx, k in enumerate(sorted_params):
        if idx == p:
            print(f"--- Top {p} selected above this line ---")
        print(f"{k}: {sensitivity_dict[k]*100:.5f}%")
    
    print(f"\nSelected top {p} most sensitive parameters: {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    # Set inferred=False for parameters NOT in top p
    if p > 0:
        disabled_count = 0
        for param_key in sorted_params[p:]:
            if "." in param_key:
                parts = param_key.split(".")
                entity_name = parts[0]
                param_name = parts[1]
                if entity_name in network_params["loads"]:
                    if param_name in network_params["loads"][entity_name]:
                        network_params["loads"][entity_name][param_name]["inferred"] = False
                        disabled_count += 1
            else:
                if param_key in network_params["cable_lengths"]:
                    network_params["cable_lengths"][param_key]["inferred"] = False
                    disabled_count += 1
                elif "fault_parameters" in network_params and param_key in network_params["fault_parameters"]:
                    network_params["fault_parameters"][param_key]["inferred"] = False
                    disabled_count += 1
        
        print(f"\nDisabled inference for {disabled_count} parameters (not in top {p})")
    
    sorted_keys = sorted_params
    return selected, sorted_keys, sensitivities
def remove_correlated_parameters(selected_params, cosine_similarity_matrix, threshold=0.94):
    """
    Given cosine similarity matrix of top p most sensitive parameters, 
    remove parameters that are highly correlated >= threshold.
    """
    remove_indices = set()
    p = len(selected_params)

    for i in range(p):
        if i in remove_indices:
            continue
        for j in range(i + 1, p):
            if j in remove_indices:
                continue

            val = cosine_similarity_matrix[i, j]
            if abs(val) >= threshold:
                print(f"High correlation: {selected_params[i]} vs {selected_params[j]} = {val:.4f}")
                print(f"Removing {selected_params[j]} (less sensitive)")
                remove_indices.add(j)

    # Turn off inference for removed parameters
    for idx in remove_indices:
        param_key = selected_params[idx]
        if "." in param_key:
            # Load parameter: "load_11.C_m_leak"
            parts = param_key.split(".")
            load_name = parts[0]
            param_name = parts[1]
            network_params["loads"][load_name][param_name]["inferred"] = False
        else:
            # Cable length parameter
            network_params["cable_lengths"][param_key]["inferred"] = False
        print(f"Set inferred=False for {param_key}")

    # Build new list (cleaner than popping)
    selected_new = [selected_params[i] for i in range(p) if i not in remove_indices]
    p_new = len(selected_new)

    print(f"Kept {p_new}/{p} parameters: {selected_new}")
    # return p, selected_s1
    return p_new, selected_new


def beta_prior_fim_closed_form(alpha):
    """Closed form prior FIM for Beta(α,α) priors."""
    if alpha <= 2:
        raise ValueError("alpha must be > 2 for finite FIM")
    
    j_pi_scalar = 4 * (2*alpha - 1) * (alpha - 1) / (alpha - 2)
    J_pi = j_pi_scalar * torch.eye(p)  # Diagonal
    return J_pi

def compute_expected_data_fim(snr_db, alpha, num_samples=100):
    """
    Compute E_π[I(θ)] via Monte Carlo.

    Returns:
        E_I: Expected data FIM [p, p]
    """
    snr_lin = 10.0 ** (snr_db / 10.0)

    # Sample parameter values from prior
    beta_dist = torch.distributions.Beta(alpha, alpha)
    theta_samples = beta_dist.sample((num_samples, p))  # [num_samples, p]
    # Accumulate FIMs
    E_I = torch.zeros(p, p)
    param_order_list, _ = get_inferred_param_order()
    for m in range(num_samples):
        print(f"{m+1} out of Monte Carlo {num_samples} for E_π[I(θ)] at {snr_db} dB")
        set_network_params_from_normalized(theta_samples[m], param_order_list)

        # Compute H_clean for this theta
        cable_lengths, load_params, fault_params = build_params_from_flat(get_true_param_flat(), param_order_list)
        if SCENARIO == "with_fault":
            H_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
        else:
            H_clean = calculate_Hnw_nofault(cable_lengths, load_params)

        # Compute var_f for this theta
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin

        # Compute FIM at this theta
        I_phi = compute_real_FIM(var_f)

        E_I += I_phi

    E_I /= num_samples
    return E_I

# Convert selected_s1 keys to param_order_list tuple format
def key_to_tuple(key):
    """Convert 'l_w_4', 'load_1.C_m_leak', or 'fault_position' to tuple format."""
    if '.' in key:
        # Load parameter: 'load_1.C_m_leak' → ('load', 'load_1', 'C_m_leak')
        parts = key.split('.')
        return ('load', parts[0], parts[1])
    elif key in network_params.get("fault_parameters", {}):
        # Fault parameter: 'fault_position' → ('fault_param', 'fault_position', None)
        return ('fault_param', key, None)
    else:
        # Cable parameter: 'l_w_4' → ('cable', 'l_w_4', None)
        return ('cable', key, None)
    

def compute_real_BCRLB(snr_db, selected_keys, alpha, num_samples=100):
    """
    Compute Bayesian CRLB.
    
    Args:
        snr_db: SNR in dB
        selected_keys: List of parameter keys to extract (in desired order)
        alpha: Beta prior parameter
        num_samples: Number of MC samples for E[I(θ)]
    
    Returns:
        bcrlb_dict: {param_name: BCRLB_value} in selected_keys order
    """

    # Compute E[I(θ)]
    E_I = compute_expected_data_fim(snr_db, alpha, num_samples)
    #print("E_I shape", E_I.shape)
    #print("E_I", E_I)
    
    # Compute J_π (prior FIM)
    J_pi = beta_prior_fim_closed_form(alpha)
    # print("J_pi", J_pi)
    # print("J_pi shape", J_pi.shape)
    # Bayesian FIM
    J_B = E_I + J_pi
    
    # Bayesian CRLB
    BCRLB = torch.linalg.inv(J_B)
    bcrlb_diag_full = torch.diag(BCRLB)  # [p] in param_order_list order
    print("bcrlb_diag", bcrlb_diag_full)
    print("shape", bcrlb_diag_full.shape)
    # Extract selected_keys subset in correct order
    print("selected keys", selected_keys)
    bcrlb_dict = {}
    param_order_list, _ = get_inferred_param_order()
    for key in selected_keys:
        key_tuple = key_to_tuple(key)
        print("key_tuple", key_tuple)
        if key_tuple in param_order_list:
            idx = param_order_list.index(key_tuple)
            bcrlb_dict[key] = bcrlb_diag_full[idx].item()
        else:
            print(f"Warning: {key} ({key_tuple}) not found in param_order_list")
    return bcrlb_dict

def calculate_bayesian_mse_monte_carlo(snr_db, selected_s1, alpha, M, seed):
    """
    Compute Bayesian MSE via Monte Carlo at specific SNR.

    For each trial:
      1. Sample θ_true ~ Beta(α, α)
      2. Generate data y ~ p(y|θ_true)
      3. Run SVI to get estimate θ̂
      4. Compute (θ̂ - θ_true)²

    Returns:
        bayesian_mse_dict: {param_name: Bayesian MSE} in selected keys order
    """
    # Set seeds for reproducibility
    #torch.manual_seed(seed)
    #pyro.set_rng_seed(seed)

    param_order_list, _ = get_inferred_param_order()
    squared_errors = {key: [] for key in selected_s1}
    snr_lin = 10.0 ** (snr_db / 10.0)

    for m in range(M):
        print(f"Trial {m+1}/{M}")

        # 1. Sample true theta from prior
        beta_dist = torch.distributions.Beta(alpha, alpha)
        theta_true_normalized = beta_dist.sample((p,))

        # 2. Set network_params to this sampled theta
        set_network_params_from_normalized(theta_true_normalized, param_order_list)

        # 3. Generate clean signal from this theta
        cable_lengths, load_params, fault_params = build_params_from_flat(get_true_param_flat(), param_order_list)
        if SCENARIO == "with_fault":
            H_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
        else:
            H_clean = calculate_Hnw_nofault(cable_lengths, load_params)
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin
        std_f = torch.sqrt(var_f / 2)
        H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                     1j * std_f * torch.randn_like(H_clean.imag)
        H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))

        # 4. Run SVI to get estimate
        pyro.clear_param_store()
        if SCENARIO == "with_fault":
            losses, param_history = run_inference(H1_noisy, model_with_fault, guide, selected_s1, std_f, snr_db, m, M)
        else:
            losses, param_history = run_inference(H1_noisy, model_no_fault, guide, selected_s1, std_f, snr_db, m, M)

        # 5. Extract posterior mean
        posterior_means = extract_posterior_means(param_history)

        # Plot TF vs reconstructed TF from these posterior means - just plot for last seed
        if m == M-1:
            plot_CI_and_pred_TF(param_history, seed, snr_db)

        # 6. Compute squared errors vs sampled true θ
        for key in selected_s1:
            # Get true value directly from network_params
            true_val = get_true_param_value(key)

            posterior_key = key.replace(".", "_")
            if posterior_key in posterior_means:
                estimate = posterior_means[posterior_key]
                squared_errors[key].append((estimate - true_val)**2)

    # Average squared errors
    bayesian_mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
    
    return bayesian_mse_dict

#Plotting stuff
def plot_rmse_vs_crlb_snr_sweep(snr_dbs, rmse_results, crlb_results, selected_keys, p, seed,
                                 is_bayesian=False, M=None, alpha=None, filename=None):
    """
    Plot RMSE vs sqrt(CRLB) across SNR for each parameter.
    
    Args:
        snr_dbs: List of SNR values in dB
        rmse_results: Dict {param_name: [rmse_values_across_snr]}
        crlb_results: Dict {param_name: [sqrt_crlb_values_across_snr]}
        selected_keys: List of parameter names
        is_bayesian: If True, use Bayesian labels; if False, use frequentist labels
        M: Number of Monte Carlo trials (for title)
        alpha: Beta prior parameter (for Bayesian plots)
        filename: Output filename (if None, auto-generated)
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # Set labels based on Bayesian vs Frequentist
    if is_bayesian:
        rmse_label = f'Bayesian RMSE (M={M})' if M else 'Bayesian RMSE'
        crlb_label = 'sqrt(BCRLB)'
        title_prefix = 'Bayesian RMSE vs sqrt(BCRLB)'
        title_suffix = f'(α={alpha}, M={M} trials per SNR)' if alpha and M else ''
    else:
        rmse_label = f'RMSE (M={M})' if M else 'RMSE'
        crlb_label = 'sqrt(CRLB)'
        title_prefix = 'RMSE vs sqrt(CRLB)'
        title_suffix = f'(M={M} trials per SNR)' if M else ''

    for idx, key in enumerate(selected_keys):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        # Plot data
        ax.plot(snr_dbs, rmse_results[key], 'bo-', label=rmse_label, markersize=6)
        ax.plot(snr_dbs, crlb_results[key], 'r--s', label=crlb_label, linewidth=2, markersize=4)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error')
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(selected_keys), len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(f'{title_prefix} - SNR Sweep {title_suffix}', fontsize=14, y=0.995)
    plt.tight_layout()

    # Generate filename if not provided
    if filename is None:
        if is_bayesian:
            filename = f"bayesian_rmse_vs_bcrlb_snr_sweep_alpha{alpha}_M{M}_p{p}_seed{seed}.pdf"
        else:
            filename = f"rmse_vs_crlb_snr_sweep_M{M}_p{p}_seed{seed}.pdf"

    if OUTPUT_DIR:
        filename = os.path.join(OUTPUT_DIR, filename)

    # Save plot data
    plot_data = {'snr_dbs': np.array(snr_dbs)}
    for key in selected_keys:
        if key in rmse_results and len(rmse_results[key]) > 0:
            plot_data[f'rmse_{key}'] = np.array(rmse_results[key])
        if key in crlb_results and len(crlb_results[key]) > 0:
            plot_data[f'crlb_{key}'] = np.array(crlb_results[key])
    np.savez(filename.replace('.pdf', '.npz'), **plot_data)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}, {filename.replace('.pdf', '.npz')}")


def plot_nll_vs_L1_complex_mtl(snr_db):
    """
    plot fault_position vs NLL.
    """
    print("snr db", snr_db)

    param_order_list, _ = get_inferred_param_order() #P = total number of network params (exluding fault)
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
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
    true_L1 = network_params["fault_parameters"]["fault_position"]["value"]

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
    plt.axvline(x=true_L1, color='r', linestyle='--', linewidth=2, label=f'True L1={true_L1:.2f}')
    plt.xlabel('L1 (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Fault Location L1 (Complex Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nll_vs_L1_complex", dpi=300, bbox_inches='tight')
    plt.close()

def plot_nll_vs_ZFre_complex_mtl(snr_db):
    """
    Plot Z_fault_real (real part of fault impedance) vs NLL.
    """
    print("snr db", snr_db)

    param_order_list, _ = get_inferred_param_order()
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    obs_tf_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigpow_s2 = torch.mean(torch.abs(obs_tf_clean)**2)
    var_f_s2 = sigpow_s2 / snr_lin
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    obs_tf_noisy = obs_tf_clean + std_f_s2 * torch.randn_like(obs_tf_clean.real) + \
                  1j * std_f_s2 * torch.randn_like(obs_tf_clean.imag)

    # Sweep Z_fault_real from 0 to 1 (normalized)
    ZFre_normalized = torch.linspace(0.01, 0.99, 199, dtype=torch.float32)
    losses = []
    original_ZFre = fault_params["Z_fault_real"]  # Save original value
    true_ZFre = network_params["fault_parameters"]["Z_fault_real"]["value"]
    with torch.no_grad():
        for ZFre in ZFre_normalized:
            fault_params["Z_fault_real"] = ZFre
            pred_tf = calculate_Hnw(cable_lengths, load_params, fault_params)
            diff = obs_tf_noisy - pred_tf
            nll = (diff.abs().pow(2) / var_f_s2).sum()
            losses.append(nll.item())
    fault_params["Z_fault_real"] = original_ZFre  # Restore original value

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ZFre_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
    plt.axvline(x=true_ZFre, color='r', linestyle='--', linewidth=2, label=f'True Z_fault_real={true_ZFre:.2f}')
    plt.xlabel('Z_fault_real (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Z_fault_real (Complex Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nll_vs_ZFre_complex", dpi=300, bbox_inches='tight')
    plt.close()

def plot_nll_vs_ZFim_complex_mtl(snr_db):
    """
    Plot Z_fault_imag (imaginary part of fault impedance) vs NLL.
    """
    print("snr db", snr_db)

    param_order_list, _ = get_inferred_param_order()
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    obs_tf_clean = calculate_Hnw(cable_lengths, load_params, fault_params)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigpow_s2 = torch.mean(torch.abs(obs_tf_clean)**2)
    var_f_s2 = sigpow_s2 / snr_lin
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    obs_tf_noisy = obs_tf_clean + std_f_s2 * torch.randn_like(obs_tf_clean.real) + \
                  1j * std_f_s2 * torch.randn_like(obs_tf_clean.imag)

    # Sweep Z_fault_imag from 0 to 1 (normalized)
    ZFim_normalized = torch.linspace(0.01, 0.99, 199, dtype=torch.float32)
    losses = []
    original_ZFim = fault_params["Z_fault_imag"]  # Save original value
    true_ZFim = network_params["fault_parameters"]["Z_fault_imag"]["value"]
    with torch.no_grad():
        for ZFim in ZFim_normalized:
            fault_params["Z_fault_imag"] = ZFim
            pred_tf = calculate_Hnw(cable_lengths, load_params, fault_params)
            diff = obs_tf_noisy - pred_tf
            nll = (diff.abs().pow(2) / var_f_s2).sum()
            losses.append(nll.item())
    fault_params["Z_fault_imag"] = original_ZFim  # Restore original value

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ZFim_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
    plt.axvline(x=true_ZFim, color='r', linestyle='--', linewidth=2, label=f'True Z_fault_imag={true_ZFim:.2f}')
    plt.xlabel('Z_fault_imag (normalized)', fontsize=12)
    plt.ylabel('Loss (NLL)', fontsize=12)
    plt.title('Loss Landscape vs Z_fault_imag (Complex Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nll_vs_ZFim_complex", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    start_time = time.time()
    # Generate load params deterministically
    total_params, load_types = generate_load_parameters_deterministic(num_loads)
    num_cable_params = len(network_params["cable_lengths"])
    num_fault_params = len(network_params["fault_parameters"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of fault parameters: {num_fault_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params + num_fault_params}")
    print(f"Load type distribution: {load_types}")

    # Turn OFF cable/load inference for stage 2
    for cable_name in network_params["cable_lengths"]:
        network_params["cable_lengths"][cable_name]["inferred"] = False
    for load_name in network_params["loads"]:
        for param_name in network_params["loads"][load_name]:
            network_params["loads"][load_name][param_name]["inferred"] = False
    # Turn OFF fault parameter inference for stage 1
    # for fault_name in network_params["fault_parameters"]:
    #     network_params["fault_parameters"][fault_name]["inferred"] = False


    # Create output folder for all plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=== Output folder: {OUTPUT_DIR} ===")

    param_order_list, P = get_inferred_param_order() 
    g = torch.Generator()
    g.manual_seed(seed)
    theta_true3 = torch.full([P], 0.25)
    a, b = 0.3, 0.7
    theta_true = a + (b - a) * torch.rand(P, generator=g)
    beta_dist = torch.distributions.Beta(alpha, alpha)
    theta_true2 = torch.zeros(P)
    for i in range(P):
        # Beta uses gamma internally, which respects torch's RNG
        # torch.manual_seed(seed + i)  # Different seed per param but reproducible
        theta_true2[i] = beta_dist.sample()

    theta_true4 = torch.zeros(P)
    for i, param_tuple in enumerate(param_order_list):
        if param_tuple[0] == "cable":  # Cable length parameter
            torch.manual_seed(seed + i)
            theta_true4[i] = beta_dist.sample()
        else:  # Load parameter
            theta_true4[i] = 0.25
    #set_network_params_from_normalized(theta_true2, param_order_list)
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    H_fault = calculate_Hnw(cable_lengths, load_params, fault_params)
    sigpow = torch.mean(torch.abs(H_fault)**2)
    # H_clean = calculate_Hnw_nofault(cable_lengths, load_params) #[F] when all inputs are scalars
    # sigpow = torch.mean(torch.abs(H_clean)**2)

    # selected_s1, sorted_keys_s1, sensitivities = perform_local_prior_averaged_sensitivity_analysis(alpha, 100, "no_fault")
    selected_s1, sorted_keys_s1, sensitivities = perform_local_sensitivity_analysis()
    # selected_s1, sorted_keys_s1, sensitivities = perform_global_sensitivity_analysis(cable_lengths, load_params)
    #selected_s1 = []
    
    #plot_nll_vs_L1_complex_mtl(40.0)
    #plot_nll_vs_ZFre_complex_mtl(40.0)
    #plot_nll_vs_ZFim_complex_mtl(40.0)
    
    #jacobians, csm = compute_jacobian_cosine_similarity(
    #    selected_s1, scenario='no_fault'
    #)
    #p, selected_s1 = remove_correlated_parameters(selected_s1, csm)

    #snr_dbs = [40]
    snr_dbs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    #snr_dbs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    rmse_results = {key: [] for key in selected_s1}
    crlb_results = {key: [] for key in selected_s1}

    bayesian_rmse_results = {key: [] for key in selected_s1}
    bayesian_crlb_results = {key: [] for key in selected_s1}

    for snr_db in snr_dbs:
        print(f"\n{'='*50}")
        print(f"SNR = {snr_db} dB")
        print('='*50)
        snr_lin = 10.0 ** (snr_db / 10.0)
        var_f = sigpow / snr_lin  # Compute var_f for this SNR only for frequentist

        # Standard CRLB + RMSE
        crlb_u1u1t_dict = compute_real_CRLB(var_f, selected_s1, sensitivities)
        mse = calculate_mse_monte_carlo(var_f, selected_s1, snr_db, M, seed)
        print(f"RMSE (M={M}):", {k: f"{math.sqrt(v):.4f}" for k, v in mse.items()})
        print(f"sqrt(CRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in crlb_u1u1t_dict.items()})

        # Bayesian CRLB + BRMSE
        # bcrlb_dict = compute_real_BCRLB(snr_db, selected_s1, alpha, M2)
        # bayesian_mse = calculate_bayesian_mse_monte_carlo(snr_db, selected_s1, alpha, M, seed)
        # print(f"Bayesian RMSE (M={M}):", {k: f"{math.sqrt(v):.4f}" for k, v in bayesian_mse.items()})
        # print(f"sqrt(BCRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in bcrlb_dict.items()})

       #  Store results for plotting
        for key in selected_s1:
            if key in mse and key in crlb_u1u1t_dict:
                rmse_results[key].append(math.sqrt(mse[key]))
                crlb_results[key].append(math.sqrt(crlb_u1u1t_dict[key]))  
        
        # Store results for plotting
        # for key in selected_s1:
        #     if key in bayesian_mse and key in bcrlb_dict:
        #         bayesian_rmse_results[key].append(math.sqrt(bayesian_mse[key]))
        #         bayesian_crlb_results[key].append(math.sqrt(bcrlb_dict[key]))  
        
    plot_rmse_vs_crlb_snr_sweep(snr_dbs, rmse_results, crlb_results, selected_s1, p, seed,
                               is_bayesian=False, M=M)
    #plot_rmse_vs_crlb_snr_sweep(snr_dbs, bayesian_rmse_results, bayesian_crlb_results, selected_s1, p, seed,
    #                           is_bayesian=True, M=M, alpha=alpha)

    
    print("My program took", time.time() - start_time, "to run")

    # Zip the output folder and remove it
    zip_filename = f"{OUTPUT_DIR}.zip"
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)  # Remove the folder, keep only the zip
    print(f"\n=== All outputs zipped to: {zip_filename} ===")