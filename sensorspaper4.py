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
OPTIMIZER = "Adagrad"  # "Adam" or "Adagrad"
LR = 0.2  # Learning rate for optimizer

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
    3,  # load_16 R2-O3  Motor
    3,  # load_17 R2-O2  Motor
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


def model_no_fault(H1_noisy, std_f):
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


    H1_pred_c = calculate_Hnw(cable_lengths, load_params, fault_params).unsqueeze(0).expand(N, -1)
    H1_pred = torch.view_as_real(H1_pred_c)

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

#Posterior analysis
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

def set_network_params_from_normalized(sampled_theta, param_order_list):
    """
    Update network_params with sampled theta. Used to calculate Bayesian CLRB.
    Args:
        sampled_theta: Sampled tensor of theta of shape [p]
        param_order_list: List of selected_keys like ["load_1.C_mleak, etc.]
    """
    counter = 0
    for params in param_order_list: 
        if params[0] == "load":
            entity_name = params[1]
            param_name = params[2]
            min, max = network_params["loads"][entity_name][param_name]["range"]
            network_params["loads"][entity_name][param_name]["value"] = denormalize(sampled_theta[counter], min, max).item()
            counter = counter + 1
        else: 
            cable_name = params[1]
            min, max = network_params["cable_lengths"][cable_name]["infer_range"]
            network_params["cable_lengths"][cable_name]["value"] = denormalize(sampled_theta[counter], min, max).item()
            counter = counter + 1    



def calculate_rmse_from_trials(posterior_means_list, sorted_keys, true_normalized_value=0.25):
    """
    Calculate RMSE across M Monte Carlo trials.

    Args:
        posterior_means_list: list of M dicts, each dict from extract_posterior_means()
        sorted_keys: list of parameter names to include (in desired order)
        true_normalized_value: true value in normalized [0,1] space (default 0.25)

    Returns:
        dict: {param_name: RMSE} in sorted_keys order
    """
    M = len(posterior_means_list)
    if M == 0:
        return {}

    rmse_dict = {}
    for key in sorted_keys:
        # Convert key format: sorted_keys uses dots, posterior_means uses underscores
        posterior_key = key.replace(".", "_")

        # Collect estimates across M trials
        estimates = []
        for posterior_means in posterior_means_list:
            if posterior_key in posterior_means:
                estimates.append(posterior_means[posterior_key])

        if estimates:
            # MSE = mean of squared errors across M trials
            squared_errors = [(est - true_normalized_value)**2 for est in estimates]
            mse = sum(squared_errors) / len(squared_errors)
            rmse_dict[key] = math.sqrt(mse)

    return rmse_dict

def run_inference(H1_noisy, model, guide, sorted_keys, std_f, num_steps):
    # H1_noisy is (N, F, 2), float NOT COMPLEX
    pyro.clear_param_store()

    if OPTIMIZER == "Adam":
        optimizer = pyro.optim.Adam({"lr": LR})
    elif OPTIMIZER == "Adagrad":
        optimizer = pyro.optim.Adagrad({"lr": LR})
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}. Use 'Adam' or 'Adagrad'.")

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))
    top_20_most_sensitive = [
        key.replace(".", "_") + "_loc"
        for key in sorted_keys[:20]
    ]    
    losses = []
    param_history = defaultdict(list)  # Contains Python floats (more memory efficient)

    # Initialize parameters by running guide once
    # guide(H1_noisy, std_f)

    # Save initial parameter values
    # param_store = pyro.get_param_store()
    # for name, value in param_store.items():
    #     param_history[name].append(value.detach().item())

    for step in range(num_steps):
        loss = svi.step(H1_noisy, std_f)
        losses.append(loss)

        # param_store = pyro.get_param_store()
        # for name, value in param_store.items():
        #     param_history[name].append(value.detach().item())
        
        if step % 25 == 0 or step == 0:
            print(f"\n===== Step {step} | ELBO: {loss:.6f} =====")
            print("\n Top 20 Most Sensitive Parameters")
            param_store = pyro.get_param_store()
            for key in top_20_most_sensitive:
                if key in param_store:
                    print(f"{key:40s} (sigmoid) = {torch.sigmoid(param_store[key])} | True value = 0.25")
    param_store = pyro.get_param_store()
    for name, value in param_store.items():
        param_history[name] = [value.detach().item()]  # Single value
    print("Inference complete.")
    return losses, param_history


def perform_load_sensitivity_analysis(load_params, fault_params, cable_lengths, threshold, scenario):
    """
    Perform sensitivity analysis on network parameters.

    Returns:
    Selected_keys: List of selected params above threshold like ['load_0.C_m_leak', 'load_1.C_m_leak', 'load_3.C_m_leak'm ...] sorted
    Sorted_keys: List of all params sorted from most sensitive to least sensitive
    Sensitivities: List of corresponding sensitivities to Sorted_keys
    """

    variations = {}

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
            key = f"{load_name}.{param_name}" #
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
                H_var = calculate_Hnw(cable_lengths, load_params, perturbed_fault)
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
def compute_real_FIM(var_f):
    """
    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately) \\
    P = number of parameters (inputs) \\
    F = number of frequencies (outputs) \\
    Compute Real Fisher Information Matrix for normalized theta in [0, 1]
    Args:
        var_f: Noise variance (determined by SNR) [] if white noise (constant)
        or [F] if frequency dependent
    Returns:
        I2_scaled: FIM [p, p] in param_order_list order
    """
    # Use float64 for FIM/CLRB computation (better numerical precision)
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

    # Build scaling/reparamterization jacobian s_p = d theta_p / d phi_p
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
    return I2

def compute_real_FIM_and_CRLB(var_f, sorted_keys_s1, sensitivities):
    """
    g = H(θ): ℝᴾ → ℂᶠ  (or ℝ²ᶠ treating real/imag separately) \\
    P = number of parameters (inputs) \\
    F = number of frequencies (outputs) \\
    Compute Real Fisher Information Matrix and CRLB for P inferred real parameters. \\
    Note FIM and CRLB are normalized here. \\
    Uses float64 precision for numerical stability. 

    Args:
        var_f: Noise variance (determined by SNR) [] if white noise (constant)
        or [F] if frequency dependent
    Returns:
        CRLB_U1U1T: [P] Dict of Cramér-Rao Lower Bounds for alpha = U1U1^T theta
        CRLB_S2: [] depends on span of null space but will be < P. Dict of Cramér-Rao Lower Bounds for alpha = S2 theta
    """
    param_order_list, _ = get_inferred_param_order()
    I2 = compute_real_FIM(var_f)
    #Normalized FIM
    eigvals2, eigvecs2 = torch.linalg.eigh(I2) 
    dimension2 = I2.shape[0]
    rank2 = torch.linalg.matrix_rank(I2)
    #print("Rank of Normalized FIM", rank2)
    singular2 = rank2 < dimension2
    #print("Normalized FIM is singular?", singular2)
    #print("Eigvals of Normalized FIM (descending)", torch.sort(eigvals2, descending=True).values)

    U, S, Vh = torch.linalg.svd(I2)
    
    #print("Singular values of Normalized FIM (descending)", torch.sort(S, descending=True).values)

    eps = torch.finfo(I2.dtype).eps
    rtol = max(I2.shape[-2], I2.shape[-1]) * eps
    tol = rtol * S.max()           # since default atol = 0

    num_zeroed = (S <= tol).sum().item()
    num_kept = (S > tol).sum().item()

    #print(f"eps       = {eps:.3e}")
    #print(f"rtol      = {rtol:.3e}")
    #print(f"sigma_max = {S.max().item():.3e}")
    #print(f"tol       = {tol.item():.3e}")
    #print(f"zeroed    = {num_zeroed}")
    #print(f"kept      = {num_kept}")
    
    # Diagnostic: parameter sensitivities
    # J_flat = J.reshape(-1, J.shape[-1])  # [F*2, P]
    # param_sensitivity = torch.abs(J_flat).max(dim=0).values

    #print(f"Jacobian range: [{param_sensitivity.min():.2e}, {param_sensitivity.max():.2e}]")
    #print(f"Ratio: {param_sensitivity.max()/param_sensitivity.min():.2e}")

    tol = 1e-8
    mask = eigvals2 > tol

    Lambda1 = eigvals2[mask]          # shape [r]
    U1 = eigvecs2[:, mask]            # shape [p, r]
    U2 = eigvecs2[:, ~mask]            #shape[p, p-r] 

    J_pinv = U1 @ torch.diag(1.0 / Lambda1) @ U1.T
    J_pinv_torch = torch.linalg.pinv(I2)
    #Sanity check they should be equal
    #print("max |manual pinv - torch pinv| =", torch.max(torch.abs(J_pinv - J_pinv_torch)).item())

    
    S2, keep_indices = build_S2_from_nullspace(U2, tol=1e-8)

    # print("keep_indices =", keep_indices)
    # print("param order", param_order_list)
    # print("S2 shape =", S2.shape)
    # print("S2 =\n", S2)
    
    SU2 = S2 @ U2
    # print("S2 @ U2 =\n", SU2)
    # print("||S2 @ U2||_F =", torch.linalg.norm(SU2).item())
    # print("max |S2 @ U2| =", torch.max(torch.abs(SU2)).item())


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


    # print("="*220)
    # print(f"{'Idx':<5} {'Parameter':<22} {'Sens':<10} {'CRLB S2':<14} {'Unc S2':<10} {'CRLB U1U1T':<14} {'Unc U1U1T':<10}")
    # print("-"*220)

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
        crlb_u1u1t_dict[key] = crlb_u1u1t
        uncert_u1u1t_pct = math.sqrt(crlb_u1u1t) * 100
        if i in keep_indices:
            s2_idx = keep_indices.index(i)
            crlb_s2 = CRLB_S2[s2_idx].item()
            crlb_s2_dict[key] = crlb_s2
            uncert_s2_pct = math.sqrt(crlb_s2) * 100
            crlb_s2_str = f"{crlb_s2:.2e}"
            uncert_s2_str = f"{uncert_s2_pct:.2f}%"
        else:
            crlb_s2_str = "N/A"
            uncert_s2_str = "N/A"

        # print(f"{i:<5} {key:<22} {sens:<10}  {crlb_s2_str:<14} {uncert_s2_str:<10} {crlb_u1u1t:<14.2e} {uncert_u1u1t_pct:>5.2f}%")

    # print("="*220)
    return crlb_u1u1t_dict, crlb_s2_dict

############Bayesian CRLB############
def beta_prior_fim_closed_form(p, alpha):
    """Closed form prior FIM for Beta(α,α) priors."""
    if alpha <= 2:
        raise ValueError("alpha must be > 2 for finite FIM")
    
    j_pi_scalar = 2 * (2*alpha - 1) * (2*alpha - 2) / (alpha - 2)
    J_pi = j_pi_scalar * torch.eye(p)  # Diagonal
    return J_pi
def compute_prior_fim_beta_mc(alpha, p, num_samples=100000):
    """
    Monte Carlo estimate of prior FIM for Beta(α,α).
    Works even for α < 2 where closed form diverges.
    """
    eps = 1e-8  # Small regularization to avoid exact 0 or 1
    # Sample from prior
    beta_dist = torch.distributions.Beta(alpha, alpha)
    theta_samples = beta_dist.sample((num_samples, p))
    # Clamp to avoid numerical issues
    theta_samples = torch.clamp(theta_samples, eps, 1-eps)
    # Compute score: ∇ log λ(θ)
    score = (alpha - 1) / theta_samples - (alpha - 1) / (1 - theta_samples)  # [num_samples, p]
    # Compute J_π = E[score @ score^T]
    J_p = torch.zeros(p, p)
    for i in range(num_samples):
        #score_i = score[i].unsqueeze(1)  # [p, 1]
        score_i_vec = score[i]  # [p]
        J_p += torch.diag(score_i_vec ** 2)  # Diagonal outer product
        #J_p += score_i @ score_i.T
    J_p /= num_samples
    return J_p

def compute_expected_data_fim(p, snr_db, param_order_list, alpha, num_samples=100):
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

    for m in range(num_samples):
        print(f"{m+1} out of Monte Carlo {num_samples} for E_π[I(θ)] at {snr_db} dB")
        set_network_params_from_normalized(theta_samples[m], param_order_list)

        # Compute H_clean for this θ
        cable_lengths, load_params = build_params_from_flat(get_true_param_flat(), param_order_list)
        H_clean = calculate_Hnw_nofault(cable_lengths, load_params)
        
        # Compute var_f for THIS θ
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin

        # Compute FIM at this theta
        I_phi = compute_real_FIM(var_f) 
        
        E_I += I_phi
    
    E_I /= num_samples
    return E_I

# Convert selected_s1 keys to param_order_list tuple format
def key_to_tuple(key):
    """Convert 'l_w_4' or 'load_1.C_m_leak' to tuple format."""
    if '.' in key:
        # Load parameter: 'load_1.C_m_leak' → ('load', 'load_1', 'C_m_leak')
        parts = key.split('.')
        return ('load', parts[0], parts[1])
    else:
        # Cable parameter: 'l_w_4' → ('cable', 'l_w_4', None)
        return ('cable', key, None)

def calculate_bayesian_mse_monte_carlo(snr_db, selected_s1, alpha, M):
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
    param_order_list, p = get_inferred_param_order()
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
        cable_lengths, load_params = build_params_from_flat(get_true_param_flat(), param_order_list)
        H_clean = calculate_Hnw_nofault(cable_lengths, load_params)
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin
        std_f = torch.sqrt(var_f / 2)
        H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                     1j * std_f * torch.randn_like(H_clean.imag)
        H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))
        # 4. Run SVI to get estimate
        pyro.clear_param_store()
        losses, param_history = run_inference(H1_noisy, model_no_fault, guide, selected_s1, std_f, num_steps=300)
        
        # 5. Extract posterior mean
        posterior_means = extract_posterior_means(param_history)

        # 6. Compute squared errors vs sampled true θ
        for key in selected_s1:
            key_tuple = key_to_tuple(key)
            idx = param_order_list.index(key_tuple)
            true_val = theta_true_normalized[idx].item()
            
            posterior_key = key.replace(".", "_")
            if posterior_key in posterior_means:
                estimate = posterior_means[posterior_key]
                squared_errors[key].append((estimate - true_val)**2)
    
    # Average squared errors
    bayesian_mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
    
    return bayesian_mse_dict
        
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

    # Turn OFF fault parameter inference for stage 1
    for fault_name in network_params["fault_parameters"]:
        network_params["fault_parameters"][fault_name]["inferred"] = False

    selected_s1, sorted_keys_s1, sensitivities = perform_load_sensitivity_analysis(
        load_params, fault_params, cable_lengths,
        threshold=0.025, scenario="no_fault"
    )
    #Initial H_true with theta = 0.25
    num_obs = 1
    params_flat = get_true_param_flat()
    param_order_list, p = get_inferred_param_order()
    cable_lengths, load_params = build_params_from_flat(params_flat, param_order_list)
    H_true = calculate_Hnw_nofault(cable_lengths, load_params)  # [F] complex
    sigpow = torch.mean(torch.abs(H_true)**2)

    current_model = model_no_fault

    # Monte Carlo configuration
    M = 100 # Number of Monte Carlo trials per SNR (set to 1 for single trial, 10+ for full MC)
    alpha = 1.1
    snr_dbs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    rmse_results = {key: [] for key in selected_s1}
    crlb_results = {key: [] for key in selected_s1}

    bayesian_rmse_results = {key: [] for key in selected_s1}
    bayesian_crlb_results = {key: [] for key in selected_s1}

    J_pi = compute_prior_fim_beta_mc(alpha, p) #not dep on SNR move out of loop

    for snr_db in snr_dbs:
        print(f"\n{'='*50}")
        print(f"SNR = {snr_db} dB")
        print('='*50)

        # Bayesian CLRB R(\phi_n, \pi) \succeq (E_{\pi} [C(\theta)]) J_B^{-1} (E_{\pi}[C(\theta)]^T)
        E_I = compute_expected_data_fim(p, snr_db, param_order_list, alpha, M)
        J_B = E_I + J_pi 
        BCRLB = torch.linalg.inv(J_B)
        # BCRLB2 = torch.linalg.inv(E_I)
        bcrlb_diag_full = torch.diag(BCRLB)  # [p] in param_order_list order
        # print("BCRLB full", bcrlb_diag_full)
        # print("BCRLB EI only", torch.diag(BCRLB2))
        

        # Build bcrlb_dict in selected_s1 order
        bcrlb_dict = {}
        for key in selected_s1:
            key_tuple = key_to_tuple(key)
            if key_tuple in param_order_list:
                idx = param_order_list.index(key_tuple)
                bcrlb_dict[key] = bcrlb_diag_full[idx].item()
            else:
                print(f"Warning: {key} ({key_tuple}) not found in param_order_list")

        
        bayesian_mse = calculate_bayesian_mse_monte_carlo(snr_db, selected_s1, alpha, M)
        print(f"Bayesian RMSE (M={M}):", {k: f"{math.sqrt(v):.4f}" for k, v in bayesian_mse.items()})
        print(f"sqrt(BCRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in bcrlb_dict.items()})

        # Store results for plotting
        for key in selected_s1:
            if key in bayesian_mse and key in bcrlb_dict:
                bayesian_rmse_results[key].append(math.sqrt(bayesian_mse[key]))
                bayesian_crlb_results[key].append(math.sqrt(bcrlb_dict[key]))  
        

    # Plot Bayesian RMSE vs BCRLB across SNR
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, key in enumerate(selected_s1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ax.plot(snr_dbs, bayesian_rmse_results[key], 'bo-', label=f'Bayesian RMSE (M={M})', markersize=6)
        ax.plot(snr_dbs, bayesian_crlb_results[key], 'r--', label='sqrt(BCRLB)', linewidth=2)
        
        # Prior std reference line
        prior_std = math.sqrt(1/12)  # For uniform [0,1]
        ax.axhline(y=prior_std, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Prior std')
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error')
        ax.set_title(key, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    # Hide unused subplots
    for idx in range(len(selected_s1), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Bayesian RMSE vs sqrt(BCRLB) - SNR Sweep (α={alpha}, M={M} trials per SNR)', fontsize=14, y=0.995)
    plt.tight_layout()

    filename = f"bayesian_rmse_vs_bcrlb_snr_sweep_alpha{alpha}_M{M}.pdf"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nSaved plot to {filename}")
        # # Standard CRLB at this SNR (computed once per SNR, same for all trials at theta = 0.25 for all params)
        # crlb_u1u1t_dict, _ = compute_real_FIM_and_CRLB(var_f, selected_s1, sensitivities)
        # # Now bcrlb_dict and crlb_u1u1t_dict are in same order (sorted_keys_s1)
        # print("Bayesian CRLB:", bcrlb_dict)
        # print("Classical CRLB:", crlb_u1u1t_dict)
        
        # # Run M Monte Carlo trials at this SNR
        # posterior_means_list = []
        # for trial in range(M):
        #     if M > 1:
        #         print(f"  Trial {trial+1}/{M}")

        #     # Generate noisy observation (different noise each trial)
        #     H1_noisy_c = H_true + std_f * torch.randn_like(H_true.real) + \
        #                     1j * std_f * torch.randn_like(H_true.imag)
        #     H1_noisy_c_expanded = H1_noisy_c.unsqueeze(0).expand(num_obs, -1)
        #     H1_noisy = torch.view_as_real(H1_noisy_c_expanded)

        #     # Run SVI inference
        #     pyro.clear_param_store()
        #     losses, param_history = run_inference(H1_noisy, current_model, guide, selected_s1, std_f, num_steps=300)

        #     # Extract posterior means for this trial
        #     posterior_means = extract_posterior_means(param_history, num_samples=2048) #dict of {paramname, posterior mean value} for all params
        #     posterior_means_list.append(posterior_means)

        # # Calculate RMSE across M trials
        # rmse_dict = calculate_rmse_from_trials(posterior_means_list, selected_s1, true_normalized_value=0.25)

        # print(f"RMSE (M={M}):", {k: f"{v:.4f}" for k, v in rmse_dict.items()})
        # print(f"sqrt(CRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in crlb_u1u1t_dict.items()})

        # Store results for plotting
        # for key in selected_s1:
        #     if key in rmse_dict and key in crlb_u1u1t_dict:
        #         rmse_results[key].append(rmse_dict[key])
        #         crlb_results[key].append(math.sqrt(crlb_u1u1t_dict[key]))  

    # Plot RMSE vs sqrt(CRLB) across SNR for each parameter
    # fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    # axes = axes.flatten()

    # for idx, key in enumerate(selected_s1):
    #     if idx >= len(axes):
    #         break
    #     ax = axes[idx]
    #     ax.plot(snr_dbs, rmse_results[key], 'bo-', label=f'RMSE (M={M})', markersize=6)
    #     ax.plot(snr_dbs, crlb_results[key], 'r--', label='sqrt(CRLB)', linewidth=2)
    #     ax.set_xlabel('SNR (dB)')
    #     ax.set_ylabel('Error')
    #     ax.set_title(key, fontsize=10)
    #     ax.legend(fontsize=8)
    #     ax.grid(True, alpha=0.3)
    #     ax.set_yscale('log')

    # # Hide unused subplots
    # for idx in range(len(selected_s1), len(axes)):
    #     axes[idx].set_visible(False)

    # fig.suptitle(f'RMSE vs sqrt(CRLB) - SNR Sweep (M={M} trials per SNR)', fontsize=14, y=0.995)
    # plt.tight_layout()

    # filename = f"rmse_vs_crlb_snr_sweep_M{M}.pdf"
    # plt.savefig(filename, dpi=150, bbox_inches='tight')
    # #plt.show()

    # print(f"\nSaved plot to {filename}")


    
    print("My program took", time.time() - start_time, "to run")