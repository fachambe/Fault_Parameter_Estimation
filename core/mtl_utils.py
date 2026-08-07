"""
Multi-conductor Transmission Line (MTL) utility functions.

This module contains core functions for MTL modeling including:
- Matrix operations (hyperbolic functions)
- Cable parameter calculations
- MTL matrix computations
- Reflection coefficients and load carry-back operations
- Network topology calculations
"""

import numpy as np
import torch
import math


def denormalize(norm_value, min_val, max_val, dist="uniform"):
    """
    Convert normalized value (0 to 1) back to its original range.

    Args:
        norm_value: Normalized value in [0, 1]
        min_val: Minimum of parameter range
        max_val: Maximum of parameter range
        dist: Distribution type - "uniform" or "log" for log-uniform

    Returns:
        Physical value in [min_val, max_val]
    """
    if dist == "log":
        # Log-uniform: value = min * (max/min)^norm_value
        return min_val * (max_val / min_val) ** norm_value
    else:
        # Linear uniform: value = min + norm_value * (max - min)
        return norm_value * (max_val - min_val) + min_val


def normalize(physical_value, min_val, max_val, dist="uniform"):
    """
    Convert physical value to normalized [0, 1] range. Inverse of denormalize.

    Args:
        physical_value: Value in physical range
        min_val: Minimum of parameter range
        max_val: Maximum of parameter range
        dist: Distribution type - "uniform" or "log" for log-uniform

    Returns:
        Normalized value in [0, 1]
    """
    if dist == "log":
        # Log-uniform: norm = log(value/min) / log(max/min)
        return torch.log(physical_value / min_val) / torch.log(torch.tensor(max_val / min_val))
    else:
        # Linear uniform
        return (physical_value - min_val) / (max_val - min_val)


def loguniform(low, high, size=None):
    """Generate samples from a log-uniform distribution."""
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))


def matrix_cosh(M):
    """
    Compute matrix hyperbolic cosine using matrix exponential.

    cosh(M) = 0.5 * (exp(M) + exp(-M))

    Args:
        M: (F, n, n) tensor

    Returns:
        (F, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) + torch.matrix_exp(-M))


def matrix_sinh(M):
    """
    Compute matrix hyperbolic sine using matrix exponential.

    sinh(M) = 0.5 * (exp(M) - exp(-M))

    Args:
        M: (F, n, n) tensor

    Returns:
        (F, n, n) tensor
    """
    return 0.5 * (torch.matrix_exp(M) - torch.matrix_exp(-M))

def calculate_cable_parameters(r_w, omega, n, device=None):
    """
    Compute R, L, C, G tensors for multiple frequencies.

    Parameters:
        r_w: Radius of MTL conductor (scalar)
        omega: Tensor of angular frequencies (F,)
        n: Number of conductors - 1 (scalar)
        device: Torch device (defaults to omega's device)

    Returns:
        R, L, C, G tensors (shape: (F, n, n))
    """
    if device is None:
        device = omega.device

    f = omega / (2 * torch.pi)  # Convert omega to frequency (Hz)
    num_freqs = len(f)

    # Constants
    mu_0 = 4 * np.pi * 1e-7
    sigma = 5.8 * 1e7
    epsilon = 3.19 * 1e-11
    dc = 4 * 1e-4 + 3.02 * r_w
    dc2 = math.sqrt(2.0) * dc
    tan_delta = 0.01

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

    # L matrix
    L = (mu_0 / (2 * np.pi)) * torch.tensor([
        [2*np.log(dc / r_w), np.log((dc * dc2) / (dc * r_w)), np.log((dc * dc) / (dc2 * r_w))],
        [np.log((dc * dc2) / (dc * r_w)), 2*np.log(dc2 / r_w), np.log((dc2 * dc) / (dc * r_w))],
        [np.log((dc * dc) / (dc2 * r_w)), np.log((dc2 * dc) / (dc * r_w)), 2*np.log(dc / r_w)]
    ], dtype=torch.complex64, device=device)

    L_new = L.unsqueeze(0).expand(num_freqs, -1, -1)
    C = mu_0 * epsilon * torch.linalg.inv(L)
    C_new = C.unsqueeze(0).expand(num_freqs, -1, -1)

    G_new = (
        omega.reshape(-1, 1, 1)
        * tan_delta
        * C_new
    ).to(torch.complex64)

    return R, L_new, C_new, G_new


def get_mtl_matrices(R, L, C, G, n, omega):
    """
    Compute MTL matrices for multiple frequencies using PyTorch.

    Parameters:
        R, L, C, G: (F, n, n) tensors
        n: Number of conductors - 1
        omega: (F,) tensor

    Returns:
        T: (F, n, n) - Eigenvectors
        Tinv: (F, n, n) - Inverse of eigenvectors
        gamma: (F, n, n) - Propagation constants (diagonal)
        ZC: (F, n, n) - Characteristic impedance
        YC: (F, n, n) - Characteristic admittance
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

    # Compute gamma as a diagonal matrix with sqrt(eigenvalues)
    gamma = torch.zeros((F, n, n), dtype=torch.complex64, device=R.device)
    gamma[:, torch.arange(n), torch.arange(n)] = torch.sqrt(eigvals)

    # Compute batch-wise inverses
    inv_YT = torch.linalg.inv(Y_T)  # (F, n, n)
    inv_eigvecs = torch.linalg.inv(eigvecs)  # (F, n, n)

    # Compute characteristic impedance Zc
    Zc = inv_YT @ eigvecs @ gamma @ inv_eigvecs  # (F, n, n)

    # Compute characteristic admittance Yc
    Yc = torch.linalg.inv(Zc)  # (F, n, n)

    return eigvecs, inv_eigvecs, gamma, Zc, Yc

def reflection_coefficient(YL, T, T_inv, ZC, YC):
    """
    Compute reflection coefficient (eq. 12 of Tonello paper).

    Args:
        YL: Load admittance. Shape: (F, n, n) or (P, F, n, n)
        T, T_inv, ZC, YC: MTL matrices. Shape: (F, n, n)

    Returns:
        rho: Reflection coefficient. Shape: (F, n, n) or (P, F, n, n)
    """
    inv_sum = torch.linalg.inv(YL + YC)
    rho = T_inv @ YC @ inv_sum @ (YL - YC) @ ZC @ T
    return rho


def carry_back_load(rhoL, T, YC, Gamma, length):
    """
    Carry load admittance back through cable (eq. 13 of Tonello paper).

    Args:
        rhoL: Reflection coefficient. Shape: (P, F, n, n), may have batch dims.
        T, YC, Gamma: MTL matrices. Shape: (F, n, n).
        length: Cable length. Scalar or shape [P, 1].

    Returns:
        Y_R(x): Admittance at input. Shape: (P, F, n, n).
    """
    # Reshape length for broadcasting with Gamma [F, n, n]
    if isinstance(length, torch.Tensor) and length.dim() > 0:
        length_bc = length.view(*length.shape, 1, 1)
    else:
        length_bc = length

    # Compute matrix exponentials
    e_pos = torch.matrix_exp(Gamma * length_bc)   # (P, F, n, n)
    e_neg = torch.matrix_exp(-Gamma * length_bc)  # (P, F, n, n)

    # Compute numerator and denominator
    num = e_pos + torch.matmul(e_neg, rhoL)  # (P, F, n, n)
    den = e_pos - torch.matmul(e_neg, rhoL)  # (P, F, n, n)
    deninv = torch.linalg.inv(den)           # (P, F, n, n)

    # Compute final YR
    YR = T @ num @ deninv @ torch.linalg.inv(T) @ YC
    return YR


def h_B(rhoL, ZC, T, T_inv, Gamma, length):
    """
    Compute h_B transfer function (eq. 14 of Tonello paper).

    Args:
        rhoL: Reflection coefficient. Shape: (P, F, n, n)
        ZC, T, T_inv, Gamma: MTL matrices. Shape: (F, n, n)
        length: Cable length. Scalar or tensor [P, 1].

    Returns:
        h_B: Transfer function. Shape: (P, F, n, n)
    """
    n = ZC.shape[-1]
    device = ZC.device

    # Identity matrix
    U = torch.eye(n, dtype=torch.complex64, device=device)

    # Reshape length for broadcasting
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

def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    """
    Compute 3x3 receiver load impedance matrix.

    Args:
        Z_RG, Z_R1, Z_R2, Z_R3: Receiver impedances (scalars)

    Returns:
        Z_rec: (3, 3) impedance matrix
    """
    Z_rec = torch.tensor([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec


def calculate_room_admittance_matrix(Y_loads_room, cable_lengths_room,
                                      T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                                      T_s, Tinv_s, ZC_s, YC_s, gamma_s):
    """
    Compute equivalent admittance matrix of a room in the network.

    Args:
        Y_loads_room: List of 4 admittance matrices (shape [P,F,n,n] or [F,n,n])
        cable_lengths_room: List of 5 cable lengths (shape [P,1] or scalar)
        T_r, Tinv_r, ZC_r, YC_r, gamma_r: Room cable MTL matrices
        T_s, Tinv_s, ZC_s, YC_s, gamma_s: Service panel cable MTL matrices

    Returns:
        Y_room_carried: Equivalent room admittance (shape [P,F,n,n] or [F,n,n])
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


def calculate_Htrans(YTalpha, YTbeta, YTgamma, Ynw, ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32):
    """
    Compute transmitter transfer function from network admittance.

    Parameters:
        YTalpha, YTbeta, YTgamma: Transmitter admittance constants (scalar)
        Ynw: Network input admittance matrix (..., F, n, n) - may have batch dims
        ZT0, ZT12, ZT21, ZT13, ZT31, ZT23, ZT32: Transmitter impedance constants (scalar)

    Returns:
        Htrans_inv: Inverse transmitter transfer function (..., F, n, n)
    """
    # Extract individual elements using ... for arbitrary batch dims
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

    # Stack into matrix
    H_trans = torch.cat([
        torch.cat([H11, H12, H13], dim=-1),
        torch.cat([H21, H22, H23], dim=-1),
        torch.cat([H31, H32, H33], dim=-1)
    ], dim=-2)  # Shape: (..., F, 3, 3)

    H_trans_inv = torch.linalg.inv(H_trans)
    return H_trans_inv
