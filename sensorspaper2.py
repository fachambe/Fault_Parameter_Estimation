import numpy as np
import random
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch

import scipy as sp 
import numpy as np
import time
start_time = time.time()

def matrix_cosh(M):
    return 0.5*(sp.linalg.expm(M) + sp.linalg.expm(-M))
def matrix_sinh(M):
    return 0.5*(sp.linalg.expm(M) - sp.linalg.expm(-M))
def reflection_coefficient(YL, T, T_inv, ZC, YC):
    """
    eq. (12) of Tonello paper
    YL, T, T_inv, ZC, YC each has shape (N,n,n).
    Returns rho of shape (N,n,n). (n = # conductors - 1)
    """
    return (T_inv @ YC @ np.linalg.inv(YL + YC) 
            @ (YL - YC) 
            @ ZC @ T) 
def carry_back_load(rhoL, T, YC, Gamma, length):
    """
    eq. (13) of Tonello paper
    rhoL, T, YC, Gamma each is (N,n,n). Returns Y_R(x) shape (N,n,n). (n = # conductors - 1)
    """
    e_pos = sp.linalg.expm(Gamma * length)
    e_neg = sp.linalg.expm(-Gamma * length)
    num   = e_pos + e_neg @ rhoL
    den   = e_pos - e_neg @ rhoL
    deninv= np.linalg.inv(den)
    return T @ num @ deninv @ np.linalg.inv(T) @ YC

def h_B(rhoLI, ZC, T, T_inv, Gamma, length):
    """
    eq. (14) of Tonello paper
    rhoL, ZC, T, T_inv, Gamma each is (N,n,n). Returns h_B shape (N,n,n). (n = # conductors - 1)
    """
    N = ZC.shape[0]
    U = np.eye(3, dtype=complex)
    U = np.repeat(U[np.newaxis, :, :], N, axis=0)
    e_pos = sp.linalg.expm(Gamma * length)
    e_neg = sp.linalg.expm(-Gamma * length)
    return ZC @ T @ (U - rhoLI) @ np.linalg.inv(e_pos - e_neg @ rhoLI) @ T_inv @ np.linalg.inv(ZC)

def get_mtl_matrices(R, L, C, G, n, omega):
    """
    Compute MTL matrices for multiple frequencies.

    Parameters:
    - R, L, C, G: (N, n, n)
    - omega: (N,)

    Returns:
    - T (N, n, n)
    - Tinv (N, n, n)
    - gamma (N, n, n)
    - ZC (N, n, n)
    - YC (N, n, n)
    """
    N, n, _ = R.shape
    omega = omega.reshape(-1, 1, 1) #Make omega (num_freq, 1, 1) so it can be added with R,L,C,G
    Z_T = R + 1j * omega * L
    Y_T = G + 1j * omega * C
    ZY = np.matmul(Z_T, Y_T)
    YZ = np.matmul(Y_T, Z_T)

    eigvals = np.zeros((N, n), dtype=complex)
    eigvecs = np.zeros((N, n, n), dtype=complex)
    for i in range(N):
        eigvals[i], eigvecs[i] = np.linalg.eig(YZ[i])

    gamma = np.zeros((N, n, n), dtype=complex)  # Initialize with zeros
    gamma[:, np.arange(n), np.arange(n)] = np.sqrt(eigvals)

    inv_YT = np.linalg.inv(Y_T)  # Shape: (N, n, n)

    # Compute batch-wise inverses
    inv_YT = np.linalg.inv(Y_T)  # (N, n, n)
    inv_eigvecs = np.linalg.inv(eigvecs)  # (N, n, n)

    # Compute Zc using batch matrix multiplications
    Zc = inv_YT @ eigvecs @ gamma @ inv_eigvecs
    #Zc = np.einsum('fij,fjk,fkl,flm->fim', inv_YT, eigvecs, gamma, inv_eigvecs)  # (N, n, n)
    Yc = np.linalg.inv(Zc)
    return eigvecs, np.linalg.inv(eigvecs), gamma, Zc, Yc

def calculate_cable_parameters(r_w, omega, n):
    """
    Compute R, L, C, G matrices for multiple frequencies.

    Parameters:
    - r_w: radius of MTL conductor (scalar)
    - omega: array of angular frequencies (num_freq,)
    - n: number of conductors - 1

    Returns:
    - R, L, C, G matrices (shape: (num_freq, n, n))
    """
    f = omega / (2 * np.pi)
    num_freqs = len(f)
    mu_0 = 4 * np.pi * 1e-7
    sigma = 5.8 * 1e7
    epsilon = 3.19 * 1e-11
    dc = 4 * 1e-4 + 3.02 * r_w
    dc2 = np.sqrt(2) * dc
    tandeltal = 1e-6
    delta = 1 / np.sqrt(np.pi * mu_0 * sigma * f)  # Skin depth (array)
    r = np.where(
        r_w <= 2 * delta,
        1 / (sigma * np.pi * r_w**2),  # Case where r_w <= 2*delta
        (1 / (2 * r_w)) * np.sqrt((mu_0 * f) / (np.pi * sigma))  # Case where r_w > 2*delta
    )
    R = np.array([
            [2 * r, r, r],
            [r, 2 * r, r],
            [r, r, 2 * r]
        ]).transpose(2, 0, 1)
    
    L = (mu_0 / 2 * np.pi) * np.array([
        [2*np.log(dc / r_w), np.log((dc * dc2) / (dc * r_w)), np.log((dc * dc) / (dc2 * r_w))],
        [np.log((dc * dc2) / (dc * r_w)), 2*np.log(dc2 / r_w), np.log((dc2 * dc) / (dc * r_w))],
        [np.log( (dc * dc) / (dc2 * r_w)), np.log((dc2 * dc) / (dc * r_w)), 2*np.log(dc / r_w)]
    ])
    L_new = np.repeat(L[np.newaxis, :, :], num_freqs, axis=0)
    C = mu_0 * epsilon * np.linalg.inv(L)
    C_new = np.repeat(C[np.newaxis, :, :], num_freqs, axis=0)
    # G_new = np.zeros((num_freqs, n, n))

    # for i in range(num_freqs):
    #     G = omega[i] * np.tan(delta[i]) * C
    #     #print("Gassa", G.shape)
    #     G_new[i] = G
    # # G = omega * tandelta1 * C_new
    # # print("G shape", G_new.shape)

    G_new = np.zeros((num_freqs, n, n))
    return R, L_new, C_new, G_new


def calculate_cable_transmission_matrix(R, L, C, G, length, omega):
    """
    Compute ABCD Matrix of MTL
    
    Parameters:
    - R, L, C, G: (N, n, n)
    - length: (scalar in meters)
    - omega: (N,)

    Returns:
    - ABCD_cable: (N, 2n, 2n)
    """
    N, n, _ = R.shape
    omega = omega.reshape(-1, 1, 1) #Make omega (num_freq, 1, 1) so it can be added with R,L,C,G
    Z_T = R + 1j * omega * L
    Y_T = G + 1j * omega * C
    ZY = np.matmul(Z_T, Y_T)
    YZ = np.matmul(Y_T, Z_T)

    eigvals1 = np.zeros((N, n), dtype=complex)
    eigvecs1 = np.zeros((N, n, n), dtype=complex)
    eigvals2 = np.zeros((N, n), dtype=complex)
    eigvecs2 = np.zeros((N, n, n), dtype=complex)
    for i in range(N):
        eigvals1[i], eigvecs1[i] = np.linalg.eig(ZY[i])
        eigvals2[i], eigvecs2[i] = np.linalg.eig(YZ[i])


    gamma = np.zeros((N, n, n), dtype=complex)  # Initialize with zeros
    gamma[:, np.arange(n), np.arange(n)] = np.sqrt(eigvals1)

    
    Gamma = eigvecs1 @ gamma @ np.linalg.inv(eigvecs1)

    Zw = np.linalg.inv(Gamma) @ Z_T
    Yw = np.linalg.inv(Zw)

    Phi11 = matrix_cosh(Gamma*length)
    Phi12 = -matrix_sinh(Gamma*length)@ Zw
    Phi21 = -Yw @ matrix_sinh(Gamma*length)
    Phi22 = Yw @ matrix_cosh(Gamma*length) @ Zw

    # Form the transmission matrix for the current frequency
    Phi_cable = np.block([
        [Phi11, Phi12],
        [Phi21, Phi22]
    ])
    return Phi_cable

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

def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    Z_rec = np.array([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec
# Function to compute constant impedance admittance matrix (type 1)
def constant_impedance(R_const, C_leak, omega):
    Z12 = Z13 = Z23 = np.full_like(omega, R_const, dtype=complex)  # Convert to array
    ZG1 = ZG2 = ZG3 = 1 / (1j * omega * C_leak)
    return Z12, Z13, Z23, ZG1, ZG2, ZG3 #(this returns a tuple not a list)

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
# Generate random parameters for each load and returns impedance values (24 loads total)
def loguniform(low, high, size=None):
    """
    Generate samples from a log-uniform distribution.

    Parameters:
    - low (float): The lower bound of the range (in real scale).
    - high (float): The upper bound of the range (in real scale).
    - size (int or tuple of ints, optional): The number of samples to draw.

    Returns:
    - samples (ndarray): Samples from the log-uniform distribution.
    """
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))
def denormalize(norm_value, min_val, max_val):
    """
    Convert normalized value (0 to 1) back to its original range.
    """
    return norm_value * (max_val - min_val) + min_val

def generate_random_parameters(load_type, omega):
    """
    Generate random values for Z_12, Z_13, Z_23, Z_G1, Z_G2, Z_G3 depending on load type

    Parameters:
    - load type(integer): 1, 2, or 3
    - omega: list of frequencies (N,)

    Returns:
    - tuple (Z12, Z13, Z23, ZG1, ZG2, ZG3) where each element in the tuple is an array of complex numbers of length N
    """
    print("load type", load_type)
    fixed_value = 0.25  # Set all network parameters to 0.25
    if load_type == 1:  # Constant impedance
        return constant_impedance(
            denormalize(fixed_value, 10, 200),  # R_const
            denormalize(fixed_value, 0.1e-9, 2.0e-9),  # C_leak
            omega
        )
        # return constant_impedance(loguniform(10, 200, 1), random.uniform(0.1e-9, 2.0e-9), omega)  # R_const, C_leak
    elif load_type == 2:  # Double RLC
        return double_RLC(
            denormalize(fixed_value, 10, 3000),  # R_s
            denormalize(fixed_value, 0, 30e6),  # omega_0s
            denormalize(fixed_value, 0.1, 2),  # zeta_s
            denormalize(fixed_value, 10, 3000),  # R_p
            denormalize(fixed_value, 0, 30e6),  # omega_0p
            denormalize(fixed_value, 0.1, 2),  # zeta_p
            denormalize(fixed_value, -0.1, 0.1),  # delta_1
            denormalize(fixed_value, -0.1, 0.1),  # delta_2
            denormalize(fixed_value, 0.1e-9, 2e-9),  # C_d_leak
            omega
        )
        # return double_RLC(loguniform(10, 3000, 1), random.uniform(0, 30e6), random.uniform(0.1, 2),
        #         loguniform(10, 3000, 1), random.uniform(0, 30e6), random.uniform(0.1, 2), 
        #         random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.1e-9, 2e-9), omega)
    elif load_type == 3:  # Motor model
         return motor_load(
            denormalize(fixed_value, 0.1e-9, 1e-9),  # C_m
            denormalize(fixed_value, 5e-3, 20e-3),  # L_m
            denormalize(fixed_value, 2000, 15000),  # R_m1
            5,  # Fixed value for R_m2
            denormalize(fixed_value, 0.2e-9, 5.0e-9),  # C_m_leak
            omega
        )
        # return motor_load(random.uniform(0.1e-9, 1e-9), random.uniform(5e-3, 20e-3), random.uniform(2000, 15000),
        #         5, random.uniform(0.2e-9, 5.0e-9), omega)
def compute_load_admittance_3d(load_params):
    """
    Compute 3x3 load admittance matrix from (Z12, Z13, Z23, ZG1, ZG2, ZG3)

    Parameters:
    - load_params (tuple): (Z12, Z13, Z23, ZG1, ZG2, ZG3)

    Returns:
    - Y_load (ndarray): Load admittance matrix (N, 3, 3)
    """
    Z12, Z13, Z23, ZG1, ZG2, ZG3 = load_params  # Unpack impedance values, (N,) array
    Y_load = np.array([
            [1/ZG1 + 1/Z12 + 1/Z13, -1/Z12, -1/Z13],
            [-1/Z12, 1/ZG2 + 1/Z12 + 1/Z23, -1/Z23],
            [-1/Z13, -1/Z23, 1/ZG3 + 1/Z13 + 1/Z23]
    ], dtype=complex).transpose(2, 0, 1) #(n, n, N) -> (N, n, n)
    return Y_load


def calculate_room_admittance_matrix(T_r, Tinv_r, ZC_r, YC_r, gamma_r, T_s, Tinv_s, ZC_s, YC_s, gamma_s, omega):
    #Ql_list = 
    load_params = [generate_random_parameters(random.choice([1,2,3]), omega) for _ in range(4)]
    Y_list = [compute_load_admittance_3d(params) for params in load_params]

    rho3 = reflection_coefficient(Y_list[0], T_r, Tinv_r, ZC_r, YC_r)
    Y_3_carried = carry_back_load(rho3, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_1new = Y_3_carried + Y_list[1]
    rho1 = reflection_coefficient(Y_1new, T_r, Tinv_r, ZC_r, YC_r)
    Y_1new_carried = carry_back_load(rho1, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))

    rho4 = reflection_coefficient(Y_list[2], T_r, Tinv_r, ZC_r, YC_r)
    Y_4_carried = carry_back_load(rho4, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_2new = Y_4_carried + Y_list[3]
    rho2 = reflection_coefficient(Y_2new, T_r, Tinv_r, ZC_r, YC_r)
    Y_2new_carried = carry_back_load(rho2, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    
    Y_room = Y_1new_carried + Y_2new_carried
    rho5 = reflection_coefficient(Y_room, T_s, Tinv_s, ZC_s, YC_s)
    Y_room_carried = carry_back_load(rho5, T_s, YC_s, gamma_s, random.uniform(6.5, 6.5))

    return Y_room_carried
def calculate_Ynw(Phi_network, n, Z_rec):
    """
    Compute input admittance matrix from ABCD parameters

    Parameters:
    - Phi_network: Overall network ABCD matrix (N, 2n, 2n)
    - n: number of conductors - 1
    - Z_rec: Receiver load matrix (N, n, n)

    Returns:
    - Ynw: Transfer function of ABCD Matrix (N, n, n)
    """
    # Split Phi_network into submatrices
    Phi_11 = Phi_network[:, :n, :n]
    Phi_12 = Phi_network[:, :n, n:]
    Phi_21 = Phi_network[:, n:, :n]
    Phi_22 = Phi_network[:, n:, n:]

    Y_nw = -np.linalg.inv(Phi_12 - Z_rec @ Phi_22) @ (Phi_11 - Z_rec @ Phi_21)
    return Y_nw
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
    Ynw11 = Ynw[:, 0, 0].reshape(N, 1, 1)
    Ynw12 = Ynw[:, 0, 1].reshape(N, 1, 1)
    Ynw13 = Ynw[:, 0, 2].reshape(N, 1, 1)
    Ynw21 = Ynw[:, 1, 0].reshape(N, 1, 1) #np.block has to work with (N, 1, 1) not (N,)
    Ynw22 = Ynw[:, 1, 1].reshape(N, 1, 1)
    Ynw23 = Ynw[:, 1, 2].reshape(N, 1, 1)
    Ynw31 = Ynw[:, 2, 0].reshape(N, 1, 1)
    Ynw32 = Ynw[:, 2, 1].reshape(N, 1, 1)
    Ynw33 = Ynw[:, 2, 2].reshape(N, 1, 1)

    H_trans = np.block([
        [1 + ZT0*Ynw11 + ZT0*YTalpha, ZT0*Ynw12 - ZT0/ZT12, ZT0*Ynw13 - ZT0/ZT13],
        [ZT0*Ynw21 - ZT0/ZT21, 1 + ZT0*Ynw22 + ZT0*YTbeta, ZT0*Ynw23 - ZT0/ZT23],
        [ZT0*Ynw31 - ZT0/ZT31, ZT0*Ynw32 - ZT0/ZT32, 1 + ZT0*Ynw33 + ZT0*YTgamma]
    ])
    return np.linalg.inv(H_trans)

def model(H1_noisy):
    R_const = pyro.sample("R_const", dist.LogNormal(torch.tensor(4.0), torch.tensor(1.0)))  
    C_leak = pyro.sample("C_leak", dist.LogNormal(torch.tensor(-20.0), torch.tensor(0.5)))
    l_w = pyro.sample("l_w", dist.Normal(torch.tensor(10.0), torch.tensor(5.0)))  # Length in meters

    # Call your existing function with sampled parameters
    H1_pred = (R_const, C_leak, l_w)  

    # Define noise model: Likelihood of the observed noisy data
    sigma_noise = pyro.sample("sigma_noise", dist.HalfNormal(torch.tensor(1.0)))  # Noise level
    pyro.sample("obs", dist.Normal(H1_pred, sigma_noise), obs=H1_noisy)

if __name__ == '__main__':
    fixed_value = 0.25
    num_rooms = 4 #Technically 6 but first and last room require manual calculation
    num_of_conductors = 4

    r_w_servicepanel = denormalize(fixed_value, 1.03e-3, 2.06e-3)
    r_w_room = denormalize(fixed_value, 0.81e-3, 1.29e-3)
    l_list = [denormalize(fixed_value, 2, 20) for _ in range(6)]  # 6 length values
    # r_w_servicepanel = random.uniform(1.03e-3, 2.06e-3)
    # r_w_room = random.uniform(0.81e-3, 1.29e-3)
    #l_list = [random.uniform(2, 20) for _ in range(6)] 
    print("l_list", l_list)
    #CONSTANTS
    #frequencies = np.arange(1e3, 8e7, 1e4)
    # frequencies = np.arange(150e3, 800e4, 1e4)
    frequencies = np.logspace(np.log10(150e3), np.log10(30e6), 2000)  # 150 kHz to 30 MHz log spacing
    freq_range_mhz = frequencies / 1e6  # Convert to MHz
    omega = 2 * np.pi * frequencies  # Angular frequency
    num_freqs = len(omega)
    print("Generating Parameters")
    #(Tonello Paper) Reflection Coefficient Approach
    Z_RG = Z_R1 = Z_R2 = 50
    Z_R3 = 50
    ZT0 = ZTG1 = ZTG2 = 50
    ZTG3 = 50
    ZT12 = 50
    ZT13 = ZT23 = 100
    Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3)
    Y_rec = np.linalg.inv(Z_rec)
    Z_rec = np.repeat(Z_rec[np.newaxis, :, :], num_freqs, axis=0)
    Y_rec = np.repeat(Y_rec[np.newaxis, :, :], num_freqs, axis=0)
    load_params = [generate_random_parameters(random.choice([1,2,3]), omega) for _ in range(6)]
    Y_list = [compute_load_admittance_3d(params) for params in load_params]
    hoverall = np.zeros((3, 3, num_freqs), dtype=complex)
    habcd = np.zeros((3,3, num_freqs), dtype=complex)

    R_s, L_s, C_s, G_s = calculate_cable_parameters(r_w_servicepanel, omega, num_of_conductors-1)
    R_r, L_r, C_r, G_r = calculate_cable_parameters(r_w_room, omega, num_of_conductors-1)
    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(R_s, L_s, C_s, G_s, num_of_conductors-1, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(R_r, L_r, C_r, G_r, num_of_conductors-1, omega) #MTL parameters of wires in room
    print("T_r", T_r[53])
    print("Tinv_r", Tinv_r[53])
    print("gamma_r", gamma_r[53])
    print("ZC_r", ZC_r[53])
    print("YC_r", YC_r[53])
    #node1 (Yrec)
    rho1 = reflection_coefficient(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, l_list[0])
    Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, l_list[0])
    #node2 (Y62)
    Y_node2 = Y_reccarried + Y_list[0]
    #print("Y_list[0]", Y_list[0])
    rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
    h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, l_list[1])
    Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, l_list[1])
    #node3 (Y61 || Y63)
    rho63 = reflection_coefficient(Y_list[1], T_r, Tinv_r, ZC_r, YC_r)
    Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_61 = Y_63 + Y_list[2]
    rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
    Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_node3 = Y_node2carried + Y_6
    rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
    h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, l_list[2])
    Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, l_list[2])
    # #node4 (4 rooms)
    Y_5 = calculate_room_admittance_matrix(T_r, Tinv_r, ZC_r, YC_r, gamma_r, T_s, Tinv_s, ZC_s, YC_s, gamma_s, omega)
    Y_4 = calculate_room_admittance_matrix(T_r, Tinv_r, ZC_r, YC_r, gamma_r, T_s, Tinv_s, ZC_s, YC_s, gamma_s, omega)
    Y_3 = calculate_room_admittance_matrix(T_r, Tinv_r, ZC_r, YC_r, gamma_r, T_s, Tinv_s, ZC_s, YC_s, gamma_s, omega)
    Y_2 = calculate_room_admittance_matrix(T_r, Tinv_r, ZC_r, YC_r, gamma_r, T_s, Tinv_s, ZC_s, YC_s, gamma_s, omega)
    Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
    rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
    h4= h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, l_list[3])
    Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, l_list[3])
    #node5 (Y12 || Y14)
    rho14 = reflection_coefficient(Y_list[3], T_r, Tinv_r, ZC_r, YC_r)
    Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_12 = Y_14 + Y_list[4]
    rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
    Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_node5 = Y_node4carried + Y_1
    rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
    h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, l_list[4])
    Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, l_list[5])
    # node6 (Y13)
    rho13 = reflection_coefficient(Y_list[5], T_r, Tinv_r, ZC_r, YC_r)
    Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, random.uniform(6.5, 6.5))
    Y_node6 = Y_node5carried + Y_13

    hoverall = h1 #@ h2 #@ h3 @ h4 @ h5

    # #ABCD Matrix Approach
    # ABCD1 = calculate_cable_transmission_matrix(R_r, L_r, C_r, G_r, l_list[0], omega)
    # I3 = np.eye(3, dtype=complex)
    # I3 = np.repeat(I3[np.newaxis, :, :], num_freqs, axis=0)
    # Zeros3 = np.zeros(( num_freqs, 3, 3), dtype=complex)
    # ABCD2 = np.block([
    #     [I3, Zeros3],
    #     [-Y_list[0], I3]
    # ])
    # ABCD3 = calculate_cable_transmission_matrix(R_r, L_r, C_r, G_r, l_list[1], omega)
    # ABCD4 = np.block([
    #     [I3, Zeros3],
    #     [-Y_6, I3]
    # ])
    # ABCD5 = calculate_cable_transmission_matrix(R_s, L_s, C_s, G_s, l_list[2], omega)
    # ABCD6 = np.block([
    #     [I3, Zeros3],
    #     [-(Y_5 + Y_4 + Y_3 + Y_2), I3]
    # ])
    # ABCD7 = calculate_cable_transmission_matrix(R_s, L_s, C_s, G_s, l_list[3], omega)
    # ABCD8 = np.block([
    #     [I3, Zeros3],
    #     [-Y_1, I3]
    # ])
    # ABCD9 = calculate_cable_transmission_matrix(R_r, L_r, C_r, G_r, l_list[4], omega)
    # ABCD10 = np.block([
    #     [I3, Zeros3],
    #     [-(Y_13), I3] #change here
    # ])

    # ABCDnw = ABCD1 @ ABCD2 @ ABCD3 @ ABCD4 @ ABCD5 @ ABCD6 @ ABCD7 @ ABCD8 @ ABCD9 @ ABCD10

    # H_abcd = calculate_H_nw(ABCDnw, num_of_conductors-1, Z_rec)
    # Y_nw = calculate_Ynw(ABCDnw, num_of_conductors-1, Z_rec)
    # print("Ynw", Y_nw[0])
    # print("Y_node6", Y_node6[0])
    # #Transmitter
    # YTalpha = (ZT12*ZT13 + ZTG1*ZT13 + ZTG1*ZT12)/(ZTG1*ZT12*ZT13)
    # YTbeta = (ZTG2*ZT12 + ZTG2*ZT23 + ZT12*ZT23)/(ZTG2*ZT12*ZT23)
    # YTgamma = (ZTG3*ZT13 + ZTG3*ZT23 + ZT13*ZT23)/(ZTG3*ZT13*ZT23)

    # H_trans = calculate_Htrans(YTalpha, YTbeta, YTgamma, Y_nw, ZT0, ZT12, ZT12, ZT13, ZT13, ZT23, ZT23)
    H1 = hoverall #@ H_trans
    #H2 = H_abcd @ H_trans

    # #Parameters: Only Load Parameters: 22 loads connected at outlets (2 of the 24 are used by transmitter + receiever)
    # #Cable parameters are not allowed to vary when performing inference because it gave poor results

    H_nw_magnitude_db_1 = 20 * np.log10(np.abs(H1[:, 0, 0]))  # Convert magnitude to dB
    #H_nw_magnitude_db_2 = 20 * np.log10(np.abs(H2[:, 0, 0]))  # Convert magnitude to dB

    # 2) Add random noise with std dev of 1 dB
    noise_dB = np.random.normal(loc=0.0, scale=1.0, size=H_nw_magnitude_db_1.shape)
    print("noisy", noise_dB)
    H_nw1_noisy = H_nw_magnitude_db_1 + noise_dB
    #H_nw2_noisy = H_nw_magnitude_db_2 + noise_dB

    # 3) Variational Inference

    print("My program took", time.time() - start_time, "to run")
    plt.figure(figsize=(8,6))
    plt.plot(freq_range_mhz, H_nw_magnitude_db_1, label=r"$H_{1,1}$ (Model)", color='b')
    #plt.xscale("log")
    #plt.plot(freq_range_mhz, H_nw2_noisy, label=r"$H_{1,1}$ (Model)(ABCD)", color='r')
    plt.xlabel("Frequency (MHz)", fontsize=12)
    plt.ylabel(r"Magnitude (dB)", fontsize=12)
    plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.figure(figsize=(8,6))
    plt.plot(freq_range_mhz, H_nw_magnitude_db_1, label=r"$H_{1,1}$ (Model)", color='b')
    plt.xscale("log")
    #plt.plot(freq_range_mhz, H_nw_magnitude_db_2, label=r"$H_{1,1}$ (Model)(ABCD)", color='r')
    plt.xlabel("Frequency (MHz)", fontsize=12)
    plt.ylabel(r"Magnitude (dB)", fontsize=12)
    plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # # Extract a representative value from each parameter (here, the (0,0) element)
    # R_values = R_s[:, 0, 0].real  # Use .real in case the values are complex
    # L_values = L_s[:, 0, 0].real
    # C_values = C_s[:, 0, 0].real
    # G_values = G_s[:, 0, 0].real

    # # Create a figure with 4 subplots
    # plt.figure(figsize=(12, 10))

    # # Plot Resistance vs Frequency
    # plt.subplot(2, 2, 1)
    # plt.plot(frequencies, R_values, 'b-')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Resistance (Ω)')
    # plt.title('Resistance vs Frequency')
    # plt.grid(True, which="both", ls="--")

    # # Plot Inductance vs Frequency
    # plt.subplot(2, 2, 2)
    # plt.loglog(frequencies, L_values, 'r-')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Inductance (H)')
    # plt.title('Inductance vs Frequency')
    # plt.grid(True, which="both", ls="--")

    # # Plot Capacitance vs Frequency
    # plt.subplot(2, 2, 3)
    # plt.loglog(frequencies, C_values, 'g-')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Capacitance (F)')
    # plt.title('Capacitance vs Frequency')
    # plt.grid(True, which="both", ls="--")

    # # Plot Conductance vs Frequency
    # plt.subplot(2, 2, 4)
    # # A small offset (e.g., 1e-20) is added in case G_values has zero values which cause issues in log scale
    # plt.loglog(frequencies, G_values + 1e-20, 'm-')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Conductance (S)')
    # plt.title('Conductance vs Frequency')
    # plt.grid(True, which="both", ls="--")

    # plt.tight_layout()
    # plt.show()



 