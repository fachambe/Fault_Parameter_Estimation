import numpy as np
import random
import matplotlib.pyplot as plt

import scipy as sp 
import numpy as np
import time
start_time = time.time()

def is_symmetric(matrix, tol=1e-8):
    """
    Check if a matrix is symmetric within a numerical tolerance.
    
    Parameters:
        matrix (ndarray): The input matrix.
        tol (float): Tolerance for symmetry check.
        
    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """
    return np.allclose(matrix, matrix.T, atol=tol)

def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite using Cholesky decomposition.
    
    Parameters:
        matrix (ndarray): The input matrix.
        
    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        # Try Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
def matrix_cosh(M):
    return 0.5*(sp.linalg.expm(M) + sp.linalg.expm(-M))
def matrix_sinh(M):
    return 0.5*(sp.linalg.expm(M) - sp.linalg.expm(-M))
    
def matrix_sqrt(M):
    """
    Compute the square root of a matrix using eigenvalue decomposition.

    Parameters:
        M (ndarray): A square matrix.

    Returns:
        ndarray: The square root of the input matrix.
    """
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eig(M)

    # Ensure eigenvalues are non-negative (important for numerical stability)
    # if np.any(eigvals < 0):  
    #     raise ValueError("Matrix has negative eigenvalues, square root is not defined.")

    # Square root of eigenvalues
    sqrt_eigvals = np.sqrt(eigvals)

    # Reconstruct the square root matrix
    sqrt_M = eigvecs @ np.diag(sqrt_eigvals) @ np.linalg.inv(eigvecs)

    return sqrt_M

def check_matrices(L, C, G):
    """
    Check if the given matrices are symmetric and positive definite.
    
    Parameters:
        L, C, G (ndarray): Input matrices to check.
        
    Returns:
        dict: Results of symmetry and positive definiteness checks.
    """
    results = {}
    for name, matrix in zip(['L', 'C', 'G'], [L, C, G]):
        results[name] = {
            'symmetric': is_symmetric(matrix),
            'positive_definite': is_positive_definite(matrix)
        }
    return results

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
# Function to compute constant impedance admittance matrix (type 1)
def constant_impedance(R_const, C_leak, omega):
    Z12 = Z13 = Z23 = np.full_like(omega, R_const, dtype=complex)  # Convert to array
    ZG1 = ZG2 = ZG3 = 1 / (1j * omega * C_leak)
    return Z12, Z13, Z23, ZG1, ZG2, ZG3 #(this returns a tuple not a list)

# Function to compute double RLC admittance matrix (type 2)
def double_RLC(R_s, omega_0s, zeta_s, R_p, omega_0p, zeta_p, delta_1, delta_2, C_d_leak, omega):
    Z12 = (2j * omega / omega_0p * R_p * zeta_p) / (1 + 2j * omega / omega_0p * R_p * zeta_p - (omega**2 / omega_0p**2))
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
def generate_random_parameters(load_type, omega):
    print("load type", load_type)
    if load_type == 1:  # Constant impedance
        print(type(constant_impedance(loguniform(10, 200, 1), random.uniform(0.1e-9, 2.0e-9), omega)))
        return constant_impedance(loguniform(10, 200, 1), random.uniform(0.1e-9, 2.0e-9), omega)  # R_const, C_leak
    elif load_type == 2:  # Double RLC
        return double_RLC(loguniform(10, 3000, 1), random.uniform(0, 30e6), random.uniform(0.1, 2),
                loguniform(10, 3000, 1), random.uniform(0, 30e6), random.uniform(0.1, 2),
                random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(0.1, 2), omega)
    elif load_type == 3:  # Motor model
        return motor_load(random.uniform(0.1e-9, 1e-9), random.uniform(5e-3, 20e-3), random.uniform(2000, 15000),
                5, random.uniform(0.2e-9, 5.0e-9), omega)
    
def invert_3d(A):
    """
    Invert a batch of 3x3 matrices, shape (3,3,N).
    Returns an array of shape (3,3,N) with each slice inverted.
    """
    out = np.zeros_like(A)
    N = A.shape[2]
    for i in range(N):
        out[:,:,i] = np.linalg.inv(A[:,:,i])
    return out

def add_3d(A, B):
    """
    Elementwise addition A + B for arrays of shape (3,3,N).
    """
    return A + B
def multiply_3d(*matrices):
    """
    Multiplies multiple 3D matrices slice-by-slice along the third dimension.
    
    Parameters:
        matrices: List of 3D NumPy arrays of shape (m, n, N),
                  where N is the number of slices (frequency samples).
    
    Returns:
        C: 3D NumPy array containing the multiplied results with shape (m, p, N),
           where p is determined by the matrix dimensions.
    """
    if len(matrices) < 2:
        raise ValueError("At least two matrices are required for multiplication.")
    
    # Get the number of slices (third dimension)
    N = matrices[0].shape[2]

    # Ensure all matrices have the same third dimension size
    for mat in matrices:
        if mat.shape[2] != N:
            raise ValueError("All matrices must have the same third dimension size.")
    
    # Start with the first matrix
    C = matrices[0].copy()

    # Multiply sequentially with the rest
    for mat in matrices[1:]:
        for i in range(N):
            C[:, :, i] = C[:, :, i] @ mat[:, :, i]  # Slice-wise matrix multiplication

    return C

def combine_series_admittance_3d(YA, YB):
    """
    Combine two 3x3xN admittances YA,YB in series.
      Z_series = Z_A + Z_B
      => Y_series = (Z_series)^-1
    """
    ZA = invert_3d(YA)
    ZB = invert_3d(YB)
    Zseries = ZA + ZB
    return invert_3d(Zseries)

def combine_parallel_admittance_3d(YA, YB):
    """
    Combine two 3x3xN admittances in parallel => Y = YA + YB.
    """
    return YA + YB
def compute_load_admittance_3d(load_params_k, omega):
    """
    load_params_k = (Z12, Z13, Z23, ZG1, ZG2, ZG3)
    Each is an array of length N freq.  Return a (3,3,N) array.
    """
    N = len(omega)
    Y_out = np.zeros((3,3,N), dtype=complex)
    for i in range(N):
        Z12_i, Z13_i, Z23_i, ZG1_i, ZG2_i, ZG3_i = (p[i] for p in load_params_k)
        Y_out[:,:,i] = compute_admittance_matrix(Z12_i, Z13_i, Z23_i, 
                                                ZG1_i, ZG2_i, ZG3_i)
    return Y_out

# Function to convert impedances to admission matrix
def compute_admittance_matrix(Z12, Z13, Z23, ZG1, ZG2, ZG3):
    Y_L = np.array([
        [1/ZG1 + 1/Z12 + 1/Z13, -1/Z12, -1/Z13],
        [-1/Z12, 1/ZG2 + 1/Z12 + 1/Z23, -1/Z23],
        [-1/Z13, -1/Z23, 1/ZG3 + 1/Z13 + 1/Z23]
    ], dtype=complex)
    return Y_L

def get_mtl_matrices(rw, omega):
    """
    Parameters:
    - length of MTL, radius of MTL, frequencies
    Returns:
    - T (transformation matrix), Tinv, gamma, YC, ZC
    """
    R, L, C, G = calculate_cable_parameters(rw, omega)
    num_freqs = len(omega)
    T = np.zeros((3, 3, num_freqs), dtype=complex)
    T_inv = np.zeros((3, 3, num_freqs), dtype=complex)
    Gamma = np.zeros((3, 3, num_freqs), dtype=complex)
    ZC = np.zeros((3, 3, num_freqs), dtype=complex)
    YC = np.zeros((3, 3, num_freqs), dtype=complex)
    for i in range(num_freqs):
        R_i = R[:, :, i]
        L_i = L[:, :, i]
        C_i = C[:, :, i]
        G_i = G[:, :, i]
        omega_i = omega[i]

        # Calculate intermediate components for gamma
        Z_T = R_i + 1j * omega_i * L_i
        Y_T = G_i + 1j * omega_i * C_i

        ZY = Z_T @ Y_T
        YZ = Y_T @ Z_T
        eigvals, eigvecs = np.linalg.eig(ZY)
        eigvals2, eigvecs2 = np.linalg.eig(YZ)
        print("eigvals from get_mtl", eigvals)
        print("eigvals2 from get_mtl", eigvals2)
        gamma = np.diag(np.sqrt(eigvals2))
        Zc = np.linalg.inv(Y_T) @ eigvecs2 @ gamma @ np.linalg.inv(eigvecs2)
        Yc = np.linalg.inv(Zc)

        T[:, :, i] = eigvecs2
        T_inv[:, :, i] = np.linalg.inv(eigvecs2)
        Gamma[:, :, i] = gamma
        ZC[:, :, i] = Zc
        YC[:, :, i] = Yc
    return T, T_inv, Gamma, ZC, YC
def reflection_coefficient_multi(YL, T, T_inv, ZC, YC):
    """
    Vectorized version of eq. (12).
    YL, T, T_inv, ZC, YC each has shape (3,3,N).
    Returns rho of shape (3,3,N).
    """
    N = YL.shape[2]
    out = np.zeros_like(YL)
    for i in range(N):
        YLi   = YL[:,:,i]
        Ti    = T[:,:,i]
        Tinv  = T_inv[:,:,i]
        ZCi   = ZC[:,:,i]
        YCi   = YC[:,:,i]
        out[:,:,i] = (Tinv @ YCi 
                      @ np.linalg.inv(YLi + YCi) 
                      @ (YLi - YCi) 
                      @ ZCi @ Ti)
    return out
def carry_back_load_multi(rhoL, T, YC, Gamma, length):
    """
    Vectorized eq. (13).
    rhoL, T, YC, Gamma each is (3,3,N). Returns Y_R(x) shape (3,3,N).
    """
    N = rhoL.shape[2]
    out = np.zeros_like(rhoL)
    for i in range(N):
        rho_i = rhoL[:,:,i]
        T_i   = T[:,:,i]
        Ti_inv= np.linalg.inv(T_i)
        YC_i  = YC[:,:,i]
        Gam_i = Gamma[:,:,i]
        e_pos = sp.linalg.expm(Gam_i * length)
        e_neg = sp.linalg.expm(-Gam_i * length)
        num   = e_pos + e_neg @ rho_i
        den   = e_pos - e_neg @ rho_i
        deninv= np.linalg.inv(den)
        out[:,:,i] = T_i @ num @ deninv @ Ti_inv @ YC_i
    return out

def h_B_multi(rhoLI, ZC, T, T_inv, Gamma, length):
    N = rhoLI.shape[2]
    out = np.zeros_like(rhoLI)
    U = np.eye(3, dtype=complex)
    for i in range(N):
        rho_Li = rhoLI[:,:,i]
        ZC_i = ZC[:, :, i]
        T_i = T[:, :, i]
        T_invi = T_inv[:, :, i]
        Gamma_i = Gamma[:, :, i]
        e_pos = sp.linalg.expm(Gamma_i * length)
        e_neg = sp.linalg.expm(-Gamma_i * length)
        out[:, :, i] = ZC_i @ T_i @ (U - rho_Li) @ np.linalg.inv(e_pos - e_neg @ rho_Li) @ T_invi @ np.linalg.inv(ZC_i)
    return out
def calculate_room_transmission_matrix(load_params, omega):
    """
    Returns a 3x3xN array Y_room_final for all frequencies at once, 
    by carrying loads 3 & 4 back in the 'room' diagram, 
    then combining in series with 1 & 2, etc.
    """
    # 1) Build 3x3xN admittance arrays for each load
    #    load_params[k] is e.g. (Z12, Z13, Z23, ZG1, ZG2, ZG3) across freq
    Y_L1 = compute_load_admittance_3d(load_params[0], omega)  # shape (3,3,N)
    Y_L2 = compute_load_admittance_3d(load_params[1], omega)
    Y_L3 = compute_load_admittance_3d(load_params[2], omega)
    Y_L4 = compute_load_admittance_3d(load_params[3], omega)

    # 2) Get MTL parameters for the entire room (all freq) => shape (3,3,N)
    wire_length_room = np.random.uniform(2, 20)
    wire_radius_room = np.random.uniform(0.81e-3, 1.29e-3)
    T, Tinv, Gamma, ZC, YC = get_mtl_matrices(wire_radius_room, omega)

    # 3) Reflection for loads 3 & 4, then carry them back
    rho4 = reflection_coefficient_multi(Y_L4, T, Tinv, ZC, YC)
    Y_R4 = carry_back_load_multi(rho4, T, YC, Gamma, wire_length_room)

    rho3 = reflection_coefficient_multi(Y_L3, T, Tinv, ZC, YC)
    Y_R3 = carry_back_load_multi(rho3, T, YC, Gamma, wire_length_room)

    # 4) Series with loads 1 & 2 => combine_series_admittance_3d
    #    Y_L2_new = series( Y_R4, Y_L2 )
    Y_L2_new = combine_series_admittance_3d(Y_R4, Y_L2)
    Y_L1_new = combine_series_admittance_3d(Y_R3, Y_L1)

    # 5) Reflection again for Y_L1_new & Y_L2_new => carry them back
    rho2 = reflection_coefficient_multi(Y_L2_new, T, Tinv, ZC, YC)
    Y_R2 = carry_back_load_multi(rho2, T, YC, Gamma, wire_length_room)

    rho1 = reflection_coefficient_multi(Y_L1_new, T, Tinv, ZC, YC)
    Y_R1 = carry_back_load_multi(rho1, T, YC, Gamma, wire_length_room)

    # 6) Finally parallel => Y_room = Y_R1 + Y_R2
    Y_room_final = combine_parallel_admittance_3d(Y_R1, Y_R2)
    return Y_room_final
    #     print("Yroom", Y_room)        
    #     # Compute transmission matrix for parallel element
    #     I3 = np.eye(3)
    #     Zeros3 = np.zeros((3, 3))
    #     Phi_room[:, :, i]= np.block([[
    #         I3, Zeros3], 
    #         [-Y_room, I3]
    #         ])
    # return Phi_room

# Calculate cable parameters
def calculate_cable_parameters(r_w, omega):
    # Constants
    f = omega/ (2 * np.pi)
    mu_0 = 4 * np.pi * 10e-7
    sigma = 5.8 * 10e7
    epsilon = 3.19 * 10e-11
    dc = 4 * 10e-4 + 3.02*r_w
    dc2 = np.sqrt(2) * dc
    tandeltal = 1e-6
    delta = 1 / np.sqrt(np.pi * mu_0 * sigma * f)  
    
    # Resistance
    r = np.zeros_like(f)
    for i in range(len(f)):
        if r_w <= 2 * delta[i]:
            r[i] = 1 / (sigma * np.pi * r_w**2)
        else:
            r[i] = (1 / (2 * r_w)) * np.sqrt((mu_0 * f[i]) / (np.pi * sigma))

    R = np.zeros((3, 3, len(f)), dtype=complex)
    for i in range(len(f)):
        R[:, :, i] = np.array([
            [2 * r[i], r[i], r[i]],
            [r[i], 2 * r[i], r[i]],
            [r[i], r[i], 2 * r[i]]
        ])
    # Inductance
    L_const = (mu_0 / 2 * np.pi) * np.array([
        [2*np.log(dc / r_w), np.log((dc * dc2) / (dc * r_w)), np.log((dc * dc) / (dc2 * r_w))],
        [np.log((dc * dc2) / (dc * r_w)), 2*np.log(dc2 / r_w), np.log((dc2 * dc) / (dc * r_w))],
        [np.log( (dc * dc) / (dc2 * r_w)), np.log((dc2 * dc) / (dc * r_w)), 2*np.log( dc / r_w)]
    ])
    L = np.repeat(L_const[:, :, np.newaxis], len(f), axis=2)  # Expand to 3x3x200
    print("L_const shape:", L_const.shape)
    print("L_const:", L_const)
    # Capacitance (constant across frequencies)
    C_const = mu_0 * epsilon * np.linalg.inv(L_const)
    C = np.repeat(C_const[:, :, np.newaxis], len(f), axis=2)  # Expand to 3x3x200
    print("C_const shape:", C_const.shape)
    print("C_const:", C_const)
    print("C.shape", C.shape)
    G = np.zeros((3, 3, len(f)), dtype=complex)
    # Conductance (assuming loss tangent = 0, constant across frequencies)
    for i in range(len(f)):
        G[:, :, i] = omega[i] * tandeltal * C_const
    #G_const = np.zeros((3, 3), dtype=complex)
    #G = np.repeat(G_const[:, :, np.newaxis], len(f), axis=2)  # Expand to 3x3x200
    print("g", G.shape)
    return R, L, C, G

def calculate_cable_transmission_matrix(R, L, C, G, length, omega):
    num_freqs = len(omega)
    Phi_cable = np.zeros((6, 6, num_freqs), dtype=complex)
    Phi_real = np.zeros((4, 4, num_freqs), dtype=complex)
    for i in range(num_freqs):
        # Extract 3x3 matrices for the current frequency
        R_i = R[:, :, i]
        L_i = L[:, :, i]
        C_i = C[:, :, i]
        G_i = G[:, :, i]
        omega_i = omega[i]

        # Calculate intermediate components for gamma
        Z_T = R_i + 1j * omega_i * L_i
        Y_T = G_i + 1j * omega_i * C_i

        ZY = Z_T @ Y_T
        YZ = Y_T @ Z_T 

        eigvals, eigvecs = np.linalg.eig(ZY)
        eigvals2, eigvecs2 = np.linalg.eig(YZ)
        print("eigvals from abcd", eigvals)
        print("eigvals2 from abcd", eigvals2)
        gamma = np.diag(np.sqrt(eigvals))

        Ywhat =  eigvecs.T @ Y_T @ eigvecs @ np.linalg.inv(gamma)
        Yw = eigvecs @ Ywhat @ np.linalg.inv(eigvecs)
        Zw = np.linalg.inv(Yw)
        Gamma = eigvecs @ gamma @ np.linalg.inv(eigvecs)
        Phi11 = matrix_cosh(Gamma*length)
        #print("A", Phi11)
        Phi12 = -matrix_sinh(Gamma*length)@ Zw
        #print("B", Phi12)
        Phi21 = -Yw @ matrix_sinh(Gamma*length)
        #print("C", Phi21)
        Phi22 = Yw @ matrix_cosh(Gamma*length) @ Zw

        # Form the transmission matrix for the current frequency
        Phi_cable[:, :, i] = np.block([
            [Phi11, Phi12],
            [Phi21, Phi22]
        ])
    return Phi_cable
        
# Define Z_rec for the receiver load
def calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3):
    Z_rec = np.array([
        [Z_RG + Z_R1, Z_RG, Z_RG],
        [Z_RG, Z_RG + Z_R2, Z_RG],
        [Z_RG, Z_RG, Z_RG + Z_R3]
    ])
    return Z_rec
# Calculate H_nw from the network transmission matrix
def calculate_H_nw(Phi_network, Z_rec):
    """
    Phi_network: Overall network transmission matrix (6x6x200)
    Z_rec: Receiver load matrix (3x3)
    Returns H_nw, the transfer function of the network. (3x3x200)
    """
    num_freqs = Phi_network.shape[2]
    H_nw = np.zeros((3, 3, num_freqs), dtype=complex)
    Phi_networkinv = np.zeros((6, 6, num_freqs), dtype=complex)
    for i in range(num_freqs):
        Phi_networkinv[:, :, i] = np.linalg.inv(Phi_network[:, :, i])
    for i in range(num_freqs):
    # Split Phi_network into submatrices
        Phi_11 = Phi_networkinv[:3, :3, i]
        Phi_12 = Phi_networkinv[:3, 3:, i]
        Phi_21 = Phi_networkinv[3:, :3, i]
        Phi_22 = Phi_networkinv[3:, 3:, i]
        #print("PHI11", Phi_11)
        # Calculate H_nw
        H_nw[:, :, i] = np.linalg.inv(Phi_11 + Phi_12 @ np.linalg.inv(Z_rec))
        #Phi_12 @ np.linalg.inv(Z_rec)
    return H_nw

if __name__ == '__main__':
    num_rooms = 4 #Technically 6 but first and last room require manual calculation
    r_w_servicepanel = random.uniform(1.03e-3, 2.06e-3)
    r_w_room = random.uniform(0.81e-3, 1.29e-3)
    l = random.uniform(10, 10) #assume all wire lengths 10m for now
    #CONSTANTS
    frequencies = np.logspace(np.log10(150e3), np.log10(30e6), 200)  # 150 kHz to 30 MHz log spacing
    omega = 2 * np.pi * frequencies  # Angular frequency
    num_freqs = len(omega)

    #(Tonello Paper) Reflection Coefficient Approach
    Y_rec = np.zeros((3,3, num_freqs), dtype=complex)
    Z_RG = Z_R1 = Z_R2 = Z_R3 = 50
    Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3)

    T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(r_w_servicepanel, omega) #MTL parameters of wires in service panel
    T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(r_w_room, omega) #MTL parameters of wires in room

    for i in range(num_freqs):
        Y_rec[:, :, i] = np.linalg.inv(Z_rec)

    load_params = [generate_random_parameters(random.choice([1, 2, 3]), omega)]
    Y_64 = compute_load_admittance_3d(load_params[0], omega)
    load_params = [generate_random_parameters(random.choice([1, 2, 3]), omega)]
    Y_62 = compute_load_admittance_3d(load_params[0], omega)
    load_params = [generate_random_parameters(random.choice([1, 2, 3]), omega)]
    Y_61 = compute_load_admittance_3d(load_params[0], omega)

    #Start at Y_rec
    rho1 = reflection_coefficient_multi(Y_rec, T_r, Tinv_r, ZC_r, YC_r)
    h1 = h_B_multi(rho1, ZC_r, T_r, Tinv_r, gamma_r, l)
    Y_rec_carried = carry_back_load_multi(rho1, T_r, YC_r, gamma_r, l)

    Y_64_new = combine_series_admittance_3d(Y_64, Y_rec_carried)
    rho2 = reflection_coefficient_multi(Y_64_new, T_r, Tinv_r, ZC_r, YC_r)
    h2 = h_B_multi(rho2, ZC_r, T_r, Tinv_r, gamma_r, l)
    Y_64_carried = carry_back_load_multi(rho2, T_r, YC_r, gamma_r, l)
    
    Y_62new = combine_series_admittance_3d(Y_62, Y_64_carried)
    rho3 = reflection_coefficient_multi(Y_62new, T_r, Tinv_r, ZC_r, YC_r)
    h3 =  h_B_multi(rho3, ZC_r, T_r, Tinv_r, gamma_r, l)
    Y_62_carried = carry_back_load_multi(rho3, T_r, YC_r, gamma_r, l)

    htemp = multiply_3d(h1, h2)
    #Carry back to Y61 then add in parallel
    
    R, L, C, G = calculate_cable_parameters(r_w_room, omega)
    ABCD1 = calculate_cable_transmission_matrix(R, L, C, G, l, omega)
    I3 = np.eye(3)
    Zeros3 = np.zeros((3, 3))  # Shape: (3, 3, num_freqs)
    ABCD2 = np.zeros((6, 6, num_freqs), dtype=complex)
    for i in range(num_freqs):
        ABCD2[:, :, i] = np.block([
            [I3, Y_64[:, :, i]],
            [Zeros3, I3]
        ])
    ABCDoverall = multiply_3d(ABCD1, ABCD2, ABCD1)
    H_abcd = calculate_H_nw(ABCD1, Z_rec)
    print("My program took", time.time() - start_time, "to run")

    freq_range_mhz = frequencies / 1e6  # Convert to MHz

    H_nw_magnitude_db_1 = 20 * np.log10(np.abs(h1[0, 0, :]))  # Convert magnitude to dB
    H_nw_magnitude_db_2 = 20 * np.log10(np.abs(H_abcd[0, 0, :]))  # Convert magnitude to dB

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

    # First plot
    ax[0].plot(freq_range_mhz, H_nw_magnitude_db_1, label=r"$H_{1,1}$ (Model)", color='b')
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Frequency (MHz)", fontsize=12)
    ax[0].set_ylabel(r"$H_{1,1}$ Magnitude (dB)", fontsize=12)
    ax[0].set_title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    ax[0].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax[0].legend(fontsize=12)

    # Second plot
    ax[1].plot(freq_range_mhz, H_nw_magnitude_db_2, label=r"$H_{1,1}$ (Model)", color='b')
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Frequency (MHz)", fontsize=12)
    ax[1].set_ylabel(r"$H_{1,1}$ Magnitude 2(dB)", fontsize=12)
    ax[1].set_title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    ax[1].grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax[1].legend(fontsize=12)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()  # Show both plots together
        

    # Initialize overall network transmission matrix as identity matrix for each frequency
    # Phi_network = np.eye(6, 6, dtype=complex)[..., np.newaxis] * np.ones(num_freqs, dtype=complex)
    # for room in range(num_rooms):
    #     print(f"Processing Room {room + 1}/{num_rooms}")
    #     # Generate random load parameters for this room
    #     load_params = [generate_random_parameters(1, omega) for _ in range(4)]
    #     # Calculate room admittance matrix
    #     Y_room = calculate_room_transmission_matrix(load_params, omega)  # Shape (6, 6, 200)
    #     #print("PHI ROOM", Phi_room[:, :, 100])
    #     # Calculate cable transmission matrix
    #     Phi_cable = calculate_cable_transmission_matrix(R, L, C, G, l, omega)  # Shape (6, 6, 200)
    #     # print("PHI CABLE", Phi_cable[:, :, 0])
    #     # # Phi_test = Phi_room[:, :, 0] @ Phi_cable[:, :, 0] @ Phi_room[:, :, 0] @ Phi_cable[:, :, 0]
    #     # # print("PHi test", Phi_test) 
    #     Phi_combined = np.zeros_like(Phi_room, dtype=complex)
    #     Z_RG = Z_R1 = Z_R2 = Z_R3 = 50
    #     Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3)

    #     H_nw = calculate_H_nw(Phi_cable, Z_rec)
    #for i in range(num_freqs):
    #   print("H_nw (Network Transfer Function):", i, H_nw[:, :, i])

    # Inspect the transfer function at specific frequencies
    #print("H_nw at first frequency:", H_nw[:, :, 0])  # Transfer function for the first frequency
    #print("H_nw at mid frequency:", H_nw[:, :, 99])  # Transfer function for the 100th frequency
    # Plot the magnitude of H_nw(1,1) across frequencies
    #     freq_range_hz = np.logspace(np.log10(150e3), np.log10(30e6), 200)  # Frequency range in Hz
    #     freq_range_mhz = freq_range_hz / 1e6  # Convert to MHz
    # #print("Frequency range (MHz):", freq_range_mhz)
    #     H_nw_magnitude_db = 20 * np.log10(np.abs(H_nw[0, 0, :]))  # Convert magnitude to dB
    # #print("HNW MAGNitude", H_nw_magnitude_db)

    # # Plot
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(freq_range_mhz, H_nw_magnitude_db, label=r"$H_{1,1}$ (Model)", color='b')

    # # Set log scale and labels
    #     plt.xscale("log")
    #     plt.xlabel("Frequency (MHz)", fontsize=12)
    #     plt.ylabel(r"$H_{1,1}$ Magnitude (dB)", fontsize=12)
    #     plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    #     plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    #     plt.legend(fontsize=12)
    #     plt.tight_layout()

    # # Show the plot
    #     plt.show()  
        # #print("0", np.max(np.abs(Phi_network)))
        # #Combine Phi_room and Phi_cable for each frequency and update Phi_network
        # for i in range(num_freqs):
        #     Phi_combined[:, :, i] = Phi_room[:, :, i] @ Phi_cable[:, :, i]  # Multiply for current frequency
            
        #     Phi_network[:, :, i] = Phi_network[:, :, i] @ Phi_combined[:, :, i]  # Update overall network matrix
        # # print(f"Phi_network max value after room {room + 1}:", np.max(np.abs(Phi_network)))
        # # print("Phi_combined[0]:", Phi_combined[:, :, 100])
        # print("Phi_network[100]:", Phi_network[:, :, 100])
    # print("0", Phi_network[:, :, 0])
    # print("50", Phi_network[:, :, 50])
    # print("100", Phi_network[:, :, 100])
    # print("150", Phi_network[:, :, 150])
    # print("200", Phi_network[:, :, 199])
    # Z_RG = Z_R1 = Z_R2 = Z_R3 = 50
    # Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3)

    # H_nw = calculate_H_nw(Phi_network, Z_rec)
    # #for i in range(num_freqs):
    # #   print("H_nw (Network Transfer Function):", i, H_nw[:, :, i])

    # # Inspect the transfer function at specific frequencies
    # #print("H_nw at first frequency:", H_nw[:, :, 0])  # Transfer function for the first frequency
    # #print("H_nw at mid frequency:", H_nw[:, :, 99])  # Transfer function for the 100th frequency
    # # Plot the magnitude of H_nw(1,1) across frequencies
    # freq_range_hz = np.logspace(np.log10(150e3), np.log10(30e6), 200)  # Frequency range in Hz
    # freq_range_mhz = freq_range_hz / 1e6  # Convert to MHz
    # #print("Frequency range (MHz):", freq_range_mhz)
    # H_nw_magnitude_db = 20 * np.log10(np.abs(H_nw[0, 0, :]))  # Convert magnitude to dB
    # #print("HNW MAGNitude", H_nw_magnitude_db)

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(freq_range_mhz, H_nw_magnitude_db, label=r"$H_{1,1}$ (Model)", color='b')

    # # Set log scale and labels
    # plt.xscale("log")
    # plt.xlabel("Frequency (MHz)", fontsize=12)
    # plt.ylabel(r"$H_{1,1}$ Magnitude (dB)", fontsize=12)
    # plt.title(r"$H_{1,1}$ Transfer Function in dB", fontsize=14)
    # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    # plt.legend(fontsize=12)
    # plt.tight_layout()

    # # Show the plot
    # plt.show()  
    # print(H_nw[:, :, 0])


