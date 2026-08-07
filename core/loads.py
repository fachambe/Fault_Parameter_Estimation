"""
Load impedance models for MTL network simulation.

This module contains functions to compute load impedance/admittance matrices
for different load types:
- Constant impedance loads
- Double RLC loads (series + parallel RLC)
- Motor loads
"""

import torch


def constant_impedance(R_const, C_leak, omega):
    """
    Compute impedance values for constant impedance load.

    Args:
        R_const: Constant resistance. Scalar or [P, 1] for batched.
        C_leak: Leakage capacitance. Scalar or [P, 1] for batched.
        omega: Angular frequencies [F].

    Returns:
        Z12, Z13, Z23, ZG1, ZG2, ZG3: Impedance values.
            Shape [F] or [P, F] depending on input.
    """
    # When vectorize_particles=True, R_const and C_leak have shape [P,1]
    # omega has shape [F]. Result should be [P, F] or [F] depending on input.
    ZG1 = 1 / (1j * omega * C_leak)
    ZG2 = ZG3 = ZG1
    # Ensure Z12 has same shape as ZG1 by using ones_like(ZG1)
    Z12 = R_const.to(torch.complex64) * torch.ones_like(ZG1)
    Z13 = Z23 = Z12
    return Z12, Z13, Z23, ZG1, ZG2, ZG3


def double_RLC(R_s, omega_0s, zeta_s, R_p, omega_0p, zeta_p,
               delta_1, delta_2, C_d_leak, omega):
    """
    Compute impedance values for double RLC load (series + parallel RLC).

    The double RLC load consists of:
    - Series RLC branch with parameters (R_s, omega_0s, zeta_s)
    - Parallel RLC branch with parameters (R_p, omega_0p, zeta_p)
    - Asymmetry factors delta_1, delta_2
    - Leakage capacitance C_d_leak

    Args:
        R_s: Series resistance. Scalar or [P, 1].
        omega_0s: Series resonance frequency. Scalar or [P, 1].
        zeta_s: Series damping factor. Scalar or [P, 1].
        R_p: Parallel resistance. Scalar or [P, 1].
        omega_0p: Parallel resonance frequency. Scalar or [P, 1].
        zeta_p: Parallel damping factor. Scalar or [P, 1].
        delta_1: Asymmetry factor 1. Scalar or [P, 1].
        delta_2: Asymmetry factor 2. Scalar or [P, 1].
        C_d_leak: Leakage capacitance. Scalar or [P, 1].
        omega: Angular frequencies [F].

    Returns:
        Z12, Z13, Z23, ZG1, ZG2, ZG3: Impedance values.
            Shape [F] or [P, F] depending on input.
    """
    x_s = omega / omega_0s
    x_p = omega / omega_0p

    # Series RLC impedance
    Z_series = (
        R_s
        + 1j * (R_s / (2.0 * zeta_s))
        * (x_s - 1.0 / x_s)
    )

    # Parallel RLC impedance
    Z_parallel = (
        2j * R_p * zeta_p * x_p
        / (1.0 + 2j * zeta_p * x_p - x_p**2)
    )

    Z12 = Z_series + Z_parallel
    Z13 = Z12 * (1.0 + delta_1)
    Z23 = Z12 * (1.0 + delta_2)

    Z_ground = 1.0 / (1j * omega * C_d_leak)
    ZG1 = Z_ground
    ZG2 = Z_ground
    ZG3 = Z_ground

    return Z12, Z13, Z23, ZG1, ZG2, ZG3


def motor_load(C_m, L_m, R_m1, R_m2, C_m_leak, omega):
    """
    Compute impedance values for motor load.

    Args:
        C_m: Motor capacitance. Scalar or [P, 1].
        L_m: Motor inductance. Scalar or [P, 1].
        R_m1: Motor resistance 1. Scalar or [P, 1].
        R_m2: Motor resistance 2 (typically fixed). Scalar or [P, 1].
        C_m_leak: Motor leakage capacitance. Scalar or [P, 1].
        omega: Angular frequencies [F].

    Returns:
        Z12, Z13, Z23, ZG1, ZG2, ZG3: Impedance values.
            Shape [F] or [P, F] depending on input.
    """
    Zprime = 1j * omega * L_m + R_m2
    Z12 = (1/3) * 1 / (
        1j * omega * (C_m + (C_m_leak/2)) + 1 / R_m1 + 1 / Zprime
    )
    Z13 = Z23 = Z12
    ZG1 = (1/3) * 1 / (
        1j * omega * C_m_leak + 1 / (
            1j * omega * C_m_leak + (R_m1 * Zprime) / (
                1j * omega * C_m * Zprime * R_m1 + Zprime + R_m1
            )
        )
    )
    ZG2 = ZG3 = ZG1
    return Z12, Z13, Z23, ZG1, ZG2, ZG3


def compute_load_admittance_3d(load_params):
    """
    Compute 3x3 load admittance matrix from impedance values.

    Args:
        load_params: Tuple of tensors (Z12, Z13, Z23, ZG1, ZG2, ZG3).
            Each impedance tensor can have shape (..., F) where ... are batch dims.

    Returns:
        Y_load: Load admittance matrix with shape (..., F, 3, 3)
    """
    Z12, Z13, Z23, ZG1, ZG2, ZG3 = load_params

    # Compute admittance elements
    Y11 = 1/ZG1 + 1/Z12 + 1/Z13
    Y12 = -1/Z12
    Y13 = -1/Z13
    Y22 = 1/ZG2 + 1/Z12 + 1/Z23
    Y23 = -1/Z23
    Y33 = 1/ZG3 + 1/Z13 + 1/Z23

    # Broadcast all elements to common shape before stacking
    elements = [Y11, Y12, Y13, Y22, Y23, Y33]
    target_shape = torch.broadcast_shapes(*[e.shape for e in elements])
    Y11, Y12, Y13, Y22, Y23, Y33 = [e.expand(target_shape) for e in elements]

    # Stack into (..., F, 3, 3) tensor
    row1 = torch.stack([Y11, Y12, Y13], dim=-1)  # (..., F, 3)
    row2 = torch.stack([Y12, Y22, Y23], dim=-1)  # (..., F, 3)
    row3 = torch.stack([Y13, Y23, Y33], dim=-1)  # (..., F, 3)
    Y_load = torch.stack([row1, row2, row3], dim=-2)  # (..., F, 3, 3)

    return Y_load


def generate_load_parameters_deterministic(network_params, load_types):
    """
    Generate FIXED load parameters matching the paper (no randomness).

    Populates network_params["loads"] in-place based on load type assignments.

    Args:
        network_params: Global network parameter dict (modified in-place).
        load_types: List of load type integers for each load.
            1 = Constant Impedance (2 params)
            2 = Double RLC (9 params)
            3 = Motor (5 params)

    Returns:
        total_parameters: Total number of load parameters generated.
        load_types: The input load_types list (for convenience).
    """
    network_params["loads"].clear()
    total_parameters = 0

    for i, load_type in enumerate(load_types):
        if load_type == 1:  # Constant Impedance (2)
            network_params["loads"][f"load_{i}"] = {
                "R_const": {"value": 0.25, "inferred": True, "range": (10, 200), "dist": "log"},
                "C_leak": {"value": 0.25, "inferred": True, "range": (0.1e-9, 2.0e-9)}
            }
            total_parameters += 2

        elif load_type == 2:  # Double RLC (9)
            network_params["loads"][f"load_{i}"] = {
                "R_s": {"value": 0.25, "inferred": True, "range": (10, 3000), "dist": "log"},
                "omega_0s": {"value": 0.25, "inferred": True, "range": (0.1e6, 30e6)},
                "zeta_s": {"value": 0.25, "inferred": True, "range": (0.1, 2)},
                "R_p": {"value": 0.25, "inferred": True, "range": (10, 3000), "dist": "log"},
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

        else:
            raise ValueError(f"Unknown load type {load_type} for load_{i}")

    return total_parameters, load_types


def compute_load_admittances(load_params_physical, omega):
    """
    Compute admittance matrices for all loads in the network.

    Args:
        load_params_physical: Dict mapping load names to parameter dicts.
            Each parameter dict contains physical (denormalized) parameter values.
        omega: Angular frequencies [F].

    Returns:
        Y_loads: Dict mapping load names to admittance matrices [P, F, 3, 3] or [F, 3, 3].
    """
    Y_loads = {}

    for load_name, params in load_params_physical.items():
        if 'R_const' in params and 'C_leak' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = constant_impedance(
                params['R_const'], params['C_leak'], omega
            )
        elif 'R_s' in params and 'omega_0s' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = double_RLC(
                params['R_s'], params['omega_0s'], params['zeta_s'],
                params['R_p'], params['omega_0p'], params['zeta_p'],
                params['delta_1'], params['delta_2'], params['C_d_leak'], omega
            )
        elif 'C_m' in params and 'L_m' in params:
            Z12, Z13, Z23, ZG1, ZG2, ZG3 = motor_load(
                params['C_m'], params['L_m'], params['R_m1'],
                params['R_m2'], params['C_m_leak'], omega
            )
        else:
            raise ValueError(f"Unknown load model in {load_name}")

        Y_loads[load_name] = compute_load_admittance_3d(
            (Z12, Z13, Z23, ZG1, ZG2, ZG3)
        )

    return Y_loads
