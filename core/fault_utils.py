"""
Fault-related utility functions for MTL forward model.

This module contains functions for computing transfer functions
through cables with shunt faults.
"""

import torch

from .mtl_utils import reflection_coefficient, carry_back_load, h_B


# Backbone cable keys (cables on the main signal path)
BACKBONE_KEYS = ["l_w_0", "l_w_1", "l_w_4", "l_w_25", "l_w_28"]


def get_total_backbone_length(cable_lengths, device):
    """Calculate total backbone length L from cable_lengths dict of tensors."""
    total = torch.tensor(0.0, device=device)
    for key in BACKBONE_KEYS:
        val = cable_lengths[key]
        total = total + val
    return total


def compute_fault_admittance_matrix(Z_fault_real, Z_fault_imag, F, n, device, k=0):
    """
    Compute shunt fault admittance matrix with conductor k to ground.

    Parameters:
        Z_fault_real: Real part of fault impedance of shape [] or [P, 1]
        Z_fault_imag: Imaginary part of fault impedance of shape [] or [P, 1]
        F: number of frequency points
        n: number of conductors - 1
        device: torch device
        k: which conductor has fault to ground (default 0)

    Returns:
        Y_fault: (F, n, n) if scalar input, or (P, F, n, n) if batched input
    """
    if not isinstance(Z_fault_real, torch.Tensor):
        Z_fault_real = torch.tensor(Z_fault_real, dtype=torch.float32, device=device)
    if not isinstance(Z_fault_imag, torch.Tensor):
        Z_fault_imag = torch.tensor(Z_fault_imag, dtype=torch.float32, device=device)

    is_batched = Z_fault_real.dim() >= 1 and Z_fault_real.shape[0] > 1

    # Compute complex admittance Y = 1/Z = 1/(R + jX)
    denom = Z_fault_real**2 + Z_fault_imag**2
    Y_real = Z_fault_real / denom
    Y_imag = -Z_fault_imag / denom

    # Create one-hot mask for position (k, k)
    mask = torch.eye(n, dtype=torch.float32, device=device)[k:k+1, :].T @ \
           torch.eye(n, dtype=torch.float32, device=device)[k:k+1, :]

    if is_batched:
        Y_real_bc = Y_real.view(-1, 1, 1, 1)
        Y_imag_bc = Y_imag.view(-1, 1, 1, 1)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(1, F, -1, -1)
        Y_fault_real = Y_real_bc * mask_expanded
        Y_fault_imag = Y_imag_bc * mask_expanded
    else:
        Y_fault_real = Y_real * mask.unsqueeze(0).expand(F, -1, -1)
        Y_fault_imag = Y_imag * mask.unsqueeze(0).expand(F, -1, -1)

    Y_fault = torch.complex(Y_fault_real, Y_fault_imag)
    return Y_fault


def get_fault_segment_and_local_position(fault_position, cable_lengths, device):
    """
    Given normalized fault position [0,1], determine which backbone segment
    and local position within that segment.

    Parameters:
        fault_position: tensor in [0, 1] of shape [] or [P, 1]
        cable_lengths: dict of cable length tensors (scalar or [P, 1])
        device: torch device

    Returns:
        segment_idx: which backbone segment (0-4) - discrete, non-differentiable
                     scalar int for scalar input, or tensor [P] for batched input
        local_position: position within that segment in meters (tensor, differentiable)
                        shape [] for scalar input, or [P, 1] for batched input
        segment_length: total length of that segment (tensor, differentiable)
                        shape [] for scalar input, or [P, 1] for batched input
    """
    is_batched = isinstance(fault_position, torch.Tensor) and \
                 fault_position.dim() >= 1 and fault_position.shape[0] > 1

    if not is_batched:
        L = get_total_backbone_length(cable_lengths, device)
        fault_position_abs = fault_position * L

        # First pass: determine segment index using detached values
        cumulative_detached = 0.0
        fault_pos_val = fault_position_abs.detach().item() if isinstance(fault_position_abs, torch.Tensor) else fault_position_abs
        segment_idx = len(BACKBONE_KEYS) - 1
        for idx, key in enumerate(BACKBONE_KEYS):
            seg_len_val = cable_lengths[key].detach().item() if isinstance(cable_lengths[key], torch.Tensor) else cable_lengths[key]
            if cumulative_detached + seg_len_val >= fault_pos_val or idx == len(BACKBONE_KEYS) - 1:
                segment_idx = idx
                break
            cumulative_detached += seg_len_val

        # Second pass: compute local_pos with gradient flow
        cumulative = torch.tensor(0.0, device=device)
        for idx, key in enumerate(BACKBONE_KEYS):
            if idx == segment_idx:
                local_pos = fault_position_abs - cumulative
                seg_len = cable_lengths[key]
                return segment_idx, local_pos, seg_len
            cumulative = cumulative + cable_lengths[key]

        last_len = cable_lengths[BACKBONE_KEYS[-1]]
        return len(BACKBONE_KEYS) - 1, fault_position_abs - (L - last_len), last_len

    else:
        # Batched path: fault_position has shape [P, 1]
        seg_lengths = torch.stack([cable_lengths[key] for key in BACKBONE_KEYS])
        num_segments = len(BACKBONE_KEYS)

        L = seg_lengths.sum()
        fault_position_abs = fault_position * L

        cumulative = torch.zeros(num_segments + 1, device=device)
        cumulative[1:] = torch.cumsum(seg_lengths, dim=0)

        fault_pos_flat = fault_position_abs.squeeze(-1).detach()
        segment_idx = torch.searchsorted(cumulative, fault_pos_flat, right=True) - 1
        segment_idx = segment_idx.clamp(0, num_segments - 1)

        cumulative_at_seg = cumulative[segment_idx]
        local_pos = fault_position_abs.squeeze(-1) - cumulative_at_seg
        local_pos = local_pos.unsqueeze(-1)

        seg_len = seg_lengths[segment_idx]
        seg_len = seg_len.unsqueeze(-1)

        return segment_idx, local_pos, seg_len


def carry_back_with_fault(Y_load, T, Tinv, ZC, YC, gamma, cable_length,
                          local_fault_pos, Z_fault_real, Z_fault_imag, device):
    """
    Carry back admittance through a cable that has a shunt fault.

    The cable is split into two segments:
      [Y_load]---(len_1)---[FAULT node]---(len_2)---[output Y_carried]

    At the fault node, Y_fault is added in parallel (shunt to ground).

    Parameters:
        Y_load: (P, F, n, n) admittance at the load end (Rx side)
        T, Tinv, ZC, YC, gamma: MTL parameters
        cable_length: total length of this segment
        local_fault_pos: distance from load end to fault (meters) [P, 1]
        Z_fault_real: real part of fault impedance (Ohms) [P, 1]
        Z_fault_imag: imaginary part of fault impedance (Ohms) [P, 1]
        device: torch device

    Returns:
        Y_carried: (P, F, n, n) admittance seen from source end (Tx side)
        h_total: (P, F, n, n) transfer function h_B through the faulted cable
    """
    n = Y_load.shape[-1]
    F = Y_load.shape[-3]

    len_1 = local_fault_pos
    len_2 = cable_length - local_fault_pos

    Y_fault = compute_fault_admittance_matrix(Z_fault_real, Z_fault_imag, F, n, device)
    rho_1 = reflection_coefficient(Y_load, T, Tinv, ZC, YC)
    h_1 = h_B(rho_1, ZC, T, Tinv, gamma, len_1)
    Y_at_fault = carry_back_load(rho_1, T, YC, gamma, len_1)

    Y_after_fault = Y_at_fault + Y_fault

    rho_2 = reflection_coefficient(Y_after_fault, T, Tinv, ZC, YC)
    h_2 = h_B(rho_2, ZC, T, Tinv, gamma, len_2)
    Y_carried = carry_back_load(rho_2, T, YC, gamma, len_2)

    h_total = h_1 @ h_2

    return Y_carried, h_total
