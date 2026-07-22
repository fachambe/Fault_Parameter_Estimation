# utils/experiment.py
"""
Shared experiment setup utilities.
Centralizes config loading, cable parameter loading, and forward model creation.
"""
import hashlib
import json
import pathlib

import numpy as np
import torch
import yaml
import scipy.io as sio

from core.forward import ForwardModel
from data.manager import DatasetManager


def config_hash(cfg_dict, length=8):
    """Create short hash of config dict for unique filenames."""
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:length]


def fmt_freq(hz):
    """Format frequency for display/filenames."""
    if hz >= 1e6:
        return f"{hz/1e6:.1f}MHz".replace(".0MHz", "MHz")
    else:
        return f"{hz/1e3:.0f}kHz"


def load_cable_params(mat_path, device):
    """
    Load cable parameters from .mat file.

    Args:
        mat_path: Path to cable_parameters_extended.mat
        device: torch device

    Returns:
        gamma_full: [F_full] complex propagation constant
        Zc_full: [F_full] complex characteristic impedance
        pul_freq: [F_full] frequency points in Hz
    """
    mat = sio.loadmat(mat_path)
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full = torch.tensor(mat["Z_C"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device)
    return gamma_full, Zc_full, pul_freq


def select_frequencies(pul_freq, gamma_full, Zc_full, fstart, fend, num_points, device):
    """
    Select frequency points from full arrays by finding nearest neighbors.

    Args:
        pul_freq: [F_full] full frequency array
        gamma_full: [F_full] full propagation constant array
        Zc_full: [F_full] full characteristic impedance array
        fstart: Start frequency in Hz
        fend: End frequency in Hz
        num_points: Number of frequency points to select
        device: torch device

    Returns:
        gamma_list: [F] selected propagation constants
        Zc_list: [F] selected characteristic impedances
    """
    desired_freqs = torch.linspace(fstart, fend, num_points, device=device, dtype=pul_freq.dtype)
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
    return gamma_full[idx], Zc_full[idx]


def setup_experiment(cfg_path="configs/benchmark.yaml",
                     mat_path="experiments/cable_parameters_extended.mat"):
    """
    Load config and create forward model + data manager.

    Args:
        cfg_path: Path to benchmark config YAML
        mat_path: Path to cable parameters .mat file

    Returns:
        dict with keys:
            cfg: Full config dict
            device: torch device
            fm: ForwardModel instance
            dm: DatasetManager instance
            gamma_full: Full propagation constant array
            Zc_full: Full characteristic impedance array
            pul_freq: Full frequency array
            freq_tag: Human-readable frequency tag (e.g., "2MHz-10MHz")
            L_tag: Human-readable length tag (e.g., "L500")
            fstart: Start frequency in Hz
            fend: End frequency in Hz
            F: Number of frequency points
            L: Cable length in meters
            Zs: Source impedance
            fixed: Dict with fixed parameter values
            true_range: Dict with parameter ranges
            snrs: List of SNR values in dB
            N: Number of observations
            seed: Random seed
            target: Target parameter (uppercase)
    """
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))

    # Load cable parameters
    gamma_full, Zc_full, pul_freq = load_cable_params(mat_path, device)

    # Extract config values
    fstart = float(cfg["freq"]["start_hz"])
    fend = float(cfg["freq"]["stop_hz"])
    F = int(cfg["freq"]["num_points"])
    L = float(cfg["L"])
    Zs = float(cfg["Zs"])
    fixed = cfg["fixed"]
    true_range = cfg["true_range"]
    snrs = cfg["snr_dbs"]
    N = int(cfg["N"])
    seed = cfg["seed"]
    target = cfg["target"].upper()

    # Human-readable tags
    freq_tag = f"{fmt_freq(fstart)}-{fmt_freq(fend)}"
    L_tag = f"L{int(L)}"

    # Select frequencies
    gamma_list, Zc_list = select_frequencies(
        pul_freq, gamma_full, Zc_full, fstart, fend, F, device
    )

    # Create forward model and data manager
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=L, Zs=Zs, device=device)

    return {
        "cfg": cfg,
        "device": device,
        "fm": fm,
        "dm": dm,
        "gamma_full": gamma_full,
        "Zc_full": Zc_full,
        "pul_freq": pul_freq,
        "freq_tag": freq_tag,
        "L_tag": L_tag,
        "fstart": fstart,
        "fend": fend,
        "F": F,
        "L": L,
        "Zs": Zs,
        "fixed": fixed,
        "true_range": true_range,
        "snrs": snrs,
        "N": N,
        "seed": seed,
        "target": target,
    }
