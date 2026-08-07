"""
Configuration loader for simple network model (3-parameter: L1, ZF, ZL).
"""

import hashlib
import json
import yaml
import torch
import scipy.io as sio
from pathlib import Path


def config_hash(cfg_dict, length=8):
    """Create short hash of config dict for unique filenames."""
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:length]


def fmt_freq(hz):
    """Format frequency for display/filenames."""
    if hz >= 1e6:
        return f"{hz/1e6:.1f}MHz".replace(".0MHz", "MHz")
    return f"{hz/1e3:.0f}kHz"


def load_config(config_path="config/simple_network_config.yaml", mat_path=None):
    """
    Load simple model configuration and create forward model.

    Args:
        config_path: Path to YAML config file.
        mat_path: Path to cable parameters .mat file. If None, uses path from config.

    Returns:
        dict with keys:
            - device: torch device
            - fm: ForwardModel instance
            - dm: DatasetManager instance
            - gamma_full: Full propagation constant array [F_full]
            - Zc_full: Full characteristic impedance array [F_full]
            - pul_freq: Full frequency array [F_full]
            - freq_tag: Human-readable frequency tag (e.g., "2MHz-10MHz")
            - L_tag: Human-readable length tag (e.g., "L1000")
            - fstart: Start frequency in Hz
            - fend: End frequency in Hz
            - F: Number of frequency points
            - L: Cable length in meters
            - Zs: Source impedance
            - fixed: Dict with fixed parameter values
            - true_range: Dict with parameter ranges
            - snrs: List of SNR values in dB
            - N: Number of observations
            - seed: Random seed
            - target: Target parameter type (uppercase)
    """
    # Import here to avoid circular imports
    from core.forward import ForwardModel
    from data.manager import DatasetManager

    # Load YAML config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['general'].get('device', 'cpu'))

    # Resolve mat file path
    if mat_path is None:
        mat_path = cfg['cable']['mat_file']

    # Load cable parameters from .mat file
    mat = sio.loadmat(mat_path)
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full = torch.tensor(mat["Z_C"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device)

    # Extract frequency config
    fstart = float(cfg['frequencies']['start_hz'])
    fend = float(cfg['frequencies']['stop_hz'])
    F = int(cfg['frequencies']['num_points'])

    # Extract cable config
    L = float(cfg['cable']['L'])
    Zs = float(cfg['cable']['Zs'])

    # Extract experiment config
    N = int(cfg['experiment']['N'])
    snrs = cfg['experiment']['snr_dbs']
    target = cfg['experiment']['target'].upper()
    seed = cfg['general']['seed']

    # Fixed values and ranges
    fixed = cfg['fixed']
    true_range = cfg['parameter_ranges']

    # Human-readable tags
    freq_tag = f"{fmt_freq(fstart)}-{fmt_freq(fend)}"
    L_tag = f"L{int(L)}"

    # Select frequencies by finding nearest neighbors in full array
    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
    gamma_list = gamma_full[idx]
    Zc_list = Zc_full[idx]

    # Create forward model and data manager
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=L, Zs=Zs, device=device)

    return {
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
