# experiments/run_1D_loss_landscape.py
"""
Generate and save observations for 1D loss landscapes.
"""
import sys, pathlib
import time
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import hashlib
import json
from core.forward import ForwardModel
from data.manager import DatasetManager

# Network parameters (true values used to generate observations)
network_params = {
    "L1": {
        "value": 250.0,
        "inferred": True,
        "range": (1.0, 999.0)
    },
    "ZF_re": {
        "value": 100.0,
        "inferred": True,
        "range": (1.0, 1000.0)
    },
    "ZF_im": {
        "value": -50.0,
        "inferred": True,
        "range": (-100.0, 100.0)
    },
    "ZL_re": {
        "value": 100.0,
        "inferred": True,
        "range": (1.0, 400.0)
    },
    "ZL_im": {
        "value": -5.0,
        "inferred": True,
        "range": (-100.0, 100.0)
    }
}

DEVICE = torch.device("cuda")
SEED = 98
N_OBS = 2500  # Number of observations to generate
K = 200  # Number of frequency points
SNR_DBS = [0, 10, 20, 30, 40]
CABLE_LENGTH = 1000.0  # L = 1000m
ZS = 50.0


def format_freq(f_hz):
    """Format frequency as kHz or MHz string."""
    if f_hz >= 1e6:
        return f"{int(f_hz / 1e6)}mhz"
    else:
        return f"{int(f_hz / 1e3)}khz"


def config_hash(cfg_dict, length=8):
    """Create short hash of config dict for unique filenames."""
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:length]


def main():
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)

    # Setup frequencies
    frequencies = torch.logspace(
        torch.log10(torch.tensor(150e3, device=DEVICE)),
        torch.log10(torch.tensor(500e3, device=DEVICE)),
        K, device=DEVICE
    )  # NB-PLC 150-500 kHz

    # Create forward model and data manager
    fm = ForwardModel(frequencies, CABLE_LENGTH, ZS, device=DEVICE)
    dm = DatasetManager(device=DEVICE)

    # Create tags for filename
    freq_start = frequencies[0].item()
    freq_end = frequencies[-1].item()
    freq_tag = f"{format_freq(freq_start)}_to_{format_freq(freq_end)}"
    L_tag = f"L{int(CABLE_LENGTH)}m"

    # Config dict for hashing
    obs_config = {
        "freq_start": freq_start,
        "freq_stop": freq_end,
        "F": K,
        "L": CABLE_LENGTH,
        "L1": network_params["L1"]["value"],
        "ZF_re": network_params["ZF_re"]["value"],
        "ZF_im": network_params["ZF_im"]["value"],
        "ZL_re": network_params["ZL_re"]["value"],
        "ZL_im": network_params["ZL_im"]["value"],
        "N": N_OBS,
        "seed": SEED,
    }
    obs_hash = config_hash(obs_config)

    print(f"Device: {DEVICE}")
    print(f"Config: {freq_tag}, {L_tag}, N={N_OBS}, seed={SEED}")

    # Create results directory
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save forward model (needed for plotting and estimation)
    fm_file = results_dir / f"forward_model_{freq_tag}_{L_tag}_{obs_hash}.pt"
    torch.save(fm, fm_file)
    print(f"Saved forward model to {fm_file}")

    # Extract cable parameters - these are [F] arrays
    pul_freq = frequencies.cpu().numpy()
    gamma_full = fm.gamma.cpu().numpy()
    Zc_full = fm.Zc.cpu().numpy()

    # Generate and save observations for each SNR
    for snr_db in SNR_DBS:
        print(f"\nGenerating observations at SNR = {snr_db} dB...")

        # Frequentist: same true values for all N trials
        L1_true = np.full(N_OBS, network_params["L1"]["value"], dtype=np.float32)
        ZF_re_true = np.full(N_OBS, network_params["ZF_re"]["value"], dtype=np.float32)
        ZF_im_true = np.full(N_OBS, network_params["ZF_im"]["value"], dtype=np.float32)
        ZL_re_true = np.full(N_OBS, network_params["ZL_re"]["value"], dtype=np.float32)
        ZL_im_true = np.full(N_OBS, network_params["ZL_im"]["value"], dtype=np.float32)

        ZF_true = ZF_re_true + 1j * ZF_im_true
        ZL_true = ZL_re_true + 1j * ZL_im_true

        # Generate observations
        h_obs, var = dm.generate_observations(snr_db, N_OBS, fm, L1_true, ZF_true, ZL_true)

        # Convert to numpy for saving
        h_obs_np = h_obs.cpu().numpy()  # [N, F] complex
        var_np = var.cpu().numpy()  # [N, 1]

        # Save observation file
        obs_file = results_dir / f"observations_{freq_tag}_{L_tag}_{obs_hash}_snr{int(snr_db)}.npz"

        np.savez(
            obs_file,
            # Config for reproducibility
            freq_start=freq_start,
            freq_stop=freq_end,
            F=K,
            L=CABLE_LENGTH,
            N=N_OBS,
            seed=SEED,
            snr_db=snr_db,
            # Observations [N, F]
            h_obs_real=h_obs_np.real,
            h_obs_imag=h_obs_np.imag,
            noise_var=var_np,
            # True parameter values [N]
            L1_true=L1_true,
            ZF_true_re=ZF_re_true,
            ZF_true_im=ZF_im_true,
            ZL_true_re=ZL_re_true,
            ZL_true_im=ZL_im_true,
            # Cable parameters (for loss landscape plots) [F]
            pul_freq=pul_freq,
            gamma_real=gamma_full.real,
            gamma_imag=gamma_full.imag,
            Zc_real=Zc_full.real,
            Zc_imag=Zc_full.imag,
        )
        print(f"  Saved to {obs_file}")

    end_time = time.perf_counter()
    print(f"\nGenerated {N_OBS} CTF observations for {len(SNR_DBS)} SNR values in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
