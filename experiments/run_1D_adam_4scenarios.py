# experiments/run_1D_adam_4scenarios.py
"""
Compare 1D gradient MLE for L1 across 4 scenarios:
- High freq (2-10 MHz) + Long cable (L=1000m)
- Low freq (150-500 kHz) + Long cable (L=1000m)
- High freq (2-10 MHz) + Short cable (L=100m)
- Low freq (150-500 kHz) + Short cable (L=100m)

Generates a 2x2 figure showing RMSE vs sqrt(CRLB) for L1 with plotting/plot_1D_adam_4scenarios.py

This file is self-contained and does not depend on benchmark.yaml.
"""
import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import scipy.io as sio

from core.forward import ForwardModel
from core.likelihoods import ComplexGaussianLik
from core.crlb import crlb_for_1_real_param
from estimators.mle_gradient import GradientMLE
from data.manager import DatasetManager
from utils.experiment import fmt_freq

# =============================================================================
# EXPERIMENT CONFIGURATION (self-contained, no yaml dependency)
# =============================================================================
DEVICE = "cuda"
Zs = 50.0  # Source impedance (ohms)
N = 100    # Number of observations
SEED = 5692
F = 200    # Number of frequency points per scenario
#SNR_DBS = [40]
SNR_DBS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

# Parameter ranges for ZF and ZL (used in true_range)
ZF_RANGE = {"re": {"min": 1.0, "max": 4000.0}, "im": {"min": -100.0, "max": 100.0}}
ZL_RANGE = {"re": {"min": 1.0, "max": 400.0}, "im": {"min": -100.0, "max": 100.0}}

# 4 scenarios: 2 freq ranges x 2 cable lengths
SCENARIOS = [
    {"name": "High_Freq_Long_Cable",  "freq_start": 2e6,   "freq_stop": 10e6,  "L": 1000, "L1": 250.0},
    {"name": "Low_Freq_Long_Cable",   "freq_start": 150e3, "freq_stop": 500e3, "L": 1000, "L1": 250.0},
    {"name": "High_Freq_Short_Cable", "freq_start": 2e6,   "freq_stop": 10e6,  "L": 100, "L1": 25.0},
    {"name": "Low_Freq_Short_Cable",  "freq_start": 150e3, "freq_stop": 500e3, "L": 100, "L1": 25.0},
]


def main():
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)
    device = torch.device(DEVICE)
    print(f"Current device: {device}")

    # Load full cable parameter arrays from .mat file
    mat = sio.loadmat("experiments/cable_parameters_extended.mat")
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full = torch.tensor(mat["Z_C"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device)
    # pul_freq covers 45kHz - 30MHz, 1305 freq points

    dm = DatasetManager(device=device)

    # Store results for each scenario
    all_results = {}

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        N_obs = N  
        if scenario_name == "Low_Freq_Short_Cable":
            N_obs = 5000
        fstart = scenario["freq_start"]
        fend = scenario["freq_stop"]
        L = scenario["L"]
        L1 = scenario["L1"]
        freq_tag = f"{fmt_freq(fstart)}-{fmt_freq(fend)}"
        L_tag = f"L{int(L)}"
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} ({freq_tag}, {L_tag})")
        print(f"{'='*60}")

        # Build scenario-specific fixed and range
        fixed = {
            "L1": L1,
            "ZF": {"re": 100.0, "im": -50.0},
            "ZL": {"re": 100.0, "im": -5.0}
        }
        true_range = {
            "L1": {"min": 1.0, "max": L - 1.0},
            "ZF": ZF_RANGE,
            "ZL": ZL_RANGE,
        }

        # Get gamma/Zc at desired frequencies
        desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)
        idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
        gamma_list = gamma_full[idx]
        Zc_list = Zc_full[idx]

        # Create forward model for this scenario
        fm = ForwardModel(gamma_list, Zc_list, L=L, Zs=Zs, device=device)

        # Only estimate L1 in this comparison
        target = "L1"

        rmse_grad = []
        crlb_sqrt = []

        for snr_db in SNR_DBS:
            print(f"  SNR = {snr_db} dB")

            # Generate observations for this scenario
            data = dm.generate_observations(
                snr_db=snr_db,
                N=N_obs,
                fm=fm,
                seed=SEED,
                target="ALL3SP",  # frequentist: same params for all N
                fixed=fixed,
                gen_cfg=true_range,
            )

            h_obs = torch.tensor(data["h_obs_real"], device=device) + 1j * torch.tensor(data["h_obs_imag"], device=device)
            var = torch.tensor(data["noise_var"], device=device)
            # Run gradient MLE for L1
            est = GradientMLE(
                fm=fm,
                likelihood=ComplexGaussianLik(),
                target=target,
                fixed=fixed,
                true_range=true_range,
                mode="1d",
                device=device,
                adam_steps=2000,
            )
            preds = est.predict(h_obs, var)  # [N]

            # Compute RMSE
            rmse = float(np.sqrt(np.mean((preds[target] - data["L1_true"]) ** 2)))
            rmse_grad.append(rmse)

            # Compute sqrt(CRLB)
            _, crlb = crlb_for_1_real_param(fm, target, fixed, var[0].squeeze(), device)
            sqrt_crlb = float(torch.sqrt(crlb).cpu())
            crlb_sqrt.append(sqrt_crlb)

            print(f"RMSE = {rmse:.4f}, sqrt(CRLB) = {sqrt_crlb:.4f}")
            
        # Store results
        all_results[scenario_name] = {
            "snrs": np.array(SNR_DBS),
            "rmse": np.array(rmse_grad),
            "crlb": np.array(crlb_sqrt),
            "freq_tag": freq_tag,
            "L_tag": L_tag,
        }

    # Save results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"1D_adam_4scenarios_seed{SEED}.npz"

    # Save with scenario metadata
    save_dict = {
        "snrs": np.array(SNR_DBS),
        "seed": SEED,
        "scenarios": [s["name"] for s in SCENARIOS],
    }
    for name, results in all_results.items():
        save_dict[f"{name}_rmse"] = results["rmse"]
        save_dict[f"{name}_crlb"] = results["crlb"]
        save_dict[f"{name}_freq_tag"] = results["freq_tag"]
        save_dict[f"{name}_L_tag"] = results["L_tag"]

    np.savez(results_file, **save_dict)
    print(f"\nSaved results to {results_file}")

    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
