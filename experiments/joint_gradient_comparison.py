# experiments/joint_gradient_comparison.py
"""
Compare joint gradient MLE for L1, ZF, ZL across 4 scenarios:
- High freq (2-10 MHz) + Long cable (L=1000m)
- Low freq (150-500 kHz) + Long cable (L=1000m)
- High freq (2-10 MHz) + Short cable (L=100m)
- Low freq (150-500 kHz) + Short cable (L=100m)

Generates results for RMSE vs sqrt(CRLB) for all 3 parameters.

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
from core.crlb import complex_partials_fullbatch, fim_from_complex_jac, get_CRLB
from estimators.mle_gradient import GradientMLE
from estimators.bfgs import BFGSEstimator
from estimators.LM import LMEstimator
from estimators.L1_profile import L1ProfileMLE
from data.manager import DatasetManager

# =============================================================================
# EXPERIMENT CONFIGURATION (self-contained, no yaml dependency)
# =============================================================================
DEVICE = "cuda"
Zs = 50.0  # Source impedance (ohms)
N = 100    # Number of observations
SEED = 5692
F = 200    # Number of frequency points per scenario
SNR_DBS = [80]

# Parameter ranges for ZF and ZL (used in true_range)
ZF_RANGE = {"re": {"min": 1.0, "max": 4000.0}, "im": {"min": -100.0, "max": 100.0}}
ZL_RANGE = {"re": {"min": 1.0, "max": 400.0}, "im": {"min": -100.0, "max": 100.0}}

# 4 scenarios: 2 freq ranges x 2 cable lengths
SCENARIOS = [
    #{"name": "High_Freq_Long_Cable",  "freq_start": 2e6,   "freq_stop": 10e6,  "L": 1000, "L1": 250.0},
    #{"name": "Low_Freq_Long_Cable",   "freq_start": 150e3, "freq_stop": 500e3, "L": 1000, "L1": 250.0},
    #{"name": "High_Freq_Short_Cable", "freq_start": 2e6,   "freq_stop": 10e6,  "L": 100, "L1": 25.0},
    {"name": "Low_Freq_Short_Cable",  "freq_start": 150e3, "freq_stop": 500e3, "L": 100, "L1": 25.0},
]

def fmt_freq(hz):
    """Format frequency for display/filenames."""
    if hz >= 1e6:
        return f"{hz/1e6:.1f}MHz".replace(".0MHz", "MHz")
    else:
        return f"{hz/1e3:.0f}kHz"


def rmse_complex(pred, true):
    """RMSE for complex arrays: sqrt(mean(|pred - true|^2))"""
    mse = np.mean(np.abs(pred - true)**2)
    return float(np.sqrt(mse))


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

    dm = DatasetManager(device=device)

    # Store results for each scenario
    all_results = {}

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
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

        # Lists to store results per SNR for each estimator
        # Est1: Adam only, Est2: Adam + Newton, Est3: Adam + LM
        rmse_L1_adam_list, rmse_ZF_adam_list, rmse_ZL_adam_list = [], [], []
        rmse_L1_newton_list, rmse_ZF_newton_list, rmse_ZL_newton_list = [], [], []
        rmse_L1_lm_list, rmse_ZF_lm_list, rmse_ZL_lm_list = [], [], []
        crlb_L1_list = []
        crlb_ZF_list = []
        crlb_ZL_list = []

        for snr_db in SNR_DBS:
            print(f"  SNR = {snr_db} dB")

            # Generate observations for this scenario
            data = dm.generate_observations(
                snr_db=snr_db,
                N=N,
                fm=fm,
                seed=SEED,
                target="ALL3SP",  # frequentist: same params for all N
                fixed=fixed,
                gen_cfg=true_range,
            )
    
            h_obs = torch.tensor(data["h_obs_real"], device=device) + 1j * torch.tensor(data["h_obs_imag"], device=device)
            var = torch.tensor(data["noise_var"], device=device)

            # Est1: Adam only, Est2: Adam + Newton, Est3: Adam + LM
            est1 = GradientMLE(
                fm = fm,
                likelihood=ComplexGaussianLik(),
                target = "L1",
                fixed = fixed,
                true_range = true_range,
                mode = "joint",
                device = device,
                adam_steps = 20000,
                # No polishing - Adam only
            )
            est2 = GradientMLE(
                fm = fm,
                likelihood=ComplexGaussianLik(),
                target = "L1",
                fixed = fixed,
                true_range = true_range,
                mode = "joint",
                device = device,
                adam_steps = 20000,
                use_Newton=True,
            )
            est3 = GradientMLE(
                fm = fm,
                likelihood=ComplexGaussianLik(),
                target = "L1",
                fixed = fixed,
                true_range = true_range,
                mode = "joint",
                device = device,
                adam_steps = 20000,
                use_LM = True,
            )
            est4 = GradientMLE(
                fm = fm,
                likelihood=ComplexGaussianLik(),
                target = "L1",
                fixed = fixed,
                true_range = true_range,
                mode = "joint",
                device = device,
                adam_steps = 20000,
                use_bfgs=True,
            )
            preds = est4.predict(h_obs, var)
            preds = est1.predict(h_obs, var)
            preds2 = est2.predict(h_obs, var)
            preds3 = est3.predict(h_obs, var)

            # Compute RMSE for each parameter and each estimator
            ZF_true = data["ZF_true_re"] + 1j * data["ZF_true_im"]
            ZL_true = data["ZL_true_re"] + 1j * data["ZL_true_im"]

            # Est1: Adam only
            rmse_L1_adam = float(np.sqrt(np.mean((preds["L1"] - data["L1_true"]) ** 2)))
            rmse_ZF_adam = rmse_complex(preds["ZF"], ZF_true)
            rmse_ZL_adam = rmse_complex(preds["ZL"], ZL_true)
            rmse_L1_adam_list.append(rmse_L1_adam)
            rmse_ZF_adam_list.append(rmse_ZF_adam)
            rmse_ZL_adam_list.append(rmse_ZL_adam)

            # Est2: Adam + Newton
            rmse_L1_newton = float(np.sqrt(np.mean((preds2["L1"] - data["L1_true"]) ** 2)))
            rmse_ZF_newton = rmse_complex(preds2["ZF"], ZF_true)
            rmse_ZL_newton = rmse_complex(preds2["ZL"], ZL_true)
            rmse_L1_newton_list.append(rmse_L1_newton)
            rmse_ZF_newton_list.append(rmse_ZF_newton)
            rmse_ZL_newton_list.append(rmse_ZL_newton)

            # Est3: Adam + LM
            rmse_L1_lm = float(np.sqrt(np.mean((preds3["L1"] - data["L1_true"]) ** 2)))
            rmse_ZF_lm = rmse_complex(preds3["ZF"], ZF_true)
            rmse_ZL_lm = rmse_complex(preds3["ZL"], ZL_true)
            rmse_L1_lm_list.append(rmse_L1_lm)
            rmse_ZF_lm_list.append(rmse_ZF_lm)
            rmse_ZL_lm_list.append(rmse_ZL_lm)

            # Compute sqrt(CRLB) for each parameter
            du_aug = complex_partials_fullbatch(fm, data, device)
            FIM = fim_from_complex_jac(du_aug, var)
            CRLB_L1, CRLB_ZF, CRLB_ZL = get_CRLB(FIM)
            sqrt_crlb_L1 = float(torch.sqrt(CRLB_L1).cpu().numpy()[0])
            sqrt_crlb_ZF = float(torch.sqrt(CRLB_ZF).cpu().numpy()[0])
            sqrt_crlb_ZL = float(torch.sqrt(CRLB_ZL).cpu().numpy()[0])
            crlb_L1_list.append(sqrt_crlb_L1)
            crlb_ZF_list.append(sqrt_crlb_ZF)
            crlb_ZL_list.append(sqrt_crlb_ZL)

            print(f"    L1: Adam={rmse_L1_adam:.4f}, Newton={rmse_L1_newton:.4f}, LM={rmse_L1_lm:.4f}, sqrt(CRLB)={sqrt_crlb_L1:.4f}")
            print(f"    ZF: Adam={rmse_ZF_adam:.4f}, Newton={rmse_ZF_newton:.4f}, LM={rmse_ZF_lm:.4f}, sqrt(CRLB)={sqrt_crlb_ZF:.4f}")
            print(f"    ZL: Adam={rmse_ZL_adam:.4f}, Newton={rmse_ZL_newton:.4f}, LM={rmse_ZL_lm:.4f}, sqrt(CRLB)={sqrt_crlb_ZL:.4f}")
            
        # Store results for this scenario
        all_results[scenario_name] = {
            "snrs": np.array(SNR_DBS),
            # Adam only
            "rmse_L1_adam": np.array(rmse_L1_adam_list),
            "rmse_ZF_adam": np.array(rmse_ZF_adam_list),
            "rmse_ZL_adam": np.array(rmse_ZL_adam_list),
            # Adam + Newton
            "rmse_L1_newton": np.array(rmse_L1_newton_list),
            "rmse_ZF_newton": np.array(rmse_ZF_newton_list),
            "rmse_ZL_newton": np.array(rmse_ZL_newton_list),
            # Adam + LM
            "rmse_L1_lm": np.array(rmse_L1_lm_list),
            "rmse_ZF_lm": np.array(rmse_ZF_lm_list),
            "rmse_ZL_lm": np.array(rmse_ZL_lm_list),
            # CRLB
            "crlb_L1": np.array(crlb_L1_list),
            "crlb_ZF": np.array(crlb_ZF_list),
            "crlb_ZL": np.array(crlb_ZL_list),
            "freq_tag": freq_tag,
            "L_tag": L_tag,
        }
    
    # Save results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"joint_gradient_comparison_seed{SEED}.npz"

    # Flatten for np.savez
    save_dict = {
        "snrs": np.array(SNR_DBS),
        "seed": SEED,
        "scenarios": [s["name"] for s in SCENARIOS],
    }
    for name, results in all_results.items():
        # Adam only
        save_dict[f"{name}_rmse_L1_adam"] = results["rmse_L1_adam"]
        save_dict[f"{name}_rmse_ZF_adam"] = results["rmse_ZF_adam"]
        save_dict[f"{name}_rmse_ZL_adam"] = results["rmse_ZL_adam"]
        # Adam + Newton
        save_dict[f"{name}_rmse_L1_newton"] = results["rmse_L1_newton"]
        save_dict[f"{name}_rmse_ZF_newton"] = results["rmse_ZF_newton"]
        save_dict[f"{name}_rmse_ZL_newton"] = results["rmse_ZL_newton"]
        # Adam + LM
        save_dict[f"{name}_rmse_L1_lm"] = results["rmse_L1_lm"]
        save_dict[f"{name}_rmse_ZF_lm"] = results["rmse_ZF_lm"]
        save_dict[f"{name}_rmse_ZL_lm"] = results["rmse_ZL_lm"]
        # CRLB
        save_dict[f"{name}_crlb_L1"] = results["crlb_L1"]
        save_dict[f"{name}_crlb_ZF"] = results["crlb_ZF"]
        save_dict[f"{name}_crlb_ZL"] = results["crlb_ZL"]
        save_dict[f"{name}_freq_tag"] = results["freq_tag"]
        save_dict[f"{name}_L_tag"] = results["L_tag"]

    np.savez(results_file, **save_dict)
    print(f"\nSaved results to {results_file}")

    end_time = time.perf_counter()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    print(f"Run plot_joint_gradient_comparison.py to generate figures.")


if __name__ == "__main__":
    main()
