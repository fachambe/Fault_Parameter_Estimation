"""
Plotting functions for both stages. 

Usage:
    python plotting/plot_bothstages.py both_stages_results/bothstages_results_149khz-10mhz_M50_S1snr40.npz 
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forward_mtl import MTLForwardModel


def plot_param_convergence(param_history, losses, sorted_keys, output_dir = None):
    # ELBO Plot
    filename_svi = f"stage1_ELBO_history.pdf"
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title("SVI ELBO Loss", fontsize=14)
    plt.xlabel("SVI step", fontsize=13)
    plt.ylabel("ELBO loss", fontsize=13)
    plt.yscale("symlog")
    plt.tick_params(axis='both', labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    filename_svi = os.path.join(output_dir, filename_svi)
    plt.savefig(filename_svi, dpi=300, bbox_inches='tight')
    plt.close()

    # Parameter Plots of top 20 most sensitive
    top_20_keys = sorted_keys[:20]
    # Convert top_20_keys to param_history format: 'load_1.C_m_leak' -> 'load_1_C_m_leak'
    top_20_base_keys = [k.replace(".", "_") for k in top_20_keys]

    _, axes = plt.subplots(1, 2, figsize=(14, 5))
    panel_labels = ["(a)", "(b)"]

    # (1) loc trajectories (normalized) - apply sigmoid since param_history stores raw loc
    ax = axes[0]
    for base_key in top_20_base_keys:
        loc_key = f"{base_key}_loc"
        if loc_key in param_history:
            vals = np.array(param_history[loc_key])
            vals_sigmoid = 1 / (1 + np.exp(-vals))  # sigmoid in numpy
            ax.plot(vals_sigmoid, alpha=0.7, label=base_key)

    ax.text(0.5, 1.05, panel_labels[0], transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("SVI step", fontsize=13)
    ax.set_ylabel("Variational mean (sigmoid(loc))", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # (2) scale trajectories
    ax = axes[1]
    for base_key in top_20_base_keys:
        scale_key = f"{base_key}_scale"
        if scale_key in param_history:
            ax.plot(param_history[scale_key], alpha=0.7, label=base_key)

    ax.text(0.5, 1.05, panel_labels[1], transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    ax.set_xlabel("SVI step", fontsize=13)
    ax.set_ylabel("Variational scale (std dev)", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    plt.tight_layout()
    filename_conv = f"svi_param_convergence.pdf"
    filename_conv = os.path.join(output_dir, filename_conv)
    plt.savefig(filename_conv, dpi=300, bbox_inches='tight')
    plt.close()

def compute_tf_posterior_ci_stage1(data, num_samples=500, device='cpu'):
    """
    Compute transfer function posterior confidence interval from Stage 1 network params.

    Stage 1 infers network parameters (cables/loads) at 40dB SNR with NO fault.

    Args:
        data: Dict from np.load with allow_pickle=True containing param_history_s1
        num_samples: Number of posterior samples
        device: Torch device

    Returns:
        tf_mean: Mean TF in dB
        tf_lower: Lower 2.5% percentile in dB
        tf_upper: Upper 97.5% percentile in dB
        H_clean_db: True TF in dB (no fault)
        freq_range_mhz: Frequencies in MHz
    """
    # Load network_params and frequencies
    network_params = data['network_params'].item()
    frequencies = torch.tensor(data['frequencies'], device=device)
    selected_keys_s1 = list(data['selected_keys_s1'])
    param_history_s1 = data['param_history_s1'].item()

    # Recreate forward model
    forward_model = MTLForwardModel(frequencies, network_params, device=device)

    # Build list of posterior samples by sampling from LogitNormal
    tf_samples = []

    for _ in range(num_samples):
        # Sample network params from posterior (final variational params)
        sampled_network_params = {}
        for key in selected_keys_s1:
            safe_key = key.replace(".", "_")
            loc_key = f"{safe_key}_loc"
            scale_key = f"{safe_key}_scale"

            if loc_key in param_history_s1 and scale_key in param_history_s1:
                # Get final values from param_history
                loc = torch.tensor(param_history_s1[loc_key][-1], device=device)
                scale = torch.tensor(param_history_s1[scale_key][-1], device=device)

                # Sample from LogitNormal: θ = sigmoid(Normal(loc, scale))
                z = torch.randn(1, device=device) * scale + loc
                theta_sample = torch.sigmoid(z).item()
                sampled_network_params[key] = theta_sample

        # Update network_params with sampled values
        for key, val in sampled_network_params.items():
            if "." in key:
                load_name, param_name = key.split(".")
                network_params["loads"][load_name][param_name]["value"] = val
            else:
                network_params["cable_lengths"][key]["value"] = val

        # Build cable_lengths dict
        cable_lengths = {}
        for name, info in network_params["cable_lengths"].items():
            cable_lengths[name] = torch.tensor(info["value"], dtype=torch.float32, device=device)

        # Build load_params dict
        load_params = {}
        for load_name, params in network_params["loads"].items():
            load_params[load_name] = {}
            for param_name, param_info in params.items():
                if isinstance(param_info, dict) and "value" in param_info:
                    load_params[load_name][param_name] = torch.tensor(
                        param_info["value"], dtype=torch.float32, device=device
                    )

        # Compute TF without fault
        H = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
        H_db = 20 * torch.log10(torch.abs(H) + 1e-12)
        tf_samples.append(H_db.detach().cpu().numpy())

    # Stack and compute statistics
    tf_samples = np.stack(tf_samples, axis=0)  # [num_samples, F]
    tf_mean = np.mean(tf_samples, axis=0)
    tf_lower = np.percentile(tf_samples, 2.5, axis=0)
    tf_upper = np.percentile(tf_samples, 97.5, axis=0)

    # Compute true TF (no fault) using true network params
    true_network_values = data['true_network_values'].item()
    for key, val in true_network_values.items():
        if "." in key:
            load_name, param_name = key.split(".")
            network_params["loads"][load_name][param_name]["value"] = val
        else:
            network_params["cable_lengths"][key]["value"] = val

    cable_lengths_true = {}
    for name, info in network_params["cable_lengths"].items():
        cable_lengths_true[name] = torch.tensor(info["value"], dtype=torch.float32, device=device)

    load_params_true = {}
    for load_name, params in network_params["loads"].items():
        load_params_true[load_name] = {}
        for param_name, param_info in params.items():
            if isinstance(param_info, dict) and "value" in param_info:
                load_params_true[load_name][param_name] = torch.tensor(
                    param_info["value"], dtype=torch.float32, device=device
                )

    H_true = forward_model.calculate_Hnw_nofault(cable_lengths_true, load_params_true)
    H_clean_db = 20 * torch.log10(torch.abs(H_true) + 1e-12)
    H_clean_db = H_clean_db.detach().cpu().numpy()

    freq_range_mhz = frequencies.cpu().numpy() / 1e6

    return tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz


def plot_stage1_reconstruction_ci(data, freq_range_str, output_dir=None, num_samples=200):
    """
    Plot Stage 1 network reconstruction with confidence interval (single plot at 40dB SNR).

    Args:
        data: Dict from np.load with allow_pickle=True
        freq_range_str: Frequency range string for filename
        output_dir: Output directory
        num_samples: Number of posterior samples for CI computation
    """
    tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz = compute_tf_posterior_ci_stage1(
        data, num_samples=num_samples
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(freq_range_mhz, tf_mean, 'k-', linewidth=1.5, label='Posterior Mean')
    ax.plot(freq_range_mhz, H_clean_db, 'r--', linewidth=1.5, label='Truth')
    ax.fill_between(freq_range_mhz, tf_lower, tf_upper,
                   alpha=0.3, color='steelblue', label='95% CI')

    ax.set_xscale('log')
    ax.set_xlabel('Frequency (MHz)', fontsize=13)
    ax.set_ylabel(r'$H_{1,1}$ (dB)', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()

    filename = f"stage1_network_reconstruction_{freq_range_str}.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


CI_SAMPLES = 200  # Number of posterior samples for CI plots


def plot_fault_params_rmse_vs_crlb(snr_dbs, results2, results, output_dir=None):

    panel_labels = ["(a)", "(b)", "(c)"]

    crlb_label = r"$\sqrt{\mathrm{BCRLB}}$"

    selected_keys = results['selected_keys']
    rmse_results_stage2only = results2['rmse_results']
    rmse_results_bothstages = results['rmse_results']
    crlb_results = results2['crlb_results']


    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Scale factors: normalized [0,1] -> physical units
    # fault_position: 5 backbone cables × 6.25m each = 31.25m
    # Z_fault_real: range [0, 1000] Ω
    # Z_fault_imag: range [-100, 100] Ω = 200 Ω width
    scale = {'fault_position': 31.25, 'Z_fault_real': 1000.0, 'Z_fault_imag': 200.0}
    units = {'fault_position': r'$L_F$ Error (m)', 'Z_fault_real': r'$Re[Z_F]$ Error ($\Omega$)', 'Z_fault_imag': r'$Im[Z_F]$ Error ($\Omega$)'}

    for idx, key in enumerate(selected_keys):
        ax = axes[idx]

        s = scale.get(key, 1.0)
        rmse_vals_stage2only = rmse_results_stage2only[key] * s
        rmse_vals_bothstages = rmse_results_bothstages[key] * s
        crlb_vals = crlb_results[key] * s

        # Panel label
        ax.text(
            0.5, 1.05,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

        ax.plot(snr_dbs, rmse_vals_stage2only, 'bo-', label=r'BRMSE (true $\theta_n$)', markersize=6)
        ax.plot(snr_dbs, rmse_vals_bothstages, 'bx-', label=r'BRMSE (est. $\theta_n$)', markersize=6)
        ax.plot(snr_dbs, crlb_vals, 'r--', label=crlb_label, linewidth=2)

        ax.set_xlabel('SNR (dB)', fontsize=13)
        ax.set_ylabel(units.get(key, 'Error'), fontsize=13)
        ax.set_yscale('log')
        # Add more y-axis tick labels on log scale
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5), numticks=10))
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.minor.formatter.set_scientific(False)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='y', which='minor', labelsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=10)

    plt.tight_layout()

    filename = f"bothstages_final.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def load_bothstages_results(npz_path):
    """
    Load two-stage results from .npz file (only has RMSE, no CRLB).

    Args:
        npz_path: Path to .npz file saved by run_bothstages.py

    Returns:
        data: Dict with all saved arrays
        results: Dict with rmse_results for each fault param
    """
    data = np.load(npz_path, allow_pickle=True)

    selected_keys = list(data['selected_keys'])

    rmse_results = {}
    for key in selected_keys:
        safe_key = key.replace(".", "_")
        rmse_results[key] = data[f"{safe_key}_rmse"]

    results = {
        'selected_keys': selected_keys,
        'rmse_results': rmse_results,
    }

    return dict(data), results


def load_stage2_results(npz_path):
    """
    Load Stage 2 only results from .npz file (has both RMSE and CRLB).

    Args:
        npz_path: Path to .npz file saved by run_stage2_mtl.py

    Returns:
        data: Dict with all saved arrays
        results: Dict with rmse_results and crlb_results for each fault param
    """
    data = np.load(npz_path, allow_pickle=True)

    selected_keys = list(data['selected_keys'])

    rmse_results = {}
    crlb_results = {}
    for key in selected_keys:
        safe_key = key.replace(".", "_")
        rmse_results[key] = data[f"{safe_key}_rmse"]
        crlb_results[key] = data[f"{safe_key}_crlb"]

    results = {
        'selected_keys': selected_keys,
        'rmse_results': rmse_results,
        'crlb_results': crlb_results,
    }

    return dict(data), results

def main():
    parser = argparse.ArgumentParser(description='Plot both stage inference results')
    parser.add_argument('npz_file', help='Path to .npz results file')
    args = parser.parse_args()

    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        sys.exit(1)

    # Load two-stage results (BRMSE only)
    print(f"Loading two-stage results from: {args.npz_file}")
    data, results = load_bothstages_results(args.npz_file)

    # Load Stage 2 only results (BRMSE + BCRLB)
    stage2_path = "stage_2_results/stage2_results_149khz-10mhz_M100_alpha3.0_fp0.0-1.0_bayesian_025parallel.npz"
    print(f"Loading Stage 2 only results from: {stage2_path}")
    _, results2 = load_stage2_results(stage2_path)


    snr_dbs = data['snr_dbs']
    M = int(data['M'])
    freq_range_str = str(data['freq_range_str'])
    selected_keys = results['selected_keys']

    # Detect mode from npz 
    mode = str(data['mode']) if 'mode' in data else "frequentist"

    # Fixed output directory
    output_dir = os.path.join("figures", "BothStagesResults")
    print(f"\nData summary:")
    print(f"  SNR values: {snr_dbs}")
    print(f"  M (Monte Carlo): {M}")
    print(f"  Frequency range: {freq_range_str}")
    print(f"  Fault parameters: {selected_keys}")
    print(f"  Mode: {mode}")
    twostage_rmse = results['rmse_results']       # Two-stage BRMSE
    stage2_rmse = results2['rmse_results']         # Stage 2 only BRMSE
    bcrlb = results2['crlb_results']               # BCRLB (from Stage 2 only)

    print("\n--- Two-stage BRMSE ---")
    for key in selected_keys:
        print(f"  {key}: {twostage_rmse[key]}")

    print("\n--- Stage 2 only BRMSE (perfect network) ---")
    for key in selected_keys:
        print(f"  {key}: {stage2_rmse[key]}")

    print("\n--- sqrt(BCRLB) ---")
    for key in selected_keys:
        print(f"  {key}: {bcrlb[key]}")
    
    # Generate plots
    print("\nGenerating plots...")

    # 1. RMSE vs CRLB for fault parameters
    plot_fault_params_rmse_vs_crlb(snr_dbs, results2, results, output_dir)


    param_history_s1 = data['param_history_s1'].item()  # Extract dict from 0-d array
    losses_s1 = data['losses_s1']
    selected_keys_s1 = list(data['selected_keys_s1'])
    plot_param_convergence(param_history_s1, losses_s1, selected_keys_s1, output_dir)

    # Plot Stage 1 network reconstruction with CI (single plot at 40dB)
    plot_stage1_reconstruction_ci(data, freq_range_str, output_dir, num_samples=CI_SAMPLES)

    print("\nDone!")

if __name__ == "__main__":
    main()
