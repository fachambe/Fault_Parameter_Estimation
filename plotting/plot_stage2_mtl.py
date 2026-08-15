"""
Plotting functions for Stage 2 MTL inference results (fault parameters only).

Usage:
    python plotting/plot_stage2_mtl.py stage_2_results/stage2_results_150khz-10mhz_M10_frequentist.npz
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


def load_stage2_results(npz_path):
    """
    Load Stage 2 results from .npz file.

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


def plot_fault_params_rmse_vs_crlb(snr_dbs, results, M, freq_range_str, alpha, fp_range, output_dir=None, mode="frequentist"):
    """
    Plot RMSE vs CRLB for the 3 fault parameters in a 1x3 layout.

    Args:
        snr_dbs: Array of SNR values in dB
        results: Dict with 'selected_keys', 'rmse_results', 'crlb_results'
        M: Number of Monte Carlo trials
        freq_range_str: Frequency range string for filename
        alpha: alpha for beta prior (bayeisan only)
        fp_range: Fault position param range
        output_dir: Output directory for saving plots
        mode: "frequentist" or "bayesian" - affects labels
    """
    panel_labels = ["(a)", "(b)", "(c)"]

    # Labels depend on mode
    if mode == "bayesian":
        rmse_label = "BRMSE"
        crlb_label = r"$\sqrt{\mathrm{BCRLB}}$"
    else:
        rmse_label = "RMSE"
        crlb_label = r"$\sqrt{\mathrm{CRLB}}$"

    selected_keys = results['selected_keys']
    rmse_results = results['rmse_results']
    crlb_results = results['crlb_results']


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
        rmse_vals = rmse_results[key] * s
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

        ax.plot(snr_dbs, rmse_vals, 'bo-', label=rmse_label, markersize=6)
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

    filename = f"stage2_fault_rmse_crlb_{freq_range_str}_{mode}_{alpha}_{fp_range}.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def compute_tf_posterior_ci_stage2(data, snr_db, num_samples=500, device='cpu'):
    """
    Compute transfer function posterior confidence interval from saved best_params (Stage 2).

    For Stage 2, we sample fault parameters from the posterior while keeping
    network params fixed.

    Args:
        data: Dict from np.load with allow_pickle=True
        snr_db: Which SNR to use
        num_samples: Number of posterior samples
        device: Torch device

    Returns:
        tf_mean: Mean TF in dB
        tf_lower: Lower 2.5% percentile in dB
        tf_upper: Upper 97.5% percentile in dB
        H_clean_db: True TF in dB
        freq_range_mhz: Frequencies in MHz
    """
    # Load network_params and frequencies
    network_params = data['network_params'].item()
    frequencies = torch.tensor(data['frequencies'], device=device)
    selected_keys = list(data['selected_keys'])

    # Recreate forward model
    forward_model = MTLForwardModel(frequencies, network_params, device=device)

    # Get saved variational parameters for this snr_db
    snr_prefix = f"snr{snr_db}"

    # Build list of posterior samples by sampling from LogitNormal
    tf_samples = []

    for _ in range(num_samples):
        # Sample from posterior for each fault parameter
        sampled_fault_params = {}
        for key in selected_keys:
            safe_key = key.replace(".", "_")
            loc_key = f"{snr_prefix}_{safe_key}_loc"
            scale_key = f"{snr_prefix}_{safe_key}_scale"

            if loc_key in data and scale_key in data:
                loc = torch.tensor(float(data[loc_key]), device=device)
                scale = torch.tensor(float(data[scale_key]), device=device)

                # Sample from LogitNormal: θ = sigmoid(Normal(loc, scale))
                z = torch.randn(1, device=device) * scale + loc
                theta_sample = torch.sigmoid(z).item()
                sampled_fault_params[key] = theta_sample

        # Update network_params with sampled fault values
        for key, val in sampled_fault_params.items():
            network_params["fault_parameters"][key]["value"] = val

        # Build cable_lengths dict (fixed from Stage 1)
        cable_lengths = {}
        for name, info in network_params["cable_lengths"].items():
            cable_lengths[name] = torch.tensor(info["value"], dtype=torch.float32, device=device)

        # Build load_params dict (fixed from Stage 1)
        load_params = {}
        for load_name, params in network_params["loads"].items():
            load_params[load_name] = {}
            for param_name, param_info in params.items():
                load_params[load_name][param_name] = torch.tensor(
                    param_info["value"], dtype=torch.float32, device=device
                )

        # Build fault_params dict
        fault_params = {}
        for fault_name, fault_info in network_params["fault_parameters"].items():
            fault_params[fault_name] = torch.tensor(
                fault_info["value"], dtype=torch.float32, device=device
            )

        # Compute TF with fault
        H = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
        H_db = 20 * torch.log10(torch.abs(H) + 1e-12)
        tf_samples.append(H_db.detach().cpu().numpy())

    # Stack and compute statistics
    tf_samples = np.stack(tf_samples, axis=0)  # [num_samples, F]
    tf_mean = np.mean(tf_samples, axis=0)
    tf_lower = np.percentile(tf_samples, 2.5, axis=0)
    tf_upper = np.percentile(tf_samples, 97.5, axis=0)

    # Compute true TF using network_params (fault values already set)
    cable_lengths_true = {}
    for name, info in network_params["cable_lengths"].items():
        cable_lengths_true[name] = torch.tensor(info["value"], dtype=torch.float32, device=device)

    load_params_true = {}
    for load_name, params in network_params["loads"].items():
        load_params_true[load_name] = {}
        for param_name, param_info in params.items():
            load_params_true[load_name][param_name] = torch.tensor(
                param_info["value"], dtype=torch.float32, device=device
            )

    fault_params_true = {}
    for fault_name, fault_info in network_params["fault_parameters"].items():
        fault_params_true[fault_name] = torch.tensor(
            fault_info["value"], dtype=torch.float32, device=device
        )

    H_true = forward_model.calculate_Hnw(cable_lengths_true, load_params_true, fault_params_true)
    H_clean_db = 20 * torch.log10(torch.abs(H_true) + 1e-12)
    H_clean_db = H_clean_db.detach().cpu().numpy()

    freq_range_mhz = frequencies.cpu().numpy() / 1e6

    return tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz


def plot_CI_grid_stage2(data, snr_dbs_to_plot, freq_range_str, output_dir=None, num_samples=200):
    """
    Plot transfer function CI for multiple SNR values in a grid layout (Stage 2).

    Args:
        data: Dict from np.load with allow_pickle=True
        snr_dbs_to_plot: List of SNR values to plot
        freq_range_str: Frequency range string for filename
        output_dir: Output directory
        num_samples: Number of posterior samples for CI computation
    """
    n_snrs = len(snr_dbs_to_plot)

    # Determine grid layout based on number of SNRs
    if n_snrs <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 4, 2

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.5 * nrows), sharex=True)
    axes = axes.flatten()

    # Compute CI data for each SNR
    for idx, snr_db in enumerate(snr_dbs_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz = compute_tf_posterior_ci_stage2(
            data, snr_db, num_samples=num_samples
        )

        # Panel label
        ax.text(
            0.5, 1.05,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )

        ax.plot(freq_range_mhz, tf_mean, 'k-', linewidth=1.0, label='Posterior Mean')
        ax.plot(freq_range_mhz, H_clean_db, 'r--', linewidth=1.0, label='Truth')
        ax.fill_between(freq_range_mhz, tf_lower, tf_upper,
                       alpha=0.3, color='steelblue', label='95% CI')

        ax.set_xscale('log')
        ax.set_xlabel('Frequency (MHz)', fontsize=10)
        ax.set_ylabel(r'$H_{1,1}$ (dB)', fontsize=10)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Hide unused subplots
    for idx in range(n_snrs, len(axes)):
        axes[idx].set_visible(False)

    # Single figure-level legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    filename = f"stage2_tf_ci_grid_{freq_range_str}.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


CI_SAMPLES = 200  # Number of posterior samples for CI plots


def main():
    parser = argparse.ArgumentParser(description='Plot Stage 2 MTL inference results')
    parser.add_argument('npz_file', help='Path to .npz results file')
    args = parser.parse_args()

    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        sys.exit(1)

    print(f"Loading results from: {args.npz_file}")
    data, results = load_stage2_results(args.npz_file)

    snr_dbs = data['snr_dbs']
    M = int(data['M'])
    freq_range_str = str(data['freq_range_str'])
    selected_keys = results['selected_keys']
    alpha = data['ALPHA']
    fp_range = data['fp_range']

    # Detect mode from npz 
    mode = str(data['mode']) if 'mode' in data else "frequentist"

    # Fixed output directory
    output_dir = os.path.join("figures", "Stage2Results")

    print(f"\nData summary:")
    print(f"  SNR values: {snr_dbs}")
    print(f"  M (Monte Carlo): {M}")
    print(f"  Frequency range: {freq_range_str}")
    print(f"  Fault parameters: {selected_keys}")
    print(f"  Mode: {mode}")
    rmse_results = results['rmse_results']
    crlb_results = results['crlb_results']
    print("RMSE results", rmse_results)
    print("CRLB results", crlb_results)

    # Generate plots
    print("\nGenerating plots...")

    # 1. RMSE vs CRLB for fault parameters
    plot_fault_params_rmse_vs_crlb(snr_dbs, results, M, freq_range_str, alpha, fp_range, output_dir, mode)

    # 2. TF Confidence Interval grid plots (frequentist mode only)
    if mode == "bayesian":
        print("\nSkipping CI plots (not available for bayesian mode)")
    else:
        print("\nGenerating TF confidence interval grid plots...")
        snr_dbs_for_ci = [snr for snr in snr_dbs if snr in snr_dbs]
        #plot_CI_grid_stage2(data, snr_dbs_for_ci, freq_range_str, output_dir, num_samples=CI_SAMPLES)

    print("\nDone!")


if __name__ == "__main__":
    main()
