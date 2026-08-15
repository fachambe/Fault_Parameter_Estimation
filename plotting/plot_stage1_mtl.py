"""
Plotting functions for Stage 1 MTL inference results.

Usage:
    python plotting/plot_stage1_mtl.py results/stage1_results_150khz-10mhz_M10.npz
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.forward_mtl import MTLForwardModel


def load_stage1_results(npz_path):
    """
    Load Stage 1 results from .npz file and reconstruct all_results dict.

    Args:
        npz_path: Path to .npz file saved by run_stage1_mtl.py

    Returns:
        data: Dict with all saved arrays
        all_results: Reconstructed dict for plotting functions
    """
    data = np.load(npz_path, allow_pickle=True)

    snr_dbs = data['snr_dbs']
    p_values = data['p_values']

    # Reconstruct all_results dict
    all_results = {}
    for p_val in p_values:
        prefix = f"p{p_val}"
        selected_keys = list(data[f"{prefix}_selected_keys"])

        rmse_results = {}
        crlb_results = {}
        for key in selected_keys:
            safe_key = key.replace(".", "_")
            rmse_results[key] = data[f"{prefix}_{safe_key}_rmse"]
            crlb_results[key] = data[f"{prefix}_{safe_key}_crlb"]

        all_results[p_val] = {
            'selected_keys': selected_keys,
            'rmse_results': rmse_results,
            'crlb_results': crlb_results,
        }

    return dict(data), all_results


def plot_stage1_p_comparison(
    snr_dbs,
    all_results,
    p_values,
    M,
    freq_range_str,
    output_dir=None,
    mode="frequentist"
):

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    if mode == "bayesian":
        rmse_label = "BRMSE"
        crlb_label = r"$\sqrt{\mathrm{BCRLB}}$"
    else:
        rmse_label = "RMSE"
        crlb_label = r"$\sqrt{\mathrm{CRLB}}$"

    top_keys = all_results[p_values[0]]["selected_keys"][:4]

    # Colorblind-friendly, high-contrast colors
    colors = {
        10: "#0072B2",   # blue
        30: "#D55E00",   # vermillion
        50: "#009E73",   # green
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.5, 8.0),
        sharex=True
    )
    axes = axes.flatten()

    for idx, key in enumerate(top_keys):
        ax = axes[idx]

        for p_val in p_values:
            if key not in all_results[p_val]["rmse_results"]:
                continue

            rmse_vals = all_results[p_val]["rmse_results"][key]
            crlb_vals = all_results[p_val]["crlb_results"][key]

            # RMSE
            ax.plot(
                snr_dbs,
                rmse_vals,
                color=colors[p_val],
                linestyle='-',
                linewidth=1.8
            )

            # CRLB
            ax.plot(
                snr_dbs,
                crlb_vals,
                color=colors[p_val],
                linestyle="--",
                linewidth=1.8,
            )

        ax.set_xlabel("SNR (dB)", fontsize=15)
        ax.set_ylabel("Normalized Error", fontsize=15)
        #ax.set_yscale('log')
        ax.tick_params(axis="both",which="major",labelsize=12,direction="in"
        )

        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)

        # Cleaner panel labels
        ax.text(0.5, 1.025,panel_labels[idx],transform=ax.transAxes,ha="center",va="bottom",fontsize=16
        )

        # Slightly thicker axes
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # ---------- Figure-level legend ----------

    # p-value colors
    p_handles = [Line2D([0], [0], color=colors[p], linewidth=2, label=rf"$p={p}$")
        for p in p_values
    ]

    # RMSE / CRLB line styles
    type_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            markersize=4.5,
            linewidth=1.8,
            label=rmse_label
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=crlb_label
        )
    ]

    handles = p_handles + type_handles

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=13,
        frameon=False
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.10,
        left=0.09,
        right=0.98,
        hspace=0.30,
        wspace=0.25
    )

    filename = (
        f"stage1_p_comparison_M{M}_"
        f"{freq_range_str}_{mode}_NEW.pdf"
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)

    plt.savefig(
        filename,
        bbox_inches="tight"
    )
    plt.close(fig)
    print(f"Saved: {filename}")



def plot_rmse_vs_crlb_all_params(snr_dbs, all_results, p_val, M, freq_range_str,
                                  num_params_to_plot=10, output_dir=None, mode="frequentist"):
    """
    Plot RMSE vs CRLB for top 10 parameters at a specific p value.

    Args:
        snr_dbs: Array of SNR values in dB
        all_results: Dict with structure {p_val: {...}}
        p_val: Which p value to plot
        M: Number of Monte Carlo trials
        freq_range_str: Frequency range string for filename
        num_params_to_plot: Number of top parameters to show
        output_dir: Output directory for saving plots
        mode: "frequentist" or "bayesian" - affects labels
    """
    # Labels depend on mode
    if mode == "bayesian":
        rmse_label = "Bayes RMSE"
        crlb_label = "√BCRLB"
    else:
        rmse_label = "RMSE"
        crlb_label = "√CRLB"

    selected_keys = all_results[p_val]['selected_keys'][:num_params_to_plot]
    rmse_results = all_results[p_val]['rmse_results']
    crlb_results = all_results[p_val]['crlb_results']

    n_params = len(selected_keys)
    ncols = 2
    nrows = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for idx, key in enumerate(selected_keys):
        ax = axes[idx]

        rmse_vals = rmse_results[key]
        crlb_vals = crlb_results[key]

        ax.plot(snr_dbs, rmse_vals, 'bo-', label=rmse_label, markersize=4)
        ax.plot(snr_dbs, crlb_vals, 'r--', label=crlb_label, linewidth=1.5)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Error')
        ax.set_title(key, fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    filename = f"stage1_rmse_crlb_p{p_val}_M{M}_{freq_range_str}_{mode}_NEW.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def compute_tf_posterior_ci(data, p_val, snr_db, num_samples=500, device='cpu'):
    """
    Compute transfer function posterior confidence interval from saved best_params.

    Args:
        data: Dict from np.load with allow_pickle=True
        p_val: Which p value to use
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
    selected_keys = list(data[f'p{p_val}_selected_keys'])

    # Recreate forward model
    forward_model = MTLForwardModel(frequencies, network_params, device=device)

    # Get saved variational parameters for this p_val and snr_db
    snr_prefix = f"p{p_val}_snr{snr_db}"

    # Build list of posterior samples by sampling from LogitNormal
    tf_samples = []

    for _ in range(num_samples):
        # Sample from posterior for each parameter
        sampled_params = {}
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
                sampled_params[key] = theta_sample

        # Update network_params with sampled values
        for key, val in sampled_params.items():
            if "." in key:
                parts = key.split(".")
                load_name, param_name = parts[0], parts[1]
                network_params["loads"][load_name][param_name]["value"] = val
            else:
                network_params["cable_lengths"][key]["value"] = val

        # Compute TF with sampled params
        cable_lengths = {}
        for name, info in network_params["cable_lengths"].items():
            cable_lengths[name] = torch.tensor(info["value"], dtype=torch.float32, device=device)

        load_params = {}
        for load_name, params in network_params["loads"].items():
            load_params[load_name] = {}
            for param_name, param_info in params.items():
                load_params[load_name][param_name] = torch.tensor(
                    param_info["value"], dtype=torch.float32, device=device
                )

        H = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
        H_db = 20 * torch.log10(torch.abs(H) + 1e-12)
        tf_samples.append(H_db.detach().cpu().numpy())

    # Stack and compute statistics
    tf_samples = np.stack(tf_samples, axis=0)  # [num_samples, F]
    tf_mean = np.mean(tf_samples, axis=0)
    tf_lower = np.percentile(tf_samples, 2.5, axis=0)
    tf_upper = np.percentile(tf_samples, 97.5, axis=0)

    # Compute true TF
    # Restore theta_true to network_params using dict (correct key->value mapping)
    theta_true_dict = data['theta_true_dict'].item()  # Unwrap 0-d numpy array
    for key, val in theta_true_dict.items():
        if "." in key:
            parts = key.split(".")
            load_name, param_name = parts[0], parts[1]
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
            load_params_true[load_name][param_name] = torch.tensor(
                param_info["value"], dtype=torch.float32, device=device
            )

    H_true = forward_model.calculate_Hnw_nofault(cable_lengths_true, load_params_true)
    H_clean_db = 20 * torch.log10(torch.abs(H_true) + 1e-12)
    H_clean_db = H_clean_db.detach().cpu().numpy()

    freq_range_mhz = frequencies.cpu().numpy() / 1e6

    return tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz


def plot_CI_grid(data, p_val, M, snr_dbs_to_plot, freq_range_str, output_dir=None, num_samples=200):
    """
    Plot transfer function CI for multiple SNR values in a grid layout.

    Shows how reconstruction improves as SNR increases.

    Args:
        data: Dict from np.load with allow_pickle=True
        p_val: Number of inferred parameters
        M: Num of Monte carlo samples
        snr_dbs_to_plot: List of SNR values to plot (e.g., [0, 5, 10, 15, 20, 25, 30, 35])
        freq_range_str: Frequency range string for filename
        output_dir: Output directory
        num_samples: Number of posterior samples for CI computation
    """
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    panel = 0
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    axes = axes.flatten()

    # Compute CI data for each SNR
    for idx, snr_db in enumerate(snr_dbs_to_plot):
        ax = axes[idx]

        tf_mean, tf_lower, tf_upper, H_clean_db, freq_range_mhz = compute_tf_posterior_ci(
            data, p_val, snr_db, num_samples=num_samples
        )

        # Panel label
        ax.text(
            0.5, 1.05,
            panel_labels[panel],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
        )
        panel += 1

        ax.plot(freq_range_mhz, tf_mean, 'k-', linewidth=1.0, label='Posterior Mean')
        ax.plot(freq_range_mhz, H_clean_db, 'r--', linewidth=1.0, label='Truth')
        ax.fill_between(freq_range_mhz, tf_lower, tf_upper,
                       alpha=0.3, color='steelblue', label='95% CI')

        ax.set_xscale('log')
        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel(r'$H_{1,1}$ (dB)', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

    # Single figure-level legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)  # Make room for legend

    filename = f"stage1_tf_ci_grid_p{p_val}_M{M}_{freq_range_str}_NEW.pdf"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


CI_SAMPLES = 200  # Number of posterior samples for CI plots

def main():
    parser = argparse.ArgumentParser(description='Plot Stage 1 MTL inference results')
    parser.add_argument('npz_file', help='Path to .npz results file')
    args = parser.parse_args()

    if not os.path.exists(args.npz_file):
        print(f"Error: File not found: {args.npz_file}")
        sys.exit(1)

    print(f"Loading results from: {args.npz_file}")
    data, all_results = load_stage1_results(args.npz_file)

    snr_dbs = data['snr_dbs']
    p_values = data['p_values']
    M = int(data['M'])
    freq_range_str = str(data['freq_range_str'])
    sorted_keys_all = list(data['sorted_keys_all'])

    # Detect mode from npz (default to frequentist for backward compatibility)
    mode = str(data['mode']) if 'mode' in data else "frequentist"

    # Fixed output directory
    output_dir = os.path.join("figures", "Stage1Results")

    print(f"\nData summary:")
    print(f"  SNR values: {snr_dbs}")
    print(f"  p values: {p_values}")
    print(f"  M (Monte Carlo): {M}")
    print(f"  Frequency range: {freq_range_str}")
    print(f"  Total parameters: {len(sorted_keys_all)}")
    print(f"  Mode: {mode}")
    print(f"\nTop 4 most sensitive parameters:")
    for i, key in enumerate(sorted_keys_all[:4]):
        print(f"  {i+1}. {key}")

    # Generate plots
    print("\nGenerating plots...")

    # 1. P-value comparison (top 3 params)
    plot_stage1_p_comparison(snr_dbs, all_results, p_values, M, freq_range_str, output_dir, mode)

    # 2. Individual RMSE vs CRLB for each p value
    for p_val in p_values:
        plot_rmse_vs_crlb_all_params(snr_dbs, all_results, p_val, M, freq_range_str,
                                      num_params_to_plot=min(10, p_val), output_dir=output_dir, mode=mode)

    # 3. TF Confidence Interval grid plots (frequentist mode only)
    if mode == "bayesian":
        print("\nSkipping CI plots (not available for bayesian mode)")
    else:
        print("\nGenerating TF confidence interval grid plots...")
        # Plot SNR 0-40 dB in 5x2 grid to show reconstruction improvement
        snr_dbs_for_ci = [snr for snr in [0, 5, 10, 15, 20, 25, 30, 35] if snr in snr_dbs]
        #just plto for p = 50
        #plot_CI_grid(data, 50, M, snr_dbs_for_ci, freq_range_str, output_dir, num_samples=CI_SAMPLES)

    print("\nDone!")


if __name__ == "__main__":
    main()
