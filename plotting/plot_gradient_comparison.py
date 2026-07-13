# plotting/plot_gradient_comparison.py
"""
Plot results from 1D_gradient_comparison.py:
- 2x2 figure showing RMSE vs sqrt(CRLB) for L1 across 4 scenarios:
  - High freq (2-10 MHz) + Long cable (L=1000m)
  - Low freq (150-500 kHz) + Long cable (L=1000m)
  - High freq (2-10 MHz) + Short cable (L=100m)
  - Low freq (150-500 kHz) + Short cable (L=100m)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_results(results_path):
    """Load saved results from .npz file."""
    data = np.load(results_path, allow_pickle=True)

    results = {
        "snrs": data["snrs"],
        "seed": int(data["seed"]),
        "scenarios": list(data["scenarios"]),
    }

    # Load per-scenario results
    # Keys are like: High_Freq_Long_Cable_rmse, High_Freq_Long_Cable_crlb, etc.
    scenario_keys = [
        "High_Freq_Long_Cable",
        "Low_Freq_Long_Cable",
        "High_Freq_Short_Cable",
        "Low_Freq_Short_Cable",
    ]

    results["scenario_data"] = {}
    for key in scenario_keys:
        results["scenario_data"][key] = {
            "rmse": data[f"{key}_rmse"],
            "crlb": data[f"{key}_crlb"],
            "freq_tag": str(data[f"{key}_freq_tag"]),
            "L_tag": str(data[f"{key}_L_tag"]),
        }

    return results


def plot_comparison(results, save_dir="figures/plot_gradient_comparison_figures"):
    """Create 2x2 figure showing RMSE vs sqrt(CRLB) for L1 across 4 scenarios."""
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    snrs = results["snrs"]

    # Scenario layout for 2x2 grid:
    # Row 0: Long Cable (L=1000m)
    # Row 1: Short Cable (L=100m)
    # Col 0: High Freq (2-10 MHz)
    # Col 1: Low Freq (150-500 kHz)
    scenario_layout = [
        ["High_Freq_Long_Cable", "Low_Freq_Long_Cable"],
        ["High_Freq_Short_Cable", "Low_Freq_Short_Cable"],
    ]

    # Display titles
    titles = {
        "High_Freq_Long_Cable": "High Freq (2-10 MHz) + Long Cable (L=1000m)",
        "Low_Freq_Long_Cable": "Low Freq (150-500 kHz) + Long Cable (L=1000m)",
        "High_Freq_Short_Cable": "High Freq (2-10 MHz) + Short Cable (L=100m)",
        "Low_Freq_Short_Cable": "Low Freq (150-500 kHz) + Short Cable (L=100m)",
    }

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel = 0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for i, row in enumerate(scenario_layout):
        for j, scenario_key in enumerate(row):
            ax = axes[i, j]
            data = results["scenario_data"][scenario_key]
            
            ax.text(
            0.5, 1.05,
            panel_labels[panel],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            )

            panel += 1

            # Plot RMSE
            ax.plot(
                snrs,
                data["rmse"],
                marker="o",
                markersize=8,
                linewidth=2.5,
                color="blue",
                label="Adam RMSE",
            )

            # Plot sqrt(CRLB)
            ax.plot(
                snrs,
                data["crlb"],
                marker="x",
                markersize=8,
                linewidth=2.5,
                linestyle="--",
                color="black",
                label=r"$\sqrt{\mathrm{CRLB}}$",
            )

            ax.set_xlabel("SNR (dB)", fontsize=14)
            ax.set_ylabel("RMSE / $\\sqrt{\\mathrm{CRLB}}$ (m)", fontsize=14)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower left", fontsize=12)
            ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save figure
    fig_path = save_path / f"gradient_comparison_seed{results['seed']}.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {fig_path}")


def main(results_path=None, save_dir="figures/plot_gradient_comparison_figures"):
    """
    Main plotting function.

    Args:
        results_path: Path to results .npz file. If None, uses default path.
        save_dir: Directory to save figures.
    """
    if results_path is None:
        # Find the most recent results file
        results_dir = pathlib.Path("results")
        results_files = list(results_dir.glob("gradient_comparison_seed*.npz"))
        if not results_files:
            print("Error: No gradient comparison results found in results/")
            print("Run 1D_gradient_comparison.py first to generate results.")
            return
        # Use most recent
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results file: {results_path}")

    print(f"Loading results from {results_path}")
    results = load_results(results_path)

    print(f"Plotting comparison for seed={results['seed']}...")
    plot_comparison(results, save_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gradient comparison results")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to results .npz file (default: most recent in results/)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="figures/plot_gradient_comparison_figures",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    main(args.results, args.save_dir)
