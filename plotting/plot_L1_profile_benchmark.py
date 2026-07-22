# plotting/plot_L1_profile_benchmark.py
"""
Plot results from run_L1_profile_benchmark.py:
- 3 separate 2x2 figures (one per parameter: L1, ZF, ZL)
- Each 2x2 shows RMSE vs sqrt(CRLB) across the 4 scenarios
- Single estimator: L1ProfileMLE
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
    scenario_keys = [
        "High_Freq_Long_Cable",
        "Low_Freq_Long_Cable",
        "High_Freq_Short_Cable",
        "Low_Freq_Short_Cable",
    ]

    results["scenario_data"] = {}
    for key in scenario_keys:
        # Check if this scenario exists in the data
        if f"{key}_rmse_L1" not in data:
            continue

        results["scenario_data"][key] = {
            "rmse_L1": data[f"{key}_rmse_L1"],
            "rmse_ZF": data[f"{key}_rmse_ZF"],
            "rmse_ZL": data[f"{key}_rmse_ZL"],
            "crlb_L1": data[f"{key}_crlb_L1"],
            "crlb_ZF": data[f"{key}_crlb_ZF"],
            "crlb_ZL": data[f"{key}_crlb_ZL"],
            "freq_tag": str(data[f"{key}_freq_tag"]),
            "L_tag": str(data[f"{key}_L_tag"]),
        }

    return results


def plot_parameter(results, param, unit, save_path):
    """
    Create 2x2 figure showing RMSE vs sqrt(CRLB) for one parameter across 4 scenarios.
    Single estimator: L1ProfileMLE.

    Args:
        results: loaded results dict
        param: "L1", "ZF", or "ZL"
        unit: unit string for y-axis label
        save_path: path to save the figure
    """
    snrs = results["snrs"]

    # Scenario layout for 2x2 grid
    scenario_layout = [
        ["High_Freq_Long_Cable", "Low_Freq_Long_Cable"],
        ["High_Freq_Short_Cable", "Low_Freq_Short_Cable"],
    ]

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    panel = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for i, row in enumerate(scenario_layout):
        for j, scenario_key in enumerate(row):
            ax = axes[i, j]

            # Check if scenario exists
            if scenario_key not in results["scenario_data"]:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                panel += 1
                continue

            data = results["scenario_data"][scenario_key]

            # Panel label
            ax.text(
                0.5, 1.05,
                panel_labels[panel],
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=18,
            )
            panel += 1

            # Get RMSE and CRLB
            rmse = data[f"rmse_{param}"]
            crlb = data[f"crlb_{param}"]

            # Plot L1ProfileMLE RMSE
            ax.plot(
                snrs, rmse,
                marker="o", markersize=8, linewidth=2,
                color="blue", label="L1Profile",
            )

            # Plot sqrt(CRLB)
            ax.plot(
                snrs, crlb,
                marker="x", markersize=8, linewidth=2.5,
                linestyle="--", color="black",
                label=r"$\sqrt{\mathrm{CRLB}}$",
            )

            ax.set_xlabel("SNR (dB)", fontsize=14)
            ax.set_ylabel(f"RMSE / $\\sqrt{{\\mathrm{{CRLB}}}}$ ({unit})", fontsize=14)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=10)
            ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main(results_path=None, save_dir="figures/L1_profile_benchmark_figures"):
    """
    Main plotting function.

    Args:
        results_path: Path to results .npz file. If None, uses most recent.
        save_dir: Directory to save figures.
    """
    if results_path is None:
        results_dir = pathlib.Path("results")
        results_files = list(results_dir.glob("L1_profile_benchmark_seed*.npz"))
        if not results_files:
            print("Error: No L1 profile benchmark results found in results/")
            print("Run run_L1_profile_benchmark.py first to generate results.")
            return
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results file: {results_path}")

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    seed = results["seed"]

    # Plot each parameter
    plot_parameter(
        results, "L1", "m",
        save_path / f"L1_profile_L1_seed{seed}.pdf"
    )
    plot_parameter(
        results, "ZF", "Ω",
        save_path / f"L1_profile_ZF_seed{seed}.pdf"
    )
    plot_parameter(
        results, "ZL", "Ω",
        save_path / f"L1_profile_ZL_seed{seed}.pdf"
    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot L1 profile benchmark results")
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to results .npz file (default: most recent in results/)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="figures/L1_profile_benchmark_figures",
        help="Directory to save figures"
    )
    args = parser.parse_args()

    main(args.results, args.save_dir)
