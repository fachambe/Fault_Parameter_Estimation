# plotting/plot_joint_gradient_comparison.py
"""
Plot results from joint_gradient_comparison.py:
- 3 separate 2x2 figures (one per parameter: L1, ZF, ZL)
- Each 2x2 shows RMSE vs sqrt(CRLB) across the 4 scenarios
- Compares 3 estimators: Adam only, Adam+Newton, Adam+LM
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
        if f"{key}_rmse_L1_adam" not in data:
            continue

        results["scenario_data"][key] = {
            # Adam only
            "rmse_L1_adam": data[f"{key}_rmse_L1_adam"],
            "rmse_ZF_adam": data[f"{key}_rmse_ZF_adam"],
            "rmse_ZL_adam": data[f"{key}_rmse_ZL_adam"],
            # Adam + Newton
            "rmse_L1_newton": data[f"{key}_rmse_L1_newton"],
            "rmse_ZF_newton": data[f"{key}_rmse_ZF_newton"],
            "rmse_ZL_newton": data[f"{key}_rmse_ZL_newton"],
            # Adam + LM
            "rmse_L1_lm": data[f"{key}_rmse_L1_lm"],
            "rmse_ZF_lm": data[f"{key}_rmse_ZF_lm"],
            "rmse_ZL_lm": data[f"{key}_rmse_ZL_lm"],
            # CRLB
            "crlb_L1": data[f"{key}_crlb_L1"],
            "crlb_ZF": data[f"{key}_crlb_ZF"],
            "crlb_ZL": data[f"{key}_crlb_ZL"],
            "freq_tag": str(data[f"{key}_freq_tag"]),
            "L_tag": str(data[f"{key}_L_tag"]),
        }

    return results


def plot_parameter_comparison(results, param, unit, save_path):
    """
    Create 2x2 figure showing RMSE vs sqrt(CRLB) for one parameter across 4 scenarios.
    Compares 3 estimators: Adam only, Adam+Newton, Adam+LM.

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

            # Check if scenario exists
            if scenario_key not in results["scenario_data"]:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                #ax.set_title(titles[scenario_key], fontsize=10)
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

            # Get RMSE for each estimator and CRLB
            rmse_adam = data[f"rmse_{param}_adam"]
            rmse_newton = data[f"rmse_{param}_newton"]
            rmse_lm = data[f"rmse_{param}_lm"]
            crlb = data[f"crlb_{param}"]

            # Plot Adam only
            ax.plot(
                snrs, rmse_adam,
                marker="o", markersize=8, linewidth=2,
                color="blue", label="Adam",
            )

            # Plot Adam + Newton
            ax.plot(
                snrs, rmse_newton,
                marker="s", markersize=8, linewidth=2,
                color="green", label="Adam+Newton",
            )

            # Plot Adam + LM
            ax.plot(
                snrs, rmse_lm,
                marker="^", markersize=8, linewidth=2,
                color="red", label="Adam+LM",
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
            #ax.set_title(titles[scenario_key], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=10)
            ax.tick_params(labelsize=12)

    #plt.suptitle(f"{param} Estimation Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main(results_path=None, save_dir="figures/joint_gradient_comparison_figures"):
    """
    Main plotting function.

    Args:
        results_path: Path to results .npz file. If None, uses most recent.
        save_dir: Directory to save figures.
    """
    if results_path is None:
        results_dir = pathlib.Path("results")
        results_files = list(results_dir.glob("joint_gradient_comparison_seed*.npz"))
        if not results_files:
            print("Error: No joint gradient comparison results found in results/")
            print("Run joint_gradient_comparison.py first to generate results.")
            return
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results file: {results_path}")

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    seed = results["seed"]

    # Plot each parameter
    plot_parameter_comparison(
        results, "L1", "m",
        save_path / f"joint_L1_comparison_seed{seed}.pdf"
    )
    plot_parameter_comparison(
        results, "ZF", "Ω",
        save_path / f"joint_ZF_comparison_seed{seed}.pdf"
    )
    plot_parameter_comparison(
        results, "ZL", "Ω",
        save_path / f"joint_ZL_comparison_seed{seed}.pdf"
    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot joint gradient comparison results")
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to results .npz file (default: most recent in results/)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="figures/joint_gradient_comparison_figures",
        help="Directory to save figures"
    )
    args = parser.parse_args()

    main(args.results, args.save_dir)
