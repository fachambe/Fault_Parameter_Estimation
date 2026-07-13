# plotting/plot_joint_benchmark.py
"""
Plot results from joint_benchmark.py:
- 3 separate figures showing RMSE vs sqrt(CRLB) for L1, ZF, ZL
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_results(results_path):
    """Load saved results from .npz file."""
    data = np.load(results_path, allow_pickle=True)
    return {
        "snrs": data["snrs"],
        "rmse_L1": data["rmse_L1"],
        "rmse_ZF": data["rmse_ZF"],
        "rmse_ZL": data["rmse_ZL"],
        "crlb_L1": data["crlb_L1"],
        "crlb_ZF": data["crlb_ZF"],
        "crlb_ZL": data["crlb_ZL"],
        "freq_tag": str(data["freq_tag"]),
        "L_tag": str(data["L_tag"]),
        "seed": int(data["seed"]),
    }


def plot_parameter(snrs, rmse, crlb, param_name, unit, save_path, freq_tag, L_tag):
    """Plot RMSE vs sqrt(CRLB) for a single parameter."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot RMSE
    ax.plot(
        snrs, rmse,
        marker="o", markersize=8, linewidth=2.5,
        color="blue", label=f"Adam RMSE"
    )

    # Plot sqrt(CRLB)
    ax.plot(
        snrs, crlb,
        marker="x", markersize=8, linewidth=2.5,
        linestyle="--", color="black",
        label=r"$\sqrt{\mathrm{CRLB}}$"
    )

    ax.set_xlabel("SNR (dB)", fontsize=14)
    ax.set_ylabel(f"RMSE / $\\sqrt{{\\mathrm{{CRLB}}}}$ ({unit})", fontsize=14)
    ax.set_yscale("log")
    ax.set_title(f"{param_name} Estimation: {freq_tag}, {L_tag}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main(results_path=None, save_dir="figures/joint_benchmark_figures"):
    """
    Main plotting function.

    Args:
        results_path: Path to results .npz file. If None, uses most recent.
        save_dir: Directory to save figures.
    """
    if results_path is None:
        results_dir = pathlib.Path("results")
        results_files = list(results_dir.glob("joint_benchmark_*.npz"))
        if not results_files:
            print("Error: No joint benchmark results found in results/")
            print("Run joint_benchmark.py first to generate results.")
            return
        results_path = max(results_files, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent results file: {results_path}")

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}")
    results = load_results(results_path)

    snrs = results["snrs"]
    freq_tag = results["freq_tag"]
    L_tag = results["L_tag"]
    seed = results["seed"]

    # Plot L1
    plot_parameter(
        snrs, results["rmse_L1"], results["crlb_L1"],
        param_name="L1", unit="m",
        save_path=save_path / f"L1_{freq_tag}_{L_tag}_seed{seed}.pdf",
        freq_tag=freq_tag, L_tag=L_tag
    )

    # Plot ZF
    plot_parameter(
        snrs, results["rmse_ZF"], results["crlb_ZF"],
        param_name="ZF", unit="Ω",
        save_path=save_path / f"ZF_{freq_tag}_{L_tag}_seed{seed}.pdf",
        freq_tag=freq_tag, L_tag=L_tag
    )

    # Plot ZL
    plot_parameter(
        snrs, results["rmse_ZL"], results["crlb_ZL"],
        param_name="ZL", unit="Ω",
        save_path=save_path / f"ZL_{freq_tag}_{L_tag}_seed{seed}.pdf",
        freq_tag=freq_tag, L_tag=L_tag
    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot joint benchmark results")
    parser.add_argument(
        "--results", type=str, default=None,
        help="Path to results .npz file (default: most recent in results/)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="figures/joint_benchmark_figures",
        help="Directory to save figures"
    )
    args = parser.parse_args()

    main(args.results, args.save_dir)
