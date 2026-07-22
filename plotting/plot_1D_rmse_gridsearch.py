# plotting/plot_rmse_gridsearch.py
"""
Plot RMSE vs sqrt(CRLB) for 1D grid search MLE.
Requires: run_1D_gridsearch.py to be run first.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt


def find_gridsearch_results(results_dir="results"):
    """Find most recent grid search results file."""
    results_path = pathlib.Path(results_dir)
    if not results_path.exists():
        return None

    files = list(results_path.glob("1D_gridsearch_*.npz"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def plot_rmse_vs_crlb(results_path, save_dir="figures/plot_rmse_gridsearch"):
    """
    Plot RMSE vs sqrt(CRLB) across SNR for all 5 parameters.

    Layout: 3 rows x 2 columns (using 4-column GridSpec)
    - Row 1: ZF_re, ZF_im
    - Row 2: ZL_re, ZL_im
    - Row 3: L1 (centered)
    """
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load results
    data = np.load(results_path)
    snr_dbs = data["snr_dbs"]

    # Extract RMSE and CRLB for each parameter
    rmse = {
        "L1": data["rmse_L1"],
        "ZF_re": data["rmse_ZF_re"],
        "ZF_im": data["rmse_ZF_im"],
        "ZL_re": data["rmse_ZL_re"],
        "ZL_im": data["rmse_ZL_im"],
    }
    crlb = {
        "L1": data["crlb_L1"],
        "ZF_re": data["crlb_ZF_re"],
        "ZF_im": data["crlb_ZF_im"],
        "ZL_re": data["crlb_ZL_re"],
        "ZL_im": data["crlb_ZL_im"],
    }

    # Quantization error (if available)
    quant_error = {}
    if "quant_L1" in data:
        quant_error = {
            "L1": float(data["quant_L1"]),
            "ZF_re": float(data["quant_ZF_re"]),
            "ZF_im": float(data["quant_ZF_im"]),
            "ZL_re": float(data["quant_ZL_re"]),
            "ZL_im": float(data["quant_ZL_im"]),
        }

    # Layout using 4 columns
    layout = [
        [("ZF_re", 0, slice(0, 2)), ("ZF_im", 0, slice(2, 4))],  # Row 1
        [("ZL_re", 1, slice(0, 2)), ("ZL_im", 1, slice(2, 4))],  # Row 2
        [("L1", 2, slice(1, 3))],                                 # Row 3 (centered)
    ]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.25)

    panel = 0
    for row in layout:
        for item in row:
            param, row_idx, col_idx = item
            ax = fig.add_subplot(gs[row_idx, col_idx])

            # Add panel label
            ax.text(
                0.5, 1.05, panel_labels[panel],
                transform=ax.transAxes, ha="center", va="bottom", fontsize=13,
            )
            panel += 1

            # RMSE curve
            ax.plot(
                snr_dbs, rmse[param],
                marker="o", markersize=6, linewidth=2,
                color="blue", label="Grid Search RMSE",
            )

            # sqrt(CRLB) curve
            ax.plot(
                snr_dbs, crlb[param],
                marker="x", markersize=6, linewidth=2,
                linestyle="--", color="black",
                label=r"$\sqrt{\mathrm{CRLB}}$",
            )

            # Quantization error line
            if param in quant_error:
                ax.axhline(
                    y=quant_error[param],
                    color="red", linestyle=":", linewidth=2,
                    label="Quant. Error",
                )

            ax.set_xlabel("SNR (dB)", fontsize=12)
            if panel % 2 != 0:
                ax.set_ylabel("RMSE", fontsize=12)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower left', fontsize=10)
            ax.tick_params(labelsize=10)

    plt.savefig(save_path / "rmse_vs_crlb_gridsearch.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE vs CRLB plot to {save_path / 'rmse_vs_crlb_gridsearch.pdf'}")


def main(results_path=None, save_dir="figures/plot_rmse_gridsearch"):
    """Main function."""
    if results_path is None:
        results_path = find_gridsearch_results()

    if results_path is None:
        print("Error: No grid search results found. Run run_1D_gridsearch.py first.")
        return

    print(f"Loading results from {results_path}")
    plot_rmse_vs_crlb(results_path, save_dir)
    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot RMSE vs CRLB for grid search")
    parser.add_argument("--results", type=str, default=None, help="Path to results file")
    parser.add_argument("--save-dir", type=str, default="figures/plot_rmse_gridsearch", help="Save directory")
    args = parser.parse_args()

    main(results_path=args.results, save_dir=args.save_dir)
