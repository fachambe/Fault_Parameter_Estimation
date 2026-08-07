# experiments/run_1D_gridsearch.py
"""
Run 1D grid search MLE benchmark.
Generates observations on-the-fly from config.
"""
import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.likelihoods import ComplexGaussianLik
from core.crlb import crlb_for_1_real_param
from estimators.mle_gridsearch import GridSearchMLE
from config.simple_config_loader import load_config, config_hash


def main(cfg_path="config/simple_network_config.yaml", quant_err=0.05, max_mem_gb=4.0):
    start_time = time.perf_counter()

    # Setup experiment (loads config, creates FM/DM)
    exp = load_config(cfg_path)
    print(f"Device: {exp['device']}")
    print(f"Config: {exp['freq_tag']}, {exp['L_tag']}, N={exp['N']}, seed={exp['seed']}")

    # Define targets
    targets = ["L1", "ZF_re", "ZF_im", "ZL_re", "ZL_im"]

    # Build grids: quant_err = step/2, so step = 2*quant_err, N = range/step + 1
    # Offset grid by step/2 so true values don't land exactly on grid points
    true_range = exp["true_range"]
    F = exp["F"]
    step = 2 * quant_err
    offset = step / 2  # shift grid so true values fall between points

    def grid_pts(rmin, rmax):
        return max(1000, int((rmax - rmin) / step) + 1)

    def batch_size(K):
        return max(50, int(max_mem_gb * 1e9 / (K * F * 8)))

    grid_sizes = {
        "L1": grid_pts(true_range["L1"]["min"], true_range["L1"]["max"]),
        "ZF_re": grid_pts(true_range["ZF"]["re"]["min"], true_range["ZF"]["re"]["max"]),
        "ZF_im": grid_pts(true_range["ZF"]["im"]["min"], true_range["ZF"]["im"]["max"]),
        "ZL_re": grid_pts(true_range["ZL"]["re"]["min"], true_range["ZL"]["re"]["max"]),
        "ZL_im": grid_pts(true_range["ZL"]["im"]["min"], true_range["ZL"]["im"]["max"]),
    }
    print(f"Target quant error: {quant_err}, grid sizes: {grid_sizes}")

    grids = {
        "L1": torch.linspace(true_range["L1"]["min"] + offset, true_range["L1"]["max"] + offset, grid_sizes["L1"], device=exp["device"]),
        "ZF_re": torch.linspace(true_range["ZF"]["re"]["min"] + offset, true_range["ZF"]["re"]["max"] + offset, grid_sizes["ZF_re"], device=exp["device"]),
        "ZF_im": torch.linspace(true_range["ZF"]["im"]["min"] + offset, true_range["ZF"]["im"]["max"] + offset, grid_sizes["ZF_im"], device=exp["device"]),
        "ZL_re": torch.linspace(true_range["ZL"]["re"]["min"] + offset, true_range["ZL"]["re"]["max"] + offset, grid_sizes["ZL_re"], device=exp["device"]),
        "ZL_im": torch.linspace(true_range["ZL"]["im"]["min"] + offset, true_range["ZL"]["im"]["max"] + offset, grid_sizes["ZL_im"], device=exp["device"]),
    }

    # Compute quantization error for each parameter
    fixed = exp["fixed"]
    true_values = {
        "L1": fixed["L1"],
        "ZF_re": fixed["ZF"]["re"],
        "ZF_im": fixed["ZF"]["im"],
        "ZL_re": fixed["ZL"]["re"],
        "ZL_im": fixed["ZL"]["im"],
    }
    quant_error = {}
    for param in targets:
        grid = grids[param]
        true_val = true_values[param]
        closest_idx = torch.argmin(torch.abs(grid - true_val))
        quant_error[param] = float(torch.abs(grid[closest_idx] - true_val).cpu())
    print(f"Quantization errors: {quant_error}")

    # Initialize result storage
    rmse_curves = {t: [] for t in targets}
    crlb_curves = {t: [] for t in targets}
    snr_list = []
    
    # Process each SNR
    for snr_db in exp["snrs"]:
        print(f"\nSNR = {snr_db} dB")
        snr_list.append(snr_db)

        # Generate observations
        data = exp["dm"].generate_observations(
            snr_db, exp["N"], exp["fm"], seed=exp["seed"],
            target=exp["target"], fixed=fixed, gen_cfg=true_range
        )

        h_obs = torch.tensor(data["h_obs_real"], device=exp["device"]) + \
                1j * torch.tensor(data["h_obs_imag"], device=exp["device"])
        var = torch.tensor(data["noise_var"], device=exp["device"])

        # Run grid search MLE for each target
        for t in targets:
            K = grid_sizes[t]
            bs = batch_size(K)
            est = GridSearchMLE(
                fm=exp["fm"],
                likelihood=ComplexGaussianLik(),
                grid=grids[t],
                target=t,
                fixed=fixed,
                device=exp["device"],
                batch_size=bs,
            )

            # Get predictions
            preds = est.predict(h_obs, var)

            # Get true values
            if t == "L1":
                true_vals = data["L1_true"]
            elif t == "ZF_re":
                true_vals = data["ZF_true_re"]
            elif t == "ZF_im":
                true_vals = data["ZF_true_im"]
            elif t == "ZL_re":
                true_vals = data["ZL_true_re"]
            elif t == "ZL_im":
                true_vals = data["ZL_true_im"]

            # Compute RMSE
            rmse = float(np.sqrt(np.mean((preds[t] - true_vals) ** 2)))
            rmse_curves[t].append(rmse)

            # Compute CRLB
            _, crlb = crlb_for_1_real_param(exp["fm"], t, fixed, var[0].squeeze(), exp["device"])
            sqrt_crlb = float(torch.sqrt(crlb).cpu())
            crlb_curves[t].append(sqrt_crlb)

            print(f"  {t}: RMSE={rmse:.4f}, sqrt(CRLB)={sqrt_crlb:.4f}")

    # Save results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create config for filename
    bench_config = {
        "freq_start": exp["fstart"], "freq_stop": exp["fend"],
        "L": exp["L"], "grid_sizes": grid_sizes,
        "N": exp["N"], "seed": exp["seed"],
    }
    bench_hash = config_hash(bench_config)

    benchmark_file = results_dir / f"1D_gridsearch_{exp['freq_tag']}_{exp['L_tag']}_{bench_hash}_seed{exp['seed']}.npz"

    np.savez(
        benchmark_file,
        # Config
        freq_tag=exp["freq_tag"],
        L_tag=exp["L_tag"],
        snr_dbs=np.array(snr_list),
        seed=exp["seed"],
        N=exp["N"],
        grid_sizes=grid_sizes,
        # RMSE results
        rmse_L1=np.array(rmse_curves["L1"]),
        rmse_ZF_re=np.array(rmse_curves["ZF_re"]),
        rmse_ZF_im=np.array(rmse_curves["ZF_im"]),
        rmse_ZL_re=np.array(rmse_curves["ZL_re"]),
        rmse_ZL_im=np.array(rmse_curves["ZL_im"]),
        # sqrt(CRLB) results
        crlb_L1=np.array(crlb_curves["L1"]),
        crlb_ZF_re=np.array(crlb_curves["ZF_re"]),
        crlb_ZF_im=np.array(crlb_curves["ZF_im"]),
        crlb_ZL_re=np.array(crlb_curves["ZL_re"]),
        crlb_ZL_im=np.array(crlb_curves["ZL_im"]),
        # Quantization error
        quant_L1=quant_error["L1"],
        quant_ZF_re=quant_error["ZF_re"],
        quant_ZF_im=quant_error["ZF_im"],
        quant_ZL_re=quant_error["ZL_re"],
        quant_ZL_im=quant_error["ZL_im"],
    )
    print(f"\nSaved grid search results to {benchmark_file}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 1D grid search MLE benchmark")
    parser.add_argument("--quant-err", type=float, default=0.05, help="Target quantization error")
    parser.add_argument("--mem", type=float, default=4.0, help="Max GPU memory in GB")
    parser.add_argument("--cfg", type=str, default="config/simple_network_config.yaml")
    args = parser.parse_args()

    main(cfg_path=args.cfg, quant_err=args.quant_err, max_mem_gb=args.mem)
