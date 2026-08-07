# experiments/run_1D_adam.py
"""
Run 1D Adam (gradient-based) MLE benchmark.
Generates observations on-the-fly from config.
"""
import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from core.likelihoods import ComplexGaussianLik
from core.crlb import crlb_for_1_real_param
from estimators.mle_gradient import GradientMLE
from config.simple_config_loader import load_config, config_hash


def main(cfg_path="config/simple_network_config.yaml", adam_steps=2000, adam_lr=1e-2):
    start_time = time.perf_counter()

    # Setup experiment (loads config, creates FM/DM)
    exp = load_config(cfg_path)
    print(f"Device: {exp['device']}")
    print(f"Config: {exp['freq_tag']}, {exp['L_tag']}, N={exp['N']}, seed={exp['seed']}")
    print(f"Adam steps: {adam_steps}, learning rate: {adam_lr}")

    # Define targets
    targets = ["L1", "ZF_re", "ZF_im", "ZL_re", "ZL_im"]

    # Initialize result storage
    rmse_curves = {t: [] for t in targets}
    crlb_curves = {t: [] for t in targets}
    snr_list = []

    fixed = exp["fixed"]
    true_range = exp["true_range"]

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

        # Run Adam MLE for each target
        for t in targets:
            est = GradientMLE(
                fm=exp["fm"],
                likelihood=ComplexGaussianLik(),
                target=t,
                fixed=fixed,
                true_range=true_range,
                mode="1d",
                device=exp["device"],
                adam_steps=adam_steps,
                adam_lr=adam_lr,
                verbose=False,
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
        "L": exp["L"], "adam_steps": adam_steps, "adam_lr": adam_lr,
        "N": exp["N"], "seed": exp["seed"],
    }
    bench_hash = config_hash(bench_config)

    benchmark_file = results_dir / f"1D_adam_{exp['freq_tag']}_{exp['L_tag']}_{bench_hash}_seed{exp['seed']}.npz"

    np.savez(
        benchmark_file,
        # Config
        freq_tag=exp["freq_tag"],
        L_tag=exp["L_tag"],
        snr_dbs=np.array(snr_list),
        seed=exp["seed"],
        N=exp["N"],
        adam_steps=adam_steps,
        adam_lr=adam_lr,
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
    )
    print(f"\nSaved Adam results to {benchmark_file}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 1D Adam MLE benchmark")
    parser.add_argument("--adam-steps", type=int, default=2000, help="Number of Adam steps")
    parser.add_argument("--adam-lr", type=float, default=1e-2, help="Adam learning rate")
    parser.add_argument("--cfg", type=str, default="config/simple_network_config.yaml", help="Config file path")
    args = parser.parse_args()

    main(cfg_path=args.cfg, adam_steps=args.adam_steps, adam_lr=args.adam_lr)
