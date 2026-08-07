# experiments/generate_observations.py
"""
Generate and save observations for 1D loss landscapes. 
"""
import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config.simple_config_loader import load_config, config_hash


def main(cfg_path="config/simple_network_config.yaml"):
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)
    # Setup experiment (loads config, creates FM/DM)
    exp = load_config(cfg_path)
    print(f"Device: {exp['device']}")
    print(f"Config: {exp['freq_tag']}, {exp['L_tag']}, N={exp['N']}, seed={exp['seed']}")
    # Config dict for observation data (for hashing)
    obs_config = {
        "freq_start": exp["fstart"], "freq_stop": exp["fend"], "F": exp["F"],
        "L": exp["L"], "L1": exp["fixed"]["L1"],
        "ZF_re": exp["fixed"]["ZF"]["re"], "ZF_im": exp["fixed"]["ZF"]["im"],
        "ZL_re": exp["fixed"]["ZL"]["re"], "ZL_im": exp["fixed"]["ZL"]["im"],
        "N": exp["N"], "seed": exp["seed"],
    }
    obs_hash = config_hash(obs_config)

    # Create results directory
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    # Save forward model (needed for plotting and estimation)
    fm_file = results_dir / f"forward_model_{exp['freq_tag']}_{exp['L_tag']}_{obs_hash}.pt"
    torch.save(exp["fm"], fm_file)
    print(f"Saved forward model to {fm_file}")

    # Generate and save observations for each SNR
    for snr_db in exp["snrs"]:
        print(f"Generating observations at SNR = {snr_db} dB...")

        data = exp["dm"].generate_observations(
            snr_db, exp["N"], exp["fm"], seed=exp["seed"],
            target=exp["target"], fixed=exp["fixed"], gen_cfg=exp["true_range"]
        )

        # Save observation file
        obs_file = results_dir / f"observations_{exp['freq_tag']}_{exp['L_tag']}_{obs_hash}_snr{int(snr_db)}.npz"

        np.savez(
            obs_file,
            # Config for reproducibility
            obs_config=obs_config,
            seed=exp["seed"],
            snr_db=snr_db,
            # Observations [N, F]
            h_obs_real=data["h_obs_real"],
            h_obs_imag=data["h_obs_imag"],
            h_true_real=data["h_true_real"],
            h_true_imag=data["h_true_imag"],
            noise_var=data["noise_var"],
            # True parameter values [N]
            L1_true=data["L1_true"],
            ZF_true_re=data["ZF_true_re"],
            ZF_true_im=data["ZF_true_im"],
            ZL_true_re=data["ZL_true_re"],
            ZL_true_im=data["ZL_true_im"],
            # Cable parameters (for loss landscape plots)
            pul_freq=exp["pul_freq"].cpu().numpy(),
            gamma_real=exp["gamma_full"].real.cpu().numpy(),
            gamma_imag=exp["gamma_full"].imag.cpu().numpy(),
            Zc_real=exp["Zc_full"].real.cpu().numpy(),
            Zc_imag=exp["Zc_full"].imag.cpu().numpy(),
        )
        print(f"  Saved to {obs_file}")

    end_time = time.perf_counter()
    print(f"\nGenerated {exp["N"]} CTF observations for {len(exp['snrs'])} SNR values in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
