# experiments/run_1D_gridsearch.py
"""
Run 1D grid search MLE benchmark for all 5 parameters for simple model. Theta is some assumed to be a fixed, deterministic value so this is frequentist only. 
"""
import sys, pathlib
import time
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
import numpy as np
import torch
import matplotlib.pyplot as plt
import hashlib
import json
from core.forward import ForwardModel
from core.likelihoods import ComplexGaussianLik
from core.crlb import crlb_for_1_real_param
from core.manager import DatasetManager
from estimators.mle_gridsearch import GridSearchMLE 


network_params = {
    "L1": {
        "value": 250.0,
        "inferred": True,
        "range": (1.0, 999.0)
    },
    "ZF_re": {
        "value": 100.0,
        "inferred": True,
        "range": (1.0, 1000.0)
    },
    "ZF_im": {
        "value": -50.0,
        "inferred": True,
        "range": (-100.0, 100.0)
    },
    "ZL_re": {
        "value": 100.0,
        "inferred": True,
        "range": (1.0, 400.0)
    },
    "ZL_im": {
        "value": -5.0, 
        "inferred": True,
        "range": (-100.0, 100.0)
    }
}

DEVICE = torch.device("cuda")
SEED = 98
N_OBS = 1 
M = 2500
K = 200 #num of frequency points 
SNR_DBS = [0, 10, 20, 30, 40]
#SNR_DBS = [40]
CABLE_LENGTH = 1000.0 #L = 1000m
ZS = 50.0
frequencies = torch.logspace(
    torch.log10(torch.tensor(150e3, device=DEVICE)),
    torch.log10(torch.tensor(500e3, device=DEVICE)),
    K, device=DEVICE
)  # NB-PLC 150-500KHz

def format_freq(f_hz):
    """Format frequency as kHz or MHz string."""
    if f_hz >= 1e6:
        return f"{int(f_hz / 1e6)}mhz"
    else:
        return f"{int(f_hz / 1e3)}khz"


def config_hash(cfg_dict, length=8):
    """Create short hash of config dict for unique filenames."""
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:length]


def main(quant_err=0.05, max_mem_gb=4.0):
    start_time = time.perf_counter()
    fm = ForwardModel(frequencies, CABLE_LENGTH, ZS, device=DEVICE)
    dm = DatasetManager(device=DEVICE)
    
    targets = [param for param, info in network_params.items() if info["inferred"]]

    # Build grids from network_params
    step = 2 * quant_err
    offset = step / 2
    
    grid_sizes = {}
    grids = {}

    for param in targets:
        rmin, rmax = network_params[param]["range"]
        n_pts = max(1000, int((rmax - rmin) / step) + 1)
        grid_sizes[param] = n_pts
        grids[param] = torch.linspace(rmin + offset, rmax + offset, n_pts, device=DEVICE)
    
    print(f"Grid sizes: {grid_sizes}")

    # Helper to determine batch size based on memory constraints
    def batch_size(K):
        """Estimate batch size from grid size and memory limit."""
        return min(K, max(1, int(max_mem_gb * 1e9 / (K * 8 * 2 * len(frequencies)))))

    # Initialize result storage
    rmse_curves = {t: [] for t in targets}
    crlb_curves = {t: [] for t in targets}

    # Process each SNR
    for snr_db in SNR_DBS:
        print(f"\nSNR = {snr_db} dB")

        # Frequentist: same true values for all M trials
        L1_true = np.full(M, network_params["L1"]["value"], dtype=np.float32)
        ZF_re_true = np.full(M, network_params["ZF_re"]["value"], dtype=np.float32)
        ZF_im_true = np.full(M, network_params["ZF_im"]["value"], dtype=np.float32)
        ZL_re_true = np.full(M, network_params["ZL_re"]["value"], dtype=np.float32)
        ZL_im_true = np.full(M, network_params["ZL_im"]["value"], dtype=np.float32)
        
        ZF_true = ZF_re_true + 1j * ZF_im_true
        ZL_true = ZL_re_true + 1j * ZL_im_true

        h_obs, var = dm.generate_observations(snr_db, M, fm, L1_true, ZF_true, ZL_true) #[M, F]

        # h_obs_np = h_obs.cpu().numpy()  # [M, F]
        # freq_np = frequencies.cpu().numpy()

        # h_obs_db = 20 * np.log10(np.abs(h_obs_np))  # [M, F]

        # plt.plot(freq_np, h_obs_db[0])
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('|H| (dB)')
        # plt.show()
        
        # Run grid search MLE for each target parameter (1D search)
        for t in targets:
            # Set inferred flags: True for target, False for others
            network_params_t = {k: {**v, "inferred": (k == t)} for k, v in network_params.items()}

            K_grid = grid_sizes[t]
            bs = batch_size(K_grid)
            est = GridSearchMLE(
                fm=fm,
                likelihood=ComplexGaussianLik(),
                grid=grids[t],
                network_params=network_params_t,
                device=DEVICE,
                batch_size=bs,
            )

            # Get predictions
            preds = est.predict(h_obs, var)

            # Get true values for this parameter
            if t == "L1":
                true_vals = L1_true
            elif t == "ZF_re":
                true_vals = ZF_re_true
            elif t == "ZF_im":
                true_vals = ZF_im_true
            elif t == "ZL_re":
                true_vals = ZL_re_true
            elif t == "ZL_im":
                true_vals = ZL_im_true

            # Compute RMSE
            rmse = float(np.sqrt(np.mean((preds[t] - true_vals) ** 2)))
            rmse_curves[t].append(rmse)

            # Build complete parameter dict for CRLB (needs ALL params at true values)
            fixed_crlb = {
                "L1": network_params["L1"]["value"],
                "ZF_re": network_params["ZF_re"]["value"],
                "ZF_im": network_params["ZF_im"]["value"],
                "ZL_re": network_params["ZL_re"]["value"],
                "ZL_im": network_params["ZL_im"]["value"]
            }

            # Compute CRLB for this parameter
            _, crlb = crlb_for_1_real_param(fm, t, fixed_crlb, var[0].squeeze(), DEVICE)
            sqrt_crlb = float(torch.sqrt(crlb).cpu())
            crlb_curves[t].append(sqrt_crlb)

            print(f"  {t}: RMSE={rmse:.4f}, sqrt(CRLB)={sqrt_crlb:.4f}")

    # Save results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create tags for filename
    freq_start = frequencies[0].item()
    freq_end = frequencies[-1].item()
    freq_tag = f"{format_freq(freq_start)}_to_{format_freq(freq_end)}"
    L_tag = f"L{int(CABLE_LENGTH)}m"

    # Create config for filename
    bench_config = {
        "freq_start": freq_start,
        "freq_stop": freq_end,
        "L": CABLE_LENGTH,
        "grid_sizes": grid_sizes,
        "M": M,
        "seed": SEED,
    }
    bench_hash = config_hash(bench_config)

    benchmark_file = results_dir / f"1D_gridsearch_{freq_tag}_{L_tag}_{bench_hash}_seed{SEED}.npz"

    np.savez(
        benchmark_file,
        # Config
        freq_tag=freq_tag,
        L_tag=L_tag,
        snr_dbs=np.array(SNR_DBS),
        seed=SEED,
        M=M,
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
    )
    print(f"\nSaved grid search results to {benchmark_file}")

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":

    main()
