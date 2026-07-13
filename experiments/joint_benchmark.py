import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml
import scipy.io as sio

from core.forward import ForwardModel
from core.likelihoods import ComplexGaussianLik
from core.crlb import complex_partials_fullbatch, fim_from_complex_jac, get_CRLB
from estimators.mle_gridsearch import GridSearchMLE
from estimators.mle_gradient import GradientMLE
from data.manager import DatasetManager


def rmse_complex(pred, true):
    """RMSE for complex arrays: sqrt(mean(|pred - true|^2))"""
    mse = np.mean(np.abs(pred - true)**2)
    return float(np.sqrt(mse))


def main(cfg_path="configs/benchmark.yaml"):
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))
    print('Current device', device)
    # Load full arrays from the .mat (assumed in /experiments)
    mat = sio.loadmat("experiments/cable_parameters_extended.mat")
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full = torch.tensor(mat["Z_C"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device)
    #From 45Khz - 30Mhz, 1305 freq points
    
    fstart = float(cfg["freq"]["start_hz"])
    fend = float(cfg["freq"]["stop_hz"])
    F = int(cfg["freq"]["num_points"])
    L = float(cfg["L"])
    Zs = float(cfg["Zs"])

    # Format freq range for filenames (use kHz if < 1MHz, else MHz)
    def fmt_freq(hz):
        if hz >= 1e6:
            return f"{hz/1e6:.1f}MHz".replace(".0MHz", "MHz")
        else:
            return f"{hz/1e3:.0f}kHz"
    freq_tag = f"{fmt_freq(fstart)}-{fmt_freq(fend)}"
    L_tag = f"L{int(L)}"

    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)  # [F]
    freq_range_mhz = desired_freqs/1e6
    # For each desired f, find index of closest pul_freq
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)     # [F]
    gamma_list = gamma_full[idx]   # [F]
    Zc_list = Zc_full[idx]      # [F]
    
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=L, Zs=Zs, device=device)

    # Target + fixed parameter values + parameter range
    fixed = cfg["fixed"]
    target = cfg["target"].upper()
    true_range = cfg["true_range"]

    #Parameters
    snrs = cfg["snr_dbs"]
    N = int(cfg["N"])
    seed = cfg["seed"]
    

    rmse_L1_list = []
    rmse_ZF_list = []
    rmse_ZL_list = []

    crlb_L1_list = []
    crlb_ZF_list = []
    crlb_ZL_list = []

    for snr_db in snrs:
        print(f"Curr SNR is {snr_db}")
        data = dm.generate_observations(snr_db, N, fm, seed=seed,
                                     target=target, fixed=fixed, gen_cfg=true_range)
        h_obs = torch.tensor(data["h_obs_real"], device=device) + 1j*torch.tensor(data["h_obs_imag"], device=device)  # [N,F]
        var = torch.tensor(data["noise_var"], device=device)   # [N,1]


        # Joint MLE
        est = GradientMLE(
            fm=fm,
            likelihood=ComplexGaussianLik(),
            target=target,
            fixed=fixed,
            true_range=true_range,
            mode="joint",
            device=device,
        )
        preds = est.predict(h_obs, var) #dict of 3 [N] estimates

        # Compute RMSE
        rmse_L1 = float(np.sqrt(np.mean((preds["L1"] - data["L1_true"]) ** 2)))
        rmse_ZF = rmse_complex(preds["ZF"], data["ZF_true_re"] + 1j*data["ZF_true_im"])
        rmse_ZL = rmse_complex(preds["ZL"], data["ZL_true_re"] + 1j*data["ZL_true_im"])
        print("Rmse L1", rmse_L1)
        print("Rmse ZF", rmse_ZF)
        print("RMSE ZL", rmse_ZL)
        rmse_L1_list.append(rmse_L1)
        rmse_ZF_list.append(rmse_ZF)
        rmse_ZL_list.append(rmse_ZL)

        # Compute sqrt(CRLB)
        du_aug = complex_partials_fullbatch(fm, data, device)
        FIM = fim_from_complex_jac(du_aug, var)
        CRLB_L1, CRLB_ZF, CRLB_ZL = get_CRLB(FIM) #[N] tensor should all be same real values across N 
        sqrt_crlb_L1 = torch.sqrt(CRLB_L1).cpu().numpy()[0]
        sqrt_crlb_ZF = torch.sqrt(CRLB_ZF).cpu().numpy()[0]
        sqrt_crlb_ZL = torch.sqrt(CRLB_ZL).cpu().numpy()[0]
        print("Sqrt crlb L1", sqrt_crlb_L1)
        print("Sqrt zf", sqrt_crlb_ZF)
        print("Sqrt zl", sqrt_crlb_ZL)
        crlb_L1_list.append(sqrt_crlb_L1)
        crlb_ZF_list.append(sqrt_crlb_ZF)
        crlb_ZL_list.append(sqrt_crlb_ZL)

    # Save results to npz
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"joint_benchmark_{freq_tag}_{L_tag}_seed{seed}.npz"
    np.savez(
        results_path,
        snrs=np.array(snrs),
        rmse_L1=np.array(rmse_L1_list),
        rmse_ZF=np.array(rmse_ZF_list),
        rmse_ZL=np.array(rmse_ZL_list),
        crlb_L1=np.array(crlb_L1_list),
        crlb_ZF=np.array(crlb_ZF_list),
        crlb_ZL=np.array(crlb_ZL_list),
        freq_tag=freq_tag,
        L_tag=L_tag,
        seed=seed,
    )
    print(f"Results saved to {results_path}")

    elapsed = time.perf_counter() - start_time
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()