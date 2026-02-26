# experiments/benchmark.py
import sys, pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, yaml, pathlib
import scipy.io as sio
import matplotlib.pyplot as plt
from core.forward import ForwardModel
from core.likelihoods import ComplexGaussianLik, RiceanLikelihood, WrappedPhaseGaussianLikelihood
from estimators.mle_gridsearch import GridSearchMLE
from estimators.elbo import ELBOArgmaxMu1D
from estimators.mle_optimized import OptimizedMLE
from estimators.mle_vanillaadam import VanillaAdam
from estimators.VI import VI
from estimators.VIoptimized import VIoptimized
from estimators.ML import MLRegressor
from data.manager import DatasetManager
from torch.func import vmap, jacrev, jacfwd
from core.crlb import complex_partials_fullbatch, fim_from_complex_jac, get_CRLB, crlb_L1_only_batch, debug_jacobian_mags, crlb_for_target_estimate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform, loguniform
import numpy as np
import torch

def rmse_complex_parts(pred_re, pred_im, true_re, true_im):
    # Euclidean error in C
    mse = np.mean((pred_re - true_re)**2 + (pred_im - true_im)**2)
    return float(np.sqrt(mse))

def rmse_joint(pred, test):
    # pred is dict from estimator; test is your NPZ dict
    out = {}
    out["L1"]    = float(np.sqrt(np.mean((pred["L1"]    - test["L1_true"])    **2)))
    #out["ZF_re"] = float(np.sqrt(np.mean((pred["ZF_re"] - test["ZF_true_re"]) **2)))
    #out["ZF_im"] = float(np.sqrt(np.mean((pred["ZF_im"] - test["ZF_true_im"]) **2)))
    #out["ZL_re"] = float(np.sqrt(np.mean((pred["ZL_re"] - test["ZL_true_re"]) **2)))
    #out["ZL_im"] = float(np.sqrt(np.mean((pred["ZL_im"] - test["ZL_true_im"]) **2)))

    # Complex RMSEs (single number each)
    out["ZF"] = rmse_complex_parts(
        pred["ZF_re"], pred["ZF_im"], test["ZF_true_re"], test["ZF_true_im"]
    )
    out["ZL"] = rmse_complex_parts(
        pred["ZL_re"], pred["ZL_im"], test["ZL_true_re"], test["ZL_true_im"]
    )
    return out

def param_labels(target, estimate):
    """
    Returns (text_label, latex_label) for legend/title strings.
    target: "L1" | "ZF" | "ZL" | "ALL3"
    estimate: None | "real" | "imag" |
    """
    t = str(target).upper()
    e = None if estimate is None or str(estimate).lower() in {"none", ""} else str(estimate).lower()

    if t == "L1":
        return "L1", r"$L_1$"

    if t == "ZF":
        if e == "real":
            return "Re[ZF]", r"$\Re\{Z_F\}$"
        if e == "imag":
            return "Im[ZF]", r"$\Im\{Z_F\}$"
        return "ZF", r"$Z_F$"  # full complex

    if t == "ZL":
        if e == "real":
            return "Re[ZL]", r"$\Re\{Z_L\}$"
        if e == "imag":
            return "Im[ZL]", r"$\Im\{Z_L\}$"
        return "ZL", r"$Z_L$"  # full complex

    if t == "ALL3SP" or "ALL3DP":
        return "ZF,ZL,L1 (joint)", r"$\{Z_F,Z_L,L_1\}$"

    return f"{t}", f"{t}"



def main(cfg_path="configs/benchmark.yaml"):
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))
    print('Current device', device)
    # Load full arrays from the .mat (assumed in /experiments)
    mat = sio.loadmat("experiments/cable_parameter.mat")
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full    = torch.tensor(mat["Z_C"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq   = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device) 

    fstart = float(cfg["freq"]["start_hz"])
    fend   = float(cfg["freq"]["stop_hz"])
    F      = int(cfg["freq"]["num_points"])

    
    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)  # [F]
    freq_range_mhz = desired_freqs/1e6
    # For each desired f, find index of closest pul_freq
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)     # [F]

    gamma_list = gamma_full[idx]   # [F]
    Zc_list    = Zc_full[idx]      # [F]
    
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=1000.0)

    # Target + fixed params + grid
    fixed = cfg["fixed"]
    target = cfg["target"].upper()
    estimate = cfg["estimate"]

    #Parameters
    snrs = cfg["snr_dbs"]
    N_train = int(cfg["train_per_snr"])
    N_test = int(cfg["test_per_snr"])
    seed = cfg["seed"]
    
    est1 = VanillaAdam(
        fm=fm,
        likelihood=ComplexGaussianLik(),
        L = 1000.0,
        device = device,
    )
    est2 = VI(
        fm = fm,
        likelihood=ComplexGaussianLik(),
        L = 1000.0,
        device = device
    )

    # est2 = OptimizedMLE(
    #     fm=fm,
    #     likelihood=ComplexGaussianLik(),
    #     L = 1000.0,
    #     device=device
    # )

    # est3 = VIoptimized(
    #     fm = fm,
    #     likelihood=ComplexGaussianLik(),
    #     L = 1000.0,
    #     device = device
    # )

    # est4 = MLRegressor(
    #     fm = fm,
    #     likelihood=ComplexGaussianLik(),
    #     estimator_type = "rfr",
    # )



    rmse_curves = {
        #"ML_Regressors": {k: [] for k in["L1", "ZF", "ZL"]},
        "MLE_w_L1_Profile": {k: [] for k in ["L1", "ZF", "ZL"]},
        "SVI_w_mu_L1_Profile": {k: [] for k in ["L1", "ZF", "ZL"]},
    }
    
    
    crlb_curves = {k: [] for k in ["L1","ZF","ZL"]}
    for snr_db in snrs:
        print(f"Curr SNR is {snr_db}")
        #X_train [N, 2F], y_train [N, 5]
        # X_train, y_train = dm.build_or_load_train_dataset(cfg["dataset_id"], snr_db, N_train, gamma_list, Zc_list, seed = seed, 
        #                                        target="ALL3DP", fixed=fixed, gen_cfg=cfg["true_range"], freq_cfg=cfg["freq"], 
        #                                        force=cfg["force"], desired_freq = pul_freq[idx], estimate=estimate, split="train")
        test = dm.build_or_load_dataset(cfg["dataset_id"], snr_db, N_test, gamma_list, Zc_list, seed = seed, 
                                     target=target, fixed=fixed, gen_cfg=cfg["true_range"], freq_cfg=cfg["freq"], 
                                     force=cfg["force"], desired_freq=pul_freq[idx], estimate=estimate, split="test")
        h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j*torch.tensor(test["h_obs_imag"], device=device)  # [M,F]
        var = torch.tensor(test["noise_var"], device=device)   # [M,F]
        # plt.figure(figsize=(8,6))
        # h_obs_1d = h_obs.squeeze()
        # plt.plot(
        #     freq_range_mhz.cpu().numpy(),
        #     20 * torch.log10(torch.abs(h_obs_1d)).cpu().numpy(),
        #     label=r"$H_{1,1}$ (Model)"
        # )

        # plt.xscale("log")
        # plt.xlabel("Frequency (MHz)", fontsize=12)
        # plt.ylabel("Magnitude (dB)", fontsize=12)
        # plt.title(r"$H$ Transfer Function in dB", fontsize=14)
        # plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        # plt.legend(fontsize=12)
        # plt.tight_layout()
        # plt.show()
        preds_mle = est1.predict2(h_obs, var)
        rmse_mle = rmse_joint(preds_mle, test)
        print("RMSE from MLE", rmse_mle)
        
        adasd
        preds_elbo = est2.predict(h_obs, var, snr_db)
        rmse_elbo = rmse_joint(preds_elbo, test)
        print("RMSE from SVI", rmse_elbo)
        
        

        du_aug = complex_partials_fullbatch(fm, test, device)   # [N, F, 5] complex
        FIM = fim_from_complex_jac(du_aug, var) # [N, 5, 5] Augmented FIM matrix
        # CRLB = torch.linalg.inv(FIM)
        # CRLB_ZF_from_inv = CRLB[:, 0, 0] #[N] CRLBs for L1
        # CRLB_ZL_from_inv = CRLB[:, 1, 1] #[N] CRLBs for L1
        # CRLB_L1_from_inv = CRLB[:, 4, 4] #[N] CRLBs for L1
        # print("CRLB_L1_from_inv", torch.mean(CRLB_L1_from_inv))
        # print("CRLB_ZF_from_inv", torch.mean(CRLB_ZF_from_inv))
        # print("CRLB_ZL_from_inv", torch.mean(CRLB_ZL_from_inv))
        
        CRLB_L1, CRLB_ZF, CRLB_ZL = get_CRLB(FIM)
        # print("CRLB_L1", CRLB_L1)
        # Take real part first, then mean, then sqrt
        m_L1 = torch.mean(CRLB_L1.real)
        m_ZF = torch.mean(CRLB_ZF.real)
        m_ZL = torch.mean(CRLB_ZL.real)

        c = {
            "L1": torch.sqrt(m_L1),
            "ZF": torch.sqrt(m_ZF),
            "ZL": torch.sqrt(m_ZL),
        }

        # Print as floats
        print("sqrt(CRLB):", {k: float(v.detach().cpu()) for k, v in c.items()})

        # Append NumPy scalars to curves
        for k, v in c.items():
            crlb_curves.setdefault(k, []).append(v.detach().cpu().numpy().item())

        for k in ["L1", "ZF", "ZL"]:
            val = rmse_optimizedmle[k]        # this is the scalar RMSE for that param
            val2 = rmse_optimizedelbo[k]        # this is the scalar RMSE for that param
            #val3 = rmse_MLReg[k]
            if torch.is_tensor(val):          
                val = val.detach().cpu().item()
                val2 = val2.detach().cpu().item()
            #    val3 = val3.detach().cpu().item()

            print(f"Parameter is {k}, RMSE is {val}")
            rmse_curves["MLE_w_L1_Profile"][k].append(val)
            print(f"Parameter is {k}, RMSE is {val2}")
            rmse_curves["SVI_w_mu_L1_Profile"][k].append(val2)
            # print(f"Parameter is {k}, RMSE is {val3}")
            # rmse_curves["ML_Regressors"][k].append(val3)
        
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Program took {elapsed_time:.4f} seconds to run.")
    plt.figure()
    params = ["L1", "ZF", "ZL"]
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    param_color = {p: palette[i] for i, p in enumerate(params)}
    
    # RMSE curves
    for est_name, per_param in rmse_curves.items():
        for param_name, series in per_param.items():   # series is the list for 'L1' / 'ZF' / 'ZL'
            if not series:        # skip if it's still empty
                continue
            marker = "o" if "MLE" in est_name else "v"
            plt.plot(
                snrs,
                series,
                marker=marker,
                color=param_color[param_name],
                label=f"{est_name} {param_name}",
            )
            
    # sqrt(CRLB) curves
    for name, series in crlb_curves.items(): 
        plt.plot(snrs, 
                 series, 
                 marker='x', 
                 linestyle='--', 
                 label=fr'$\sqrt{{\mathrm{{CRLB}}}}$ for {name}', 
                 color=param_color[name])

    plt.xlabel("SNR (dB)")
    plt.yscale("log")
    plt.ylabel("RMSE and sqrt(CRLB)")
    plt.title(fr"Estimator RMSE vs $\sqrt{{\mathrm{{CRLB}}}}$ across SNR for (L1, ZF, ZL)")
    plt.grid(True)
    #Linear Scale
    plt.yscale("linear")
    plt.legend(
        fontsize=7,
        labelspacing=0.15,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.2,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig("rmse_vs_crlb_linear_scenarioA.png", dpi=300, bbox_inches="tight")

    #Log Scale
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("rmse_vs_crlb_log_scenarioA.png", dpi=300, bbox_inches="tight")

    #plt.show()


if __name__ == "__main__":
    main()