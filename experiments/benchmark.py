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
from data.manager import DatasetManager
from torch.func import vmap, jacrev, jacfwd
from core.crlb import complex_partials_fullbatch, fim_from_complex_jac, get_CRLB, crlb_L1_only_batch, debug_jacobian_mags, crlb_for_target_estimate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint, uniform, loguniform

def _to_c(val):
    if isinstance(val, dict):
        return complex(float(val["re"]), float(val["im"]))
    return complex(val)

@torch.no_grad()
def ll_curve_1d_real(
    fm,
    likelihood,
    obs_f,              # [F] or [N,F]; if [F] it will be expanded
    var_f,              # [F] or [N,F]; if [F] it will be expanded
    target: str,        # "L1" | "ZF" | "ZL"
    estimate: str=None, # None | "real" | "imag" (for ZF/ZL)
    grid=None,          # 1-D tensor of candidate values (float for L1/real/imag; complex for full complex)
    fixed: dict=None,   # {"L1": float, "ZF": {"re":..,"im":..}, "ZL": {...}}
):
    device = fm.gamma.device
    t = target.upper()
    e = estimate

    if t in ("ZF", "ZL") and e not in ("real", "imag"):
        raise ValueError("For ZF/ZL you must set estimate='real' or 'imag' for a 1-D plot.")
    
    # Normalize obs_f/var_f to [N,F]
    if obs_f.dim() == 1:
        obs_f = obs_f.unsqueeze(0)    # [1,F]
    if var_f.dim() == 1:
        var_f = var_f.unsqueeze(0)    # [1,F]
    K = grid.numel()
    

    # Fixed params broadcasted to [K]
    L1_fix = torch.full((K,), float(fixed["L1"]), device=device, dtype=torch.float32)
    ZF_fix = torch.full((K,), _to_c(fixed["ZF"]), device=device, dtype=torch.cfloat)
    ZL_fix = torch.full((K,), _to_c(fixed["ZL"]), device=device, dtype=torch.cfloat)

    if t == "L1":
        L1v, ZFv, ZLv = grid, ZF_fix, ZL_fix
    elif t == "ZF":
        if e == "real":
            ZFv = torch.complex(grid, ZF_fix.imag.to(torch.float32))
        else:  # "imag"
            ZFv = torch.complex(ZF_fix.real.to(torch.float32), grid)
        L1v, ZLv = L1_fix, ZL_fix
    else:  # "ZL"
        if e == "real":
            ZLv = torch.complex(grid, ZL_fix.imag.to(torch.float32))
        else:  # "imag"
            ZLv = torch.complex(ZL_fix.real.to(torch.float32), grid)
        L1v, ZFv = L1_fix, ZF_fix

    # Forward over all candidates -> [K,F]
    H = fm.compute_H_complex(L1=L1v, ZF=ZFv, ZL=ZLv)  # [K,F]
    ll = likelihood.score_matrix(obs_f, H, var_f) # -> [K]; sums over N,F inside

    idx_hat = torch.argmax(ll).item()
    x = grid.detach().cpu().numpy()
    x_hat = float(x[idx_hat])

    return x, ll.detach().cpu().numpy(), x_hat, idx_hat


def complex_from_dict(d):
    if isinstance(d, dict):
        return complex(float(d["re"]), float(d["im"]))
    return complex(float(d), 0.0)

def build_grid(cfg_grid, target, device, estimate=None):
    """
    estimate: None | "real" | "imag"
      - None  -> return 1-D complex cfloat candidates (full complex search)
      - "real"-> return 1-D float32 candidates for Re only
      - "imag"-> return 1-D float32 candidates for Im only
    """
    t = target.upper()
    if t == "L1":
        g = cfg_grid["L1"]
        return torch.linspace(g["min"], g["max"], g["num"], device=device, dtype=torch.float32)

    if t in ("ZF", "ZL"):
        g = cfg_grid[t]
        if estimate == "real":
            re = torch.linspace(g["re"]["min"], g["re"]["max"], g["re"]["num"], device=device, dtype=torch.float32)
            return re  # float grid of Re candidates
        if estimate == "imag":
            im = torch.linspace(g["im"]["min"], g["im"]["max"], g["im"]["num"], device=device, dtype=torch.float32)
            return im  # float grid of Im candidates
        # Full complex grid (flattened 2D -> 1D complex)
        re = torch.linspace(g["re"]["min"], g["re"]["max"], g["re"]["num"], device=device, dtype=torch.float32)
        im = torch.linspace(g["im"]["min"], g["im"]["max"], g["im"]["num"], device=device, dtype=torch.float32)
        return (re.unsqueeze(1) + 1j * im.unsqueeze(0)).reshape(-1).to(torch.cfloat)

    raise ValueError(f"Unknown target {target}")

def rmse_for_target(preds, test, target, estimate):
    """
    preds: np.array [N] (float32 for L1, ZF/ZL real/im only, cfloat for ZF/ZL)
    test:  dict from NPZ
    target: "L1" | "ZF" | "ZL"
    estimate: None | "real" | "imag"
    """
    if target == "L1":
        y = test["L1_true"]                      # [N], real
        return float(np.sqrt(np.mean((preds - y)**2)))
    else: 
        if estimate == "real":
            y = test[f"{target}_true_re"]
            return float(np.sqrt(np.mean((preds - y)**2)))
        elif estimate == "imag":
            y = test[f"{target}_true_im"]
            return float(np.sqrt(np.mean((preds - y)**2)))
        else:
            y = test[f"{target}_true_re"] + 1j * test[f"{target}_true_im"]
            return float(np.sqrt(np.mean(np.abs(preds - y)**2)))


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

    if t == "ALL3":
        return "ZF,ZL,L1 (joint)", r"$\{Z_F,Z_L,L_1\}$"

    return f"{t}", f"{t}"

def range_scale(true_range, target, estimate=None):
    """Return s_k = (max - min) for the requested targeted parameter/component."""
    t = str(target).upper()
    if t == "L1":
        return float(true_range["L1"]["max"] - true_range["L1"]["min"])

    sec = true_range[t]
    if estimate == "real":
        return float(sec["re"]["max"] - sec["re"]["min"])
    if estimate == "imag":
        return float(sec["im"]["max"] - sec["im"]["min"])
    # full complex (not used in this “component” comparison)
    re_w = float(sec["re"]["max"] - sec["re"]["min"])
    im_w = float(sec["im"]["max"] - sec["im"]["min"])
    # if ever needed: a single scale for complex could be sqrt(re_w^2 + im_w^2) or max(re_w, im_w)
    return max(re_w, im_w)

def get_true_param_for_sample(test, target: str, estimate: str | None, i: int = 0) -> float:
    tgt = target.upper()
    if tgt == "L1":
        return float(test["L1_true"][i])
    elif tgt == "ZF":
        if estimate == "real":
            return float(test["ZF_true_re"][i])
        elif estimate == "imag":
            return float(test["ZF_true_im"][i])
        else:
            raise ValueError("1-D LL curve needs a single real dimension (use estimate='real' or 'imag').")
    elif tgt == "ZL":
        if estimate == "real":
            return float(test["ZL_true_re"][i])
        elif estimate == "imag":
            return float(test["ZL_true_im"][i])
        else:
            raise ValueError("1-D LL curve needs a single real dimension (use estimate='real' or 'imag').")
    else:
        raise ValueError(f"Unknown target: {target}")
def bounds_from_grid(cfg_grid, target, estimate):
    t = target.upper()
    if t == "L1":
        g = cfg_grid["L1"]
        return float(g["min"]), float(g["max"])
    sec = cfg_grid[t]
    if estimate == "real":
        return float(sec["re"]["min"]), float(sec["re"]["max"])
    elif estimate == "imag":
        return float(sec["im"]["min"]), float(sec["im"]["max"])
    else:
        raise ValueError("ELBOArgmaxMu1D is 1-D; for ZF/ZL set estimate='real' or 'imag'.")

def sigma_from_snr(snr_db, *, sigma_max, sigma_min, alpha, ref_db):
    sigma = sigma_max * 10 ** (-alpha * (snr_db - ref_db) / 20.0)
    return float(np.clip(sigma, sigma_min, sigma_max))

def plot_log_likelihood(fm, likelihood, h_obs, var, target, estimate, grid, fixed, true_val, snr_db):
    # 1-D LL curve w.r.t. target parameter
    xs, ll, xhat, _ = ll_curve_1d_real(
        fm, likelihood, h_obs, var,
        target=target, estimate=estimate,
        grid=grid, fixed=fixed
    )

    # nice axis/label per parameter
    axis_label = {
        ("L1", None):        "L1",
        ("ZF", "real"):      "Re{ZF}",
        ("ZF", "imag"):      "Im{ZF}",
        ("ZL", "real"):      "Re{ZL}",
        ("ZL", "imag"):      "Im{ZL}",
    }.get((target.upper(), estimate), f"{target} ({estimate})")

    plt.figure()
    plt.plot(xs, ll, label=f"{snr_db} dB")
    plt.axvline(true_val, linestyle="--", color="red",   label=f"true {axis_label} = {true_val:g}")
    plt.axvline(xhat,     linestyle="--", color="green", label=f"MLE = {xhat:.3g}")
    plt.xlabel(axis_label)
    plt.ylabel("log-likelihood")
    plt.title(f"Log-likelihood vs {axis_label} @ {snr_db} dB")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

def make_X_from_test(test_dict, snr_db=None, include_snr_feature=True):
    ri = np.stack([test_dict["h_obs_real"], test_dict["h_obs_imag"]], axis=2)  # [N,F,2]
    X  = ri.reshape(ri.shape[0], -1).astype(np.float32)                         # [N,2F]
    if include_snr_feature:
        snr_col = np.full((X.shape[0], 1), float(snr_db), dtype=np.float32)
        X = np.concatenate([X, snr_col], axis=1)
    return X


def main(cfg_path="configs/benchmark.yaml"):
    start_time = time.perf_counter()
    torch.set_printoptions(precision=8, sci_mode=False)
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))
    print('device', device)
    # --- Load full arrays from the .mat (assumed in /experiments) ---
    mat = sio.loadmat("experiments/cable_parameter.mat")
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    Zc_full    = torch.tensor(mat["Z_C"].squeeze(),   dtype=torch.cfloat, device=device)
    pul_freq   = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device) 

    fstart = float(cfg["freq"]["start_hz"])
    fend   = float(cfg["freq"]["stop_hz"])
    F      = int(cfg["freq"]["num_points"])

    
    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)  # [F]
    
    # For each desired f, find index of closest pul_freq
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)     # [F]

    gamma_list = gamma_full[idx]   # [F]
    Zc_list    = Zc_full[idx]      # [F]
    
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=1000.0)

    # Target + fixed params + grid
    fixed = cfg["fixed"]
    target = cfg.get("target").upper()
    estimate = cfg["estimate"]
    text_param, latex_param = param_labels(target, estimate)
    est_key = f"MLE H(f) Likelihood for {text_param}"
    grid = build_grid(cfg["grid"], target, device, estimate=estimate)  #MLE Grid Search
    

    # VI config
    mu_grid = torch.linspace(float(cfg["vi_elbo1d"]["mu_grid"]["min"]), float(cfg["vi_elbo1d"]["mu_grid"]["max"]), int(cfg["vi_elbo1d"]["mu_grid"]["num"]),
                             device=device, dtype=torch.float32)
    fixed_sigma = float(cfg["vi_elbo1d"]["fixed_sigma"])       
    M = int(cfg["vi_elbo1d"]["M"])                   
    K = int(cfg["vi_elbo1d"]["K"])                   
    bounds = bounds_from_grid(cfg["grid"], target, estimate)

    #Parameters
    snrs = cfg["snr_dbs"]
    N_train = int(cfg["train_per_snr"])
    N_test = int(cfg["test_per_snr"])
    true_rng = cfg["true_range"]
    seed = cfg["seed"]
    sched = cfg["vi_elbo1d"]["sigma_schedule"]
    plot_likelihood = cfg["plot_loglikelihood"]

    #ML Training Stage
    X_train, y_train = dm.build_or_load_pooled_dataset(
    dataset_id=cfg["dataset_id"],
    snr_list=snrs, N_per_snr=N_train,
    gamma=fm.gamma, Zc=fm.Zc, L=fm.L,
    seed=cfg["seed"], target=target, fixed=fixed,
    gen_cfg=cfg["grid"], freq_cfg=cfg["freq"],
    force=cfg["force"], estimate=estimate, split="train", include_snr_feature=True
)

    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        max_features='sqrt',
        random_state=42,
        verbose=1
    )
    rf  = RandomForestRegressor(
        n_estimators=500,
        max_features='sqrt',
        random_state=42,
        verbose=1
    )

    # print("Training GBR...")
    # gbr.fit(X_train, y_train)
    # print("Training RF...")
    # rf.fit(X_train, y_train)
    # print("Done Training...")
    # ml_models = {f"GBR (ML) for {text_param}": gbr, f"RF (ML) for {text_param}": rf}


    # Estimators
    estimators = {
    est_key: GridSearchMLE(
        fm, ComplexGaussianLik(),
        grid=grid, target=target, fixed=fixed, device=device, estimate=estimate
    ),
    f"ELBO-argmax for ({text_param})": ELBOArgmaxMu1D(
        fm=fm, target=target, estimate=estimate,
        mu_grid=mu_grid, fixed_sigma=fixed_sigma,
        bounds=bounds, fixed=fixed, M=M, K=K, device=device
    ),
}

    # # 5 tasks: (target, estimate, label)
    # tasks = [
    #     ("L1", None,        r"$L_1$"),
    #     ("ZF", "real",      r"$\Re\{Z_F\}$"),
    #     ("ZF", "imag",      r"$\Im\{Z_F\}$"),
    #     ("ZL", "real",      r"$\Re\{Z_L\}$"),
    #     ("ZL", "imag",      r"$\Im\{Z_L\}$"),
    # ]


    # results = {lab: {"nrmse": [], "nsqrtcrlb": []} for _, _, lab in tasks}

    # # color map: one color per parameter
    # color_for = {
    #     r"$L_1$":        "C0",
    #     r"$\Re\{Z_F\}$": "C2",
    #     r"$\Im\{Z_F\}$": "C3",
    #     r"$\Re\{Z_L\}$": "C4",
    #     r"$\Im\{Z_L\}$": "C5",
    # }

    # for (tgt, est, lab) in tasks:
    #     # scale for normalization
    #     s_k = range_scale(true_rng, tgt, est)

    #     # Build a search grid for THIS task
    #     grid_k = build_grid(cfg["grid"], tgt, device, estimate=est)

    #     # Estimator (others fixed to cfg["fixed"])
    #     est_key = f"MLE H(f) for {lab}"
    #     estimator = GridSearchMLE(
    #         fm, ComplexGaussianLik(), grid=grid_k,
    #         target=tgt, fixed=fixed, device=device, estimate=est
    #     )

    #     for snr_db in snrs:
    #         # Generate/Load test set for THIS task.
    #         # IMPORTANT: use gen_cfg=true_range (not the search grid).
    #         # To keep caches disjoint across tasks, vary dataset_id or force=True.
    #         test = dm.build_or_load_test(
    #             dataset_id = cfg["dataset_id"] + f"_{tgt}_{est or 'full'}",
    #             snr_db = snr_db, N = N, gamma = fm.gamma, Zc = fm.Zc, L = fm.L,
    #             seed = cfg["seed"], target = tgt, fixed = fixed,
    #             gen_cfg = true_rng, freq_cfg = cfg["freq"],
    #             force = cfg["force"], desired_freq = None, estimate = est
    #         )

    #         h_obs = (torch.tensor(test["h_obs_real"], device=device)
    #                  + 1j*torch.tensor(test["h_obs_imag"], device=device))  # [N,F]
    #         varNF = torch.tensor(test["noise_var"], device=device)          # [N,F]

    #         du_aug = complex_partials_fullbatch(fm, test, device)           # [N,F,5]
    #         crlb_i = crlb_for_target_estimate(du_aug, varNF, target=tgt, estimate=est)  # [N]
    #         nsqrtcrlb = float(np.sqrt(crlb_i.mean().item()) / s_k)

    #         # MLE predictions and component RMSE
    #         preds = estimator.predict_batch(h_obs, varNF)  # np array
    #         rmse_val = rmse_for_target(preds, test, tgt, est)
    #         nrmse    = rmse_val / s_k

    #         results[lab]["nsqrtcrlb"].append(nsqrtcrlb)
    #         results[lab]["nrmse"].append(nrmse)
    #         print(f"[{lab}] SNR={snr_db:>2}dB  NRMSE={nrmse:.4g}  N√CRLB={nsqrtcrlb:.4g}")
    

    #  # ------------ Plot (percent of true range) -------------
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # style legend (solid = NRMSE, dashed = N√CRLB)
    # style_handles = [
    #     Line2D([0],[0], color="black", linestyle="-",  marker="o", label="NRMSE"),
    #     Line2D([0],[0], color="black", linestyle="--", marker="x", label="N√CRLB"),
    # ]

    # for (_, _, lab) in tasks:
    #     c = color_for[lab]
    #     y_nrmse_pct   = [100.0*x for x in results[lab]["nrmse"]]
    #     y_ncrlb_pct   = [100.0*x for x in results[lab]["nsqrtcrlb"]]
    #     ax.plot(snrs, y_nrmse_pct, color=c, linestyle="-",  marker="o", label=lab)  # solid: NRMSE
    #     ax.plot(snrs, y_ncrlb_pct, color=c, linestyle="--", marker="x")             # dashed: N√CRLB

    # ax.set_yscale("log")
    # ax.set_xlabel("SNR (dB)")
    # ax.set_ylabel("Error (% of true range)")
    # ax.set_title("Normalized RMSE vs Normalized √CRLB\n(L1, Re/Im ZF, Re/Im ZL)")

    # # Legends: colors -> parameter, line style -> metric
    # legend_params = ax.legend(title="Parameter", loc="upper right")
    # ax.add_artist(legend_params)
    # ax.legend(handles=style_handles, title="Metric", loc="lower left")

    # # percentage tick labels (values already in %)
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))

    # ax.grid(True, which="both", ls=":", alpha=0.6)
    # fig.tight_layout()
    # plt.show()
    

    rmse = {}
    rmse.update({name: [] for name in estimators.keys()})
    # rmse.update({name: [] for name in ml_models.keys()})
    crlb_line = []


    for snr_db in snrs:
        test = dm.build_or_load_dataset(cfg["dataset_id"], snr_db, N_test, gamma_list, Zc_list, seed = seed, 
                                     target=target, fixed=fixed, gen_cfg=cfg["true_range"], freq_cfg=cfg["freq"], 
                                     force=cfg["force"], desired_freq=pul_freq[idx], estimate=estimate, split="test")
        h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j*torch.tensor(test["h_obs_imag"], device=device)  # [N,F]
        var = torch.tensor(test["noise_var"], device=device)   # [N,F]

        #ML Models Prediction
        # X_test = make_X_from_test(test, snr_db=snr_db, include_snr_feature=True)
        # for name, model in ml_models.items():
        #     y_pred = model.predict(X_test).astype(np.float32)
        #     e = rmse_for_target(y_pred, test, target, estimate)
        #     rmse[name].append(e)
        #     print(f"SNR {snr_db:>2} dB | {name} RMSE = {e:.3f}")
        #Plot Log likelihood
        # if(plot_likelihood):
        #     true_val = get_true_param_for_sample(test, target=target, estimate=estimate, i=0)
        #     plot_log_likelihood(fm, ComplexGaussianLik(), h_obs[0], var[0], 
        #                         target=target, estimate=estimate, grid=grid, fixed=cfg["fixed"], true_val=true_val, snr_db = snr_db)
            
        #CRLB Calculations
        du_aug = complex_partials_fullbatch(fm, test, device)   # [N, F, 5] complex
        # pick the matching CRLB per (target, estimate)
        #_, CRLB_L1_tight = crlb_L1_only_batch(fm, test, var)
        #print("sqrt CRLB should match:", np.sqrt(CRLB_L1_tight.mean().item()))
        crlb_per_sample = crlb_for_target_estimate(du_aug, var, target=target, estimate=estimate)  # [N]
        sqrt_crlb_mean = np.sqrt(crlb_per_sample.mean().item())
        print("sqrt CRLB:", sqrt_crlb_mean)
        crlb_line.append(sqrt_crlb_mean) 


        #MLE and ELBO
        for name, est in estimators.items():
            #Variational distribution sigma scheduling
            if isinstance(est, ELBOArgmaxMu1D):
                est.fixed_sigma = sigma_from_snr(
                snr_db,
                sigma_max=sched["sigma_max"],
                sigma_min=sched["sigma_min"],
                alpha=sched["alpha"],
                ref_db=sched["ref_db"],
            )
                print("Curr Sigma", est.fixed_sigma)
            preds = est.predict(h_obs, var)        # numpy array [N] predict batchwise to save on memory
            #z_hat = preds
            #z_true = test[f"{target}_true_re"] + 1j*test[f"{target}_true_im"]
            #print(f"Current SNR is {snr_db}") 
            #print("zhat", z_hat)
            #print("ztrue", z_true)
            # err   = z_hat - z_true
            # mse   = np.mean(np.abs(err)**2)
            # bias  = np.mean(err)              # complex
            # variance   = np.mean(np.abs(err - bias)**2)
            # print(f"MSE={mse:.3g},  Var={variance:.3g},  |Bias|^2={(abs(bias)**2):.3g},  Var+Bias^2={variance + abs(bias)**2:.3g}")

            e = rmse_for_target(preds, test, target, estimate)
            rmse[name].append(e)
            print(f"SNR {snr_db:>2} dB | {name} RMSE = {e:.3f}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Program took {elapsed_time:.4f} seconds to run.")
    plt.figure()
    for name, series in rmse.items():
        plt.plot(snrs, series, marker='o', label=name)
    plt.plot(snrs, crlb_line, marker='x', linestyle='--', label=fr'$\sqrt{{\mathrm{{CRLB}}}}$ for {latex_param}')
    plt.xlabel("SNR (dB)")
    plt.yscale("log")
    plt.ylabel("RMSE / sqrt(CRLB)")
    plt.title(fr"Estimator RMSE vs $\sqrt{{\mathrm{{CRLB}}}}$ across SNR for {latex_param}")
    plt.grid(True)
    plt.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()