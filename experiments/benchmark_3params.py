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
from estimators.mle_optimized import ContinuousMLE
from data.manager import DatasetManager
from torch.func import vmap, jacrev, jacfwd
from core.crlb import complex_partials_fullbatch, fim_from_complex_jac, get_CRLB, crlb_L1_only_batch, debug_jacobian_mags, crlb_for_target_estimate
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint, uniform, loguniform
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_rmse_vs_restarts_40db(
    dm,
    cfg,
    fm,
    gamma_list,
    Zc_list,
    desired_freq,                 # e.g. pul_freq[idx] from your main()
    device,
    restarts_list=(1, 5, 10, 20, 30, 50),
    adam_steps=400,
    adam_lr=1e-3,
    lbfgs_steps=150,
):
    """
    Fix SNR=40 dB. For each S in restarts_list:
      - run ContinuousMLE with n_starts=S on the SAME test set
      - compute RMSE for {L1, ZF_re, ZF_im, ZL_re, ZL_im}
    Then plot RMSE vs S.
    """
    snr_db = 40
    N_test = int(cfg["test_per_snr"])
    seed    = cfg["seed"]
    target  = cfg["target"].upper()

    # --- Build a SINGLE test set we will reuse for all S ---
    test = dm.build_or_load_dataset(
        cfg["dataset_id"], snr_db, N_test,
        gamma_list, Zc_list,
        seed=seed,
        target=target,
        fixed=cfg["fixed"],
        gen_cfg=cfg["true_range"],
        freq_cfg=cfg["freq"],
        force=cfg["force"],
        desired_freq=desired_freq,
        estimate=cfg["estimate"],
        split="test",
    )

    # Observations and noise
    h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j * torch.tensor(test["h_obs_imag"], device=device)  # [N,F]
    var   = torch.tensor(test["noise_var"], device=device)                                                          # [N,F]

    # Storage for curves
    params = ["L1","ZF_re","ZF_im","ZL_re","ZL_im"]
    rmse_curves = {k: [] for k in params}

    for S in restarts_list:
        # Fresh estimator with the chosen number of restarts
        est = ContinuousMLE(
            fm=fm,
            likelihood=ComplexGaussianLik(),
            L=float(fm.L) if hasattr(fm, "L") else 1000.0,
            device=device,
            n_starts=int(S),
            adam_steps=int(adam_steps),
            adam_lr=float(adam_lr),
            use_lbfgs=True,
            lbfgs_steps=int(lbfgs_steps),
            verbose=True,                # keep output clean
        )

        # Predict on the fixed test set
        preds = est.predict(
            h_obs, var,
            test["L1_true"],
            test["ZF_true_re"], test["ZF_true_im"],
            test["ZL_true_re"], test["ZL_true_im"]
        )

        # Compute RMSE for this S (uses your existing helper)
        r = rmse_joint(preds, test)   # returns dict with keys in `params`
        for k in params:
            rmse_curves[k].append(r[k])

        print(f"S={S:>3}: " +
              ", ".join([f"{k} RMSE={rmse_curves[k][-1]:.3f}" for k in params]))

    # --- Plot ---
    plt.figure(figsize=(8,5))
    for k in params:
        plt.plot(restarts_list, rmse_curves[k], marker="o", label=k)
    #plt.xscale("log")            # restart counts often span decades
    #plt.yscale("log")            # RMSE typically shrinks roughly exponentially with S
    plt.xlabel("Number of random restarts (S)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs number of restarts at 40 dB SNR")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.show()

def rmse_complex_parts(pred_re, pred_im, true_re, true_im):
    # Euclidean error in C
    mse = np.mean((pred_re - true_re)**2 + (pred_im - true_im)**2)
    return float(np.sqrt(mse))

def rmse_joint(pred, test):
    # pred is dict from estimator; test is your NPZ dict
    out = {}
    out["L1"]    = float(np.sqrt(np.mean((pred["L1"]    - test["L1_true"])    **2)))
    out["ZF_re"] = float(np.sqrt(np.mean((pred["ZF_re"] - test["ZF_true_re"]) **2)))
    out["ZF_im"] = float(np.sqrt(np.mean((pred["ZF_im"] - test["ZF_true_im"]) **2)))
    out["ZL_re"] = float(np.sqrt(np.mean((pred["ZL_re"] - test["ZL_true_re"]) **2)))
    out["ZL_im"] = float(np.sqrt(np.mean((pred["ZL_im"] - test["ZL_true_im"]) **2)))

    # # Complex RMSEs (single number each)
    out["ZF_complex"] = rmse_complex_parts(
        pred["ZF_re"], pred["ZF_im"], test["ZF_true_re"], test["ZF_true_im"]
    )
    out["ZL_complex"] = rmse_complex_parts(
        pred["ZL_re"], pred["ZL_im"], test["ZL_true_re"], test["ZL_true_im"]
    )
    return out


def plot_L1_sweeps_across_snrs(est, dm, cfg, gamma_list, Zc_list, snrs,
                               device, example_idx=0, num_L1=600):
    """
    For each SNR in `snrs`, take one observation (example_idx),
    hold ZF,ZL at that observation's *true* values, and sweep L1 in [0, L].
    Plot NLL(L1) in up to 9 subplots (3x3).
    """
    import math
    # Panels: up to 9
    n_plot = min(9, len(snrs))
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    # L1 grid
    Lmin, Lmax = 0.0, float(est.L)
    L1_grid = torch.linspace(Lmin, Lmax, num_L1, device=device, dtype=torch.float32)

    for si, snr_db in enumerate(snrs[:n_plot]):
        # Build/load dataset for this SNR
        test = dm.build_or_load_dataset(
            cfg["dataset_id"], snr_db, int(cfg["test_per_snr"]),
            gamma_list, Zc_list, seed=cfg["seed"],
            target=cfg["target"].upper(), fixed=cfg["fixed"],
            gen_cfg=cfg["true_range"], freq_cfg=cfg["freq"],
            force=cfg["force"], desired_freq=gamma_list.new_tensor([]),  # not used
            estimate=cfg["estimate"], split="test"
        )

        # Grab obs + noise for the chosen example
        h_obs = torch.tensor(test["h_obs_real"], device=device) \
              + 1j * torch.tensor(test["h_obs_imag"], device=device)   # [N,F]
        var   = torch.tensor(test["noise_var"], device=device)          # [N,F]

        i = min(example_idx, h_obs.shape[0]-1)
        y_i = h_obs[i]      # [F] complex
        v_i = var[i]        # [F] float

        # True ZF, ZL, L1 for this example
        ZF_true = torch.complex(
            torch.tensor(float(test["ZF_true_re"][i]), device=device, dtype=torch.float32),
            torch.tensor(float(test["ZF_true_im"][i]), device=device, dtype=torch.float32)
        )
        ZL_true = torch.complex(
            torch.tensor(float(test["ZL_true_re"][i]), device=device, dtype=torch.float32),
            torch.tensor(float(test["ZL_true_im"][i]), device=device, dtype=torch.float32)
        )
        L1_true = float(test["L1_true"][i])

        # Sweep NLL over L1 with ZF,ZL fixed at truth
        nll_vals = []
        with torch.no_grad():
            for L1_val in L1_grid:
                u = est._theta_to_u(L1_val, ZF_true, ZL_true)  # turn (L1, ZF_true, ZL_true) -> u
                nll = est._nll_one(y_i, v_i, u)                # scalar tensor
                nll_vals.append(float(nll.item()))
        nll_vals = torch.tensor(nll_vals).cpu().numpy()

        # Plot
        ax = axes[si]
        ax.plot(L1_grid.cpu().numpy(), nll_vals, label="NLL vs L1 (ZF,ZL at truth)")
        ax.axvline(L1_true, color="red", linestyle="--", linewidth=1.5, label="true L1")
        ax.set_title(f"SNR = {snr_db} dB")
        ax.set_xlabel("L1")
        ax.set_ylabel("NLL")
        ax.grid(True)
        ax.legend(fontsize=8)

    # Hide any unused axes (e.g., if len(snrs) < 9)
    for ax in axes[n_plot:]:
        ax.set_visible(False)

    fig.tight_layout()


    fig.suptitle("NLL vs $L_1$ with $Z_F,Z_L$ fixed at truth (one example per SNR)", y=1.02)
    plt.tight_layout()
    plt.show()

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

def _range_to_sigmoid_x(x: torch.Tensor, lo: float, hi: float, eps: float = 1e-6) -> torch.Tensor:
    # Inverse of _sigmoid_to_range (for seeding)
    z = ((x - lo) / (hi - lo)).clamp(eps, 1.0 - eps)  # avoid 0/1
    return torch.log(z) - torch.log1p(-z)  # logit

def verify_L1_peak_stability_subplots(
    est,
    dm, cfg,
    gamma_list, Zc_list,
    device,
    example_idx=0,
    num_L1=800,
    n_random=4,
    snr_db=40
):
    import numpy as np
    import torch, matplotlib.pyplot as plt
    from math import ceil

    # --- One test sample at the chosen SNR ---
    test = dm.build_or_load_dataset(
        cfg["dataset_id"], snr_db, int(cfg["test_per_snr"]),
        gamma_list, Zc_list,
        seed=cfg["seed"],
        target=cfg["target"].upper(),
        fixed=cfg["fixed"],
        gen_cfg=cfg["true_range"],
        freq_cfg=cfg["freq"],
        force=cfg["force"],
        desired_freq=gamma_list.new_tensor([]),
        estimate=cfg["estimate"],
        split="test"
    )
    h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j*torch.tensor(test["h_obs_imag"], device=device)
    var   = torch.tensor(test["noise_var"],  device=device)
    i = min(example_idx, h_obs.shape[0]-1)
    y_i, v_i = h_obs[i], var[i]

    L1_true = float(test["L1_true"][i])
    ZF_true = torch.complex(
        torch.tensor(float(test["ZF_true_re"][i]), device=device, dtype=torch.float32),
        torch.tensor(float(test["ZF_true_im"][i]), device=device, dtype=torch.float32)
    )
    ZL_true = torch.complex(
        torch.tensor(float(test["ZL_true_re"][i]), device=device, dtype=torch.float32),
        torch.tensor(float(test["ZL_true_im"][i]), device=device, dtype=torch.float32)
    )

    # --- L1 grid ---
    Lmin, Lmax = est.L1_lo, est.L1_hi
    L1_grid = torch.linspace(Lmin, Lmax, num_L1, device=device, dtype=torch.float32)

    # Helper: build uZ (u[1:]) from physical ZF with ZL fixed at truth
    def uZ_from_ZF_with_true_ZL(ZF, L1_for_inverse=None):
        if L1_for_inverse is None:
            L1_for_inverse = torch.tensor((Lmin + Lmax) / 2, device=device, dtype=torch.float32)
        u_all = est._theta_to_u(L1_for_inverse, ZF, ZL_true)
        return u_all[1:].detach().clone()  # u[1:] corresponds to ZF,ZL

    # ZF choices (ZL always fixed to ZL_true)
    ReZF_mid = (est.ReZF_lo + est.ReZF_hi)/2
    ZF_mid = torch.complex(torch.tensor(ReZF_mid, device=device), torch.tensor(0.0, device=device))
    ZF_wall = torch.complex(torch.tensor(est.ReZF_hi, device=device),
                            torch.tensor(est.ImZF_max, device=device))

    # Random interior ZF
    def rand_interior(lo, hi, margin_frac=0.05):
        span = hi - lo
        lo2, hi2 = lo + margin_frac*span, hi - margin_frac*span
        return lo2 + (hi2 - lo2) * torch.rand((), device=device)

    rand_ZF_list = []
    for _ in range(n_random):
        ReZF_r = rand_interior(est.ReZF_lo, est.ReZF_hi)
        ImZF_r = (2*torch.rand((), device=device)-1) * (0.9*est.ImZF_max)
        rand_ZF_list.append(torch.complex(ReZF_r, ImZF_r))

    # Scenarios (ZL fixed = true)
    scenarios = []
    scenarios.append(("True ZF (ZL fixed true)", uZ_from_ZF_with_true_ZL(ZF_true)))
    scenarios.append(("Mid-box ZF (ZL fixed true)", uZ_from_ZF_with_true_ZL(ZF_mid)))
    scenarios.append(("Wall ZF (ZL fixed true)", uZ_from_ZF_with_true_ZL(ZF_wall)))
    for j, ZF_r in enumerate(rand_ZF_list):
        scenarios.append((f"Random ZF #{j+1} (ZL fixed true)", uZ_from_ZF_with_true_ZL(ZF_r)))

    # Compute NLL(L1) per scenario (ZF varying, ZL fixed)
    curves = []
    with torch.no_grad():
        for name, uZ in scenarios:
            nll_vals = []
            for L1_val in L1_grid:
                # u0 depends only on L1; using true Z here is fine to get the right transform element
                u0 = est._theta_to_u(L1_val, ZF_true, ZL_true)[0]
                u = torch.cat([u0.view(1), uZ], dim=0)
                nll_vals.append(float(est._nll_one(y_i, v_i, u).item()))
            curves.append((name, np.array(nll_vals)))

    # Print argmins
    L1_grid_np = L1_grid.detach().cpu().numpy()
    print("\nArgmin L1 per scenario (meters), ZL fixed to truth:")
    for name, nll_vals in curves:
        idx = int(np.argmin(nll_vals))
        L1_hat = float(L1_grid_np[idx])
        print(f"{name:>28s}: L1_hat = {L1_hat:.2f}   (|Δ| = {abs(L1_hat - L1_true):.2f}),  NLL_min = {nll_vals[idx]:.2f}")
    print(f"True L1: {L1_true:.2f}\n")

    # ---- Separate subplots ----
    n = len(curves)
    if n <= 8:
        rows, cols = 2, int(ceil(n/2))
    else:
        cols, rows = 3, int(ceil(n/3))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.6*rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    for ax_idx, (name, nll_vals) in enumerate(curves):
        ax = axes[ax_idx]
        ax.plot(L1_grid_np, nll_vals, linewidth=1.2)
        imin = int(np.argmin(nll_vals))
        ax.axvline(L1_true, linestyle="--", linewidth=1.2, label="true L1")
        ax.plot(L1_grid_np[imin], nll_vals[imin], marker="o", ms=4, label="argmin")
        ax.set_title(name)
        ax.grid(True, alpha=0.4)
        if ax_idx % cols == 0:
            ax.set_ylabel("NLL")
        ax.legend(fontsize="x-small")

    for ax in axes[len(curves):]:
        ax.set_visible(False)

    fig.suptitle(f"NLL vs L1 with ZL fixed at truth (SNR={snr_db} dB, sample {i})", y=0.98)
    for ax in axes[-cols:]:
        ax.set_xlabel("L1 (m)")
    plt.tight_layout()
    plt.show()


def plot_ZF_sweeps_side_by_side_40dB(
    est, dm, cfg, gamma_list, Zc_list,
    device, example_idx=0, num_ZF=600, n_random=3, save_path=None
):
    """
    One wide figure with subplots placed side-by-side (1 row).
    Each subplot corresponds to a different L1 scenario:
      [L1=true, L1=mid, L1=wall, L1=random #1..n_random]
    """
    import numpy as np
    import torch, matplotlib.pyplot as plt

    # Which ZF component to sweep
    estimate = str(cfg.get("estimate", "real")).lower()
    if estimate not in ("real", "imag"):
        estimate = "real"

    # --- Load one test split at SNR=40 dB and pick a sample
    snr_db = 40
    test = dm.build_or_load_dataset(
        cfg["dataset_id"], snr_db, int(cfg["test_per_snr"]),
        gamma_list, Zc_list,
        seed=cfg["seed"],
        target=cfg["target"].upper(),
        fixed=cfg["fixed"],
        gen_cfg=cfg["true_range"],
        freq_cfg=cfg["freq"],
        force=cfg["force"],
        desired_freq=gamma_list.new_tensor([]),
        estimate=cfg["estimate"],
        split="test"
    )
    h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j*torch.tensor(test["h_obs_imag"], device=device)
    var   = torch.tensor(test["noise_var"],  device=device)
    i = min(example_idx, h_obs.shape[0]-1)
    y_i, v_i = h_obs[i], var[i]

    # Truths
    L1_true = float(test["L1_true"][i])
    ZF_true = torch.complex(
        torch.tensor(float(test["ZF_true_re"][i]), device=device, dtype=torch.float32),
        torch.tensor(float(test["ZF_true_im"][i]), device=device, dtype=torch.float32)
    )
    ZL_true = torch.complex(
        torch.tensor(float(test["ZL_true_re"][i]), device=device, dtype=torch.float32),
        torch.tensor(float(test["ZL_true_im"][i]), device=device, dtype=torch.float32)
    )
    Re_true = float(torch.real(ZF_true).item())
    Im_true = float(torch.imag(ZF_true).item())

    # L1 scenarios
    L1_lo, L1_hi = float(est.L1_lo), float(est.L1_hi)
    L1_mid  = 0.5 * (L1_lo + L1_hi)
    L1_wall = L1_hi

    def rand_L1(margin_frac=0.05):
        span = L1_hi - L1_lo
        lo2, hi2 = L1_lo + margin_frac*span, L1_hi - margin_frac*span
        return float(lo2 + (hi2 - lo2) * torch.rand((), device=device))

    scenarios = [("L1 true", L1_true), ("L1 mid", L1_mid), ("L1 wall", L1_wall)]
    for r in range(n_random):
        scenarios.append((f"L1 random #{r+1}", rand_L1()))

    # ZF grid along chosen component
    Re_lo, Re_hi = float(est.ReZF_lo), float(est.ReZF_hi)
    Im_max = float(est.ImZF_max)
    if estimate == "real":
        ZF_grid = torch.linspace(Re_lo, Re_hi, num_ZF, device=device, dtype=torch.float32)
        grid_label = "Re{Z_F} (Ω)"
        true_comp  = Re_true
    else:
        ZF_grid = torch.linspace(-Im_max, Im_max, num_ZF, device=device, dtype=torch.float32)
        grid_label = "Im{Z_F} (Ω)"
        true_comp  = Im_true
    ZF_grid_np = ZF_grid.detach().cpu().numpy()

    # Figure: 1 row, N columns
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(4.8*n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]

    # Compute & plot per scenario
    for ax, (name, L1_val) in zip(axes, scenarios):
        nll_vals = []
        with torch.no_grad():
            for zval in ZF_grid:
                if estimate == "real":
                    ZF_cand = torch.complex(zval.to(torch.float32), torch.tensor(Im_true, device=device))
                else:
                    ZF_cand = torch.complex(torch.tensor(Re_true, device=device), zval.to(torch.float32))
                u_all = est._theta_to_u(torch.tensor(L1_val, device=device, dtype=torch.float32),
                                        ZF_cand, ZL_true)
                nll_vals.append(float(est._nll_one(y_i, v_i, u_all).item()))
        nll_vals = np.array(nll_vals)
        imin = int(np.argmin(nll_vals))
        zhat = float(ZF_grid_np[imin])

        # Console summary
        print(f"{name:>14s} | SNR=40dB | {grid_label.split()[0]}_hat = {zhat:.3f} | "
              f"NLL_min = {nll_vals[imin]:.2f} | L1={L1_val:.2f} (true L1={L1_true:.2f})")

        # Plot
        ax.plot(ZF_grid_np, nll_vals, linewidth=1.3)
        ax.plot(zhat, nll_vals[imin], "o", ms=4, label="argmin")
        ax.axvline(true_comp, linestyle="--", linewidth=1.2, label="true comp")
        ax.set_title(name)
        ax.grid(True, alpha=0.4)
        ax.set_xlabel(grid_label)

    axes[0].set_ylabel("NLL")
    fig.suptitle(f"NLL vs {grid_label.split()[0]} | ZL fixed true | SNR=40 dB | sample {i}", y=0.98)
    # single legend for the whole row (from last axes)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()




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

    print("Text_param", text_param)
    print("Latex param", latex_param)

    #Parameters
    snrs = cfg["snr_dbs"]
    N_train = int(cfg["train_per_snr"])
    N_test = int(cfg["test_per_snr"])
    true_rng = cfg["true_range"]
    seed = cfg["seed"]
    sched = cfg["vi_elbo1d"]["sigma_schedule"]
    plot_likelihood = cfg["plot_loglikelihood"]
    

    est = ContinuousMLE(
    fm=fm,
    likelihood=ComplexGaussianLik(),
    L=1000.0,              # only L1 is boxed
    device=device,
    n_starts=10,
    adam_steps=400,
    adam_lr=1e-3,
    use_lbfgs=True,
    lbfgs_steps=100,
    # profiling knobs
    profile_L1=True,
    L1_grid_points=500,   
    inner_steps=12,       
    inner_lr=1e-2,
    inner_use_lbfgs=False,
    inner_lbfgs_steps=8,
    profile_topk=3        # keep the 3 best L1 wells as joint seeds
)   
  


    rmse_curves = {k: [] for k in ["L1","ZF_re","ZF_im","ZL_re","ZL_im", "ZF_complex", "ZL_complex"]} #dict of RMSE values for each parameter
    crlb_curves = {k: [] for k in ["L1","ZF","ZL"]}
    for snr_db in snrs:
        test = dm.build_or_load_dataset(cfg["dataset_id"], snr_db, N_test, gamma_list, Zc_list, seed = seed, 
                                     target=target, fixed=fixed, gen_cfg=cfg["true_range"], freq_cfg=cfg["freq"], 
                                     force=cfg["force"], desired_freq=pul_freq[idx], estimate=estimate, split="test")
        h_obs = torch.tensor(test["h_obs_real"], device=device) + 1j*torch.tensor(test["h_obs_imag"], device=device)  # [N,F]
        var = torch.tensor(test["noise_var"], device=device)   # [N,F]


        print(f"Curr SNR is {snr_db}")

        # Joint MLE 
        #preds = est.predict(h_obs, var, test["L1_true"], test["ZF_true_re"], test["ZF_true_im"], test["ZL_true_re"], test["ZL_true_im"]) #dict of arrays
        #r = rmse_joint(preds, test) #dict of floats - one single RMSE per SNR as we want

        du_aug = complex_partials_fullbatch(fm, test, device)   # [N, F, 5] complex
        FIM = fim_from_complex_jac(du_aug, var) # [N, 5, 5] Augmented FIM matrix
        CRLB = torch.linalg.inv(FIM)
        CRLB_ZF_from_inv = CRLB[:, 0, 0] #[N] CRLBs for L1
        CRLB_ZL_from_inv = CRLB[:, 1, 1] #[N] CRLBs for L1
        CRLB_L1_from_inv = CRLB[:, 4, 4] #[N] CRLBs for L1
        # print("CRLB_L1_from_inv", torch.mean(CRLB_L1_from_inv))
        # print("CRLB_ZF_from_inv", torch.mean(CRLB_ZF_from_inv))
        # print("CRLB_ZL_from_inv", torch.mean(CRLB_ZL_from_inv))
        
        CRLB_L1, CRLB_ZF, CRLB_ZL = get_CRLB(FIM)
        # Take real part first, then mean, clamp to ≥0, then sqrt
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
        import numpy as np
        for k, v in c.items():
            crlb_curves.setdefault(k, []).append(v.detach().cpu().numpy().item())
    #     for k, v in c.items():
    #         crlb_curves.setdefault(k, []).append(v.detach().cpu().numpy())  # dtype float64, shape ()

    #     # Optional: nice print as floats
    #     print("sqrt(CRLB):", {k: float(v.detach().cpu()) for k, v in c.items()})
        # for k in rmse_curves: 
        #     print(f"Parameter is {k}, RMSE is {r[k]}")
        #     rmse_curves[k].append(r[k])
        
        
    

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Program took {elapsed_time:.4f} seconds to run.")
    plt.figure()
    # for name, series in rmse_curves.items():
    #     plt.plot(snrs, series, marker='o', label=name)
    for name, series in crlb_curves.items():
        plt.plot(snrs, series, marker='x', linestyle='--', label=fr'$\sqrt{{\mathrm{{CRLB}}}}$ for {name}')
    #plt.plot(snrs, crlb_line, marker='x', linestyle='--', label=fr'$\sqrt{{\mathrm{{CRLB}}}}$ for {latex_param}')
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