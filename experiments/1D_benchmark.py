# experiments/1D_benchmark.py
import sys, pathlib
import time
import hashlib
import json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
import scipy.io as sio
import matplotlib.pyplot as plt

from core.forward import ForwardModel
from core.likelihoods import ComplexGaussianLik
from core.crlb import crlb_for_1_real_param
from estimators.mle_gridsearch import GridSearchMLE
from estimators.mle_gradient import GradientMLE
from data.manager import DatasetManager


def config_hash(cfg_dict, length=8):
    """Create short hash of config dict for unique filenames."""
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:length]


def fmt_freq(hz):
    """Format frequency for display/filenames."""
    if hz >= 1e6:
        return f"{hz/1e6:.1f}MHz".replace(".0MHz", "MHz")
    else:
        return f"{hz/1e3:.0f}kHz"

def plot_gamma_vs_frequency(cfg_path="configs/benchmark.yaml"):
    """Plot propagation constant gamma vs frequency."""
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))

    # Load from .mat file
    mat = sio.loadmat("experiments/cable_parameter.mat")
    gamma_full = torch.tensor(mat["gamma"].squeeze(), dtype=torch.cfloat, device=device)
    pul_freq = torch.tensor(mat["pulFreq"].squeeze(), dtype=torch.float32, device=device)

    fstart = float(cfg["freq"]["start_hz"])
    fend = float(cfg["freq"]["stop_hz"])
    F = int(cfg["freq"]["num_points"])

    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)
    freq_range_mhz = desired_freqs / 1e6

    # Get gamma at desired frequencies
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)
    gamma_list = gamma_full[idx]

    # Extract real (attenuation) and imaginary (phase constant) parts
    alpha = gamma_list.real.cpu().numpy()  # Attenuation constant
    beta = gamma_list.imag.cpu().numpy()   # Phase constant
    freq_mhz = freq_range_mhz.cpu().numpy()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot attenuation constant (real part)
    ax1.plot(freq_mhz, alpha, 'b-', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (MHz)', fontsize=12)
    ax1.set_ylabel(r'$\alpha$ (Np/m) - Attenuation', fontsize=12)
    ax1.set_title(r'Attenuation Constant $\alpha = \Re\{\gamma\}$', fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    # Add min/max text box
    alpha_text = f'Min: {alpha.min():.6f}\nMax: {alpha.max():.6f}'
    ax1.text(0.95, 0.05, alpha_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot phase constant (imaginary part)
    ax2.plot(freq_mhz, beta, 'r-', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (MHz)', fontsize=12)
    ax2.set_ylabel(r'$\beta$ (rad/m) - Phase', fontsize=12)
    ax2.set_title(r'Phase Constant $\beta = \Im\{\gamma\}$', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    # Add min/max text box
    beta_text = f'Min: {beta.min():.4f}\nMax: {beta.max():.4f}'
    ax2.text(0.95, 0.05, beta_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('gamma_vs_frequency.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Frequency range: {freq_mhz[0]:.2f} - {freq_mhz[-1]:.2f} MHz")
    print(f"Alpha range: {alpha.min():.6f} - {alpha.max():.6f} Np/m")
    print(f"Beta range: {beta.min():.4f} - {beta.max():.4f} rad/m")


def calculate_gamma_Zc_from_first_principles(frequencies, device="cpu"):
    """
    Calculate propagation constant gamma and characteristic impedance Zc
    from first principles for HELUKABEL N2XSEY 6/10 kV cable.

    Models the cable as an equivalent two-conductor transmission line with:
    - Complex permittivity for XLPE insulation (εr = 2.3, εi = 0.001)
    - Skin effect with DC/AC threshold behavior
    - Stranding correction factor for multi-strand conductors

    Args:
        frequencies: torch.Tensor [F] of frequencies in Hz
        device: torch device

    Returns:
        gamma: torch.Tensor [F] complex propagation constant (α + jβ)
        Zc: torch.Tensor [F] complex characteristic impedance
    """
    # Physical constants
    mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of free space)
    eps_0 = 8.854e-12  # F/m (permittivity of free space)
    sigma = 5.8e7  # S/m (copper conductivity)

    # Cable geometry parameters (50 mm² conductor)
    a = 3.9894e-3  # m (conductor radius from A_c = 50 mm², a = sqrt(50e-6/pi))
    #a = 2.82e-3
    D = 15e-3  # m (centre-to-centre conductor spacing)

    # Complex permittivity: ε̂ = ε₀(εr - jεi) for XLPE
    eps_r = 2.3  # real part of relative permittivity
    eps_i = 0.001  # imaginary part (loss)
    eps_hat = eps_0 * (eps_r - 1j * eps_i)  # complex permittivity

    # Stranding parameters
    r_s = 0.915e-3  # m (radius of single strand)
    n_0 = 12  # number of strands in outer ring

    # Per-unit-length inductance (frequency independent)
    # L' = (μ₀/π) × ln(D/a)  [MATLAB uses log, not acosh]
    log_term = np.log(D / a)
    L_prime = (mu_0 / np.pi) * log_term  # H/m

    # Complex per-unit-length capacitance using TEM relationship: L'C' = μ₀ε
    # Ĉ' = μ₀ε₀ε̂ / L'  [matches MATLAB: C_Complex = MU0*EPS0*e_xlpe / l_entry]
    C_hat = (mu_0 * eps_hat) / L_prime

    # Extract real capacitance: C' = Re{Ĉ'}
    C_prime = np.real(C_hat)  # F/m

    # Convert frequencies to numpy for calculation
    f = frequencies.cpu().numpy()  # Hz
    omega = 2 * np.pi * f  # rad/s

    # Frequency-dependent conductance: G' = -ω × Im{Ĉ'}
    # [matches MATLAB: G_Matrix = -imag(C_Complex) * 2*pi*pulFreq(f)]
    G_prime = -omega * np.imag(C_hat)  # S/m

    # Skin depth: δ(f) = 1 / √(πfμ₀σ)
    delta = 1.0 / np.sqrt(np.pi * f * mu_0 * sigma)  # m

    # Solid conductor PUL resistance with skin effect threshold
    # If δ > 2a: r_solid = 1/(σπa²) (DC resistance)
    # If δ ≤ 2a: r_solid = √(μ₀f/πσ) / (2a) (AC resistance with skin effect)
    r_solid = np.where(
        delta > 2 * a,
        1.0 / (sigma * np.pi * a**2),  # DC regime
        np.sqrt(mu_0 * f / (np.pi * sigma)) / (2 * a)  # AC regime with skin effect
    )

    # Stranding correction factor x_c(f)
    # x_c = n₀ × [arccos((r_s-δ)/r_s) × r_s² - (r_s-δ) × √(r_s² - (r_s-δ)²)] / (2πaδ)
    # Handle case where δ > r_s (full penetration of strand)
    delta_clipped = np.clip(delta, 1e-12, r_s - 1e-12)  # Avoid numerical issues
    term1 = r_s - delta_clipped
    arccos_arg = np.clip(term1 / r_s, -1, 1)  # Ensure valid arccos input
    sqrt_term = np.sqrt(np.maximum(r_s**2 - term1**2, 0))

    x_c = n_0 * (np.arccos(arccos_arg) * r_s**2 - term1 * sqrt_term) / (2 * np.pi * a * delta)

    # Handle edge cases where x_c might be invalid
    x_c = np.where(x_c > 0, x_c, 1.0)  # Default to 1 if correction is invalid

    # Stranded conductor resistance: r_stranded = r_solid / x_c
    # Total series resistance: R' = 2 × r_stranded (two conductors)
    R_prime = 2 * r_solid / x_c  # Ω/m

    # Series impedance per unit length: Z' = R' + jωL'
    Z_prime = R_prime + 1j * omega * L_prime

    # Shunt admittance per unit length: Y' = G' + jωC'
    Y_prime = G_prime + 1j * omega * C_prime

    # Propagation constant: γ = √(Z' × Y')
    gamma = np.sqrt(Z_prime * Y_prime)

    # Characteristic impedance: Z₀ = √(Z' / Y')
    Zc = np.sqrt(Z_prime / Y_prime)

    # Convert to torch tensors
    gamma_torch = torch.tensor(gamma, dtype=torch.cfloat, device=device)
    Zc_torch = torch.tensor(Zc, dtype=torch.cfloat, device=device)

    return gamma_torch, Zc_torch


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

    # Human-readable tags
    freq_tag = f"{fmt_freq(fstart)}-{fmt_freq(fend)}"
    L_tag = f"L{int(L)}"

    # Fixed true values
    fixed = cfg["fixed"]
    true_range = cfg["true_range"]
    L1_true = float(fixed["L1"])
    ZF_re_true = float(fixed["ZF"]["re"])
    ZF_im_true = float(fixed["ZF"]["im"])
    ZL_re_true = float(fixed["ZL"]["re"])
    ZL_im_true = float(fixed["ZL"]["im"])

    snrs = cfg["snr_dbs"]
    N = int(cfg["N"])
    seed = cfg["seed"]

    # Config dict for observation data (affects H_true and noise)
    obs_config = {
        "freq_start": fstart, "freq_stop": fend, "F": F,
        "L": L, "L1": L1_true,
        "ZF_re": ZF_re_true, "ZF_im": ZF_im_true,
        "ZL_re": ZL_re_true, "ZL_im": ZL_im_true,
        "N": N, "seed": seed,
    }
    obs_hash = config_hash(obs_config)

    # Config dict for benchmark (includes estimator range)
    bench_config = {**obs_config, "true_range": true_range}
    bench_hash = config_hash(bench_config)

    desired_freqs = torch.linspace(fstart, fend, F, device=device, dtype=pul_freq.dtype)  # [F]
    # For each desired f, find index of closest pul_freq
    idx = torch.abs(pul_freq.unsqueeze(0) - desired_freqs.unsqueeze(1)).argmin(dim=1)     # [F]

    #gamma_list = gamma_analytical[idx]
    #Zc_list = Zc_analytical[idx]
    gamma_list = gamma_full[idx]   # [F]
    Zc_list = Zc_full[idx]      # [F]
    
    dm = DatasetManager(device=device)
    fm = ForwardModel(gamma_list, Zc_list, L=L, Zs=Zs, device=device)

    # Target from config (for dataset generation)
    target = cfg["target"].upper()

    # Define the 5 1D estimation targets
    targets = ["L1", "ZF_re", "ZF_im", "ZL_re", "ZL_im"]

    # Build grids for each target
    num_grid_pts = 500
    grids = {
        "L1": torch.linspace(cfg["true_range"]["L1"]["min"], cfg["true_range"]["L1"]["max"], num_grid_pts, device=device),
        "ZF_re": torch.linspace(cfg["true_range"]["ZF"]["re"]["min"], cfg["true_range"]["ZF"]["re"]["max"], num_grid_pts, device=device),
        "ZF_im": torch.linspace(cfg["true_range"]["ZF"]["im"]["min"], cfg["true_range"]["ZF"]["im"]["max"], num_grid_pts, device=device),
        "ZL_re": torch.linspace(cfg["true_range"]["ZL"]["re"]["min"], cfg["true_range"]["ZL"]["re"]["max"], num_grid_pts, device=device),
        "ZL_im": torch.linspace(cfg["true_range"]["ZL"]["im"]["min"], cfg["true_range"]["ZL"]["im"]["max"], num_grid_pts, device=device),
    }

    # Compute quantization error for each parameter (distance from true value to closest grid point)
    true_values = {
        "L1": L1_true,
        "ZF_re": ZF_re_true,
        "ZF_im": ZF_im_true,
        "ZL_re": ZL_re_true,
        "ZL_im": ZL_im_true,
    }
    quant_error = {}
    for param in targets:
        grid = grids[param]
        true_val = true_values[param]
        # Find closest grid point and compute distance
        closest_idx = torch.argmin(torch.abs(grid - true_val))
        quant_error[param] = float(torch.abs(grid[closest_idx] - true_val).cpu())
    print(f"Quantization errors: {quant_error}")

    rmse_curves_grid = {t: [] for t in targets}
    rmse_curves_gradient = {t: [] for t in targets}
    crlb_curves = {t: [] for t in targets}

    # SNR to use for loss landscape plotting (high SNR for clear convergence)
    LANDSCAPE_SNR = 40  

    for snr_db in snrs:
        print(f"Curr SNR is {snr_db}")
        data = dm.generate_observations(snr_db, N, fm, seed=seed,
                                     target=target, fixed=fixed, gen_cfg=true_range)
        h_obs = torch.tensor(data["h_obs_real"], device=device) + 1j*torch.tensor(data["h_obs_imag"], device=device)  # [N,F]
        var = torch.tensor(data["noise_var"], device=device)   # [N,1]
        
        # Plot h_obs magnitude in dB vs frequency
        # h_obs_mag_db = 20 * torch.log10(torch.abs(h_obs))  # [N, F]
        # plt.figure(figsize=(10, 6))
        # # Plot first observation
        # plt.plot(freq_range_mhz.cpu().numpy(), h_obs_mag_db[0].cpu().numpy(), 'b-', linewidth=2, label='Observation')
        # plt.xlabel('Frequency (MHz)', fontsize=12)
        # plt.ylabel('|H(f)| (dB)', fontsize=12)
        # plt.title(f'Observed Transfer Function Magnitude at SNR = {snr_db} dB', fontsize=14)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        

        # Save observation file at specific SNR
        if snr_db == LANDSCAPE_SNR:
            results_dir = pathlib.Path("results")
            results_dir.mkdir(exist_ok=True)
            obs_file = results_dir / f"observation_{freq_tag}_{L_tag}_{obs_hash}_snr{int(snr_db)}.npz"

            np.savez(
                obs_file,
                # Config for reproducibility
                obs_config=obs_config,
                seed=seed,
                snr_db=snr_db,
                # Sample observation at high SNR for loss landscape plotting
                h_obs_sample_real=h_obs[0:1, :].real.cpu().numpy(),
                h_obs_sample_imag=h_obs[0:1, :].imag.cpu().numpy(),
                noise_var_sample=var[0:1, :].cpu().numpy(),
                # Full frequency and cable parameters (for freq impact analysis)
                pul_freq=pul_freq.cpu().numpy(),
                gamma_real=gamma_full.real.cpu().numpy(),
                gamma_imag=gamma_full.imag.cpu().numpy(),
                Zc_real=Zc_full.real.cpu().numpy(),
                Zc_imag=Zc_full.imag.cpu().numpy(),
            )
            print(f"Saved observation data at {snr_db}dB SNR to {obs_file}")

            # Save forward model
            fm_file = results_dir / f"forward_model_{freq_tag}_{L_tag}_{obs_hash}.pt"
            torch.save(fm, fm_file)
            print(f"Saved forward model to {fm_file}")
        
        
        # Run grid search MLE for each target parameter
        for target in targets:
            # Create estimator for this target
            est = GridSearchMLE(
                fm=fm,
                likelihood=ComplexGaussianLik(),
                grid=grids[target],
                target=target,
                fixed=fixed,
                device=device,
            )
            est2 = GradientMLE(
                fm = fm,
                likelihood=ComplexGaussianLik(),
                target=target,
                fixed=fixed,
                true_range=true_range,
                mode="1d"
            )

            # Get predictions
            preds_grid = est.predict(h_obs, var)
            preds_gradient = est2.predict(h_obs, var)

            # Compute RMSE for this parameter
            true_key = "L1_true" if target == "L1" else target.replace("_", "_true_")
            rmse_grid = float(np.sqrt(np.mean((preds_grid[target] - data[true_key]) ** 2)))
            rmse_curves_grid[target].append(rmse_grid)

            rmse_gradient = float(np.sqrt(np.mean((preds_gradient[target] - data[true_key]) ** 2)))
            rmse_curves_gradient[target].append(rmse_gradient)
            
            # Compute CRLB for this parameter at true values from config
            _, crlb = crlb_for_1_real_param(fm, target, fixed, var[0].squeeze(), device)
            sqrt_crlb = float(torch.sqrt(crlb).cpu())
            crlb_curves[target].append(sqrt_crlb)

            print(f"  {target}: RMSE Grid Search={rmse_grid:.4f}, sqrt(CRLB)={sqrt_crlb:.4f}")
            print(f"  {target}: RMSE Gradient ={rmse_gradient:.4f}, sqrt(CRLB)={sqrt_crlb:.4f}")

    # Save benchmark results
    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    benchmark_file = results_dir / f"1D_benchmark_{freq_tag}_{L_tag}_{bench_hash}_seed{seed}.npz"

    np.savez(
        benchmark_file,
        # Config for reproducibility
        bench_config=bench_config,
        freq_tag=freq_tag,
        L_tag=L_tag,
        snr_dbs=np.array(snrs),
        seed=seed,
        # 1D RMSE Grid Search results
        rmse_grid_L1=np.array(rmse_curves_grid["L1"]),
        rmse_grid_ZF_re=np.array(rmse_curves_grid["ZF_re"]),
        rmse_grid_ZF_im=np.array(rmse_curves_grid["ZF_im"]),
        rmse_grid_ZL_re=np.array(rmse_curves_grid["ZL_re"]),
        rmse_grid_ZL_im=np.array(rmse_curves_grid["ZL_im"]),
        # 1D RMSE Gradient Descent results
        rmse_grad_L1=np.array(rmse_curves_gradient["L1"]),
        rmse_grad_ZF_re=np.array(rmse_curves_gradient["ZF_re"]),
        rmse_grad_ZF_im=np.array(rmse_curves_gradient["ZF_im"]),
        rmse_grad_ZL_re=np.array(rmse_curves_gradient["ZL_re"]),
        rmse_grad_ZL_im=np.array(rmse_curves_gradient["ZL_im"]),
        # sqrt(CRLB) results for all 5 parameters
        crlb_L1=np.array(crlb_curves["L1"]),
        crlb_ZF_re=np.array(crlb_curves["ZF_re"]),
        crlb_ZF_im=np.array(crlb_curves["ZF_im"]),
        crlb_ZL_re=np.array(crlb_curves["ZL_re"]),
        crlb_ZL_im=np.array(crlb_curves["ZL_im"]),
        # Quantization error for grid search (distance from true value to closest grid point)
        quant_L1=quant_error["L1"],
        quant_ZF_re=quant_error["ZF_re"],
        quant_ZF_im=quant_error["ZF_im"],
        quant_ZL_re=quant_error["ZL_re"],
        quant_ZL_im=quant_error["ZL_im"],
    )
    print(f"Saved benchmark results to {benchmark_file}")

    end_time = time.perf_counter()
    print(f"Program took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()