# plotting/plot_loss_landscapes.py
"""
Plot loss landscapes for all 5 parameters.
Requires: run_1D_loss_landscape.py to be run first.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from core.likelihoods import ComplexGaussianLik


def find_observation_file(results_dir="results", snr_db=40):
    """Find observation file for specific SNR."""
    results_path = pathlib.Path(results_dir)
    if not results_path.exists():
        return None

    files = list(results_path.glob(f"observations_*_snr{int(snr_db)}.npz"))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def find_forward_model(results_dir="results"):
    """Find most recent forward model file."""
    results_path = pathlib.Path(results_dir)
    if not results_path.exists():
        return None

    fm_files = list(results_path.glob("forward_model_*.pt"))
    return max(fm_files, key=lambda p: p.stat().st_mtime) if fm_files else None


def compute_nll(lik, pred_tf, obs_tf, noise_var):
    """Compute NLL using ComplexGaussianLik."""
    if noise_var.shape[-1] == 1:
        F = pred_tf.shape[-1]
        noise_var = noise_var.expand(-1, F)
    return lik(obs_tf, pred_tf, noise_var)


def plot_all_loss_landscapes(fm, h_obs, noise_var, fixed, true_range, device,
                             save_path="loss_landscapes_all.pdf"):
    """Plot loss landscapes for all 5 parameters in a single figure.

    Layout: 3 rows x 2 columns (using 4-column GridSpec)
    - Row 1: ZF_re, ZF_im
    - Row 2: ZL_re, ZL_im
    - Row 3: L1 (centered)

    Args:
        fm: Forward model with compute_H_complex method
        h_obs: [1, F] complex observation tensor
        noise_var: [1, 1] or [1, F] noise variance tensor
        fixed: Dict with true parameter values
        true_range: Dict with parameter ranges
        device: torch device
        save_path: Where to save the figure
    """
    lik = ComplexGaussianLik()

    # Extract fixed values
    L1_fixed = torch.tensor(fixed["L1"], device=device, dtype=torch.float32)
    ZF_fixed = torch.tensor(
        complex(fixed["ZF"]["re"], fixed["ZF"]["im"]),
        device=device, dtype=torch.cfloat
    )
    ZL_fixed = torch.tensor(
        complex(fixed["ZL"]["re"], fixed["ZL"]["im"]),
        device=device, dtype=torch.cfloat
    )

    # Target configs: (target_name, lo, hi, true_val, unit)
    targets_config = [
        ("ZF_re", true_range["ZF"]["re"]["min"], true_range["ZF"]["re"]["max"], fixed["ZF"]["re"], "Ω"),
        ("ZF_im", true_range["ZF"]["im"]["min"], true_range["ZF"]["im"]["max"], fixed["ZF"]["im"], "Ω"),
        ("ZL_re", true_range["ZL"]["re"]["min"], true_range["ZL"]["re"]["max"], fixed["ZL"]["re"], "Ω"),
        ("ZL_im", true_range["ZL"]["im"]["min"], true_range["ZL"]["im"]["max"], fixed["ZL"]["im"], "Ω"),
        ("L1", true_range["L1"]["min"], true_range["L1"]["max"], fixed["L1"], "m"),
    ]

    # Precompute loss landscapes for all targets
    all_data = []
    for target, target_lo, target_hi, true_val, unit in targets_config:
        sweep_normalized = torch.linspace(0.01, 0.99, 199, device=device, dtype=torch.float32)
        sweep_physical = target_lo + (target_hi - target_lo) * sweep_normalized
        losses = []

        with torch.no_grad():
            for val in sweep_physical:
                if target == "L1":
                    L1, ZF, ZL = val, ZF_fixed, ZL_fixed
                elif target == "ZF_re":
                    L1, ZF, ZL = L1_fixed, torch.complex(val, ZF_fixed.imag), ZL_fixed
                elif target == "ZF_im":
                    L1, ZF, ZL = L1_fixed, torch.complex(ZF_fixed.real, val), ZL_fixed
                elif target == "ZL_re":
                    L1, ZF, ZL = L1_fixed, ZF_fixed, torch.complex(val, ZL_fixed.imag)
                elif target == "ZL_im":
                    L1, ZF, ZL = L1_fixed, ZF_fixed, torch.complex(ZL_fixed.real, val)

                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, h_obs, noise_var)
                losses.append(loss.item())

        all_data.append({
            "target": target, "sweep": sweep_physical.cpu().numpy(),
            "losses": losses, "true_val": true_val, "unit": unit
        })

    # Layout using 4 columns
    layout = [
        [(0, slice(0, 2)), (1, slice(2, 4))],  # Row 1: ZF_re, ZF_im
        [(2, slice(0, 2)), (3, slice(2, 4))],  # Row 2: ZL_re, ZL_im
        [(4, slice(1, 3))],                     # Row 3: L1 (centered)
    ]

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.25)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    panel = 0

    for row_idx, row in enumerate(layout):
        for item in row:
            data_idx, col_idx = item
            ax = fig.add_subplot(gs[row_idx, col_idx])
            data = all_data[data_idx]

            ax.text(
                0.5, 1.05, panel_labels[panel],
                transform=ax.transAxes, ha="center", va="bottom", fontsize=13,
            )
            panel += 1

            ax.plot(data["sweep"], data["losses"], 'b-', linewidth=2)
            ax.axvline(x=data["true_val"], color='r', linestyle='--', linewidth=2,
                       label=f'True={data["true_val"]:.1f} {data["unit"]}')
            ax.set_xlabel(f'{data["target"]} ({data["unit"]})', fontsize=12)
            if panel % 2 != 0:
                ax.set_ylabel('NLL', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            ax.tick_params(labelsize=10)

            ymax = np.nanmax(np.abs(data["losses"]))
            exp = int(np.floor(np.log10(ymax)))
            if exp >= 5:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined loss landscape plot to {save_path}")


def analyze_gamma_impact_on_loss(fm, L1_lo, L1_hi, fixed,
                                  snr_db=40, device=None,
                                  save_path="gamma_impact_loss_landscape_onL1.pdf"):
    """
    Analyze how gamma (propagation constant) affects the NLL vs L1 loss landscape.

    For each gamma scaling, we REGENERATE the observation at true L1 using that gamma,
    then sweep L1 to see the loss landscape. This correctly shows how gamma affects
    the identifiability of L1.

    Args:
        fm: Forward model with compute_H_complex method and gamma attribute
        L1_lo: Lower bound for L1
        L1_hi: Upper bound for L1
        fixed: Dict from config with true values, e.g.:
        {"L1": 250.0, "ZF": {"re": 100.0, "im": -50.0}, "ZL": {"re": 100.0, "im": -5.0}}
        snr_db: SNR for generating observations
        device: torch device
        save_path: Where to save the figure
    """
    lik = ComplexGaussianLik()
    if device is None:
        device = torch.device('cpu')

    # Fix ZF and ZL to true values
    L1_true = torch.tensor(fixed["L1"], device=device, dtype=torch.float32)
    ZF = torch.tensor(
        complex(fixed["ZF"]["re"], fixed["ZF"]["im"]),
        device=device, dtype=torch.cfloat
    )
    ZL = torch.tensor(
        complex(fixed["ZL"]["re"], fixed["ZL"]["im"]),
        device=device, dtype=torch.cfloat
    )

    # Store original gamma
    gamma_orig = fm.gamma.clone()
    alpha_orig = gamma_orig.real.clone()
    beta_orig = gamma_orig.imag.clone()

    # L1 grid for loss landscape
    L1_normalized = torch.linspace(0.01, 0.99, 199, device=device, dtype=torch.float32)
    L1_physical = L1_lo + (L1_hi - L1_lo) * L1_normalized
    L1_true_normalized = ((L1_true - L1_lo) / (L1_hi - L1_lo)).item()

    # Scaling factors to test
    alpha_scales = [0.1, 0.5, 1.0, 2.0, 10.0]
    beta_scales = [0.1, 0.5, 1.0, 2.0, 10.0]

    # Create figure with 5 rows x 3 columns (alpha, beta, both)
    fig, axes = plt.subplots(5, 3, figsize=(21, 30))

    # Column 0: Vary alpha (attenuation), keep beta fixed
    for i, alpha_scale in enumerate(alpha_scales):
        # Modify gamma: scale alpha, keep beta
        fm.gamma = (alpha_scale * alpha_orig) + 1j * beta_orig

        # Generate observation at TRUE L1 with THIS gamma
        with torch.no_grad():
            H_true = fm.compute_H_complex(L1_true, ZF, ZL)
            sig_pow = torch.mean(torch.abs(H_true) ** 2)
            var_f = sig_pow / (10 ** (snr_db / 10))
            std_f = torch.sqrt(var_f / 2)
            noise = std_f * (torch.randn_like(H_true.real) + 1j * torch.randn_like(H_true.imag))
            obs_tf_local = H_true + noise

        # Now sweep L1 and compute NLL
        losses = []
        with torch.no_grad():
            for L1 in L1_physical:
                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, obs_tf_local, var_f.expand_as(pred_tf))
                losses.append(loss.item())

        ax = axes[i, 0]
        ax.plot(L1_normalized.cpu().numpy(), losses, 'b-', linewidth=2)
        ax.axvline(x=L1_true_normalized, color='r', linestyle='--', linewidth=2, label='True L1')
        ax.set_xlabel('L1 (normalized)', fontsize=20)
        ax.set_ylabel('NLL', fontsize=20)
        ax.set_title(fr'$\alpha \times {alpha_scale}$', fontsize=20)
        ax.grid(True, alpha=0.3)

        ymax = np.nanmax(np.abs(losses))
        exp = int(np.floor(np.log10(ymax)))
        if exp >= 5:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.tick_params(axis='both', labelsize=20)

        # Add min loss info
        min_idx = np.argmin(losses)
        min_L1 = L1_normalized[min_idx].item()
        ax.axvline(x=min_L1, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(0.6, 0.95, f'Min at L1={min_L1:.2f}', transform=ax.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Column 1: Vary beta (phase constant), keep alpha fixed
    for i, beta_scale in enumerate(beta_scales):
        # Modify gamma: keep alpha, scale beta
        fm.gamma = alpha_orig + 1j * (beta_scale * beta_orig)

        # Generate observation at TRUE L1 with THIS gamma
        with torch.no_grad():
            H_true = fm.compute_H_complex(L1_true, ZF, ZL)
            sig_pow = torch.mean(torch.abs(H_true) ** 2)
            var_f = sig_pow / (10 ** (snr_db / 10))
            std_f = torch.sqrt(var_f / 2)
            noise = std_f * (torch.randn_like(H_true.real) + 1j * torch.randn_like(H_true.imag))
            obs_tf_local = H_true + noise

        losses = []
        with torch.no_grad():
            for L1 in L1_physical:
                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, obs_tf_local, var_f.expand_as(pred_tf))
                losses.append(loss.item())

        ax = axes[i, 1]
        ax.plot(L1_normalized.cpu().numpy(), losses, 'r-', linewidth=2)
        ax.axvline(x=L1_true_normalized, color='b', linestyle='--', linewidth=2, label='True L1')
        ax.set_xlabel('L1 (normalized)', fontsize=20)
        ax.set_ylabel('NLL', fontsize=20)
        ax.set_title(fr'$\beta \times {beta_scale}$', fontsize=20)
        ax.grid(True, alpha=0.3)
        
        ymax = np.nanmax(np.abs(losses))
        exp = int(np.floor(np.log10(ymax)))
        if exp >= 5:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.tick_params(axis='both', labelsize=20)


        # Add min loss info
        min_idx = np.argmin(losses)
        min_L1 = L1_normalized[min_idx].item()
        ax.axvline(x=min_L1, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(0.6, 0.95, f'Min at L1={min_L1:.2f}', transform=ax.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Column 2: Vary both alpha and beta together
    for i, scale in enumerate(alpha_scales):
        # Modify gamma: scale both alpha and beta by the same factor
        fm.gamma = (scale * alpha_orig) + 1j * (scale * beta_orig)

        # Generate observation at TRUE L1 with THIS gamma
        with torch.no_grad():
            H_true = fm.compute_H_complex(L1_true, ZF, ZL)
            sig_pow = torch.mean(torch.abs(H_true) ** 2)
            var_f = sig_pow / (10 ** (snr_db / 10))
            std_f = torch.sqrt(var_f / 2)
            noise = std_f * (torch.randn_like(H_true.real) + 1j * torch.randn_like(H_true.imag))
            obs_tf_local = H_true + noise

        losses = []
        with torch.no_grad():
            for L1 in L1_physical:
                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, obs_tf_local, var_f.expand_as(pred_tf))
                losses.append(loss.item())

        ax = axes[i, 2]
        ax.plot(L1_normalized.cpu().numpy(), losses, 'purple', linewidth=2)
        ax.axvline(x=L1_true_normalized, color='r', linestyle='--', linewidth=2, label='True L1')
        ax.set_xlabel('L1 (normalized)', fontsize=20)
        ax.set_ylabel('NLL', fontsize=20)
        ax.set_title(fr'$\gamma \times {scale}$', fontsize=20)
        ax.grid(True, alpha=0.3)

        ymax = np.nanmax(np.abs(losses))
        exp = int(np.floor(np.log10(ymax)))
        if exp >= 5:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))
        ax.yaxis.get_offset_text().set_fontsize(20)
        ax.tick_params(axis='both', labelsize=20)

        # Add min loss info
        min_idx = np.argmin(losses)
        min_L1 = L1_normalized[min_idx].item()
        ax.axvline(x=min_L1, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(0.6, 0.95, f'Min at L1={min_L1:.2f}', transform=ax.transAxes,
                fontsize=18, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Restore original gamma
    fm.gamma = gamma_orig

    # Add column labels at top
    fig.text(0.19, 0.98, 'Attenuation (α scaling)', fontsize=20, fontweight='bold',
             ha='center', va='top')
    fig.text(0.50, 0.98, 'Phase (β scaling)', fontsize=20, fontweight='bold',
             ha='center', va='top')
    fig.text(0.81, 0.98, 'Both (γ scaling)', fontsize=20, fontweight='bold',
             ha='center', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved gamma impact analysis to {save_path}")

    # Print statistics about original gamma
    print(f"\nOriginal gamma statistics:")
    print(f"  Alpha (attenuation): min={alpha_orig.min().item():.6f}, max={alpha_orig.max().item():.6f}")
    print(f"  Beta (phase):        min={beta_orig.min().item():.4f}, max={beta_orig.max().item():.4f}")


def analyze_L_impact_on_loss(fm, fixed,
                                  snr_db=40, device=None,
                                  save_path="L_impact_loss_landscape_onL1.pdf"):
    """
    Analyze how total cable length L affects the NLL vs L1 loss landscape.

    Tests specific cable lengths [100, 250, 500, 1000, 2000] meters with a fault
    at a fixed RELATIVE position of 0.25. This means the fault location scales
    with cable length (e.g., 25m for 100m cable, 250m for 1000m cable), isolating
    the effect of cable length from the effect of fault position.

    Args:
        fm: Forward model with compute_H_complex method and L attribute
        fixed: Dict from config with true values for ZF and ZL, e.g.:
               {"ZF": {"re": 100.0, "im": -50.0}, "ZL": {"re": 100.0, "im": -5.0}}
        snr_db: SNR for generating observations
        device: torch device
        save_path: Where to save the figure
    """
    lik = ComplexGaussianLik()
    if device is None:
        device = torch.device('cpu')

    # Fix ZF and ZL to true values from config
    ZF = torch.tensor(
        complex(fixed["ZF"]["re"], fixed["ZF"]["im"]),
        device=device, dtype=torch.cfloat
    )
    ZL = torch.tensor(
        complex(fixed["ZL"]["re"], fixed["ZL"]["im"]),
        device=device, dtype=torch.cfloat
    )

    # Store original L
    if isinstance(fm.L, torch.Tensor):
        L_orig = fm.L.clone()
    else:
        L_orig = float(fm.L)

    # Cable lengths to test (in meters)
    L_values = [100.0, 250.0, 500.0, 1000.0, 2000.0]

    # Fixed relative position for fault (0.25 = 25% of cable length)
    L1_relative = 0.25

    # Layout using 4 columns: each plot spans 2 columns
    # Row 1: L=100 (cols 0-1), L=250 (cols 2-3)
    # Row 2: L=500 (cols 0-1), L=1000 (cols 2-3)
    # Row 3: L=2000 (cols 1-2, centered)
    layout = [
        [(0, slice(0, 2)), (1, slice(2, 4))],  # Row 1: L=100, L=250
        [(2, slice(0, 2)), (3, slice(2, 4))],  # Row 2: L=500, L=1000
        [(4, slice(1, 3))],                     # Row 3: L=2000 (centered)
    ]

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.25)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    panel = 0

    # Precompute all losses for each cable length
    all_data = []
    for new_L in L_values:
        fm.L = new_L
        L1_true = torch.tensor(L1_relative * new_L, device=device, dtype=torch.float32)
        L1_lo = 1.0
        L1_hi = new_L

        with torch.no_grad():
            H_true = fm.compute_H_complex(L1_true, ZF, ZL)
            sig_pow = torch.mean(torch.abs(H_true) ** 2)
            var_f = sig_pow / (10 ** (snr_db / 10))
            std_f = torch.sqrt(var_f / 2)
            noise = std_f * (torch.randn_like(H_true.real) + 1j * torch.randn_like(H_true.imag))
            obs_tf_local = H_true + noise

        L1_normalized = torch.linspace(0.01, 0.99, 199, device=device, dtype=torch.float32)
        L1_physical = L1_lo + (L1_hi - L1_lo) * L1_normalized

        losses = []
        with torch.no_grad():
            for L1 in L1_physical:
                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, obs_tf_local, var_f.expand_as(pred_tf))
                losses.append(loss.item())

        all_data.append({
            "L": new_L, "L1_true": L1_true.item(),
            "L1_physical": L1_physical.cpu().numpy(), "losses": losses
        })

    # Plot using layout
    for row_idx, row in enumerate(layout):
        for item in row:
            data_idx, col_idx = item
            ax = fig.add_subplot(gs[row_idx, col_idx])
            data = all_data[data_idx]

            ax.text(
                0.5, 1.05, panel_labels[panel],
                transform=ax.transAxes, ha="center", va="bottom", fontsize=13,
            )
            panel += 1

            ax.plot(data["L1_physical"], data["losses"], 'b-', linewidth=2)
            ax.axvline(x=data["L1_true"], color='r', linestyle='--', linewidth=2,
                       label=f'True L1={data["L1_true"]:.1f} m')
            ax.set_xlabel('L1 (m)', fontsize=12)
            if panel % 2 != 0:
                ax.set_ylabel('NLL', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            ax.tick_params(labelsize=10)

            ymax = np.nanmax(np.abs(data["losses"]))
            exp = int(np.floor(np.log10(ymax)))
            if exp >= 5:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))

    # Restore original L
    fm.L = L_orig

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved L impact analysis to {save_path}")
    print(f"\nTested {len(L_values)} cable lengths with fault at relative position {L1_relative:.2f}:")
    for L_val in L_values:
        L1_at_this_L = L1_relative * L_val
        print(f"  L = {L_val:.1f} m, L1 = {L1_at_this_L:.1f} m")



def analyze_freq_impact_on_loss(fm, fixed, pul_freq_full, gamma_full, Zc_full,
                                snr_db=40, device=None,
                                save_path="freq_impact_loss_landscape_onL1.pdf",
                                num_points=200):
    """
    Analyze how frequency range affects the NLL vs L1 loss landscape.

    Tests different frequency ranges with progressively lower starting frequencies
    while keeping the upper bound at 10 MHz. Lower frequencies provide more
    information about longer wavelength features, potentially improving fault
    localization.

    Frequency ranges tested:
    - 2 MHz - 10 MHz (baseline, BB-PLC typical)
    - 1 MHz - 10 MHz
    - 500 kHz - 10 MHz
    - 100 kHz - 10 MHz
    - 45 kHz - 10 MHz (full range)

    Args:
        fm: Forward model with compute_H_complex method and gamma, Zc attributes
        fixed: Dict with true values for L1, ZF, and ZL
        pul_freq_full: Full frequency array [F] in Hz
        gamma_full: Full propagation constant array [F] (complex)
        Zc_full: Full characteristic impedance array [F] (complex)
        snr_db: SNR for generating observations
        device: torch device
        save_path: Where to save the figure
        num_points: Number of frequency points to subsample to
    """
    lik = ComplexGaussianLik()
    if device is None:
        device = torch.device('cpu')

    # Fix ZF and ZL to true values from config
    ZF = torch.tensor(
        complex(fixed["ZF"]["re"], fixed["ZF"]["im"]),
        device=device, dtype=torch.cfloat
    )
    ZL = torch.tensor(
        complex(fixed["ZL"]["re"], fixed["ZL"]["im"]),
        device=device, dtype=torch.cfloat
    )

    # Get L1 true value from fixed dict
    L1_true_val = fixed["L1"]
    L1_true = torch.tensor(L1_true_val, device=device, dtype=torch.float32)

    # Store original forward model parameters
    gamma_orig = fm.gamma.clone() if isinstance(fm.gamma, torch.Tensor) else fm.gamma
    Zc_orig = fm.Zc.clone() if isinstance(fm.Zc, torch.Tensor) else fm.Zc

    # Frequency ranges to test: (f_min_hz, f_max_hz, label)
    freq_ranges = [
        (2e6, 10e6, "2 MHz - 10 MHz (baseline)"),
        (1e6, 10e6, "1 MHz - 10 MHz"),
        (500e3, 10e6, "500 kHz - 10 MHz"),
        (100e3, 10e6, "100 kHz - 10 MHz"),
        (45e3, 10e6, "45 kHz - 10 MHz"),
    ]

    # L1 search range
    L1_lo = 1.0
    L1_hi = fm.L if isinstance(fm.L, (int, float)) else fm.L.item()

    # Layout using 4 columns: each plot spans 2 columns
    # Row 1: freq_range[0] (cols 0-1), freq_range[1] (cols 2-3)
    # Row 2: freq_range[2] (cols 0-1), freq_range[3] (cols 2-3)
    # Row 3: freq_range[4] (cols 1-2, centered)
    layout = [
        [(0, slice(0, 2)), (1, slice(2, 4))],  # Row 1
        [(2, slice(0, 2)), (3, slice(2, 4))],  # Row 2
        [(4, slice(1, 3))],                     # Row 3 (centered)
    ]

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.25)

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    # Convert full arrays to numpy for indexing
    pul_freq_np = pul_freq_full.cpu().numpy()

    # Precompute all losses for each frequency range
    all_data = []
    for f_min, f_max, label in freq_ranges:
        mask = (pul_freq_np >= f_min) & (pul_freq_np <= f_max)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"Warning: No frequencies found in range {label}")
            all_data.append(None)
            continue

        if len(indices) > num_points:
            subsample_idx = np.linspace(0, len(indices) - 1, num_points, dtype=int)
            indices = indices[subsample_idx]

        gamma_subset = gamma_full[indices]
        Zc_subset = Zc_full[indices]
        fm.gamma = gamma_subset
        fm.Zc = Zc_subset

        with torch.no_grad():
            H_true = fm.compute_H_complex(L1_true, ZF, ZL)
            sig_pow = torch.mean(torch.abs(H_true) ** 2)
            var_f = sig_pow / (10 ** (snr_db / 10))
            std_f = torch.sqrt(var_f / 2)
            noise = std_f * (torch.randn_like(H_true.real) + 1j * torch.randn_like(H_true.imag))
            obs_tf_local = H_true + noise

        L1_normalized = torch.linspace(0.01, 0.99, 199, device=device, dtype=torch.float32)
        L1_physical = L1_lo + (L1_hi - L1_lo) * L1_normalized

        losses = []
        with torch.no_grad():
            for L1 in L1_physical:
                pred_tf = fm.compute_H_complex(L1, ZF, ZL)
                loss = compute_nll(lik, pred_tf, obs_tf_local, var_f.expand_as(pred_tf))
                losses.append(loss.item())

        actual_f_min = pul_freq_np[indices[0]] / 1e6
        actual_f_max = pul_freq_np[indices[-1]] / 1e6
        print(f"  {label}: {len(indices)} points, f=[{actual_f_min:.3f}, {actual_f_max:.3f}] MHz")

        all_data.append({
            "label": label,
            "L1_physical": L1_physical.cpu().numpy(),
            "losses": losses,
        })

    # Plot using layout
    panel = 0
    for row_idx, row in enumerate(layout):
        for item in row:
            data_idx, col_idx = item
            if all_data[data_idx] is None:
                continue
            ax = fig.add_subplot(gs[row_idx, col_idx])
            data = all_data[data_idx]

            ax.text(
                0.5, 1.05, panel_labels[panel],
                transform=ax.transAxes, ha="center", va="bottom", fontsize=13,
            )
            panel += 1

            ax.plot(data["L1_physical"], data["losses"], 'b-', linewidth=2)
            ax.axvline(x=L1_true.item(), color='r', linestyle='--', linewidth=2,
                       label=f'True L1={L1_true.item():.1f} m')
            ax.set_xlabel('L1 (m)', fontsize=12)
            if panel % 2 != 0:
                ax.set_ylabel('NLL', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            ax.tick_params(labelsize=10)

            ymax = np.nanmax(np.abs(data["losses"]))
            exp = int(np.floor(np.log10(ymax)))
            if exp >= 5:
                ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))

    # Restore original forward model parameters
    fm.gamma = gamma_orig
    fm.Zc = Zc_orig

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved frequency impact analysis to {save_path}")
    print(f"Tested {len(freq_ranges)} frequency ranges with L1_true={L1_true_val:.1f} m, SNR={snr_db} dB")


def main(cfg_path="configs/benchmark.yaml", snr_db=40,
         save_dir="figures/plot_loss_landscapes"):
    """Main function to plot all loss landscape analyses."""
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device(cfg.get("device", "cpu"))

    # Find and load forward model
    fm_path = find_forward_model()
    if fm_path is None:
        print("Error: No forward model found. Run generate_observations.py first.")
        return
    print(f"Loading forward model from {fm_path}")
    fm = torch.load(fm_path, weights_only=False)

    # Find and load observation file
    obs_path = find_observation_file(snr_db=snr_db)
    if obs_path is None:
        print(f"Error: No observation file found for SNR={snr_db}dB.")
        print("Run generate_observations.py first.")
        return
    print(f"Loading observations from {obs_path}")

    obs_data = np.load(obs_path)
    h_obs = torch.tensor(
        obs_data["h_obs_real"][0:1] + 1j * obs_data["h_obs_imag"][0:1],
        device=device, dtype=torch.cfloat
    )
    noise_var = torch.tensor(obs_data["noise_var"][0:1], device=device, dtype=torch.float32)

    fixed = cfg["fixed"]
    true_range = cfg["true_range"]

    # Create save directory
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Plot loss landscapes for all 5 parameters
    print("Plotting loss landscapes for all parameters...")
    plot_all_loss_landscapes(
        fm, h_obs, noise_var, fixed, true_range, device,
        save_path=str(save_path / "loss_landscapes_all.pdf")
    )

    # 2. Plot gamma impact analysis
    print("Plotting gamma impact analysis...")
    analyze_gamma_impact_on_loss(
        fm, true_range["L1"]["min"], true_range["L1"]["max"], fixed,
        snr_db=snr_db, device=device,
        save_path=str(save_path / "gamma_impact_loss_landscape.pdf")
    )

    # 3. Plot L impact analysis
    print("Plotting L impact analysis...")
    analyze_L_impact_on_loss(
        fm, fixed, snr_db=snr_db, device=device,
        save_path=str(save_path / "L_impact_loss_landscape.pdf")
    )

    # 4. Plot frequency impact analysis
    if "pul_freq" in obs_data:
        print("Plotting frequency impact analysis...")
        pul_freq_full = torch.tensor(obs_data["pul_freq"], device=device, dtype=torch.float32)
        gamma_full = torch.tensor(obs_data["gamma_real"] + 1j * obs_data["gamma_imag"], device=device, dtype=torch.cfloat)
        Zc_full = torch.tensor(obs_data["Zc_real"] + 1j * obs_data["Zc_imag"], device=device, dtype=torch.cfloat)

        analyze_freq_impact_on_loss(
            fm, fixed, pul_freq_full, gamma_full, Zc_full,
            snr_db=snr_db, device=device,
            save_path=str(save_path / "freq_impact_loss_landscape.pdf"),
            num_points=cfg["freq"]["num_points"]
        )
    else:
        print("Warning: Frequency data not in observation file, skipping freq impact plot.")

    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot loss landscapes")
    parser.add_argument("--snr", type=int, default=40, help="SNR in dB")
    parser.add_argument("--cfg", type=str, default="configs/benchmark.yaml", help="Config file path")
    parser.add_argument("--save-dir", type=str, default="figures/plot_loss_landscapes", help="Save directory")
    args = parser.parse_args()

    main(cfg_path=args.cfg, snr_db=args.snr, save_dir=args.save_dir)
