"""
Plot results from saved benchmark data.
Run 1D_benchmark.py first to generate results, then run this to create plots.
All figures are saved to figures/plot_1D_benchmark_figures/.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from core.likelihoods import ComplexGaussianLik


# =============================================================================
# Loss landscape functions (merged from loss_landscape.py)
# =============================================================================

def compute_nll(lik, pred_tf, obs_tf, noise_var):
    """
    Compute NLL using ComplexGaussianLik, handling noise_var broadcasting.

    Args:
        lik: ComplexGaussianLik instance
        pred_tf: [K, F] or [1, F] predicted transfer function
        obs_tf: [1, F] observed transfer function
        noise_var: [1, 1] or [1, F] noise variance

    Returns:
        NLL values [K] or scalar if K=1
    """
    # Expand noise_var to [1, F] if it's [1, 1]
    if noise_var.shape[-1] == 1:
        F = pred_tf.shape[-1]
        noise_var = noise_var.expand(-1, F)

    # ComplexGaussianLik returns NLL [K] directly
    return lik(obs_tf, pred_tf, noise_var)  # Note: obs first, pred second


def plot_loss_landscape(fm, obs_tf, noise_var, fixed, true_range, target="L1",
                        device=None, save_path=None):
    """
    Plot loss vs a target parameter to visualize the optimization landscape.
    Other parameters are fixed to their true values from config.

    Args:
        fm: Forward model with compute_H_complex method
        obs_tf: [1, F] complex observation tensor
        noise_var: [1, 1] or [1, F] noise variance tensor
        fixed: Dict from config with true values, e.g.:
               {"L1": 250.0, "ZF": {"re": 100.0, "im": -50.0}, "ZL": {"re": 100.0, "im": -5.0}}
        true_range: Dict from config with parameter ranges, e.g.:
               {"L1": {"min": 1.0, "max": 1000.0}, "ZF": {"re": {...}, "im": {...}}, ...}
        target: Which parameter to sweep: "L1", "ZF_re", "ZF_im", "ZL_re", "ZL_im"
        device: torch device (defaults to cpu)
        save_path: Where to save the figure (auto-generated if None)
    """
    lik = ComplexGaussianLik()
    if device is None:
        device = torch.device('cpu')

    if save_path is None:
        save_path = f"loss_landscape_{target}.pdf"

    # Extract fixed values from config
    L1_fixed = torch.tensor(fixed["L1"], device=device, dtype=torch.float32)
    ZF_fixed = torch.tensor(
        complex(fixed["ZF"]["re"], fixed["ZF"]["im"]),
        device=device, dtype=torch.cfloat
    )
    ZL_fixed = torch.tensor(
        complex(fixed["ZL"]["re"], fixed["ZL"]["im"]),
        device=device, dtype=torch.cfloat
    )

    # Get range for target parameter
    if target == "L1":
        target_lo = true_range["L1"]["min"]
        target_hi = true_range["L1"]["max"]
        true_val = fixed["L1"]
    elif target == "ZF_re":
        target_lo = true_range["ZF"]["re"]["min"]
        target_hi = true_range["ZF"]["re"]["max"]
        true_val = fixed["ZF"]["re"]
    elif target == "ZF_im":
        target_lo = true_range["ZF"]["im"]["min"]
        target_hi = true_range["ZF"]["im"]["max"]
        true_val = fixed["ZF"]["im"]
    elif target == "ZL_re":
        target_lo = true_range["ZL"]["re"]["min"]
        target_hi = true_range["ZL"]["re"]["max"]
        true_val = fixed["ZL"]["re"]
    elif target == "ZL_im":
        target_lo = true_range["ZL"]["im"]["min"]
        target_hi = true_range["ZL"]["im"]["max"]
        true_val = fixed["ZL"]["im"]
    else:
        raise ValueError(f"Unknown target: {target}. Use L1, ZF_re, ZF_im, ZL_re, or ZL_im")

    # Sweep target from 0.01 to 0.99 (normalized)
    sweep_normalized = torch.linspace(0.01, 0.99, 199, device=device, dtype=torch.float32)
    sweep_physical = target_lo + (target_hi - target_lo) * sweep_normalized
    losses = []

    with torch.no_grad():
        for val in sweep_physical:
            # Build parameters based on target
            if target == "L1":
                L1 = val
                ZF = ZF_fixed
                ZL = ZL_fixed
            elif target == "ZF_re":
                L1 = L1_fixed
                ZF = torch.complex(val, ZF_fixed.imag)
                ZL = ZL_fixed
            elif target == "ZF_im":
                L1 = L1_fixed
                ZF = torch.complex(ZF_fixed.real, val)
                ZL = ZL_fixed
            elif target == "ZL_re":
                L1 = L1_fixed
                ZF = ZF_fixed
                ZL = torch.complex(val, ZL_fixed.imag)
            elif target == "ZL_im":
                L1 = L1_fixed
                ZF = ZF_fixed
                ZL = torch.complex(ZL_fixed.real, val)

            pred_tf = fm.compute_H_complex(L1, ZF, ZL) #[1, F]
            loss = compute_nll(lik, pred_tf, obs_tf, noise_var)
            losses.append(loss.item())

    # Map target to units
    unit_map = {
        "L1": "m",
        "ZF_re": "Ω",
        "ZF_im": "Ω",
        "ZL_re": "Ω",
        "ZL_im": "Ω",
    }
    unit = unit_map.get(target, "")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_physical.cpu().numpy(), losses, 'b-', linewidth=2)
    plt.axvline(x=true_val, color='r', linestyle='--', linewidth=2,
            label=f'True {target}={true_val:.1f} {unit}')
    plt.xlabel(f'{target} ({unit})', fontsize=15)
    plt.ylabel('Loss (NLL)', fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=15)
    ax = plt.gca()
    ymax = np.nanmax(np.abs(losses))
    exp = int(np.floor(np.log10(ymax)))
    if exp >= 5:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(exp, exp))
    ax.yaxis.get_offset_text().set_fontsize(15)
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss landscape plot to {save_path}")


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
                                L1_true_val=None, snr_db=40, device=None,
                                save_path="freq_impact_loss_landscape_onL1.pdf",
                                fixed_num_points=True, num_points=200):
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
        L1_true_val: True fault location in meters (if None, uses fixed["L1"])
        snr_db: SNR for generating observations
        device: torch device
        save_path: Where to save the figure
        fixed_num_points: If True, subsample all ranges to have the same number of
                         points (num_points). This isolates the effect of frequency
                         content from the effect of having more data points.
        num_points: Number of frequency points to use when fixed_num_points=True.
                   Should match freq.num_points from config (default 200).
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

    # Get L1 true value from fixed dict if not provided
    if L1_true_val is None:
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

    # If using fixed number of points, use num_points from config
    if fixed_num_points:
        print(f"Using fixed {num_points} frequency points for all ranges (from config)")

    # Precompute all losses for each frequency range
    all_data = []
    for f_min, f_max, label in freq_ranges:
        mask = (pul_freq_np >= f_min) & (pul_freq_np <= f_max)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"Warning: No frequencies found in range {label}")
            all_data.append(None)
            continue

        if fixed_num_points and len(indices) > num_points:
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
    mode_str = "fixed points (isolating freq content)" if fixed_num_points else "variable points (realistic)"
    print(f"Tested {len(freq_ranges)} frequency ranges with L1_true={L1_true_val:.1f} m, SNR={snr_db} dB, mode={mode_str}")


# =============================================================================
# Benchmark plotting functions
# =============================================================================

def load_results(obs_path, benchmark_path=None, cfg_path="configs/benchmark.yaml"):
    """
    Load saved results from .npz files and config from yaml.

    Args:
        obs_path: Path to observation file (contains computed data only)
        benchmark_path: Optional path to benchmark file (contains RMSE + CRLB results)
        cfg_path: Path to config yaml file (contains fixed params, true_range, etc.)
    """
    # Load config from yaml (fixed params, true_range, etc.)
    cfg = yaml.safe_load(open(cfg_path))

    # Load observation data (computed values only)
    obs_data = np.load(obs_path)

    results = {
        "seed": int(obs_data["seed"]),
        "fixed": cfg["fixed"],
        "true_range": cfg["true_range"],
        "h_obs_sample": obs_data["h_obs_sample_real"] + 1j * obs_data["h_obs_sample_imag"],
        "noise_var_sample": obs_data["noise_var_sample"],
    }

    # Load full frequency and cable parameters if available (for freq impact analysis)
    if "pul_freq" in obs_data:
        results["pul_freq"] = obs_data["pul_freq"]
        results["gamma_full"] = obs_data["gamma_real"] + 1j * obs_data["gamma_imag"]
        results["Zc_full"] = obs_data["Zc_real"] + 1j * obs_data["Zc_imag"]

    # Load benchmark data if provided
    if benchmark_path is not None and pathlib.Path(benchmark_path).exists():
        bench_data = np.load(benchmark_path)
        results["snr_dbs"] = bench_data["snr_dbs"]
        # 5 parameters: L1, ZF_re, ZF_im, ZL_re, ZL_im
        # 1D Grid Search RMSE
        results["rmse_grid"] = {
            "L1": bench_data["rmse_grid_L1"],
            "ZF_re": bench_data["rmse_grid_ZF_re"],
            "ZF_im": bench_data["rmse_grid_ZF_im"],
            "ZL_re": bench_data["rmse_grid_ZL_re"],
            "ZL_im": bench_data["rmse_grid_ZL_im"],
        }
        # 1D Gradient Descent RMSE
        results["rmse_grad"] = {
            "L1": bench_data["rmse_grad_L1"],
            "ZF_re": bench_data["rmse_grad_ZF_re"],
            "ZF_im": bench_data["rmse_grad_ZF_im"],
            "ZL_re": bench_data["rmse_grad_ZL_re"],
            "ZL_im": bench_data["rmse_grad_ZL_im"],
        }
        # sqrt(CRLB)
        results["crlb"] = {
            "L1": bench_data["crlb_L1"],
            "ZF_re": bench_data["crlb_ZF_re"],
            "ZF_im": bench_data["crlb_ZF_im"],
            "ZL_re": bench_data["crlb_ZL_re"],
            "ZL_im": bench_data["crlb_ZL_im"],
        }
        # Quantization error for grid search (if available)
        if "quant_L1" in bench_data:
            results["quant_error"] = {
                "L1": float(bench_data["quant_L1"]),
                "ZF_re": float(bench_data["quant_ZF_re"]),
                "ZF_im": float(bench_data["quant_ZF_im"]),
                "ZL_re": float(bench_data["quant_ZL_re"]),
                "ZL_im": float(bench_data["quant_ZL_im"]),
            }
        else:
            results["quant_error"] = None
    else:
        # Benchmark data not available
        results["snr_dbs"] = None
        results["rmse_grid"] = None
        results["rmse_grad"] = None
        results["crlb"] = None
        results["quant_error"] = None

    return results


def plot_rmse_vs_crlb_1D(results, save_dir="figures/plot_1D_benchmark_figures"):
    """Plot RMSE vs sqrt(CRLB) across SNR for all 5 parameters.

    Layout: 3 rows x 2 columns
    - Row 1: Re[ZF], Im[ZF]
    - Row 2: Re[ZL], Im[ZL]
    - Row 3: L1 (centered, same size as other plots)
    """
    if results["snr_dbs"] is None or results["rmse_grid"] is None:
        print("Warning: Benchmark data not available. Skipping RMSE vs CRLB plots.")
        print("Run 1D_benchmark.py to generate benchmark results first.")
        return

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    snr_dbs = results["snr_dbs"]

    # Layout using 4 columns: each plot spans 2 columns
    # Row 1: ZF_re (cols 0-1), ZF_im (cols 2-3)
    # Row 2: ZL_re (cols 0-1), ZL_im (cols 2-3)
    # Row 3: L1 (cols 1-2, centered)
    layout = [
        [("ZF_re", 0, slice(0, 2)), ("ZF_im", 0, slice(2, 4))],  # Row 1
        [("ZL_re", 1, slice(0, 2)), ("ZL_im", 1, slice(2, 4))],  # Row 2
        [("L1", 2, slice(1, 3))],                                 # Row 3 (centered)
    ]

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    def create_figure_and_plot(rmse_data, rmse_color, rmse_label, filename, quant_error=None):
        fig = plt.figure(figsize=(10, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.25)

        panel = 0
        for row in layout:
            for item in row:
                param, row_idx, col_idx = item
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # Add panel label at top center
                ax.text(
                    0.5, 1.05, panel_labels[panel],
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=13,
                )
                panel += 1

                # RMSE curve
                ax.plot(
                    snr_dbs,
                    rmse_data[param],
                    marker="o",
                    markersize=6,
                    linewidth=2,
                    color=rmse_color,
                    label=rmse_label,
                )
                # sqrt(CRLB) curve
                ax.plot(
                    snr_dbs,
                    results["crlb"][param],
                    marker="x",
                    markersize=6,
                    linewidth=2,
                    linestyle="--",
                    color="black",
                    label=r"$\sqrt{\mathrm{CRLB}}$",
                )
                # Quantization error horizontal line (for grid search only)
                if quant_error is not None and param in quant_error:
                    ax.axhline(
                        y=quant_error[param],
                        color="red",
                        linestyle=":",
                        linewidth=2,
                        label="Quant. Error",
                    )
                ax.set_xlabel("SNR (dB)", fontsize=12)
                if panel % 2 != 0:
                    ax.set_ylabel("RMSE", fontsize=12)
                ax.set_yscale('log')
                #ax.set_title(param_labels[param], fontsize=14)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='lower left', fontsize=10)
                ax.tick_params(labelsize=10)

        plt.savefig(save_path / filename, dpi=300, bbox_inches="tight")
        plt.close()

    # Grid Search figure (with quantization error)
    create_figure_and_plot(
        results["rmse_grid"], "blue", "Grid Search",
        "rmse_vs_crlb_1d_gridsearch.pdf",
        quant_error=results.get("quant_error")
    )

    # Gradient Descent figure
    create_figure_and_plot(
        results["rmse_grad"], "green", "Gradient Descent",
        "rmse_vs_crlb_1d_grad_descent.pdf"
    )

    print(f"Saved 1D RMSE vs CRLB plots")


def plot_all_loss_landscapes(results, fm, device, save_dir="figures/plot_1D_benchmark_figures"):
    """Plot loss landscapes for all 5 parameters in a single figure.

    Layout: 3 rows x 2 columns (using 4-column GridSpec)
    - Row 1: ZF_re, ZF_im
    - Row 2: ZL_re, ZL_im
    - Row 3: L1 (centered)
    """
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Convert sample observation to tensor
    h_obs = torch.tensor(results["h_obs_sample"], device=device, dtype=torch.cfloat)
    noise_var = torch.tensor(results["noise_var_sample"], device=device, dtype=torch.float32)

    lik = ComplexGaussianLik()
    fixed = results["fixed"]
    true_range = results["true_range"]

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

    plt.savefig(save_path / "loss_landscapes_all.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined loss landscape plot to {save_path / 'loss_landscapes_all.pdf'}")


def plot_gamma_impact_on_L1(results, fm, device, snr_db=40, save_dir="figures/plot_1D_benchmark_figures"):
    """Plot gamma impact analysis on L1 loss landscape."""
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    analyze_gamma_impact_on_loss(
        fm=fm,
        L1_lo=results["true_range"]["L1"]["min"],
        L1_hi=results["true_range"]["L1"]["max"],
        fixed=results["fixed"],
        snr_db=snr_db,
        device=device,
        save_path=str(save_path / "gamma_impact_loss_landscape.pdf"),
    )


def plot_L_impact_on_L1(results, fm, device, snr_db=40, save_dir="figures/plot_1D_benchmark_figures"):
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    analyze_L_impact_on_loss(
        fm=fm,
        fixed=results["fixed"],
        snr_db=snr_db,
        device=device,
        save_path=str(save_path / "L_impact_loss_landscape.pdf"),
    )


def plot_freq_impact_on_L1(results, fm, device, snr_db=40, save_dir="figures/plot_1D_benchmark_figures", fixed_num_points=True, num_points=200):
    """
    Plot frequency range impact analysis on L1 loss landscape.

    Args:
        fixed_num_points: If True, all frequency ranges use same number of points
                         (isolates effect of frequency content vs. data count)
        num_points: Number of frequency points to use when fixed_num_points=True
                   (should match freq.num_points from config)
    """
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Check if frequency data is available
    if "pul_freq" not in results:
        print("Warning: Frequency data not available in results. Skipping freq impact plot.")
        print("Re-run 1D_benchmark.py to generate results with frequency data.")
        return

    # Convert numpy arrays to torch tensors
    pul_freq_full = torch.tensor(results["pul_freq"], device=device, dtype=torch.float32)
    gamma_full = torch.tensor(results["gamma_full"], device=device, dtype=torch.cfloat)
    Zc_full = torch.tensor(results["Zc_full"], device=device, dtype=torch.cfloat)

    analyze_freq_impact_on_loss(
        fm=fm,
        fixed=results["fixed"],
        pul_freq_full=pul_freq_full,
        gamma_full=gamma_full,
        Zc_full=Zc_full,
        snr_db=snr_db,
        device=device,
        save_path=str(save_path / "freq_impact_loss_landscape.pdf"),
        fixed_num_points=fixed_num_points,
        num_points=num_points,
    )


def find_most_recent_files(results_dir="results"):
    """
    Find the most recent observation, forward model, and benchmark files.
    Returns tuple of (obs_path, fm_path, benchmark_path) or None for missing files.
    """
    results_path = pathlib.Path(results_dir)
    if not results_path.exists():
        return None, None, None

    # Find most recent observation file
    obs_files = list(results_path.glob("observation_*.npz"))
    obs_path = max(obs_files, key=lambda p: p.stat().st_mtime) if obs_files else None

    # Find most recent forward model file
    fm_files = list(results_path.glob("forward_model_*.pt"))
    fm_path = max(fm_files, key=lambda p: p.stat().st_mtime) if fm_files else None

    # Find most recent benchmark file
    bench_files = list(results_path.glob("1D_benchmark_*.npz"))
    benchmark_path = max(bench_files, key=lambda p: p.stat().st_mtime) if bench_files else None

    return obs_path, fm_path, benchmark_path


def main(obs_path=None, fm_path=None, benchmark_path=None, cfg_path="configs/benchmark.yaml"):
    """
    Main plotting function.
    Args:
        obs_path: Path to observation .npz file. If None, uses most recent.
        fm_path: Path to forward model .pt file. If None, uses most recent.
        benchmark_path: Path to benchmark .npz file. If None, uses most recent.
        cfg_path: Path to config yaml file (fixed params, true_range, num_points)
    """
    # Auto-find most recent files if not provided
    if obs_path is None or fm_path is None or benchmark_path is None:
        auto_obs, auto_fm, auto_bench = find_most_recent_files()
        if obs_path is None:
            obs_path = auto_obs
        if fm_path is None:
            fm_path = auto_fm
        if benchmark_path is None:
            benchmark_path = auto_bench

    # Check required files exist
    if obs_path is None or not pathlib.Path(obs_path).exists():
        print("Error: No observation file found. Run 1D_benchmark.py first.")
        return
    if fm_path is None or not pathlib.Path(fm_path).exists():
        print("Error: No forward model file found. Run 1D_benchmark.py first.")
        return

    print(f"Loading observation data from {obs_path}")
    if benchmark_path and pathlib.Path(benchmark_path).exists():
        print(f"Loading benchmark data from {benchmark_path}")
    else:
        print(f"Benchmark data not found at {benchmark_path}")
        print("Loss landscapes will be plotted, but RMSE vs CRLB plots will be skipped.")
        benchmark_path = None

    results = load_results(obs_path, benchmark_path, cfg_path)

    # Load forward model directly
    print(f"Loading forward model from {fm_path}")
    fm = torch.load(fm_path, weights_only=False)
    device = fm.device

    # Load config for num_points
    cfg = yaml.safe_load(open(cfg_path))

    print("Plotting RMSE vs CRLB for Grid Search and Grad Descent...")
    plot_rmse_vs_crlb_1D(results)

    print("Plotting loss landscapes...")
    plot_all_loss_landscapes(results, fm, device)

    print("Plotting gamma impact analysis on L1...")
    plot_gamma_impact_on_L1(results, fm, device, snr_db=40)

    print("Plotting L impact analysis on L1...")
    plot_L_impact_on_L1(results, fm, device, snr_db=40)

    print("Plotting frequency impact analysis on L1...")
    plot_freq_impact_on_L1(results, fm, device, snr_db=40, num_points=cfg["freq"]["num_points"])

    print("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot results from saved data")
    parser.add_argument(
        "--obs",
        type=str,
        default=None,
        help="Path to observation .npz file (default: most recent in results/)"
    )
    parser.add_argument(
        "--fm",
        type=str,
        default=None,
        help="Path to forward model .pt file (default: most recent in results/)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark .npz file (default: most recent in results/)"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/benchmark.yaml",
        help="Path to config yaml file"
    )
    args = parser.parse_args()

    main(args.obs, args.fm, args.benchmark, args.cfg)
