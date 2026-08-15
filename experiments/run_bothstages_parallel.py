"""
Parallelized version of run_bothstages.py

Parallelizes Stage 2 Monte Carlo trials using joblib (same as run_stage2_mtl_parallel.py).
Stage 1 remains sequential (single inference run).
"""
import sys, pathlib
import time
import math
import os
import copy
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from torch.func import jacfwd
from joblib import Parallel, delayed
from core.mtl_utils import *
from core.loads import *
from core.forward_mtl import MTLForwardModel
from core.crlb import (
    compute_real_CRLB,
    beta_prior_fim_closed_form,
    compute_expected_data_fim,
    key_to_tuple,
)
from core.inference import SVIEngine, InferenceConfig
torch.set_printoptions(precision=8)

# ==================== CONFIGURATION ====================
OUTPUT_DIR = "both_stages_results"
OPTIMIZER = "Adam"
LR = 0.02
NUM_PARTICLES = 12
VECTORIZE_PARTICLES = True
SEED = 98
M = 100  # Number of Monte Carlo trials per SNR
M2 = 100
ALPHA = 3.0
N_JOBS = -1  # Number of parallel jobs (-1 = use all cores)

FIXED_LOAD_TYPES = [
    3, 3, 1, 3, 2, 1, 3, 2, 3, 3, 2, 3, 1, 1, 1, 3, 2, 1, 3, 1, 3, 1,
]

# Network constants
num_loads = 22
num_of_conductors = 4
device = torch.device("cpu")
frequencies = torch.logspace(
    torch.log10(torch.tensor(150e3, device=device)),
    torch.log10(torch.tensor(10e6, device=device)), 200, device=device
)
freq_range_mhz = frequencies / 1e6
omega = 2 * torch.pi * frequencies
num_freqs = len(omega)

def format_freq(f_hz):
    if f_hz >= 1e6:
        return f"{int(f_hz / 1e6)}mhz"
    else:
        return f"{int(f_hz / 1e3)}khz"

f_start_str = format_freq(frequencies[0].item())
f_end_str = format_freq(frequencies[-1].item())
freq_range_str = f"{f_start_str}-{f_end_str}"

# Transmitter/Receiver Constants
Z_RG = Z_R1 = Z_R2 = 50.0
Z_R3 = 50.0
ZT0 = ZTG1 = ZTG2 = 50.0
ZTG3 = 50.0
ZT12 = 100.0
ZT13 = ZT23 = 100.0
Z_rec = calculate_receiver_load(Z_RG, Z_R1, Z_R2, Z_R3).to(device)
Y_rec = torch.linalg.inv(Z_rec)
Z_rec = Z_rec.unsqueeze(0).repeat(num_freqs, 1, 1)
Y_rec = Y_rec.unsqueeze(0).repeat(num_freqs, 1, 1)

BACKBONE_KEYS = ["l_w_0", "l_w_1", "l_w_4", "l_w_25", "l_w_28"]

# ---- Global Network Parameter Dictionary ----
network_params = {
    "cable_lengths": {
        f"l_w_{i}": {"value": 0.25, "inferred": True, "range": (5, 10)}
        for i in range(30)
    },
    "conductor_radii": {
        "r_w_servicepanel": {"value": 0.25, "inferred": False, "range": (1.03e-3, 2.06e-3)},
        "r_w_room": {"value": 0.25, "inferred": False, "range": (0.81e-3, 1.29e-3)}
    },
    "fault_parameters": {
        "fault_position": {"value": 0.25, "inferred": False, "range": (0.0, 1.0)},
        "Z_fault_real": {"value": 0.1, "inferred": False, "range": (0.0, 1000.0)},
        "Z_fault_imag": {"value": 0.25, "inferred": False, "range": (-100.0, 100.0)}
    },
    "loads": {}
}


def build_params_from_flat(params_flat, param_order):
    """Unpack flat parameter tensor into dictionaries."""
    cable_lengths = {}
    for cable_name, cable_info in network_params["cable_lengths"].items():
        cable_lengths[cable_name] = torch.tensor(cable_info["value"], dtype=torch.float32, device=device)

    load_params = {}
    for load_name, params in network_params["loads"].items():
        load_params[load_name] = {}
        for param_name, param_info in params.items():
            load_params[load_name][param_name] = torch.tensor(param_info["value"], dtype=torch.float32, device=device)

    fault_params = {}
    for fault_name, fault_info in network_params["fault_parameters"].items():
        fault_params[fault_name] = torch.tensor(fault_info["value"], dtype=torch.float32, device=device)

    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            cable_lengths[name] = params_flat[i]
        elif ptype == "load":
            load_params[name][subname] = params_flat[i]
        elif ptype == "fault_param":
            fault_params[name] = params_flat[i]

    return cable_lengths, load_params, fault_params


def get_true_param_flat():
    """Get flat tensor of true parameter values."""
    param_order, num_params = get_inferred_param_order()
    params_flat = torch.zeros(num_params, dtype=torch.float32, device=device)

    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            params_flat[i] = network_params["cable_lengths"][name]["value"]
        elif ptype == "load":
            params_flat[i] = network_params["loads"][name][subname]["value"]
        elif ptype == "fault_param":
            params_flat[i] = network_params["fault_parameters"][name]["value"]

    return params_flat


def get_inferred_param_order():
    """Get ordered list of inferred parameters."""
    param_order = []

    for cable_name in sorted(network_params["cable_lengths"].keys(), key=lambda x: int(x.split("_")[-1])):
        if network_params["cable_lengths"][cable_name]["inferred"]:
            param_order.append(("cable", cable_name, None))

    for load_name in sorted(network_params["loads"].keys(), key=lambda x: int(x.split("_")[-1])):
        for param_name in sorted(network_params["loads"][load_name].keys()):
            if network_params["loads"][load_name][param_name]["inferred"]:
                param_order.append(("load", load_name, param_name))

    for fault_name in network_params["fault_parameters"]:
        if network_params["fault_parameters"][fault_name]["inferred"]:
            param_order.append(("fault_param", fault_name, None))

    return param_order, len(param_order)


def set_network_params_from_normalized(sampled_theta, param_order_list):
    """Update global network_params dict with sampled theta."""
    counter = 0
    for params in param_order_list:
        if params[0] == "load":
            entity_name = params[1]
            param_name = params[2]
            network_params["loads"][entity_name][param_name]["value"] = sampled_theta[counter].item()
            counter += 1
        elif params[0] == "cable":
            cable_name = params[1]
            network_params["cable_lengths"][cable_name]["value"] = sampled_theta[counter].item()
            counter += 1
        elif params[0] == "fault_param":
            fault_name = params[1]
            network_params["fault_parameters"][fault_name]["value"] = sampled_theta[counter].item()
            counter += 1


# ---- Forward Model Instance ----
forward_model = MTLForwardModel(frequencies, network_params, device=device)

# ---- SVI Inference Engine ----
inference_config = InferenceConfig(
    alpha=ALPHA,
    num_particles=NUM_PARTICLES,
    vectorize_particles=VECTORIZE_PARTICLES,
    optimizer=OPTIMIZER,
    learning_rate=LR,
    device=device,
)
svi_engine = SVIEngine(forward_model, network_params, inference_config)


def H_nofault_wrapper(params_flat):
    """Wrapper for no-fault forward model."""
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params, _ = build_params_from_flat(params_flat, param_order)
    H_complex = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
    return torch.view_as_real(H_complex)


def perform_local_sensitivity_analysis(scenario):
    """Perform LOCAL sensitivity analysis at θ_true."""
    params_flat = get_true_param_flat()
    param_order, p = get_inferred_param_order()

    if scenario == "no_fault":
        wrapper = H_nofault_wrapper
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    J = jacfwd(wrapper)(params_flat)
    n_freq = J.shape[0]
    n_params = J.shape[2]
    J_flat = J.reshape(n_freq * 2, n_params)

    sensitivities_raw = torch.norm(J_flat, dim=0)
    sensitivities_normalized = sensitivities_raw / sensitivities_raw.sum()

    param_keys = []
    for item in param_order:
        if item[0] == 'cable':
            param_keys.append(item[1])
        elif item[0] == 'load':
            param_keys.append(f"{item[1]}.{item[2]}")
        elif item[0] == 'fault_param':
            param_keys.append(item[1])

    sensitivity_dict = {param_keys[i]: sensitivities_normalized[i].item() for i in range(n_params)}
    sorted_params = sorted(sensitivity_dict.keys(), key=lambda k: sensitivity_dict[k], reverse=True)
    selected = sorted_params[:p]
    sensitivities = [sensitivity_dict[k] * 100 for k in sorted_params]

    print("\n--- LOCAL Sensitivity Analysis (at theta_true) ---")
    for idx, k in enumerate(sorted_params):
        if idx == p:
            print(f"--- Top {p} selected above this line ---")
        print(f"{k}: {sensitivity_dict[k]*100:.5f}%")

    print(f"\nSelected top {p} most sensitive parameters: {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    return selected, sorted_params, sensitivities


def run_single_twostage_trial(m, theta_true_fault, snr_db, selected_keys_s2,
                               true_network_values, estimated_network_values,
                               num_steps_s2, p_fault):
    """
    Run a single two-stage Bayesian MC trial for Stage 2.

    Each worker creates its own forward_model and svi_engine to avoid shared state.
    Includes multi-start optimization for fault_position.

    Args:
        m: Trial index
        theta_true_fault: True fault params for this trial [p_fault] as list
        snr_db: SNR in dB
        selected_keys_s2: List of fault parameter keys
        true_network_values: Dict of true network param values
        estimated_network_values: Dict of Stage 1 estimated network param values
        num_steps_s2: Number of SVI steps
        p_fault: Number of fault parameters

    Returns:
        trial_errors: Dict {key: squared_error} for this trial
    """
    # Each worker needs its own copy of network_params
    local_network_params = copy.deepcopy(network_params)

    # Re-generate load parameters structure for this worker
    generate_load_parameters_deterministic(local_network_params, FIXED_LOAD_TYPES)

    # Set network params to ESTIMATED values (from Stage 1) for inference
    for key, val in estimated_network_values.items():
        if key.startswith("l_w_"):
            local_network_params["cable_lengths"][key]["value"] = val
            local_network_params["cable_lengths"][key]["inferred"] = False
        elif "." in key:
            load_name, param_name = key.split(".")
            local_network_params["loads"][load_name][param_name]["value"] = val
            local_network_params["loads"][load_name][param_name]["inferred"] = False

    # Enable fault params for inference
    for fault_name in local_network_params["fault_parameters"]:
        local_network_params["fault_parameters"][fault_name]["inferred"] = True

    # Create worker's forward model and SVI engine
    local_forward_model = MTLForwardModel(frequencies, local_network_params, device=device)
    local_config = InferenceConfig(
        alpha=ALPHA,
        num_particles=NUM_PARTICLES,
        vectorize_particles=VECTORIZE_PARTICLES,
        optimizer=OPTIMIZER,
        learning_rate=LR,
        device=device,
    )
    local_svi_engine = SVIEngine(local_forward_model, local_network_params, local_config)

    # ===== Generate data using TRUE network params (directly, no global state swap) =====
    # Build cable_lengths and load_params from true_network_values
    cable_lengths = {name: torch.tensor(val, dtype=torch.float32, device=device)
                   for name, val in true_network_values.items() if name.startswith("l_w_")}
    load_params = {}
    for key, val in true_network_values.items():
        if "." in key:
            load_name, param_name = key.split(".")
            if load_name not in load_params:
                load_params[load_name] = {}
            load_params[load_name][param_name] = torch.tensor(val, dtype=torch.float32, device=device)

    # Build fault_params from this MC sample's true fault values
    fault_params = {
        "fault_position": torch.tensor(theta_true_fault[0], dtype=torch.float32, device=device),
        "Z_fault_real": torch.tensor(theta_true_fault[1], dtype=torch.float32, device=device),
        "Z_fault_imag": torch.tensor(theta_true_fault[2], dtype=torch.float32, device=device),
    }

    H_clean_s2 = local_forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)

    sigpow_s2 = torch.mean(torch.abs(H_clean_s2)**2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    var_f_s2 = sigpow_s2 / snr_lin
    std_f_s2 = torch.sqrt(var_f_s2 / 2)

    # Add noise
    H2_noisy_c = H_clean_s2 + std_f_s2 * torch.randn_like(H_clean_s2.real) + \
                1j * std_f_s2 * torch.randn_like(H_clean_s2.imag)
    H2_noisy = torch.view_as_real(H2_noisy_c.unsqueeze(0))

    # ===== Inference uses ESTIMATED network params (already set in local_network_params) =====
    # Multi-start optimization
    init_values = [0.0, -1.386, 1.386]  # sigmoid: 0.5, 0.2, 0.8
    best_loss = float('inf')
    best_params_s2 = None

    for init_val in init_values:
        losses, params, _ = local_svi_engine.run_inference(
            H2_noisy, "with_fault", selected_keys_s2, std_f_s2, num_steps_s2,
            snr_db=snr_db, m=m, M=M, p_val=p_fault,
            verbose=False,
            fault_position_init=init_val
        )
        final_loss = losses[-1] if losses else float('inf')
        if final_loss < best_loss:
            best_loss = final_loss
            best_params_s2 = params

    # Extract posterior means and compute squared errors
    posterior_means_s2 = local_svi_engine.extract_posterior_means(best_params_s2)

    trial_errors = {}
    for key in selected_keys_s2:
        true_val = theta_true_fault[selected_keys_s2.index(key)]
        posterior_key = key.replace(".", "_")
        if posterior_key in posterior_means_s2:
            estimate = posterior_means_s2[posterior_key]
            trial_errors[key] = (estimate - true_val)**2

    return trial_errors


def main():
    start_time = time.perf_counter()

    # ==================== STAGE 1 CONFIG ====================
    snr_db_s1 = 40
    num_steps_s1 = 2000

    # ==================== STAGE 2 CONFIG ====================
    snr_dbs_s2 = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    num_steps_s2 = 250

    # ==================== SETUP ====================
    total_params, load_types = generate_load_parameters_deterministic(network_params, FIXED_LOAD_TYPES)
    num_cable_params = len(network_params["cable_lengths"])
    num_fault_params = len(network_params["fault_parameters"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of fault parameters: {num_fault_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params + num_fault_params}")
    print(f"Load type distribution: {load_types}")
    print(f"Using N_JOBS={N_JOBS} for parallel Stage 2")

    # Get param order for all network parameters (Stage 1)
    param_order_list_full, p_tot = get_inferred_param_order()

    # Store true network param values
    true_network_values = {
        key: svi_engine.get_true_param_value(key)
        for key in [entry[1] if entry[0] == "cable" else f"{entry[1]}.{entry[2]}"
                   for entry in param_order_list_full]
    }

    # ==================== STAGE 1: Infer Network Params ====================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Inferring {p_tot} network parameters at {snr_db_s1}dB SNR")
    print('='*60)

    _, sorted_keys_all, _ = perform_local_sensitivity_analysis("no_fault")
    selected_keys_s1 = sorted_keys_all

    # Generate clean TF and add noise
    params_flat = get_true_param_flat()
    cable_lengths, load_params, _ = build_params_from_flat(params_flat, param_order_list_full)
    H_clean_s1 = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
    sigpow_s1 = torch.mean(torch.abs(H_clean_s1)**2)
    snr_lin_s1 = 10.0 ** (snr_db_s1 / 10.0)
    var_f_s1 = sigpow_s1 / snr_lin_s1
    std_f_s1 = torch.sqrt(var_f_s1 / 2)

    H1_noisy_c = H_clean_s1 + std_f_s1 * torch.randn_like(H_clean_s1.real) + \
                    1j * std_f_s1 * torch.randn_like(H_clean_s1.imag)
    H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))

    # Run SVI inference for Stage 1
    losses_s1, best_params_s1, param_history_s1 = svi_engine.run_inference(
        H1_noisy, "no_fault", selected_keys_s1, std_f_s1, num_steps_s1,
        snr_db=snr_db_s1, m=0, M=1, p_val=p_tot
    )

    posterior_means_s1 = svi_engine.extract_posterior_means(best_params_s1)

    print(f"\n--- Stage 1 Results (first 10 params) ---")
    for i, key in enumerate(selected_keys_s1[:10]):
        true_val = true_network_values[key]
        est_key = key.replace(".", "_")
        est_val = posterior_means_s1.get(est_key, float('nan'))
        error = abs(est_val - true_val)
        print(f"  {key:20s}: true={true_val:.4f}, est={est_val:.4f}, error={error:.4f}")

    # ==================== TRANSITION ====================
    print(f"\n{'='*60}")
    print("Updating network_params with Stage 1 estimates for Stage 2...")
    print('='*60)

    for entry in param_order_list_full:
        ptype, name, subname = entry
        if ptype == "cable":
            key = name
            est_key = name
        elif ptype == "load":
            key = f"{name}.{subname}"
            est_key = f"{name}_{subname}"
        else:
            continue

        if est_key in posterior_means_s1:
            if ptype == "cable":
                network_params["cable_lengths"][name]["value"] = posterior_means_s1[est_key]
                network_params["cable_lengths"][name]["inferred"] = False
            elif ptype == "load":
                network_params["loads"][name][subname]["value"] = posterior_means_s1[est_key]
                network_params["loads"][name][subname]["inferred"] = False

    for fault_name in network_params["fault_parameters"]:
        network_params["fault_parameters"][fault_name]["inferred"] = True

    # ==================== STAGE 2: Parallel Inference ====================
    param_order_list_s2, p_fault = get_inferred_param_order()
    selected_keys_s2 = [entry[1] for entry in param_order_list_s2]
    print(f"\nSTAGE 2: Inferring {p_fault} fault parameters: {selected_keys_s2}")

    # Sample fault parameter values from prior
    torch.manual_seed(SEED-1)
    beta_dist = torch.distributions.Beta(ALPHA, ALPHA)
    theta_bayesian_fault = beta_dist.sample((M, p_fault))

    rmse_results = {key: [] for key in selected_keys_s2}

    # Store estimated network values
    estimated_network_values = {}
    for entry in param_order_list_full:
        ptype, name, subname = entry
        if ptype == "cable":
            estimated_network_values[name] = network_params["cable_lengths"][name]["value"]
        elif ptype == "load":
            estimated_network_values[f"{name}.{subname}"] = network_params["loads"][name][subname]["value"]

    # SNR sweep for Stage 2 (parallelized)
    for snr_db in snr_dbs_s2:
        print(f"\n{'='*50}")
        print(f"Stage 2 | SNR = {snr_db} dB | Bayesian (Two-Stage) | Parallel")
        print('='*50)
        print(f"Running {M} MC trials in parallel...")

        # Run trials in parallel using joblib
        results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
            delayed(run_single_twostage_trial)(
                m,
                theta_bayesian_fault[m].tolist(),
                snr_db,
                selected_keys_s2,
                true_network_values,
                estimated_network_values,
                num_steps_s2,
                p_fault
            )
            for m in range(M)
        )

        # Aggregate results
        squared_errors_all = {key: [] for key in selected_keys_s2}
        for trial_errors in results:
            for key, err in trial_errors.items():
                squared_errors_all[key].append(err)

        # Compute BRMSE for this SNR
        for key in selected_keys_s2:
            if squared_errors_all[key]:
                brmse = math.sqrt(sum(squared_errors_all[key]) / len(squared_errors_all[key]))
                rmse_results[key].append(brmse)
                print(f"  {key}: Two-stage BRMSE = {brmse:.4f}")
            else:
                rmse_results[key].append(float('nan'))

    # ==================== SAVE RESULTS ====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fp_range = network_params["fault_parameters"]["fault_position"]["range"]
    save_data = {
        'snr_dbs': np.array(snr_dbs_s2),
        'selected_keys': np.array(selected_keys_s2, dtype=object),
        'selected_keys_s1': np.array(selected_keys_s1, dtype=object),
        'M': M,
        'ALPHA': ALPHA,
        'SEED': SEED,
        'snr_db_s1': snr_db_s1,
        'freq_range_str': freq_range_str,
        'fp_range': fp_range,
        'mode': 'bayesian',
        'frequencies': frequencies.numpy(),
        'network_params': network_params,
        'true_network_values': true_network_values,
        'estimated_network_values': estimated_network_values,
        'theta_true_network': np.full(p_tot, 0.25),
    }

    # Save Stage 1 losses and param_history as dicts
    save_data['losses_s1'] = np.array(losses_s1)
    save_data['param_history_s1'] = param_history_s1  # dict, saved with allow_pickle

    # Save Stage 2 BRMSE results
    for key in selected_keys_s2:
        safe_key = key.replace(".", "_")
        save_data[f"{safe_key}_rmse"] = np.array(rmse_results[key])

    save_path = os.path.join(OUTPUT_DIR, f"bothstages_parallel_{freq_range_str}_M{M}_S1snr{snr_db_s1}.npz")
    np.savez(save_path, **save_data)
    print(f"\nResults saved to: {save_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
