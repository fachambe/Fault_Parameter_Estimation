import sys, pathlib
import time
import math
import os
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from torch.func import jacfwd
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
torch.set_printoptions(precision=8)  # Show 8 decimal places

OUTPUT_DIR = "both_stages_results"
#OUTPUT_DIR = "two_stage_results_S1=20dB_bayesian" #Name of output folder to save plots
OPTIMIZER = "Adam"  # "Adam" or "Adagrad"
LR = 0.02 #Learning rate for optimizer
NUM_PARTICLES = 12  # Number of particles for SVI
VECTORIZE_PARTICLES = True # Whether to vectorize particles (faster but uses more memory)
SEED = 98 #Seed for theta_true for Bayesian Results
M = 50 #Number of Monte Carlo trials per SNR to calculate RMSE (number of SVI runs)
M2 = 100 #Number of Monte Carlo samples for expectation of FIM and expectation of prior
ALPHA = 3.0 #Hyperparameter of beta prior
# 1 = Constant, 2 = Double RLC, 3 = Motor
FIXED_LOAD_TYPES = [
    3,  # load_0 R6-O3  Motor
    3,  # load_1 R6-O2  Motor
    1,  # load_2 R6-O1  Constant
    3,  # load_3 R5-O4  Motor
    2,  # load_4 R5-O3  Double RLC
    1,  # load_5 R5-O2  Constant
    3,  # load_6 R5-O1  Motor
    2,  # load_7 R4-O4  Double RLC
    3,  # load_8 R4-O3  Motor
    3,  # load_9 R4-O2  Motor
    2,  # load_10 R4-O1  Double RLC
    3,  # load_11 R3-O4  Motor
    1,  # load_12 R3-O3  Constant
    1,  # load_13 R3-O2  Constant
    1,  # load_14 R3-O1  Constant
    3,  # load_15 R2-O4  Motor
    2,  # load_16 R2-O3  Double RLC
    1,  # load_17 R2-O2  Constant
    3,  # load_18 R2-O1  Motor
    1,  # load_19 R1-O4  Constant
    3,  # load_20 R1-O3  Motor
    1,  # load_21 R1-O2  Constant
]

# Network constants
num_loads = 22
num_of_conductors = 4
device = torch.device("cpu")
#frequencies = torch.logspace(torch.log10(torch.tensor(2e6)), torch.log10(torch.tensor(10e6)), 500) #2-10MHz
#frequencies = torch.logspace(torch.log10(torch.tensor(150e3)), torch.log10(torch.tensor(30e6)), 200) #150KHz - 30MHz
#frequencies = torch.logspace(torch.log10(torch.tensor(150e3, device=device)),
#                              torch.log10(torch.tensor(10e6, device=device)), 200, device=device) #150KHz - 500KHz
frequencies = torch.logspace(torch.log10(torch.tensor(150e3, device=device)),
                               torch.log10(torch.tensor(10e6, device=device)), 200, device=device) #150KHz - 500KHz
freq_range_mhz = frequencies / 1e6
omega = 2 * torch.pi * frequencies
num_freqs = len(omega)

# Filename prefix for saving figures (includes config info)
def format_freq(f_hz):
    """Format frequency as kHz or MHz string."""
    if f_hz >= 1e6:
        return f"{int(f_hz / 1e6)}mhz"
    else:
        return f"{int(f_hz / 1e3)}khz"

f_start_str = format_freq(frequencies[0].item())
f_end_str = format_freq(frequencies[-1].item())
freq_range_str = f"{f_start_str}-{f_end_str}"
FILENAME_PREFIX = f"{f_start_str}-{f_end_str}_{OPTIMIZER}_lr{LR}"


#Transmitter/Receiver Constants
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
    "cable_lengths": {  # 30 parameters, set all to 0.25
        f"l_w_{i}": {"value": 0.25, "inferred": True, "range": (5, 10)}
        for i in range(30)
    },
    "conductor_radii": {  # Fixed values, not inferred
        "r_w_servicepanel": {"value": 0.25, "inferred": False, "range": (1.03e-3, 2.06e-3)},
        "r_w_room": {"value": 0.25, "inferred": False, "range": (0.81e-3, 1.29e-3)}
    },
    "fault_parameters": {
        # Normalized position [0.3, 0.7], will be scaled to [0.3L, 0.7L] in forward model
        # For Stage 1 (no fault): inferred=False. For Stage 2: inferred=True.
        "fault_position": {"value": 0.25, "inferred": False, "range": (0.0, 1.0)},
        # Complex fault impedance Z_fault = Z_fault_real + j*Z_fault_imag
        "Z_fault_real": {"value": 0.1, "inferred": False, "range": (0.0, 1000.0)},
        "Z_fault_imag": {"value": 0.25, "inferred": False, "range": (-100.0, 100.0)}
    },
    "loads": {}  # Dynamically generated based on load type
}

def build_params_from_flat(params_flat, param_order):
    """
    Unpack flat parameter tensor into cable_lengths, load_params, and fault_params dictionaries.

    Args:
        params_flat: [p] tensor of parameter values
        param_order: List from get_inferred_param_order()

    Returns:
        cable_lengths: Dict of cable length tensors
        load_params: Dict of load parameter dicts
        fault_params: Dict of fault parameter tensors
    """
    # Initialize with non-inferred (fixed) values
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

    # Override with values from flat tensor
    for i, (ptype, name, subname) in enumerate(param_order):
        if ptype == "cable":
            cable_lengths[name] = params_flat[i]
        elif ptype == "load":
            load_params[name][subname] = params_flat[i]
        elif ptype == "fault_param":
            fault_params[name] = params_flat[i]

    return cable_lengths, load_params, fault_params

def get_true_param_flat():
    """
    Get flat tensor of true inferred parameter values in the order defined by get_inferred_param_order().

    Returns:
        params_flat: [p] tensor of true parameter values in normalized units
    """
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
    """
    Get ordered list of inferred parameters for consistent flat tensor indexing from network_params dict. 

    Returns:
        param_order: List of tuples describing each parameter:
            - ("cable", cable_name, None) for cable lengths
            - ("load", load_name, param_name) for load parameters
        
        num_params: Total number of inferred parameters (p)
    """
    param_order = []

    # Cable lengths first (in sorted order for consistency)
    for cable_name in sorted(network_params["cable_lengths"].keys(), key=lambda x: int(x.split("_")[-1])):
        if network_params["cable_lengths"][cable_name]["inferred"]:
            param_order.append(("cable", cable_name, None))

    # Load parameters (sorted by load number, then by param name)
    for load_name in sorted(network_params["loads"].keys(), key=lambda x: int(x.split("_")[-1])):
        for param_name in sorted(network_params["loads"][load_name].keys()):
            if network_params["loads"][load_name][param_name]["inferred"]:
                param_order.append(("load", load_name, param_name))

    for fault_name in network_params["fault_parameters"]:
        if network_params["fault_parameters"][fault_name]["inferred"]:
            param_order.append(("fault_param", fault_name, None))

    return param_order, len(param_order)

def set_network_params_from_normalized(sampled_theta, param_order_list):
    """
    Update global network_params dict with sampled theta only for inferred keys in param_order_list.
    Args:
        sampled_theta: Sampled tensor of theta of shape [p] where theta in [0, 1]
    """
    counter = 0
    for params in param_order_list:
        if params[0] == "load":
            entity_name = params[1]
            param_name = params[2]
            network_params["loads"][entity_name][param_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1
        elif params[0] == "cable":
            cable_name = params[1]
            network_params["cable_lengths"][cable_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1
        elif params[0] == "fault_param":
            fault_name = params[1]
            network_params["fault_parameters"][fault_name]["value"] = sampled_theta[counter].item()
            counter = counter + 1

def set_top_p_params_inferred(sorted_keys, p_value):
    """
    Set only the top p_value parameters as inferred, disable the rest.

    Args:
        sorted_keys: List of parameter keys sorted by sensitivity (most to least)
        p_value: Number of top parameters to keep inferred
    """
    # First disable all
    for cable_name in network_params["cable_lengths"]:
        network_params["cable_lengths"][cable_name]["inferred"] = False
    for load_name in network_params["loads"]:
        for param_name in network_params["loads"][load_name]:
            if param_name == "R_m2":
                continue
            network_params["loads"][load_name][param_name]["inferred"] = False

    # Enable only top p_value
    for key in sorted_keys[:p_value]:
        if "." in key:
            # Load parameter: "load_0.C_m_leak"
            parts = key.split(".")
            load_name = parts[0]
            param_name = parts[1]
            network_params["loads"][load_name][param_name]["inferred"] = True
        else:
            # Cable length parameter
            network_params["cable_lengths"][key]["inferred"] = True


# ---- Forward Model Instance ----
# Created after network_params is defined. Since network_params is passed by reference,
# any updates (e.g., from generate_load_parameters_deterministic or from changing inferred flag) will be reflected.
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
    """
    Wrapper for no-fault forward model that takes flat parameter tensor.

    Used for Jacobian computation with jacfwd.

    Args:
        params_flat: [p] tensor of normalized parameter values

    Returns:
        H_real: [F, 2] tensor (real and imaginary parts stacked)
    """
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params, _ = build_params_from_flat(params_flat, param_order)

    H_complex = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
    return torch.view_as_real(H_complex)  # [F, 2]


def H_fault_wrapper(params_flat):
    """
    Wrapper for fault forward model that takes flat parameter tensor.

    Used for Jacobian computation with jacfwd.

    Args:
        params_flat: [p] tensor of normalized parameter values

    Returns:
        H_real: [F, 2] tensor (real and imaginary parts stacked)
    """
    param_order, _ = get_inferred_param_order()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order)

    H_complex = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
    return torch.view_as_real(H_complex)  # [F, 2]


def perform_local_sensitivity_analysis(scenario):
    """
    Perform LOCAL sensitivity analysis at the specific θ_true point using Jacobians.

    Unlike global sensitivity which sweeps the entire range, this computes
    the gradient ∂H/∂θ at the current parameter values.

    Args:
        scenario: "no_fault" or "with_fault"

    Returns:
        selected_keys: List of top p most sensitive params sorted by sensitivity
        sorted_keys: List of all params sorted from most sensitive to least sensitive
        sensitivities: List of sensitivity values (%) corresponding to sorted_keys
    """
    params_flat = get_true_param_flat()
    param_order, p = get_inferred_param_order()

    # Select wrapper based on scenario
    if scenario == "no_fault":
        wrapper = H_nofault_wrapper
    elif scenario == "with_fault":
        wrapper = H_fault_wrapper
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Use 'no_fault' or 'with_fault'.")

    # Compute Jacobian dH/dtheta
    J = jacfwd(wrapper)(params_flat)  # [F, 2, P]
    n_freq = J.shape[0]
    n_params = J.shape[2]
    J_flat = J.reshape(n_freq * 2, n_params)  # [F*2, P]
    
    # Sensitivity = norm of each column (each parameter's Jacobian)
    sensitivities_raw = torch.norm(J_flat, dim=0)  # [P]
    # Normalize to percentages
    sensitivities_normalized = sensitivities_raw / sensitivities_raw.sum()

    # Build parameter key list from param_order
    param_keys = []
    for item in param_order:
        if item[0] == 'cable':
            # ('cable', 'l_w_0', None)
            param_keys.append(item[1])
        elif item[0] == 'load':
            # ('load', 'load_0', 'C_m_leak')
            param_keys.append(f"{item[1]}.{item[2]}")
        elif item[0] == 'fault_param':
            # ('fault_param', 'Z_fault_real', None)
            param_keys.append(item[1])
    
    # Create dict mapping param_key -> sensitivity
    sensitivity_dict = {param_keys[i]: sensitivities_normalized[i].item() 
                        for i in range(n_params)}
    
    # Sort by sensitivity (descending)
    sorted_params = sorted(sensitivity_dict.keys(), 
                          key=lambda k: sensitivity_dict[k], 
                          reverse=True)
    
    # Select top p
    selected = sorted_params[:p]

    # Build sensitivities list in sorted order (as percentages)
    sensitivities = [sensitivity_dict[k] * 100 for k in sorted_params]
    
    # Print results
    print("\n--- LOCAL Sensitivity Analysis (at theta_true) ---")
    for idx, k in enumerate(sorted_params):
        if idx == p:
            print(f"--- Top {p} selected above this line ---")
        print(f"{k}: {sensitivity_dict[k]*100:.5f}%")
    
    print(f"\nSelected top {p} most sensitive parameters: {selected}")
    print(f"Number of selected parameters: {len(selected)}")

    return selected, sorted_params, sensitivities

def calculate_bayesian_mse_monte_carlo(snr_db, selected_keys, all_thetas, num_steps, scenario, p_val=None):
    """
    Compute Bayesian MSE via Monte Carlo at specific SNR.

    For each trial:
      1. Use pre-generated θ_true
      2. Generate data y ~ p(y|θ_true)
      3. Run SVI to get estimate θ̂
      4. Compute (θ̂ - θ_true)²

    Args:
        snr_db: SNR level in dB
        selected_keys: List of parameter keys to infer
        all_thetas: Pre-generated theta samples [M, p] (same as used for BCRLB)
        num_steps: Num of SVI Steps
        scenario: "no_fault" or "with_fault"
        p_val: number of inferred parameters


    Returns:
        bayesian_mse_dict: {param_name: Bayesian MSE} in selected keys order
    """
    param_order_list, _ = get_inferred_param_order()
    squared_errors = {key: [] for key in selected_keys}
    snr_lin = 10.0 ** (snr_db / 10.0)

    for m in range(M):
        print(f"Run {m+1}/{M}")
        theta_true_normalized = all_thetas[m]
        set_network_params_from_normalized(theta_true_normalized, param_order_list)

        # 1. Generate clean signal from this theta
        cable_lengths, load_params, fault_params = build_params_from_flat(get_true_param_flat(), param_order_list)
        if scenario == "with_fault":
            H_clean = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
        else:
            H_clean = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
        sigpow = torch.mean(torch.abs(H_clean)**2)
        var_f = sigpow / snr_lin
        std_f = torch.sqrt(var_f / 2)
        H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                    1j * std_f * torch.randn_like(H_clean.imag)
        H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))

        # 2. Run SVI inference
        _, best_params, _ = svi_engine.run_inference(
            H1_noisy, scenario, selected_keys, std_f, num_steps,
            snr_db=snr_db, m=m, M=M, p_val=p_val
        )

        # 3. Extract posterior means for this run
        posterior_means = svi_engine.extract_posterior_means(best_params)

        # 4. Compute squared errors vs true theta
        for key in selected_keys:
            # Get true value directly from network_params
            true_val = svi_engine.get_true_param_value(key)

            posterior_key = key.replace(".", "_")
            if posterior_key in posterior_means:
                estimate = posterior_means[posterior_key]
                squared_errors[key].append((estimate - true_val)**2)

    # Average squared errors
    bayesian_mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
        
    return bayesian_mse_dict

def calculate_mse_monte_carlo(var_f, selected_keys, snr_db, num_steps, scenario, p_val=None):
    """
    Compute Frequentist MSE via Monte Carlo at specific SNR.
    For each trial:
      1. Generate data y ~ p(y|θ_true) at theta_true
      2. Run SVI to get estimate θ̂
      3. Compute (θ̂ - θ_true)²

    Args:
        var_f: Noise variance
        selected_keys: List of parameter keys to infer
        snr_db: SNR in dB
        num_steps: Num of SVI steps
        scenario: "no_fault" or "with_fault"
        p_val: number of inferred parameters

    Returns:
        mse_dict: {param_name: MSE} in selected keys order
        last_best_params: best_params {param_name: best_param} from the last Monte Carlo run (for CI plotting). 
        Note this is not in selected_keys order and param_name not same as param_name in mse_dict. 
    """
    param_order_list, _ = get_inferred_param_order()
    params_flat = get_true_param_flat()

    squared_errors = {key: [] for key in selected_keys}
    std_f = torch.sqrt(var_f / 2)
    last_best_params = None

    # Compute H_clean and then H_noisy for SVI from true network parameters not inferred
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    if scenario == 'with_fault':
        H_clean = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
    else:
        H_clean = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)

    for m in range(M):
        print(f"Run {m+1}/{M}")

        # 1. Generate noisy observation (different observation each run because of noise)
        H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                     1j * std_f * torch.randn_like(H_clean.imag)
        H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))  # [1, F, 2]

        # 2. Run SVI inference
        _, best_params, _ = svi_engine.run_inference(
            H1_noisy, scenario, selected_keys, std_f, num_steps,
            snr_db=snr_db, m=m, M=M, p_val=p_val
        )

        # 3. Extract posterior means for this run
        posterior_means = svi_engine.extract_posterior_means(best_params)

        # 4. Compute squared errors vs true theta
        for key in selected_keys:
            # Get true value directly from network_params
            true_val = svi_engine.get_true_param_value(key)

            posterior_key = key.replace(".", "_")
            if posterior_key in posterior_means:
                estimate = posterior_means[posterior_key]
                squared_errors[key].append((estimate - true_val)**2)

        # Keep last run's best_params for CI plotting
        last_best_params = best_params

    # Average squared errors
    mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
    return mse_dict, last_best_params

def main():
    start_time = time.perf_counter()

    # ==================== STAGE 1 CONFIG ====================
    snr_db_s1 = 40
    num_steps_s1 = 1000

    # ==================== STAGE 2 CONFIG ====================
    snr_dbs_s2 = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    num_steps_s2 = 250
    M_s2 = 50  # Monte Carlo trials for Stage 2 BRMSE

    # ==================== SETUP ====================
    total_params, load_types = generate_load_parameters_deterministic(network_params, FIXED_LOAD_TYPES)
    num_cable_params = len(network_params["cable_lengths"])
    num_fault_params = len(network_params["fault_parameters"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of fault parameters: {num_fault_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params + num_fault_params}")
    print(f"Load type distribution: {load_types}")

    # Get param order for all network parameters (Stage 1)
    param_order_list_full, p_tot = get_inferred_param_order()

    # TRUE network params stay at default (0.25) to match run_stage2_mtl.py
    # This allows fair comparison: Stage 2 standalone uses 0.25, two-stage estimates it

    # Store true network param values (for later comparison) - all should be 0.25
    true_network_values = {key: svi_engine.get_true_param_value(key)
                          for key in [entry[1] if entry[0] == "cable" else f"{entry[1]}.{entry[2]}"
                                     for entry in param_order_list_full]}

    # ==================== STAGE 1: Infer Network Params ====================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Inferring {p_tot} network parameters at {snr_db_s1}dB SNR")
    print('='*60)

    # Perform sensitivity analysis
    _, sorted_keys_all, sensitivities_all = perform_local_sensitivity_analysis("no_fault")
    selected_keys_s1 = sorted_keys_all

    # Generate clean TF (no fault) and add noise
    params_flat = get_true_param_flat()
    cable_lengths, load_params, _ = build_params_from_flat(params_flat, param_order_list_full)
    H_clean_s1 = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)
    sigpow_s1 = torch.mean(torch.abs(H_clean_s1)**2)
    snr_lin_s1 = 10.0 ** (snr_db_s1 / 10.0)
    var_f_s1 = sigpow_s1 / snr_lin_s1
    std_f_s1 = torch.sqrt(var_f_s1 / 2)

    H1_noisy_c = H_clean_s1 + std_f_s1 * torch.randn_like(H_clean_s1.real) + \
                    1j * std_f_s1 * torch.randn_like(H_clean_s1.imag)
    H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))  # [1, F, 2]

    # Run SVI inference for Stage 1
    _, best_params_s1, _ = svi_engine.run_inference(
        H1_noisy, "no_fault", selected_keys_s1, std_f_s1, num_steps_s1,
        snr_db=snr_db_s1, m=0, M=1, p_val=p_tot
    )

    # Extract Stage 1 estimates
    posterior_means_s1 = svi_engine.extract_posterior_means(best_params_s1)

    # Print Stage 1 results (compare estimates to true values)
    print(f"\n--- Stage 1 Results (first 10 params) ---")
    for i, key in enumerate(selected_keys_s1[:10]):
        true_val = true_network_values[key]
        est_key = key.replace(".", "_")
        est_val = posterior_means_s1.get(est_key, float('nan'))
        error = abs(est_val - true_val)
        print(f"  {key:20s}: true={true_val:.4f}, est={est_val:.4f}, error={error:.4f}")

    # ==================== TRANSITION: Update network_params with Stage 1 estimates ====================
    print(f"\n{'='*60}")
    print("Updating network_params with Stage 1 estimates for Stage 2...")
    print('='*60)

    # Update network_params with estimated values (these become "fixed" for Stage 2)
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
                network_params["cable_lengths"][name]["inferred"] = False  # Fixed for Stage 2
            elif ptype == "load":
                network_params["loads"][name][subname]["value"] = posterior_means_s1[est_key]
                network_params["loads"][name][subname]["inferred"] = False  # Fixed for Stage 2

    # Enable fault parameters for inference in Stage 2
    for fault_name in network_params["fault_parameters"]:
        network_params["fault_parameters"][fault_name]["inferred"] = True

    # ==================== STAGE 2: Infer Fault Params (Bayesian) ====================
    # Get new param order (only fault params now)
    param_order_list_s2, p_fault = get_inferred_param_order()
    selected_keys_s2 = [entry[1] for entry in param_order_list_s2]  # fault_position, Z_fault_real, Z_fault_imag
    print(f"\nSTAGE 2: Inferring {p_fault} fault parameters: {selected_keys_s2}")

    # Sample fault parameter values from prior for Bayesian
    torch.manual_seed(SEED)
    beta_dist = torch.distributions.Beta(ALPHA, ALPHA)
    theta_bayesian_fault = beta_dist.sample((M_s2, p_fault))

    # Initialize results storage (only BRMSE, BCRLB comes from ideal run_stage2_mtl.py)
    rmse_results = {key: [] for key in selected_keys_s2}

    # Store estimated network values for restoring after data generation
    estimated_network_values = {}
    for entry in param_order_list_full:
        ptype, name, subname = entry
        if ptype == "cable":
            estimated_network_values[name] = network_params["cable_lengths"][name]["value"]
        elif ptype == "load":
            estimated_network_values[f"{name}.{subname}"] = network_params["loads"][name][subname]["value"]

    # SNR sweep for Stage 2
    for snr_db in snr_dbs_s2:
        print(f"\n{'='*50}")
        print(f"Stage 2 | SNR = {snr_db} dB | Bayesian (Two-Stage)")
        print('='*50)

        if snr_db >= 30:
            num_steps_s2 = 500
        else:
            num_steps_s2 = 250

        squared_errors = {key: [] for key in selected_keys_s2}
        snr_lin = 10.0 ** (snr_db / 10.0)

        for m in range(M_s2):
            print(f"  Run {m+1}/{M_s2}")

            # Set fault params to this MC sample's true value
            theta_true_this = theta_bayesian_fault[m]
            for i, entry in enumerate(param_order_list_s2):
                fault_name = entry[1]
                network_params["fault_parameters"][fault_name]["value"] = theta_true_this[i].item()

            # ===== Generate data using TRUE network params =====
            # Temporarily restore TRUE network values for data generation
            for entry in param_order_list_full:
                ptype, name, subname = entry
                if ptype == "cable":
                    key = name
                    network_params["cable_lengths"][name]["value"] = true_network_values[key]
                elif ptype == "load":
                    key = f"{name}.{subname}"
                    network_params["loads"][name][subname]["value"] = true_network_values[key]

            # Generate H_clean using TRUE network + TRUE fault
            params_flat_s2 = get_true_param_flat()
            cable_lengths, load_params, fault_params = build_params_from_flat(params_flat_s2, param_order_list_s2)
            H_clean_s2 = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
            sigpow_s2 = torch.mean(torch.abs(H_clean_s2)**2)
            var_f_s2 = sigpow_s2 / snr_lin
            std_f_s2 = torch.sqrt(var_f_s2 / 2)

            H2_noisy_c = H_clean_s2 + std_f_s2 * torch.randn_like(H_clean_s2.real) + \
                        1j * std_f_s2 * torch.randn_like(H_clean_s2.imag)
            H2_noisy = torch.view_as_real(H2_noisy_c.unsqueeze(0))

            # ===== Restore ESTIMATED network params for inference =====
            for entry in param_order_list_full:
                ptype, name, subname = entry
                if ptype == "cable":
                    key = name
                    network_params["cable_lengths"][name]["value"] = estimated_network_values[key]
                elif ptype == "load":
                    key = f"{name}.{subname}"
                    network_params["loads"][name][subname]["value"] = estimated_network_values[key]

            # Run SVI inference with multi-start for high SNR
            # (forward model now uses ESTIMATED network params)
            if snr_db >= 10:
                init_values = [0.0, -1.386, 1.386]  # sigmoid: 0.5, 0.2, 0.8
                best_loss = float('inf')
                best_params_s2 = None
                for init_val in init_values:
                    losses, params, _ = svi_engine.run_inference(
                        H2_noisy, "with_fault", selected_keys_s2, std_f_s2, num_steps_s2,
                        snr_db=snr_db, m=m, M=M_s2, p_val=p_fault,
                        verbose=(init_val == 0.0),
                        fault_position_init=init_val
                    )
                    final_loss = losses[-1] if losses else float('inf')
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_params_s2 = params
            else:
                _, best_params_s2, _ = svi_engine.run_inference(
                    H2_noisy, "with_fault", selected_keys_s2, std_f_s2, num_steps_s2,
                    snr_db=snr_db, m=m, M=M_s2, p_val=p_fault
                )

            # Extract posterior means and compute squared errors
            posterior_means_s2 = svi_engine.extract_posterior_means(best_params_s2)
            for key in selected_keys_s2:
                true_val = theta_true_this[selected_keys_s2.index(key)].item()
                posterior_key = key.replace(".", "_")
                if posterior_key in posterior_means_s2:
                    estimate = posterior_means_s2[posterior_key]
                    squared_errors[key].append((estimate - true_val)**2)

        # Compute BRMSE for this SNR
        for key in selected_keys_s2:
            brmse = math.sqrt(sum(squared_errors[key]) / len(squared_errors[key]))
            rmse_results[key].append(brmse)
            print(f"  {key}: Two-stage BRMSE = {brmse:.4f}")

    # ==================== SAVE RESULTS ====================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare save data
    fp_range = network_params["fault_parameters"]["fault_position"]["range"]
    save_data = {
        'snr_dbs': np.array(snr_dbs_s2),
        'selected_keys': np.array(selected_keys_s2, dtype=object),
        'selected_keys_s1': np.array(selected_keys_s1, dtype=object),
        'M': M_s2,
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
        # True network params are all 0.25 (default) - store as tensor for consistency
        'theta_true_network': np.full(p_tot, 0.25),
    }

    # Add Stage 1 variational params (for CI plots)
    for param_name, param_val in best_params_s1.items():
        if hasattr(param_val, 'detach'):
            save_data[f"s1_{param_name}"] = param_val.detach().cpu().numpy()
        else:
            save_data[f"s1_{param_name}"] = np.array(param_val)

    # Add RMSE results (BCRLB comes from ideal run_stage2_mtl.py)
    for key in selected_keys_s2:
        safe_key = key.replace(".", "_")
        save_data[f"{safe_key}_rmse"] = np.array(rmse_results[key])

    save_path = os.path.join(OUTPUT_DIR, f"bothstages_results_{freq_range_str}_M{M_s2}_S1snr{snr_db_s1}.npz")
    np.savez(save_path, **save_data)
    print(f"\nResults saved to: {save_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()