import sys, pathlib
import time
import math
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from torch.func import jacfwd
from joblib import Parallel, delayed
from core.mtl_utils import *
from core.loads import *
from core.forward_mtl import MTLForwardModel
from core.crlb import (
    compute_real_CRLB,
    compute_real_BCRLB,
    beta_prior_fim_closed_form,
    compute_expected_data_fim,
    key_to_tuple,
)
from core.inference import SVIEngine, InferenceConfig
torch.set_printoptions(precision=8)  # Show 8 decimal places

OUTPUT_DIR = "stage_2_results"
#OUTPUT_DIR = "two_stage_results_S1=20dB_bayesian" #Name of output folder to save plots
OPTIMIZER = "Adam"  # "Adam" or "Adagrad"
LR = 0.02 #Learning rate for optimizer
NUM_PARTICLES = 12  # Number of particles for SVI
VECTORIZE_PARTICLES = True # Whether to vectorize particles (faster but uses more memory)
SEED = 98 #Seed for theta_true for Bayesian Results
M = 100 #Number of Monte Carlo trials per SNR to calculate RMSE (number of SVI runs)
M2 = 100 #Number of Monte Carlo samples for expectation of FIM and expectation of prior
ALPHA = 3.0 #Hyperparameter of beta prior
N_JOBS = -1  # Number of parallel jobs (-1 = use all cores)
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


def set_only_fault_params_inferred():
    """
    Set only fault parameters as inferred, disable all cable and load parameters.

    Used for Stage 2 inference where network topology is assumed known from Stage 1,
    and only fault location/impedance needs to be estimated.
    """
    # Disable all cable lengths
    for cable_name in network_params["cable_lengths"]:
        network_params["cable_lengths"][cable_name]["inferred"] = False

    # Disable all load parameters
    for load_name in network_params["loads"]:
        for param_name in network_params["loads"][load_name]:
            network_params["loads"][load_name][param_name]["inferred"] = False

    # Enable all fault parameters
    for fault_name in network_params["fault_parameters"]:
        network_params["fault_parameters"][fault_name]["inferred"] = True


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



def run_single_bayesian_trial_stage2(m, theta_true_normalized, snr_db, selected_keys, num_steps, scenario, p_val):
    """
    Run a single Bayesian MC trial for Stage 2. This function is designed to be called in parallel.

    Each worker initializes its own forward_model and svi_engine to avoid sharing issues.

    Args:
        m: Trial index
        theta_true_normalized: True theta for this trial [p]
        snr_db: SNR in dB
        selected_keys: List of parameter keys to infer
        num_steps: Number of SVI steps
        scenario: "with_fault" for Stage 2
        p_val: Number of inferred parameters

    Returns:
        trial_errors: Dict {key: squared_error} for this trial
    """
    # Each worker needs its own copy of network_params to avoid race conditions
    import copy
    local_network_params = copy.deepcopy(network_params)

    # Initialize local forward model and SVI engine
    local_forward_model = MTLForwardModel(frequencies, local_network_params, device=device)
    local_inference_config = InferenceConfig(
        alpha=ALPHA,
        num_particles=NUM_PARTICLES,
        vectorize_particles=VECTORIZE_PARTICLES,
        optimizer=OPTIMIZER,
        learning_rate=LR,
        device=device,
    )
    local_svi_engine = SVIEngine(local_forward_model, local_network_params, local_inference_config)

    # Helper to set network params from normalized theta
    def local_set_network_params(sampled_theta, param_order_list):
        counter = 0
        for params in param_order_list:
            if params[0] == "load":
                entity_name = params[1]
                param_name = params[2]
                local_network_params["loads"][entity_name][param_name]["value"] = sampled_theta[counter].item()
                counter += 1
            elif params[0] == "cable":
                cable_name = params[1]
                local_network_params["cable_lengths"][cable_name]["value"] = sampled_theta[counter].item()
                counter += 1
            elif params[0] == "fault_param":
                fault_name = params[1]
                local_network_params["fault_parameters"][fault_name]["value"] = sampled_theta[counter].item()
                counter += 1

    def local_get_true_param_flat():
        param_order = []
        for cable_name in sorted(local_network_params["cable_lengths"].keys(), key=lambda x: int(x.split("_")[-1])):
            if local_network_params["cable_lengths"][cable_name]["inferred"]:
                param_order.append(("cable", cable_name, None))
        for load_name in sorted(local_network_params["loads"].keys(), key=lambda x: int(x.split("_")[-1])):
            for param_name in sorted(local_network_params["loads"][load_name].keys()):
                if local_network_params["loads"][load_name][param_name]["inferred"]:
                    param_order.append(("load", load_name, param_name))
        for fault_name in local_network_params["fault_parameters"]:
            if local_network_params["fault_parameters"][fault_name]["inferred"]:
                param_order.append(("fault_param", fault_name, None))

        params_flat = torch.zeros(len(param_order), dtype=torch.float32, device=device)
        for i, (ptype, name, subname) in enumerate(param_order):
            if ptype == "cable":
                params_flat[i] = local_network_params["cable_lengths"][name]["value"]
            elif ptype == "load":
                params_flat[i] = local_network_params["loads"][name][subname]["value"]
            elif ptype == "fault_param":
                params_flat[i] = local_network_params["fault_parameters"][name]["value"]
        return params_flat, param_order

    def local_build_params_from_flat(params_flat, param_order):
        cable_lengths = {}
        for cable_name, cable_info in local_network_params["cable_lengths"].items():
            cable_lengths[cable_name] = torch.tensor(cable_info["value"], dtype=torch.float32, device=device)
        load_params = {}
        for load_name, params in local_network_params["loads"].items():
            load_params[load_name] = {}
            for param_name, param_info in params.items():
                load_params[load_name][param_name] = torch.tensor(param_info["value"], dtype=torch.float32, device=device)
        fault_params = {}
        for fault_name, fault_info in local_network_params["fault_parameters"].items():
            fault_params[fault_name] = torch.tensor(fault_info["value"], dtype=torch.float32, device=device)
        for i, (ptype, name, subname) in enumerate(param_order):
            if ptype == "cable":
                cable_lengths[name] = params_flat[i]
            elif ptype == "load":
                load_params[name][subname] = params_flat[i]
            elif ptype == "fault_param":
                fault_params[name] = params_flat[i]
        return cable_lengths, load_params, fault_params

    # Get param order
    param_order_list = []
    for fault_name in local_network_params["fault_parameters"]:
        if local_network_params["fault_parameters"][fault_name]["inferred"]:
            param_order_list.append(("fault_param", fault_name, None))

    # Set theta for this trial
    local_set_network_params(theta_true_normalized, param_order_list)

    # Generate clean signal
    params_flat, param_order = local_get_true_param_flat()
    cable_lengths, load_params, fault_params = local_build_params_from_flat(params_flat, param_order)
    H_clean = local_forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)

    sigpow = torch.mean(torch.abs(H_clean)**2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    var_f = sigpow / snr_lin
    std_f = torch.sqrt(var_f / 2)

    # Add noise
    H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                1j * std_f * torch.randn_like(H_clean.imag)
    H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))

    # Run SVI (multi-start for high SNR)
    if snr_db >= 0:
        init_values = [0.0, -1.386, 1.386]
        best_loss = float('inf')
        best_params = None
        for init_val in init_values:
            losses, params, _ = local_svi_engine.run_inference(
                H1_noisy, scenario, selected_keys, std_f, num_steps,
                snr_db=snr_db, m=m, M=M, p_val=p_val,
                verbose=False,
                fault_position_init=init_val
            )
            final_loss = losses[-1] if losses else float('inf')
            if final_loss < best_loss:
                best_loss = final_loss
                best_params = params
    else:
        _, best_params, _ = local_svi_engine.run_inference(
            H1_noisy, scenario, selected_keys, std_f, num_steps,
            snr_db=snr_db, m=m, M=M, p_val=p_val,
            verbose=False
        )

    # Extract posterior means
    posterior_means = local_svi_engine.extract_posterior_means(best_params)

    # Compute squared errors
    trial_errors = {}
    for key in selected_keys:
        true_val = local_svi_engine.get_true_param_value(key)
        posterior_key = key.replace(".", "_")
        if posterior_key in posterior_means:
            estimate = posterior_means[posterior_key]
            trial_errors[key] = (estimate - true_val)**2

    return trial_errors


def run_single_frequentist_trial_stage2(m, H_clean, std_f, snr_db, selected_keys, num_steps, scenario, p_val):
    """
    Run a single Frequentist MC trial for Stage 2. This function is designed to be called in parallel.

    Each worker initializes its own forward_model and svi_engine to avoid sharing issues.

    Args:
        m: Trial index
        H_clean: Clean transfer function (shared across trials)
        std_f: Noise standard deviation
        snr_db: SNR in dB
        selected_keys: List of parameter keys to infer
        num_steps: Number of SVI steps
        scenario: "with_fault" for Stage 2
        p_val: Number of inferred parameters

    Returns:
        trial_errors: Dict {key: squared_error} for this trial
        best_params: Best params from SVI for CI plotting
    """
    # Each worker needs its own copy of network_params to avoid race conditions
    import copy
    local_network_params = copy.deepcopy(network_params)

    # Initialize local forward model and SVI engine
    local_forward_model = MTLForwardModel(frequencies, local_network_params, device=device)
    local_inference_config = InferenceConfig(
        alpha=ALPHA,
        num_particles=NUM_PARTICLES,
        vectorize_particles=VECTORIZE_PARTICLES,
        optimizer=OPTIMIZER,
        learning_rate=LR,
        device=device,
    )
    local_svi_engine = SVIEngine(local_forward_model, local_network_params, local_inference_config)

    # Generate noisy observation
    H1_noisy_c = H_clean + std_f * torch.randn_like(H_clean.real) + \
                 1j * std_f * torch.randn_like(H_clean.imag)
    H1_noisy = torch.view_as_real(H1_noisy_c.unsqueeze(0))

    # Run SVI inference
    _, best_params, _ = local_svi_engine.run_inference(
        H1_noisy, scenario, selected_keys, std_f, num_steps,
        snr_db=snr_db, m=m, M=M, p_val=p_val,
        verbose=False
    )

    # Extract posterior means
    posterior_means = local_svi_engine.extract_posterior_means(best_params)

    # Compute squared errors
    trial_errors = {}
    for key in selected_keys:
        true_val = local_svi_engine.get_true_param_value(key)
        posterior_key = key.replace(".", "_")
        if posterior_key in posterior_means:
            estimate = posterior_means[posterior_key]
            trial_errors[key] = (estimate - true_val)**2

    return trial_errors, best_params


def calculate_bayesian_mse_monte_carlo(snr_db, selected_keys, all_thetas, num_steps, scenario, p_val=None):
    """
    Compute Bayesian MSE via Monte Carlo at specific SNR using parallel execution.

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
    print(f"Running {M} Bayesian MC trials in parallel...")

    # Run trials in parallel
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
        delayed(run_single_bayesian_trial_stage2)(
            m, all_thetas[m], snr_db, selected_keys, num_steps, scenario, p_val
        )
        for m in range(M)
    )

    # Aggregate results
    squared_errors = {key: [] for key in selected_keys}
    for trial_errors in results:
        for key, err in trial_errors.items():
            squared_errors[key].append(err)

    # Average squared errors
    bayesian_mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}

    return bayesian_mse_dict


def calculate_mse_monte_carlo(var_f, selected_keys, snr_db, num_steps, scenario, p_val=None):
    """
    Compute Frequentist MSE via Monte Carlo at specific SNR using parallel execution.
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
    std_f = torch.sqrt(var_f / 2)

    # Compute H_clean (shared across all trials)
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    if scenario == 'with_fault':
        H_clean = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
    else:
        H_clean = forward_model.calculate_Hnw_nofault(cable_lengths, load_params)

    print(f"Running {M} Frequentist MC trials in parallel...")

    # Run trials in parallel
    results = Parallel(n_jobs=N_JOBS, backend='loky', verbose=10)(
        delayed(run_single_frequentist_trial_stage2)(
            m, H_clean, std_f, snr_db, selected_keys, num_steps, scenario, p_val
        )
        for m in range(M)
    )

    # Aggregate results
    squared_errors = {key: [] for key in selected_keys}
    last_best_params = None
    for i, (trial_errors, best_params) in enumerate(results):
        for key, err in trial_errors.items():
            squared_errors[key].append(err)
        if i == len(results) - 1:
            last_best_params = best_params

    # Average squared errors
    mse_dict = {key: sum(errs)/len(errs) for key, errs in squared_errors.items() if errs}
    return mse_dict, last_best_params


def main():
    start_time = time.perf_counter()
    snr_dbs = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    scenario = "with_fault"  # Stage 2 always uses fault scenario
    #mode = "frequentist"
    mode = "bayesian"
    num_steps = 250


    total_params, load_types = generate_load_parameters_deterministic(network_params, FIXED_LOAD_TYPES)
    num_cable_params = len(network_params["cable_lengths"])
    num_fault_params = len(network_params["fault_parameters"])
    print(f"Total number of load parameters: {total_params}")
    print(f"Total number of cable length parameters: {num_cable_params}")
    print(f"Total number of fault parameters: {num_fault_params}")
    print(f"Total number of network parameters: {total_params + num_cable_params + num_fault_params}")
    print(f"Load type distribution: {load_types}")

    # For Stage 2: Set only fault parameters as inferred
    set_only_fault_params_inferred()

    # Get param order for fault parameters only
    param_order_list, p_fault = get_inferred_param_order()
    print(f"\n{'='*60}")
    print(f"Stage 2 Inference: Fault Parameters Only")
    print(f"Number of fault parameters to infer: {p_fault}")
    print('='*60)

    # Get selected keys (fault parameters only)
    selected_keys = [entry[1] for entry in param_order_list]  # fault_position, Z_fault_real, Z_fault_imag
    print(f"Inferring parameters: {selected_keys}")

    if mode == "frequentist":
        # Frequentist: use default values from network_params (0.25, 0.1, 0.25)
        print(f"Using default theta_true from network_params:")
        for key in selected_keys:
            print(f"  {key}: {network_params['fault_parameters'][key]['value']}")
        theta_bayesian = None
    else:
        # Bayesian: sample theta for each MC trial from prior
        torch.manual_seed(SEED-1)
        beta_dist = torch.distributions.Beta(ALPHA, ALPHA)
        theta_bayesian = beta_dist.sample((M, p_fault))

    # Get true params and compute clean transfer function (with fault)
    params_flat = get_true_param_flat()
    cable_lengths, load_params, fault_params = build_params_from_flat(params_flat, param_order_list)
    H_clean = forward_model.calculate_Hnw(cable_lengths, load_params, fault_params)
    sigpow = torch.mean(torch.abs(H_clean)**2)

    # Initialize results storage
    rmse_results = {key: [] for key in selected_keys}
    crlb_results = {key: [] for key in selected_keys}
    best_params_per_snr = {}  # Store best_params for CI plotting

    # SNR sweep
    for snr_db in snr_dbs:
        print(f"\n{'='*50}")
        print(f"Stage 2 | SNR = {snr_db} dB | Mode = {mode}")
        print('='*50)
        # if snr_db <= 20:
        #     num_steps = 250
        # else:
        #     num_steps = 500

        snr_lin = 10.0 ** (snr_db / 10.0)
        var_f = sigpow / snr_lin
        wrapper_fn = H_fault_wrapper  # Stage 2 always uses fault model

        sensitivities = [1.0 / p_fault] * p_fault  # Equal weights for fault params

        if mode == "frequentist":
            # Frequentist: fixed θ_true, CRLB
            crlb_dict = compute_real_CRLB(
                var_f, selected_keys, sensitivities, scenario,
                wrapper_fn, get_true_param_flat, get_inferred_param_order
            )
            mse_dict, last_best_params = calculate_mse_monte_carlo(
                var_f, selected_keys, snr_db, num_steps, scenario, p_fault
            )
            print(f"RMSE (M={M}):", {k: f"{math.sqrt(v):.4f}" for k, v in mse_dict.items()})
            print(f"sqrt(CRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in crlb_dict.items()})

            # Store results
            for key in selected_keys:
                if key in mse_dict and key in crlb_dict:
                    rmse_results[key].append(math.sqrt(mse_dict[key]))
                    crlb_results[key].append(math.sqrt(crlb_dict[key]))

            # Store best_params for CI plotting (from last MC run)
            best_params_per_snr[snr_db] = last_best_params

        elif mode == "bayesian":
            # Bayesian: θ ~ π(θ) each run, BCRLB
            bcrlb_dict = compute_real_BCRLB(
                snr_db, selected_keys, theta_bayesian, scenario, ALPHA, forward_model,
                network_params, wrapper_fn, get_true_param_flat, get_inferred_param_order,
                set_network_params_from_normalized, build_params_from_flat
            )

            bayesian_mse_dict = calculate_bayesian_mse_monte_carlo(
                snr_db, selected_keys, theta_bayesian, num_steps, scenario, p_fault
            )
            print(f"Bayesian RMSE (M={M}):", {k: f"{math.sqrt(v):.4f}" for k, v in bayesian_mse_dict.items()})
            print(f"sqrt(BCRLB):", {k: f"{math.sqrt(v):.4f}" for k, v in bcrlb_dict.items()})

            # Store results
            for key in selected_keys:
                if key in bayesian_mse_dict and key in bcrlb_dict:
                    rmse_results[key].append(math.sqrt(bayesian_mse_dict[key]))
                    crlb_results[key].append(math.sqrt(bcrlb_dict[key]))

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'frequentist' or 'bayesian'.")

    # ---- Save results to .npz ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert results to numpy-compatible format
    results_to_save = {}
    for key in selected_keys:
        safe_key = key.replace(".", "_")
        results_to_save[f"{safe_key}_rmse"] = np.array(rmse_results[key])
        results_to_save[f"{safe_key}_crlb"] = np.array(crlb_results[key])

    # Save best_params for each SNR (for CI plotting)
    for snr_db, best_params in best_params_per_snr.items():
        snr_prefix = f"snr{snr_db}"
        # Convert Pyro params to numpy
        for param_name, param_val in best_params.items():
            if hasattr(param_val, 'detach'):
                results_to_save[f"{snr_prefix}_{param_name}"] = param_val.detach().cpu().numpy()
            else:
                results_to_save[f"{snr_prefix}_{param_name}"] = np.array(param_val)

    # Extract fault_position range for filename
    fp_range = network_params["fault_parameters"]["fault_position"]["range"]
    fp_range_str = f"fp{fp_range[0]}-{fp_range[1]}"

    save_path = os.path.join(OUTPUT_DIR, f"stage2_results_{freq_range_str}_M{M}_alpha{ALPHA}_{fp_range_str}_{mode}_025parallel.npz")
    np.savez(
        save_path,
        snr_dbs=np.array(snr_dbs),
        selected_keys=np.array(selected_keys, dtype=object),
        frequencies=frequencies.numpy(),
        network_params=network_params,
        M=M,
        ALPHA=ALPHA,
        SEED=SEED,
        scenario=scenario,
        freq_range_str=freq_range_str,
        fp_range=np.array(fp_range),
        mode=mode,
        **results_to_save
    )
    print(f"\nResults saved to: {save_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
