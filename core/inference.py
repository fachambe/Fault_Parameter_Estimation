"""
Stochastic Variational Inference (SVI) for MTL network parameter estimation.

This module provides Pyro-based probabilistic inference for estimating
network parameters from transfer function measurements.
"""

import math
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import SigmoidTransform
from pyro.distributions import TransformedDistribution, constraints
from pyro.infer import SVI, Trace_ELBO


class InferenceConfig:
    """Configuration for SVI inference."""

    def __init__(
        self,
        alpha: float = 5.0,
        num_particles: int = 12,
        vectorize_particles: bool = True,
        optimizer: str = "Adam",
        learning_rate: float = 0.02,
        device: torch.device = None,
    ):
        """
        Args:
            alpha: Beta distribution hyperparameter for prior Beta(alpha, alpha)
            num_particles: Number of particles for SVI ELBO estimation
            vectorize_particles: Whether to vectorize particle computation
            optimizer: Optimizer type ("Adam" or "Adagrad")
            learning_rate: Learning rate for optimizer
            device: Torch device for computation
        """
        self.alpha = alpha
        self.num_particles = num_particles
        self.vectorize_particles = vectorize_particles
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = device or torch.device("cpu")


class SVIEngine:
    """
    Stochastic Variational Inference engine for MTL network.

    This class encapsulates the Pyro-based inference for estimating
    network parameters from noisy transfer function measurements.
    """

    def __init__(self, forward_model, network_params, config: InferenceConfig):
        """
        Args:
            forward_model: MTLForwardModel instance
            network_params: Network parameters dictionary (modified in-place during inference)
            config: InferenceConfig instance
        """
        self.forward_model = forward_model
        self.network_params = network_params
        self.config = config
        self.device = config.device

    def model_no_fault(self, H1_noisy, std_f):
        """
        Pyro model for Stage 1: Network without fault.

        Samples load and cable parameters from Beta priors,
        computes predicted transfer function, and conditions on observations.

        Args:
            H1_noisy: Noisy observations [N, F, 2] (real/imag stacked)
            std_f: Noise standard deviation (scalar or [F])
        """
        N, F, _ = H1_noisy.shape

        # Sample/fix load parameters
        load_params = {}
        for load_name, params in self.network_params["loads"].items():
            load_dict = {}
            for param_name, param_info in params.items():
                if param_info["inferred"]:
                    norm_sample = pyro.sample(
                        f"{load_name}_{param_name}",
                        dist.Beta(self.config.alpha, self.config.alpha)
                    )
                    load_dict[param_name] = norm_sample
                else:
                    load_dict[param_name] = torch.tensor(
                        param_info["value"], device=self.device
                    )
            load_params[load_name] = load_dict

        # Sample/fix cable parameters
        cable_lengths = {}
        for cable_name, cable_info in self.network_params["cable_lengths"].items():
            if cable_info["inferred"]:
                norm_sample = pyro.sample(
                    f"{cable_name}",
                    dist.Beta(self.config.alpha, self.config.alpha)
                )
                cable_lengths[cable_name] = norm_sample
            else:
                cable_lengths[cable_name] = torch.tensor(
                    cable_info["value"], device=self.device
                )

        # Compute predicted transfer function
        H1_pred_c = self.forward_model.calculate_Hnw_nofault(cable_lengths, load_params)

        # Handle both vectorized [P, F] and non-vectorized [F] cases
        if H1_pred_c.dim() == 1:
            H1_pred_c = H1_pred_c.unsqueeze(0).expand(N, -1)
        else:
            H1_pred_c = H1_pred_c.unsqueeze(-2).expand(
                *H1_pred_c.shape[:-1], N, H1_pred_c.shape[-1]
            )

        H1_pred = torch.view_as_real(H1_pred_c)  # [P, N, F, 2] or [N, F, 2]

        with pyro.plate("data", N):
            pyro.sample(
                "obs",
                dist.Independent(
                    dist.Normal(loc=H1_pred, scale=std_f),
                    reinterpreted_batch_ndims=2
                ),
                obs=H1_noisy
            )

    def model_with_fault(self, H1_noisy, std_f):
        """
        Pyro model for Stage 2: Network with fault.

        Samples load, cable, and fault parameters from Beta priors,
        computes predicted transfer function, and conditions on observations.

        Args:
            H1_noisy: Noisy observations [N, F, 2] (real/imag stacked)
            std_f: Noise standard deviation (scalar or [F])
        """
        N, F, _ = H1_noisy.shape

        # Sample/fix load parameters
        load_params = {}
        for load_name, params in self.network_params["loads"].items():
            load_dict = {}
            for param_name, param_info in params.items():
                if param_info["inferred"]:
                    norm_sample = pyro.sample(
                        f"{load_name}_{param_name}",
                        dist.Beta(self.config.alpha, self.config.alpha)
                    )
                    load_dict[param_name] = norm_sample
                else:
                    load_dict[param_name] = torch.tensor(
                        param_info["value"], device=self.device
                    )
            load_params[load_name] = load_dict

        # Sample/fix cable parameters
        cable_lengths = {}
        for cable_name, cable_info in self.network_params["cable_lengths"].items():
            if cable_info["inferred"]:
                norm_sample = pyro.sample(
                    cable_name,
                    dist.Beta(self.config.alpha, self.config.alpha)
                )
                cable_lengths[cable_name] = norm_sample
            else:
                cable_lengths[cable_name] = torch.tensor(
                    cable_info["value"], device=self.device
                )

        # Sample/fix fault parameters
        fault_params = {}
        for fault_name, fault_info in self.network_params["fault_parameters"].items():
            if fault_info["inferred"]:
                norm_sample = pyro.sample(
                    fault_name,
                    dist.Beta(self.config.alpha, self.config.alpha)
                )
                fault_params[fault_name] = norm_sample
            else:
                fault_params[fault_name] = torch.tensor(
                    fault_info["value"], device=self.device
                )

        # Compute predicted transfer function
        H1_pred_c = self.forward_model.calculate_Hnw(
            cable_lengths, load_params, fault_params
        )

        # Handle both vectorized [P, F] and non-vectorized [F] cases
        if H1_pred_c.dim() == 1:
            H1_pred_c = H1_pred_c.unsqueeze(0).expand(N, -1)
        else:
            H1_pred_c = H1_pred_c.unsqueeze(-2).expand(
                *H1_pred_c.shape[:-1], N, H1_pred_c.shape[-1]
            )

        H1_pred = torch.view_as_real(H1_pred_c)  # [P, N, F, 2] or [N, F, 2]

        with pyro.plate("data", N):
            pyro.sample(
                "obs",
                dist.Independent(
                    dist.Normal(loc=H1_pred, scale=std_f),
                    reinterpreted_batch_ndims=2
                ),
                obs=H1_noisy
            )

    def guide(self, H1_noisy, std_f):
        """
        Variational guide using LogitNormal (sigmoid-transformed Normal).

        Each inferred parameter has a learnable loc and scale, with the
        posterior approximated as sigmoid(Normal(loc, scale)).

        Args:
            H1_noisy: Noisy observations (unused but required for Pyro signature)
            std_f: Noise standard deviation (unused but required for Pyro signature)
        """
        # Load parameters
        for load_name, params in self.network_params["loads"].items():
            for param_name, param_info in params.items():
                if not param_info["inferred"]:
                    continue

                full_name = f"{load_name}_{param_name}"

                loc = pyro.param(
                    f"{full_name}_loc",
                    torch.tensor(0.0, device=self.device)
                )
                scale = pyro.param(
                    f"{full_name}_scale",
                    torch.tensor(0.1, device=self.device),
                    constraint=constraints.positive
                )

                q = TransformedDistribution(
                    dist.Normal(loc, scale),
                    [SigmoidTransform()]
                )
                pyro.sample(full_name, q)

        # Cable parameters
        for key, info in self.network_params["cable_lengths"].items():
            if not info["inferred"]:
                continue

            loc = pyro.param(
                f"{key}_loc",
                torch.tensor(0.0, device=self.device)
            )
            scale = pyro.param(
                f"{key}_scale",
                torch.tensor(0.1, device=self.device),
                constraint=constraints.positive
            )

            q = TransformedDistribution(
                dist.Normal(loc, scale),
                [SigmoidTransform()]
            )
            pyro.sample(key, q)

        # Fault parameters
        for key, info in self.network_params["fault_parameters"].items():
            if not info["inferred"]:
                continue

            loc = pyro.param(
                f"{key}_loc",
                torch.tensor(0.0, device=self.device)
            )
            scale = pyro.param(
                f"{key}_scale",
                torch.tensor(0.1, device=self.device),
                constraint=constraints.positive
            )

            q = TransformedDistribution(
                dist.Normal(loc, scale),
                [SigmoidTransform()]
            )
            pyro.sample(key, q)

    def get_true_param_value(self, key):
        """
        Get the true normalized value for a parameter key.

        Args:
            key: Parameter key in format "load_name.param_name", "cable_name",
                 or "fault_param_name"

        Returns:
            Normalized parameter value in [0, 1]
        """
        if "." in key:
            load_name, param_name = key.split(".")
            return self.network_params["loads"][load_name][param_name]["value"]
        elif key in self.network_params.get("fault_parameters", {}):
            return self.network_params["fault_parameters"][key]["value"]
        else:
            return self.network_params["cable_lengths"][key]["value"]

    def run_inference(
        self,
        H1_noisy,
        scenario,
        sorted_keys,
        std_f,
        num_steps,
        snr_db=None,
        m=0,
        M=1,
        p_val=None,
        verbose=True,
        fault_position_init=0.0
    ):
        """
        Run SVI inference.

        Args:
            H1_noisy: Noisy observations [N, F, 2] (real/imag stacked)
            scenario: "no_fault" or "with_fault"
            sorted_keys: List of parameter keys sorted by sensitivity
            std_f: Noise standard deviation
            num_steps: Number of SVI optimization steps
            snr_db: SNR in dB (for logging)
            m: Current Monte Carlo trial index (for logging)
            M: Total Monte Carlo trials (for logging)
            p_val: Number of inferred parameters (for logging)
            verbose: Whether to print progress
            fault_position_init: Initial loc value for fault_position (pre-sigmoid).
                0.0 -> sigmoid(0) = 0.5, -1.386 -> 0.2, 1.386 -> 0.8

        Returns:
            losses: List of ELBO losses at each step
            best_params: Dict of best parameter values (scalars)
            param_history: Dict of parameter trajectories
        """
        pyro.clear_param_store()

        # Select model based on scenario
        if scenario == "no_fault":
            model = self.model_no_fault
        elif scenario == "with_fault":
            model = self.model_with_fault
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Setup optimizer
        if self.config.optimizer == "Adam":
            optimizer = pyro.optim.Adam({"lr": self.config.learning_rate})
        elif self.config.optimizer == "Adagrad":
            optimizer = pyro.optim.Adagrad({"lr": self.config.learning_rate})
        else:
            raise ValueError(
                f"Unknown optimizer: {self.config.optimizer}. Use 'Adam' or 'Adagrad'."
            )

        # Setup SVI
        svi = SVI(
            model,
            self.guide,
            optimizer,
            loss=Trace_ELBO(
                num_particles=self.config.num_particles,
                vectorize_particles=self.config.vectorize_particles,
                max_plate_nesting=1
            )
        )

        # Initialize guide parameters
        self.guide(H1_noisy, std_f)

        # Override fault_position initialization if specified
        if fault_position_init != 0.0:
            param_store = pyro.get_param_store()
            if "fault_position_loc" in param_store:
                param_store["fault_position_loc"].data.fill_(fault_position_init)

        if verbose:
            param_store = pyro.get_param_store()
            print(f"\n===== STEP 0 (INITIALIZATION) =====")
            for key in sorted_keys[:20]:
                store_key = key.replace(".", "_") + "_loc"
                if store_key in param_store:
                    true_norm = self.get_true_param_value(key)
                    loc = param_store[store_key].detach()
                    sig = torch.sigmoid(loc)
                    print(f"{key:40s} (sigmoid) = {sig.item():.4f} | True = {true_norm:.4f}")

        losses = []
        best_loss = float("inf")
        best_params = None
        param_history = {}

        for step in range(num_steps):
            loss = svi.step(H1_noisy, std_f)
            losses.append(loss)

            # Track parameter history
            param_store = pyro.get_param_store()
            for name, value in param_store.items():
                param_history.setdefault(name, []).append(value.detach().cpu().item())
            # Track best parameters
            if loss < best_loss:
                best_loss = loss
                best_params = {
                    name: value.detach().clone()
                    for name, value in pyro.get_param_store().items()
                }

            # Logging
            if verbose and step % 25 == 0:
                print(f"\n===== p = {p_val} | SNR {snr_db} | m = {m+1}/{M} | "
                      f"Step {step} | loss = -ELBO: {loss:.6f} =====")
                print("\n Top 20 Most Sensitive Parameters")
                param_store = pyro.get_param_store()
                for key in sorted_keys[:20]:
                    store_key = key.replace(".", "_") + "_loc"
                    if store_key in param_store:
                        true_norm = self.get_true_param_value(key)
                        loc = param_store[store_key].detach()
                        sig = torch.sigmoid(loc)
                        print(f"{key:40s} (sigmoid) = {sig.item():.4f} | True = {true_norm:.4f}")

        # Restore best params into Pyro param store
        param_store = pyro.get_param_store()
        for name, value in best_params.items():
            param_store[name] = value.clone()

        # Print final best estimates vs true values (only when verbose=True to avoid jumbled output in parallel mode)
        if verbose:
            print(f"\n===== FINAL BEST ESTIMATES | p = {p_val} | SNR {snr_db} | m = {m+1}/{M} | best_loss = {best_loss:.2f} =====")
            for key in sorted_keys[:20]:
                store_key = key.replace(".", "_") + "_loc"
                if store_key in best_params:
                    true_norm = self.get_true_param_value(key)
                    loc = best_params[store_key]
                    if hasattr(loc, 'item'):
                        loc_val = loc.item()
                    else:
                        loc_val = loc
                    sig = 1.0 / (1.0 + math.exp(-loc_val))  # sigmoid
                    print(f"  {key:40s}: estimate = {sig:.4f} | true = {true_norm:.4f}")

        # Convert best_params to scalars
        best_params = {
            name: value.detach().cpu().item()
            for name, value in best_params.items()
        }

        if verbose:
            print("Inference complete.")

        return losses, best_params, param_history

    def extract_posterior_means(self, best_params, num_samples=2048):
        """
        Extract posterior means (normalized [0,1]) from best_params after inference.

        Uses the loc and scale values to construct the variational distribution,
        then computes the MC mean of the sigmoid-transformed Normal.

        Args:
            best_params: Dict of parameter values from run_inference
            num_samples: Number of MC samples for computing the mean

        Returns:
            Dict: {param_name: normalized_mean} for all inferred parameters
        """
        posterior_means = {}

        for key, values in best_params.items():
            if "_loc" not in key:
                continue

            param_name = key.replace("_loc", "")
            loc = values[-1] if isinstance(values, list) else values

            # Look up corresponding scale
            scale_key = f"{param_name}_scale"
            scale_val = best_params[scale_key]
            scale = scale_val[-1] if isinstance(scale_val, list) else scale_val

            with torch.no_grad():
                loc_t = torch.tensor(loc, device=self.device)
                scale_t = torch.tensor(scale, device=self.device)
                q = TransformedDistribution(
                    dist.Normal(loc_t, scale_t),
                    [SigmoidTransform()]
                )
                mc_mean = q.sample((num_samples,)).mean().item()

            posterior_means[param_name] = mc_mean

        return posterior_means

    def update_network_params_from_posterior(self, posterior_means, verbose=True):
        """
        Update network_params with inferred posterior means and mark as fixed.

        Used after Stage 1 to fix load/cable parameters before Stage 2.

        Args:
            posterior_means: Dict from extract_posterior_means() - normalized [0,1] values
            verbose: Whether to print updates

        Note:
            Values are stored as normalized [0,1]. Denormalization happens in forward model.
        """
        updated_count = 0

        # Update load parameters
        for load_name, params in self.network_params["loads"].items():
            for param_name, param_info in params.items():
                full_name = f"{load_name}_{param_name}"

                if full_name in posterior_means:
                    norm_val = posterior_means[full_name]
                    old_val = param_info["value"]
                    param_info["value"] = norm_val
                    param_info["inferred"] = False
                    updated_count += 1
                    if verbose:
                        print(f"  Updated {full_name}: {old_val:.4f} -> {norm_val:.4f}")

        # Update cable parameters
        for cable_name, cable_info in self.network_params["cable_lengths"].items():
            if cable_name in posterior_means:
                norm_val = posterior_means[cable_name]
                old_val = cable_info["value"]
                cable_info["value"] = norm_val
                cable_info["inferred"] = False
                updated_count += 1
                if verbose:
                    print(f"  Updated {cable_name}: {old_val:.4f} -> {norm_val:.4f}")

        if verbose:
            print(f"\nUpdated {updated_count} parameters from posterior means.")


# Convenience functions for backwards compatibility with run_stage1_mtl.py

def create_svi_engine(forward_model, network_params, config=None, **kwargs):
    """
    Create an SVIEngine instance with default or custom configuration.

    Args:
        forward_model: MTLForwardModel instance
        network_params: Network parameters dictionary
        config: InferenceConfig instance (optional)
        **kwargs: Override config parameters (alpha, num_particles, etc.)

    Returns:
        SVIEngine instance
    """
    if config is None:
        config = InferenceConfig(**kwargs)
    return SVIEngine(forward_model, network_params, config)
