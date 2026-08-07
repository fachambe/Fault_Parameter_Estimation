"""
Multi-conductor Transmission Line (MTL) Forward Model.

This module provides the MTLForwardModel class for computing transfer functions
of complex power line networks with multiple loads and optional faults.
"""

import torch
import numpy as np

from .mtl_utils import (
    denormalize,
    calculate_cable_parameters,
    get_mtl_matrices,
    reflection_coefficient,
    carry_back_load,
    h_B,
    calculate_room_admittance_matrix,
    calculate_Htrans,
)
from .loads import compute_load_admittances
from .fault_utils import (
    compute_fault_admittance_matrix,
    get_fault_segment_and_local_position,
    carry_back_with_fault,
)


class MTLForwardModel:
    """
    Forward model for MTL network transfer function computation.

    This class encapsulates:
    - Network configuration (frequencies, transmitter/receiver parameters)
    - Cable and load parameter ranges
    - Methods to compute transfer functions with/without faults

    Attributes:
        frequencies: Tensor of frequencies [F]
        omega: Angular frequencies [F]
        device: Torch device
        num_conductors: Number of conductors (typically 4 for 3-phase + neutral)
        network_params: Dict containing parameter ranges and values
    """

    # Backbone cable keys (cables on the main signal path)
    BACKBONE_KEYS = ["l_w_0", "l_w_1", "l_w_4", "l_w_25", "l_w_28"]

    def __init__(self, frequencies, network_params, device=None):
        """
        Initialize the MTL forward model.

        Args:
            frequencies: Tensor of frequencies in Hz [F]
            network_params: Dict containing:
                - cable_lengths: Dict of cable length configs
                - loads: Dict of load parameter configs
                - conductor_radii: Dict of conductor radius configs
                - fault_parameters: Dict of fault parameter configs (optional)
            device: Torch device (defaults to CPU)
        """
        self.device = device or torch.device("cpu")
        self.frequencies = frequencies.to(self.device)
        self.omega = 2 * torch.pi * self.frequencies
        self.num_freqs = len(self.omega)
        self.num_conductors = 4  # 3-phase + neutral
        self.n = self.num_conductors - 1  # Matrix dimension

        self.network_params = network_params

        # Initialize transmitter parameters (fixed values from paper)
        self._init_transmitter_params()

        # Initialize receiver admittance matrix
        self._init_receiver()

    def _init_transmitter_params(self):
        """Initialize transmitter impedance parameters."""
        self.ZT0 = 50.0
        self.ZTG1 = 50.0
        self.ZTG2 = 50.0
        self.ZTG3 = 50.0
        self.ZT12 = 100.0
        self.ZT13 = 100.0
        self.ZT23 = 100.0

        # Pre-compute transmitter admittances
        self.YTalpha = (self.ZT12 * self.ZT13 + self.ZTG1 * self.ZT13 +
                        self.ZTG1 * self.ZT12) / (self.ZTG1 * self.ZT12 * self.ZT13)
        self.YTbeta = (self.ZTG2 * self.ZT12 + self.ZTG2 * self.ZT23 +
                       self.ZT12 * self.ZT23) / (self.ZTG2 * self.ZT12 * self.ZT23)
        self.YTgamma = (self.ZTG3 * self.ZT13 + self.ZTG3 * self.ZT23 +
                        self.ZT13 * self.ZT23) / (self.ZTG3 * self.ZT13 * self.ZT23)

    def _init_receiver(self):
        """Initialize receiver admittance matrix."""
        Z_RG = 50.0
        Z_R1 = Z_R2 = Z_R3 = 50.0

        Z_rec = torch.tensor([
            [Z_RG + Z_R1, Z_RG, Z_RG],
            [Z_RG, Z_RG + Z_R2, Z_RG],
            [Z_RG, Z_RG, Z_RG + Z_R3]
        ], dtype=torch.complex64, device=self.device)

        Y_rec = torch.linalg.inv(Z_rec)
        # Expand to [F, 3, 3] for frequency dimension
        self.Y_rec = Y_rec.unsqueeze(0).repeat(self.num_freqs, 1, 1)

    def _convert_cable_lengths_to_physical(self, cable_lengths):
        """Convert normalized cable lengths [0,1] to physical values."""
        cable_lengths_physical = {}
        for cable_name, norm_val in cable_lengths.items():
            lo, hi = self.network_params["cable_lengths"][cable_name]["range"]
            cable_lengths_physical[cable_name] = lo + norm_val * (hi - lo)
        return cable_lengths_physical

    def _convert_load_params_to_physical(self, load_params):
        """Convert normalized load parameters [0,1] to physical values."""
        load_params_physical = {}
        for load_name, params in load_params.items():
            load_params_physical[load_name] = {}
            for param_name, norm_val in params.items():
                if param_name == 'R_m2':
                    # R_m2 is fixed, not normalized
                    load_params_physical[load_name][param_name] = norm_val
                else:
                    param_info = self.network_params["loads"][load_name][param_name]
                    lo, hi = param_info["range"]
                    dist = param_info.get("dist", "uniform")
                    load_params_physical[load_name][param_name] = denormalize(
                        norm_val, lo, hi, dist
                    )
        return load_params_physical

    def _get_conductor_radii(self):
        """Get physical conductor radii from network params."""
        sp_info = self.network_params["conductor_radii"]["r_w_servicepanel"]
        lo, hi = sp_info["range"]
        r_w_servicepanel = denormalize(sp_info["value"], lo, hi)

        room_info = self.network_params["conductor_radii"]["r_w_room"]
        lo, hi = room_info["range"]
        r_w_room = denormalize(room_info["value"], lo, hi)

        return r_w_servicepanel, r_w_room

    def _compute_mtl_params(self, r_w_servicepanel, r_w_room):
        """Compute MTL matrices for service panel and room cables."""
        n = self.n

        # Service panel cables
        R_s, L_s, C_s, G_s = calculate_cable_parameters(
            r_w_servicepanel, self.omega, n, self.device
        )
        T_s, Tinv_s, gamma_s, ZC_s, YC_s = get_mtl_matrices(
            R_s, L_s, C_s, G_s, n, self.omega
        )

        # Room cables
        R_r, L_r, C_r, G_r = calculate_cable_parameters(
            r_w_room, self.omega, n, self.device
        )
        T_r, Tinv_r, gamma_r, ZC_r, YC_r = get_mtl_matrices(
            R_r, L_r, C_r, G_r, n, self.omega
        )

        return {
            'service_panel': (T_s, Tinv_s, gamma_s, ZC_s, YC_s),
            'room': (T_r, Tinv_r, gamma_r, ZC_r, YC_r)
        }

    def calculate_Hnw_nofault(self, cable_lengths, load_params):
        """
        Compute network transfer function without fault.

        Takes normalized parameters in [0,1] range, converts to physical inside.

        Args:
            cable_lengths: Dict of normalized cable lengths {name: tensor}
            load_params: Dict of normalized load params {load_name: {param_name: tensor}}

        Returns:
            H_nw: Transfer function [F] or [P, F] if batched
        """
        # Convert to physical values
        cable_lengths_physical = self._convert_cable_lengths_to_physical(cable_lengths)
        load_params_physical = self._convert_load_params_to_physical(load_params)

        # Get conductor radii and compute MTL params
        r_w_sp, r_w_room = self._get_conductor_radii()
        mtl = self._compute_mtl_params(r_w_sp, r_w_room)
        T_s, Tinv_s, gamma_s, ZC_s, YC_s = mtl['service_panel']
        T_r, Tinv_r, gamma_r, ZC_r, YC_r = mtl['room']

        # Compute load admittances
        Y_loads = compute_load_admittances(load_params_physical, self.omega)

        # ===== Network topology traversal =====
        # Start from receiver end, carry back through network

        # Node 0: Receiver
        rho1 = reflection_coefficient(self.Y_rec, T_r, Tinv_r, ZC_r, YC_r)
        h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_0"])
        Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_0"])

        # Node 1: load_0
        Y_node2 = Y_reccarried + Y_loads["load_0"]
        rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
        h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_1"])
        Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_1"])

        # Node 2: Junction box (load_1, load_2)
        rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
        Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_2"])
        Y_61 = Y_63 + Y_loads["load_2"]
        rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
        Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_3"])

        Y_node3 = Y_node2carried + Y_6
        rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
        h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_4"])
        Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_4"])

        # Node 3: Service panel (4 rooms)
        Y_5 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(3, 7)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(5, 10)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_4 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(7, 11)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(10, 15)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_3 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(11, 15)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(15, 20)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_2 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(15, 19)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(20, 25)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )

        Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2
        rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
        h4 = h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_25"])
        Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_25"])

        # Node 4: load_19, load_20 branch
        rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
        Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_26"])
        Y_12 = Y_14 + Y_loads["load_20"]
        rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
        Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_27"])

        Y_node5 = Y_node4carried + Y_1
        rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
        h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_28"])
        Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_28"])

        # Node 5: load_21 (near transmitter)
        rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
        Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_29"])

        Y_node6 = Y_node5carried + Y_13

        # Compute transmitter transfer function
        H_trans = calculate_Htrans(
            self.YTalpha, self.YTbeta, self.YTgamma, Y_node6,
            self.ZT0, self.ZT12, self.ZT12, self.ZT13, self.ZT13, self.ZT23, self.ZT23
        )

        # Overall transfer function
        hoverall = h1 @ h2 @ h3 @ h4 @ h5
        H1 = hoverall @ H_trans

        # Extract scalar transfer function (element [0,0])
        H_nw = H1[..., 0, 0]

        return H_nw

    def get_total_backbone_length(self, cable_lengths):
        """Calculate total backbone length from cable_lengths dict."""
        total = torch.tensor(0.0, device=self.device)
        for key in self.BACKBONE_KEYS:
            val = cable_lengths[key]
            total = total + val
        return total

    def _convert_fault_params_to_physical(self, fault_params):
        """Convert normalized fault parameters [0,1] to physical values."""
        fault_params_physical = {}
        for param_name, norm_val in fault_params.items():
            param_info = self.network_params["fault_parameters"][param_name]
            lo, hi = param_info["range"]
            fault_params_physical[param_name] = lo + norm_val * (hi - lo)
        return fault_params_physical

    def calculate_Hnw(self, cable_lengths, load_params, fault_params, num_particles=None):
        """
        Compute network transfer function with fault.

        Takes normalized parameters in [0,1] range, converts to physical inside.

        Args:
            cable_lengths: Dict of normalized cable lengths {name: tensor}
            load_params: Dict of normalized load params {load_name: {param_name: tensor}}
            fault_params: Dict of normalized fault params {param_name: tensor}
            num_particles: Number of particles for batched computation (optional)

        Returns:
            H_nw: Transfer function [F] or [P, F] if batched
        """
        # Convert to physical values
        cable_lengths_physical = self._convert_cable_lengths_to_physical(cable_lengths)
        load_params_physical = self._convert_load_params_to_physical(load_params)

        # Get conductor radii and compute MTL params
        r_w_sp, r_w_room = self._get_conductor_radii()
        mtl = self._compute_mtl_params(r_w_sp, r_w_room)
        T_s, Tinv_s, gamma_s, ZC_s, YC_s = mtl['service_panel']
        T_r, Tinv_r, gamma_r, ZC_r, YC_r = mtl['room']

        # Compute load admittances
        Y_loads = compute_load_admittances(load_params_physical, self.omega)

        # Convert fault params to physical
        fp = self.network_params["fault_parameters"]
        lo, hi = fp["fault_position"]["range"]
        fault_location = lo + fault_params["fault_position"] * (hi - lo)
        lo, hi = fp["Z_fault_real"]["range"]
        Z_fault_real = lo + fault_params["Z_fault_real"] * (hi - lo)
        lo, hi = fp["Z_fault_imag"]["range"]
        Z_fault_imag = lo + fault_params["Z_fault_imag"] * (hi - lo)

        is_batched = fault_location.dim() >= 1 and fault_location.shape[0] > 1

        # Get fault segment and local position
        fault_seg_idx, local_fault_pos, _ = get_fault_segment_and_local_position(
            fault_location, cable_lengths_physical, self.device
        )

        if not is_batched:
            return self._calculate_Hnw_scalar(
                cable_lengths_physical, Y_loads, fault_seg_idx, local_fault_pos,
                Z_fault_real, Z_fault_imag,
                T_s, Tinv_s, gamma_s, ZC_s, YC_s,
                T_r, Tinv_r, gamma_r, ZC_r, YC_r
            )
        else:
            return self._calculate_Hnw_batched(
                cable_lengths_physical, Y_loads, fault_seg_idx, local_fault_pos,
                Z_fault_real, Z_fault_imag, num_particles,
                T_s, Tinv_s, gamma_s, ZC_s, YC_s,
                T_r, Tinv_r, gamma_r, ZC_r, YC_r
            )

    def _calculate_Hnw_scalar(self, cable_lengths_physical, Y_loads, fault_seg_idx,
                               local_fault_pos, Z_fault_real, Z_fault_imag,
                               T_s, Tinv_s, gamma_s, ZC_s, YC_s,
                               T_r, Tinv_r, gamma_r, ZC_r, YC_r):
        """Compute transfer function with fault for scalar (non-batched) case."""
        # Node 0: Receiver - Segment 0: l_w_0
        if fault_seg_idx == 0:
            Y_reccarried, h1 = carry_back_with_fault(
                self.Y_rec, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                cable_lengths_physical["l_w_0"], local_fault_pos,
                Z_fault_real, Z_fault_imag, self.device
            )
        else:
            rho1 = reflection_coefficient(self.Y_rec, T_r, Tinv_r, ZC_r, YC_r)
            h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_0"])
            Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_0"])

        # Node 1: load_0 - Segment 1: l_w_1
        Y_node2 = Y_reccarried + Y_loads["load_0"]
        if fault_seg_idx == 1:
            Y_node2carried, h2 = carry_back_with_fault(
                Y_node2, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                cable_lengths_physical["l_w_1"], local_fault_pos,
                Z_fault_real, Z_fault_imag, self.device
            )
        else:
            rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
            h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_1"])
            Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_1"])

        # Node 2: Junction box
        rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
        Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_2"])
        Y_61 = Y_63 + Y_loads["load_2"]
        rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
        Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_3"])

        Y_node3 = Y_node2carried + Y_6

        # Segment 2: l_w_4
        if fault_seg_idx == 2:
            Y_node3carried, h3 = carry_back_with_fault(
                Y_node3, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
                cable_lengths_physical["l_w_4"], local_fault_pos,
                Z_fault_real, Z_fault_imag, self.device
            )
        else:
            rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
            h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_4"])
            Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_4"])

        # Node 3: Service panel (4 rooms)
        Y_5 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(3, 7)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(5, 10)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_4 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(7, 11)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(10, 15)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_3 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(11, 15)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(15, 20)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_2 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(15, 19)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(20, 25)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )

        Y_node4 = Y_node3carried + Y_5 + Y_4 + Y_3 + Y_2

        # Segment 3: l_w_25
        if fault_seg_idx == 3:
            Y_node4carried, h4 = carry_back_with_fault(
                Y_node4, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
                cable_lengths_physical["l_w_25"], local_fault_pos,
                Z_fault_real, Z_fault_imag, self.device
            )
        else:
            rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
            h4 = h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_25"])
            Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_25"])

        # Node 4: load_19, load_20 branch
        rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
        Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_26"])
        Y_12 = Y_14 + Y_loads["load_20"]
        rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
        Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_27"])

        Y_node5 = Y_node4carried + Y_1

        # Segment 4: l_w_28
        if fault_seg_idx == 4:
            Y_node5carried, h5 = carry_back_with_fault(
                Y_node5, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                cable_lengths_physical["l_w_28"], local_fault_pos,
                Z_fault_real, Z_fault_imag, self.device
            )
        else:
            rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
            h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_28"])
            Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_28"])

        # Node 5: load_21 (near transmitter)
        rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
        Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_29"])
        Y_node6 = Y_node5carried + Y_13

        # Compute transmitter transfer function
        H_trans = calculate_Htrans(
            self.YTalpha, self.YTbeta, self.YTgamma, Y_node6,
            self.ZT0, self.ZT12, self.ZT12, self.ZT13, self.ZT13, self.ZT23, self.ZT23
        )

        hoverall = h1 @ h2 @ h3 @ h4 @ h5
        H1 = hoverall @ H_trans
        H_nw = H1[:, 0, 0]
        return H_nw

    def _calculate_Hnw_batched(self, cable_lengths_physical, Y_loads, fault_seg_idx,
                                local_fault_pos, Z_fault_real, Z_fault_imag, num_particles,
                                T_s, Tinv_s, gamma_s, ZC_s, YC_s,
                                T_r, Tinv_r, gamma_r, ZC_r, YC_r):
        """Compute transfer function with fault for batched (particle) case."""
        P = num_particles if num_particles else fault_seg_idx.shape[0]
        num_backbone_segments = 5

        # Pre-compute side branches that don't depend on fault location
        rho63 = reflection_coefficient(Y_loads["load_1"], T_r, Tinv_r, ZC_r, YC_r)
        Y_63 = carry_back_load(rho63, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_2"])
        Y_61 = Y_63 + Y_loads["load_2"]
        rho61 = reflection_coefficient(Y_61, T_r, Tinv_r, ZC_r, YC_r)
        Y_6 = carry_back_load(rho61, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_3"])

        # 4 rooms
        Y_5 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(3, 7)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(5, 10)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_4 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(7, 11)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(10, 15)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_3 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(11, 15)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(15, 20)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_2 = calculate_room_admittance_matrix(
            [Y_loads[f"load_{i}"] for i in range(15, 19)],
            [cable_lengths_physical[f"l_w_{i}"] for i in range(20, 25)],
            T_r, Tinv_r, ZC_r, YC_r, gamma_r,
            T_s, Tinv_s, ZC_s, YC_s, gamma_s
        )
        Y_rooms = Y_5 + Y_4 + Y_3 + Y_2

        # Node 4 branch
        rho14 = reflection_coefficient(Y_loads["load_19"], T_r, Tinv_r, ZC_r, YC_r)
        Y_14 = carry_back_load(rho14, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_26"])
        Y_12 = Y_14 + Y_loads["load_20"]
        rho12 = reflection_coefficient(Y_12, T_r, Tinv_r, ZC_r, YC_r)
        Y_1 = carry_back_load(rho12, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_27"])

        # Transmitter branch
        rho13 = reflection_coefficient(Y_loads["load_21"], T_r, Tinv_r, ZC_r, YC_r)
        Y_13 = carry_back_load(rho13, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_29"])

        H_all_segments = []

        for assumed_seg in range(num_backbone_segments):
            # Segment 0: l_w_0
            if assumed_seg == 0:
                Y_reccarried, h1 = carry_back_with_fault(
                    self.Y_rec, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                    cable_lengths_physical["l_w_0"], local_fault_pos,
                    Z_fault_real, Z_fault_imag, self.device
                )
            else:
                rho1 = reflection_coefficient(self.Y_rec, T_r, Tinv_r, ZC_r, YC_r)
                h1 = h_B(rho1, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_0"])
                Y_reccarried = carry_back_load(rho1, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_0"])

            Y_node2 = Y_reccarried + Y_loads["load_0"]

            # Segment 1: l_w_1
            if assumed_seg == 1:
                Y_node2carried, h2 = carry_back_with_fault(
                    Y_node2, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                    cable_lengths_physical["l_w_1"], local_fault_pos,
                    Z_fault_real, Z_fault_imag, self.device
                )
            else:
                rho2 = reflection_coefficient(Y_node2, T_r, Tinv_r, ZC_r, YC_r)
                h2 = h_B(rho2, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_1"])
                Y_node2carried = carry_back_load(rho2, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_1"])

            Y_node3 = Y_node2carried + Y_6

            # Segment 2: l_w_4
            if assumed_seg == 2:
                Y_node3carried, h3 = carry_back_with_fault(
                    Y_node3, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
                    cable_lengths_physical["l_w_4"], local_fault_pos,
                    Z_fault_real, Z_fault_imag, self.device
                )
            else:
                rho3 = reflection_coefficient(Y_node3, T_s, Tinv_s, ZC_s, YC_s)
                h3 = h_B(rho3, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_4"])
                Y_node3carried = carry_back_load(rho3, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_4"])

            Y_node4 = Y_node3carried + Y_rooms

            # Segment 3: l_w_25
            if assumed_seg == 3:
                Y_node4carried, h4 = carry_back_with_fault(
                    Y_node4, T_s, Tinv_s, ZC_s, YC_s, gamma_s,
                    cable_lengths_physical["l_w_25"], local_fault_pos,
                    Z_fault_real, Z_fault_imag, self.device
                )
            else:
                rho4 = reflection_coefficient(Y_node4, T_s, Tinv_s, ZC_s, YC_s)
                h4 = h_B(rho4, ZC_s, T_s, Tinv_s, gamma_s, cable_lengths_physical["l_w_25"])
                Y_node4carried = carry_back_load(rho4, T_s, YC_s, gamma_s, cable_lengths_physical["l_w_25"])

            Y_node5 = Y_node4carried + Y_1

            # Segment 4: l_w_28
            if assumed_seg == 4:
                Y_node5carried, h5 = carry_back_with_fault(
                    Y_node5, T_r, Tinv_r, ZC_r, YC_r, gamma_r,
                    cable_lengths_physical["l_w_28"], local_fault_pos,
                    Z_fault_real, Z_fault_imag, self.device
                )
            else:
                rho5 = reflection_coefficient(Y_node5, T_r, Tinv_r, ZC_r, YC_r)
                h5 = h_B(rho5, ZC_r, T_r, Tinv_r, gamma_r, cable_lengths_physical["l_w_28"])
                Y_node5carried = carry_back_load(rho5, T_r, YC_r, gamma_r, cable_lengths_physical["l_w_28"])

            Y_node6 = Y_node5carried + Y_13

            H_trans = calculate_Htrans(
                self.YTalpha, self.YTbeta, self.YTgamma, Y_node6,
                self.ZT0, self.ZT12, self.ZT12, self.ZT13, self.ZT13, self.ZT23, self.ZT23
            )

            hoverall = h1 @ h2 @ h3 @ h4 @ h5
            H1 = hoverall @ H_trans
            H_seg = H1[:, :, 0, 0]

            H_all_segments.append(H_seg)

        # Stack: [5, P, F]
        H_stacked = torch.stack(H_all_segments, dim=0)

        # Select based on fault_seg_idx for each particle
        P_idx = torch.arange(P, device=self.device)
        H_nw = H_stacked[fault_seg_idx, P_idx, :]

        return H_nw
