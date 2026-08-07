"""
Core module for MTL network forward model.
"""

from .loads import (
    constant_impedance,
    double_RLC,
    motor_load,
    compute_load_admittance_3d,
    compute_load_admittances,
)

from .mtl_utils import (
    denormalize,
    normalize,
    loguniform,
    calculate_cable_parameters,
    get_mtl_matrices,
    reflection_coefficient,
    carry_back_load,
    h_B,
    calculate_room_admittance_matrix,
    calculate_Htrans,
    matrix_cosh,
    matrix_sinh,
)

from .forward_mtl import MTLForwardModel

from .inference import (
    InferenceConfig,
    SVIEngine,
    create_svi_engine,
)

__all__ = [
    # Loads
    'constant_impedance',
    'double_RLC',
    'motor_load',
    'compute_load_admittance_3d',
    'compute_load_admittances',
    # MTL utilities
    'denormalize',
    'normalize',
    'loguniform',
    'calculate_cable_parameters',
    'get_mtl_matrices',
    'reflection_coefficient',
    'carry_back_load',
    'h_B',
    'calculate_room_admittance_matrix',
    'calculate_Htrans',
    'matrix_cosh',
    'matrix_sinh',
    # Forward model
    'MTLForwardModel',
    # Inference
    'InferenceConfig',
    'SVIEngine',
    'create_svi_engine',
]
