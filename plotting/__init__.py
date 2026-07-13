"""
Plotting utilities for CRLB analysis and visualization.
"""

from .plot_1D_benchmark import (
    compute_nll,
    plot_loss_landscape,
    analyze_gamma_impact_on_loss,
    analyze_L_impact_on_loss,
    analyze_freq_impact_on_loss,
    load_results,
    plot_rmse_vs_crlb_1D,
    plot_all_loss_landscapes,
    plot_gamma_impact_on_L1,
    plot_L_impact_on_L1,
    plot_freq_impact_on_L1,
)

from .plot_gradient_comparison import (
    load_results as load_gradient_results,
    plot_comparison,
)

from .plot_joint_benchmark import (
    load_results as load_joint_results,
    plot_parameter,
)

from .plot_joint_gradient_comparison import (
    load_results as load_joint_gradient_results,
    plot_parameter_comparison,
)

__all__ = [
    # Loss landscape functions
    'compute_nll',
    'plot_loss_landscape',
    'analyze_gamma_impact_on_loss',
    'analyze_L_impact_on_loss',
    'analyze_freq_impact_on_loss',
    # 1D benchmark plotting
    'load_results',
    'plot_rmse_vs_crlb_1D',
    'plot_all_loss_landscapes',
    'plot_gamma_impact_on_L1',
    'plot_L_impact_on_L1',
    'plot_freq_impact_on_L1',
    # Gradient comparison plotting
    'load_gradient_results',
    'plot_comparison',
    # Joint benchmark plotting
    'load_joint_results',
    'plot_parameter',
    # Joint gradient comparison plotting
    'load_joint_gradient_results',
    'plot_parameter_comparison',
]
