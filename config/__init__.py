"""
Configuration module for network models.

Usage:
    # For simple model (3 parameters: L1, ZF, ZL)
    from config.simple_config_loader import load_config
    exp = load_config("config/simple_network_config.yaml")
"""

from .simple_config_loader import load_config as load_simple_config
from .simple_config_loader import config_hash, fmt_freq

__all__ = [
    'load_simple_config',
    'config_hash',
    'fmt_freq',
]
