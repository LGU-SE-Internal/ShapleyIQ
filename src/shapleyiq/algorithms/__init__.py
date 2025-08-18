"""
Algorithm implementations for root cause analysis.

This module contains the ShapleyIQ algorithm and baseline algorithms.
"""

try:
    from .base import BaseRCAAlgorithm
    from .microhecl import MicroHECL
    from .microrank import MicroRank
    from .microrca import MicroRCA
    from .shapley_value_rca import ShapleyValueRCA
    from .ton import TON

    __all__ = [
        "BaseRCAAlgorithm",
        "ShapleyValueRCA",
        "MicroHECL",
        "MicroRCA",
        "MicroRank",
        "TON",
    ]
except ImportError as e:
    # Handle import errors gracefully
    import warnings

    warnings.warn(f"Failed to import some algorithms: {e}")

    # Try to import at least the base classes
    try:
        from .base import BaseRCAAlgorithm

        __all__ = ["BaseRCAAlgorithm"]
    except ImportError:
        __all__ = []
