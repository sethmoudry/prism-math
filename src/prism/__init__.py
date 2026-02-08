"""
Prism Math - Composable mathematical method blocks.

718 methods across geometry, algebra, number theory, combinatorics,
analysis, and more. Each method implements generate_parameters(),
compute(), and can_invert() for composable problem generation.

Usage:
    from prism.techniques import MethodRegistry, MethodBlock, MethodResult

    methods = MethodRegistry.get_all()
    print(f"{len(methods)} methods loaded")
"""

__version__ = "0.1.0"
