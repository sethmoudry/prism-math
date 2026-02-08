"""
Primitives and decompositions for the method system.

This module defines:
1. PRIMITIVES: The ~53 core methods that cannot be decomposed further
2. DECOMPOSITIONS: Mappings from composite methods to primitive expressions (1000+)

Decompositions are split by category across multiple files for maintainability.
"""

from ..decomposition import Decomposition, DecompositionRegistry

from .arithmetic import ARITHMETIC_DECOMPOSITIONS
from .number_theory_defs import NUMBER_THEORY_DECOMPOSITIONS
from .combinatorics_defs import COMBINATORICS_DECOMPOSITIONS
from .geometry_defs import GEOMETRY_DECOMPOSITIONS
from .algebra_defs import ALGEBRA_DECOMPOSITIONS

# =============================================================================
# PRIMITIVE METHODS (~53 core operations)
# =============================================================================

PRIMITIVES = {
    # ==========================================================================
    # ORIGINAL 38 PRIMITIVES (core operations)
    # ==========================================================================

    # Arithmetic (8)
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "mod",
    "floor",
    "negate",

    # Combinatorics (3)
    "factorial",
    "binomial",
    "permutation",

    # Number theory (8)
    "gcd",
    "lcm",
    "totient",
    "is_prime",
    "padic_valuation",
    "mod_inverse",
    "mobius_function",
    "primitive_root",

    # Sequences (6)
    "fibonacci",
    "lucas_number",
    "sum_range",
    "product_range",
    "arithmetic_sum",
    "nth_prime",

    # Comparison (4)
    "min_value",
    "max_value",
    "compare_values",
    "equal_to",

    # List operations (4)
    "sum",
    "product",
    "length",
    "count_elements",

    # Constants (3)
    "constant_zero",
    "constant_one",
    "constant_result",

    # Iteration helpers (2)
    "sum_digits",
    "count_divisors",

    # ==========================================================================
    # ADDED PRIMITIVES (~15 truly necessary additions)
    # ==========================================================================

    # Logarithms (2) - commonly needed
    "log2",
    "floor_log2",

    # Roots (1) - needed for geometry
    "sqrt",

    # Recurrence sequences (1) - general linear recurrence
    "linear_recurrence",

    # Algorithmic - truly cannot be decomposed (4)
    "prime_factorization",    # Returns list of prime factors
    "divisor_list",           # Returns list of divisors
    "chinese_remainder",      # CRT algorithm
    "count_partitions",       # Integer partitions (no closed form)

    # Divisor functions (2)
    "sum_divisors",           # sigma(n) - sum of divisors
    "divisor_sigma",          # sigma_k(n) - generalized divisor function

    # Number theory algorithmic (3)
    "count_primes_leq",       # pi(n) - prime counting function
    "jacobi_symbol",          # Jacobi/Legendre symbol
    "n_order",                # Multiplicative order
}

# =============================================================================
# MERGED DECOMPOSITIONS
# =============================================================================

DECOMPOSITIONS = {
    **ARITHMETIC_DECOMPOSITIONS,
    **NUMBER_THEORY_DECOMPOSITIONS,
    **COMBINATORICS_DECOMPOSITIONS,
    **GEOMETRY_DECOMPOSITIONS,
    **ALGEBRA_DECOMPOSITIONS,
}


def register_all_primitives():
    """Register all primitives with the DecompositionRegistry."""
    DecompositionRegistry.register_primitives(list(PRIMITIVES))


def register_all_decompositions():
    """Register all decompositions with the DecompositionRegistry."""
    for method_name, decomposition in DECOMPOSITIONS.items():
        DecompositionRegistry.register_decomposition(method_name, decomposition)


def initialize_decomposition_system():
    """Initialize the full decomposition system."""
    register_all_primitives()
    register_all_decompositions()


# Export all public symbols
__all__ = [
    "PRIMITIVES",
    "DECOMPOSITIONS",
    "ARITHMETIC_DECOMPOSITIONS",
    "NUMBER_THEORY_DECOMPOSITIONS",
    "COMBINATORICS_DECOMPOSITIONS",
    "GEOMETRY_DECOMPOSITIONS",
    "ALGEBRA_DECOMPOSITIONS",
    "register_all_primitives",
    "register_all_decompositions",
    "initialize_decomposition_system",
    "Decomposition",
    "DecompositionRegistry",
]
