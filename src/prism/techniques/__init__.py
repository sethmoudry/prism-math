"""
Techniques package for PRISM - Mathematical method blocks for problem generation.

This package provides composable mathematical techniques organized by category:
- geometry: Triangles, circles, polygons, coordinates, transformations
- number_theory: Divisibility, primes, modular arithmetic, sequences, valuation
- algebra: Vieta, polynomials, equations, inequalities, series, matrices
- combinatorics: Counting, special numbers, graphs, generating functions, probability
- primitives: Core primitive methods and decomposition mappings
"""

# =============================================================================
# Base classes and registry (always required)
# =============================================================================

from .base import MethodBlock, MethodResult
from .registry import MethodRegistry, register_technique, is_compatible
from .decomposition import Decomposition, DecompositionRegistry

# =============================================================================
# Primitives and decompositions
# =============================================================================

from .primitives import (
    PRIMITIVES,
    DECOMPOSITIONS,
    ARITHMETIC_DECOMPOSITIONS,
    NUMBER_THEORY_DECOMPOSITIONS,
    COMBINATORICS_DECOMPOSITIONS,
    GEOMETRY_DECOMPOSITIONS,
    ALGEBRA_DECOMPOSITIONS,
    register_all_primitives,
    register_all_decompositions,
    initialize_decomposition_system,
)

# =============================================================================
# Technique submodules - import all techniques from each category
# =============================================================================

# Geometry techniques (triangles, circles, polygons, coordinates, etc.)
from .geometry import *

# Number theory techniques (primes, modular, valuations, sequences, etc.)
from .number_theory import *

# Algebra techniques (vieta, polynomials, equations, inequalities, etc.)
from .algebra import *

# Combinatorics techniques (counting, graphs, generating functions, etc.)
from .combinatorics import *

# =============================================================================
# Optional modules - import if available
# =============================================================================

# Basic primitives (optional - may be merged into primitives)
try:
    from .basic_primitives import *
except ImportError:
    pass

# Analysis methods (optional)
try:
    from .analysis_methods import *
except ImportError:
    pass

# Deep insights (functional equations, sequences, number theory, combinatorics)
try:
    from .advanced_techniques import *
except ImportError:
    pass

# Operations (generic operational methods for pipeline)
try:
    from .operations import *
except ImportError:
    pass

# Operations part 2 (control flow, extraction, aliases)
try:
    from .operations_2 import *
except ImportError:
    pass

# Method aliases (maps alternate names to canonical prism methods)
try:
    from .aliases import *
except ImportError:
    pass

# =============================================================================
# Core exports (always available)
# =============================================================================

__all__ = [
    # Base classes
    "MethodBlock",
    "MethodResult",
    # Registry
    "MethodRegistry",
    "register_technique",
    "is_compatible",
    # Decomposition
    "Decomposition",
    "DecompositionRegistry",
    # Primitives
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
]
