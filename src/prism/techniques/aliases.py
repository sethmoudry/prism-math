"""
Method aliases and small utility techniques.

Aliases map names used in generated problem code to their prism equivalents.
This lets adjudication scripts find methods regardless of naming convention.
"""

import math
import random
from .base import MethodBlock, MethodResult
from .registry import MethodRegistry, register_technique

# ---------------------------------------------------------------------------
# Alias map: old/alternate name -> canonical prism name
# Only entries whose alias doesn't already exist will be registered.
# ---------------------------------------------------------------------------
ALIASES = {
    "bell_number": "bell",
    "find_modular_inverse": "mod_inverse",
    "sum_of_divisors": "divisor_sum",
    "lifting_exponent": "lifting_exponent_valuation",
    "gcd_euclidean": "gcd_compute",
    "chinese_remainder": "chinese_remainder_solve",
    "stirling": "stirling_first",
    "derangements": "derangement",
    "arithmetic_progression_analysis": "arithmetic_progression",
    "determinant_3x3": "matrix_det",
    "roots_of_unity_ops": "roots_of_unity",
}


def register_aliases():
    """Register all aliases. Safe to call multiple times."""
    for alias, target in ALIASES.items():
        MethodRegistry.register_alias(alias, target)


# ---------------------------------------------------------------------------
# difference_of_squares: (a+b)(a-b)
# ---------------------------------------------------------------------------
@register_technique
class DifferenceOfSquaresMethod(MethodBlock):
    """Compute (a+b)*(a-b) = a^2 - b^2."""

    def __init__(self):
        super().__init__()
        self.name = "difference_of_squares"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "factoring"]

    def generate_parameters(self, input_value=None):
        a = random.randint(2, 100)
        b = random.randint(1, a - 1)
        return {"a": a, "b": b}

    def compute(self, input_value=None, params=None):
        if params is None:
            params = {}
        a = params.get("a", input_value if input_value is not None else 0)
        b = params.get("b", 0)
        result = (a + b) * (a - b)
        return MethodResult(
            value=result,
            description=f"({a}+{b})*({a}-{b}) = {a}^2 - {b}^2 = {result}",
            params=params,
        )

    def can_invert(self):
        return False


# ---------------------------------------------------------------------------
# cyclotomic_field_degree: degree of Q(zeta_n)/Q = phi(n)
# ---------------------------------------------------------------------------
@register_technique
class CyclotomicFieldDegreeMethod(MethodBlock):
    """Compute degree of cyclotomic field Q(zeta_n) over Q, which is phi(n)."""

    def __init__(self):
        super().__init__()
        self.name = "cyclotomic_field_degree"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebraic_geometry", "number_theory", "galois_theory"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(2, 30)}

    def _totient(self, n):
        """Euler's totient without sympy dependency."""
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    def compute(self, input_value=None, params=None):
        if params is None:
            params = {}
        n = input_value if input_value is not None else params.get("n", 5)
        if n <= 0:
            n = abs(n) or 1
        result = self._totient(n)
        return MethodResult(
            value=result,
            description=f"[Q(zeta_{n}):Q] = phi({n}) = {result}",
            params=params,
        )

    def can_invert(self):
        return False


# ---------------------------------------------------------------------------
# Auto-register aliases on import
# ---------------------------------------------------------------------------
register_aliases()
