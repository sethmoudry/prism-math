"""
Series and sequence algebra techniques.

Contains:
- Arithmetic and geometric series (5)
- Telescoping (1)
- Partial fractions (1)
"""

import random
import math
from typing import Any, Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# SERIES (5 techniques)
# ============================================================================

@register_technique
class ArithmeticSum(MethodBlock):
    """Sum of arithmetic series."""

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_sum"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "series", "arithmetic"]
        self.is_primitive = True

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate arithmetic series parameters."""
        a = random.randint(1, 10)
        d = random.randint(1, 5)
        n = random.randint(5, 20)
        return {"a": a, "d": d, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute sum = n/2 * (2a + (n-1)d)."""
        a, d, n = params.get("a", 10), params.get("d", 5), params.get("n", 10)

        sum_val = n * (2*a + (n-1)*d) // 2

        return MethodResult(
            value=sum_val,
            description=f"Sum of {n} terms with a={a}, d={d} is {sum_val}",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ArithmeticSumInverseN(MethodBlock):
    """Find n given arithmetic sum."""

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_sum_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "series", "arithmetic", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters."""
        if input_value is not None and input_value > 1000:
            a = random.randint(1, 5)
            d = random.randint(1, 3)
            target = random.randint(5000, 20000)
        else:
            a = random.randint(1, 5)
            d = random.randint(1, 3)
            target = random.randint(50, 200)
        return {"a": a, "d": d, "target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find n where n/2 * (2a + (n-1)d) >= target."""
        a, d = params.get("a", 10), params.get("d", 5)
        target = params.get("target", 10)

        A = d
        B = 2*a - d
        C = -2 * target

        disc = B**2 - 4*A*C
        if disc >= 0:
            n = (-B + math.sqrt(disc)) / (2*A)
            n = int(math.ceil(n))
        else:
            n = -1

        description = f"Need n={n} terms to reach sum>={target}"

        return MethodResult(
            value=n,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SumRange(MethodBlock):
    """Sum of integers from start to end (inclusive)."""

    def __init__(self):
        super().__init__()
        self.name = "sum_range"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "series", "arithmetic", "sum"]
        self.is_primitive = True

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate range parameters."""
        start = random.randint(1, 20)
        end = start + random.randint(5, 50)
        return {"start": start, "end": end}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute sum of integers from start to end (inclusive)."""
        start = params.get("start", 1)
        end = params.get("end", input_value if input_value is not None else 100)

        if start is None:
            start = 1

        n = end - start + 1
        sum_val = n * (start + end) // 2

        return MethodResult(
            value=sum_val,
            description=f"Sum of integers from {start} to {end} = {sum_val}",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class GeometricSum(MethodBlock):
    """Sum of geometric series."""

    def __init__(self):
        super().__init__()
        self.name = "geometric_sum"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "series", "geometric"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate geometric series parameters."""
        a = random.randint(1, 5)
        r = random.randint(2, 4)
        n = random.randint(4, 8)
        return {"a": a, "r": r, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute sum = a * (r^n - 1) / (r - 1)."""
        a, r, n = params.get("a", 10), params.get("r", 5), params.get("n", 10)

        sum_val = a * (r**n - 1) // (r - 1)

        return MethodResult(
            value=sum_val,
            description=f"Sum of {n} terms with a={a}, r={r} is {sum_val}",
            params=params
        )

    def validate_params(self, params, prev_value=None):
        """Common ratio r must not be 1."""
        r = params.get("r", 2)
        n = params.get("n", 1)
        return r != 1 and n >= 1

    def can_invert(self) -> bool:
        return False


@register_technique
class GeometricSumInverseN(MethodBlock):
    """Find n given geometric sum."""

    def __init__(self):
        super().__init__()
        self.name = "geometric_sum_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "series", "geometric", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters."""
        if input_value is not None and input_value > 1000:
            a = random.randint(1, 3)
            r = random.randint(2, 3)
            target = random.randint(10000, 50000)
        else:
            a = random.randint(1, 3)
            r = random.randint(2, 3)
            target = random.randint(50, 200)
        return {"a": a, "r": r, "target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find n where a*(r^n-1)/(r-1) >= target."""
        a, r = params.get("a", 10), params.get("r", 5)
        target = params.get("target", 10)

        threshold = target * (r - 1) / a + 1
        n = math.ceil(math.log(threshold) / math.log(r))

        description = f"Need n={n} terms to reach sum>={target}"

        return MethodResult(
            value=n,
            description=description,
            params=params
        )

    def validate_params(self, params, prev_value=None):
        """Common ratio r must be > 1."""
        r = params.get("r", 2)
        a = params.get("a", 1)
        target = params.get("target", 1)
        return r > 1 and a > 0 and target > 0

    def can_invert(self) -> bool:
        return False


# ============================================================================
# BASIC ADVANCED TECHNIQUES (2)
# ============================================================================

@register_technique
class Telescoping(MethodBlock):
    """Simplify telescoping sum."""

    def __init__(self):
        super().__init__()
        self.name = "telescoping"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "series", "telescoping"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate telescoping series."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            n = random.randint(100, 500)
        else:
            n = random.randint(5, 20)
        return {"n": n, "type": "harmonic"}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute telescoping sum."""
        n = params.get("n", 10)

        result = 1 - 1/(n+1)
        result_int = n * (n+1) // (n+1)

        description = f"Telescoping sum Sigma 1/(i(i+1)) from 1 to {n} = {result:.4f}"

        return MethodResult(
            value=result_int,
            description=description,
            params=params, metadata={"n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PartialFractions(MethodBlock):
    """Decompose rational function into partial fractions."""

    def __init__(self):
        super().__init__()
        self.name = "partial_fractions"
        self.input_type = "polynomial"
        self.output_type = "sequence"
        self.difficulty = 4
        self.tags = ["algebra", "partial_fractions", "rational"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate rational function with distinct linear roots.

        Produces N/(prod of (x - r_i)) where roots r_i are distinct integers.
        """
        num_roots = random.randint(2, 4)
        roots = random.sample(range(-5, 6), num_roots)
        # Numerator is a constant (degree 0)
        numerator_const = random.choice([1, 2, 3, -1, -2])
        return {"roots": roots, "numerator_const": numerator_const}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Decompose N / prod((x - r_i)) into partial fractions A_i / (x - r_i).

        For f(x) = N / prod_{i}(x - r_i), each coefficient is:
            A_k = N / prod_{j != k}(r_k - r_j)
        """
        roots = params.get("roots", [0, 1])
        N = params.get("numerator_const", 1)

        coefficients = []
        for k, rk in enumerate(roots):
            denom_product = 1
            for j, rj in enumerate(roots):
                if j != k:
                    denom_product *= (rk - rj)
            coefficients.append(N / denom_product)

        # Round to clean floats if they are integers
        clean_coeffs = []
        for c in coefficients:
            if abs(c - round(c)) < 1e-9:
                clean_coeffs.append(int(round(c)))
            else:
                clean_coeffs.append(round(c, 6))

        # Build description
        terms = []
        for coeff, root in zip(clean_coeffs, roots):
            if root == 0:
                terms.append(f"{coeff}/x")
            elif root > 0:
                terms.append(f"{coeff}/(x-{root})")
            else:
                terms.append(f"{coeff}/(x+{-root})")
        desc = f"{N} / prod(x - r_i) = " + " + ".join(terms)

        return MethodResult(
            value=clean_coeffs,
            description=desc,
            params=params,
            metadata={"roots": roots, "numerator_const": N, "coefficients": clean_coeffs}
        )

    def can_invert(self) -> bool:
        return False
