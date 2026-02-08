"""
Limit computation and analysis methods.
"""

import random
from typing import Any

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class LimitIntegralExpression(MethodBlock):
    """Compute limit of integral expression as n→∞."""

    def __init__(self):
        super().__init__()
        self.name = "limit_integral_expression"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "integrals"]

    def generate_parameters(self, input_value=None):
        """Generate random integral limit parameters."""
        return {
            "a": random.randint(1, 10),
            "b": random.randint(1, 10),
            "n": random.randint(10, 100)
        }

    def compute(self, input_value, params):
        """
        Compute limit of integral expression as n→∞.

        For simple cases, returns the limit value.
        Example: lim (a/n + b/n^2) as n→∞ = 0

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'b', 'n' keys

        Returns:
            MethodResult with limit value
        """
        a = params.get("a", 1)
        b = params.get("b", 1)
        n = params.get("n", 100)

        if n == 0:
            raise ValueError("n cannot be zero")

        # Simple case: a/n + b/n^2 → 0 as n→∞
        # For finite n, we compute the expression
        result = a / n + b / (n * n)

        return MethodResult(
            value=result,
            description=f"Limit expression: {a}/n + {b}/n² at n={n} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AnalyzeLimit(MethodBlock):
    """Compute limit of ratio (L'Hopital style for simple cases)."""

    def __init__(self):
        super().__init__()
        self.name = "analyze_limit"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "ratios"]

    def generate_parameters(self, input_value=None):
        """Generate random numerator and denominator."""
        return {
            "numerator": random.uniform(-100, 100),
            "denominator": random.uniform(1, 100)  # Avoid zero
        }

    def compute(self, input_value, params):
        """
        Compute limit of ratio numerator/denominator.

        For simple cases, returns numerator/denominator if denominator != 0, else 0.

        Args:
            input_value: Not used
            params: Dictionary with 'numerator' and 'denominator' keys

        Returns:
            MethodResult with limit value
        """
        numerator = params.get("numerator", 10)
        denominator = params.get("denominator", 5)

        # Safely extract numeric values - handle dict/list/string params gracefully
        def safe_numeric(val, default):
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, dict):
                # Try to extract a numeric value from dict
                for key in ['value', 'result', 'limit']:
                    if key in val and isinstance(val[key], (int, float)):
                        return val[key]
                return default
            if isinstance(val, (list, tuple)) and val:
                if isinstance(val[0], (int, float)):
                    return val[0]
                return default
            try:
                return float(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return default

        numerator = safe_numeric(numerator, 10.0)
        denominator = safe_numeric(denominator, 5.0)

        if abs(denominator) < 1e-10:
            result = 0.0
            description = f"Limit: {numerator}/{denominator} → 0 (indeterminate)"
        else:
            result = numerator / denominator
            description = f"Limit: {numerator}/{denominator} = {result:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ComposeLimits(MethodBlock):
    """Compose two limits: lim_{x→a} f(g(x))."""

    def __init__(self):
        super().__init__()
        self.name = "compose_limits"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "composition"]

    def generate_parameters(self, input_value=None):
        """Generate random composition parameters."""
        return {
            "f_coeff": random.uniform(1, 10),
            "g_coeff": random.uniform(1, 10),
            "point": random.uniform(1, 10)
        }

    def compute(self, input_value, params):
        """
        Compose two limits: lim_{x→a} f(g(x)).

        For simplicity, uses f(x) = f_coeff * x and g(x) = g_coeff * x.
        Result: lim_{x→a} (f_coeff * g_coeff * x)

        Args:
            input_value: Not used
            params: Dictionary with 'f_coeff', 'g_coeff', 'point' keys

        Returns:
            MethodResult with composed limit value
        """
        f_coeff = params.get("f_coeff", 2.0)
        g_coeff = params.get("g_coeff", 3.0)
        point = params.get("point", 1.0)

        # lim_{x→a} f(g(x)) = f(lim_{x→a} g(x)) if g is continuous
        # With g(x) = g_coeff * x, lim_{x→a} g(x) = g_coeff * a
        # Then f(g_coeff * a) = f_coeff * g_coeff * a
        result = f_coeff * g_coeff * point

        return MethodResult(
            value=result,
            description=f"lim_{{x→{point:.2f}}} f(g(x)) where f(u)={f_coeff:.2f}u, g(x)={g_coeff:.2f}x = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LimitOfReciprocal(MethodBlock):
    """Compute limit of reciprocal: lim 1/f(x)."""

    def __init__(self):
        super().__init__()
        self.name = "limit_of_reciprocal"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "reciprocal"]

    def generate_parameters(self, input_value=None):
        """Generate random value for reciprocal."""
        return {"value": random.uniform(0.1, 10)}

    def compute(self, input_value, params):
        """
        Compute limit of reciprocal: lim 1/f(x).

        Args:
            input_value: f(x) value (if provided)
            params: Dictionary with 'value' key

        Returns:
            MethodResult with 1/f(x)
        """
        value = input_value if input_value is not None else params.get("value", 1.0)

        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = 1.0

        if abs(value) < 1e-10:
            raise ValueError("Cannot compute reciprocal of zero")

        result = 1.0 / value

        return MethodResult(
            value=result,
            description=f"lim 1/f where f={value:.6f} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LimitAtInfinity(MethodBlock):
    """Compute limit at infinity: lim_{x→∞} f(x)."""

    def __init__(self):
        super().__init__()
        self.name = "limit_at_infinity"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "infinity"]

    def generate_parameters(self, input_value=None):
        """Generate random function parameters for limit at infinity."""
        return {
            "coeff": random.uniform(1, 10),
            "power": random.randint(-3, 2),
            "x": random.uniform(100, 1000)  # Large x to approximate infinity
        }

    def compute(self, input_value, params):
        """
        Compute limit at infinity: lim_{x→∞} c*x^p.

        - If p > 0: diverges to ±∞ (return large value)
        - If p = 0: limit = c
        - If p < 0: limit = 0

        Args:
            input_value: Not used
            params: Dictionary with 'coeff', 'power', 'x' keys

        Returns:
            MethodResult with limit value
        """
        coeff = params.get("coeff", 1.0)
        power = params.get("power", -1)
        x = params.get("x", 100.0)

        if power > 0:
            # Diverges - return value at large x
            result = coeff * (x ** power)
            description = f"lim_{{x→∞}} {coeff:.2f}x^{power} → ∞ (value at x={x:.0f}: {result:.6f})"
        elif power == 0:
            # Constant
            result = coeff
            description = f"lim_{{x→∞}} {coeff:.2f} = {result:.6f}"
        else:
            # Converges to 0
            result = coeff * (x ** power)
            description = f"lim_{{x→∞}} {coeff:.2f}x^{power} = 0 (value at x={x:.0f}: {result:.6f})"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LimitExistenceConclusion(MethodBlock):
    """Check if limit exists by comparing left and right limits."""

    def __init__(self):
        super().__init__()
        self.name = "limit_existence_conclusion"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "existence"]

    def generate_parameters(self, input_value=None):
        """Generate random left and right limit values."""
        base = random.uniform(1, 10)
        # Sometimes make them equal, sometimes different
        if random.random() < 0.5:
            return {"left_lim": base, "right_lim": base}
        else:
            return {"left_lim": base, "right_lim": base + random.uniform(0.1, 2)}

    def compute(self, input_value, params):
        """
        Check if limit exists: left_lim == right_lim.

        Returns 1 if limit exists (equal), 0 if not.

        Args:
            input_value: Not used
            params: Dictionary with 'left_lim', 'right_lim' keys

        Returns:
            MethodResult with 1 (exists) or 0 (doesn't exist)
        """
        left_lim = params.get("left_lim", 1.0)
        right_lim = params.get("right_lim", 1.0)

        # Check if equal (with tolerance)
        exists = abs(left_lim - right_lim) < 1e-9
        result = 1 if exists else 0

        if exists:
            description = f"lim exists: left={left_lim:.6f}, right={right_lim:.6f} → limit = {left_lim:.6f}"
        else:
            description = f"lim does not exist: left={left_lim:.6f} ≠ right={right_lim:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name], "limit_value": left_lim if exists else None}
        )

    def can_invert(self):
        return False


@register_technique
class LimitImpliesConstant(MethodBlock):
    """If limit equals L everywhere, then function is constant."""

    def __init__(self):
        super().__init__()
        self.name = "limit_implies_constant"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "constant"]

    def generate_parameters(self, input_value=None):
        """Generate random limit value."""
        return {"limit_val": random.uniform(1, 10)}

    def compute(self, input_value, params):
        """
        If limit equals L everywhere, then function is constant L.

        Args:
            input_value: Limit value (if provided)
            params: Dictionary with 'limit_val' key

        Returns:
            MethodResult with constant value L
        """
        limit_val = input_value if input_value is not None else params.get("limit_val", 5.0)

        if isinstance(limit_val, str):
            try:
                limit_val = float(limit_val)
            except ValueError:
                limit_val = 5.0

        result = limit_val

        return MethodResult(
            value=result,
            description=f"If lim_{{x→a}} f(x) = {limit_val:.6f} for all a, then f(x) = {result:.6f} (constant)",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LimitCombinationFromLiminfLimsup(MethodBlock):
    """Combine liminf and limsup to determine limit existence."""

    def __init__(self):
        super().__init__()
        self.name = "limit_combination_from_liminf_limsup"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "liminf", "limsup"]

    def generate_parameters(self, input_value=None):
        """Generate random liminf and limsup values."""
        base = random.uniform(1, 10)
        # Sometimes equal (limit exists), sometimes different
        if random.random() < 0.5:
            return {"liminf": base, "limsup": base}
        else:
            return {"liminf": base, "limsup": base + random.uniform(0.1, 2)}

    def compute(self, input_value, params):
        """
        Combine liminf and limsup to check if limit exists.

        If liminf = limsup, then limit exists and equals this value.
        Otherwise, limit doesn't exist (return average).

        Args:
            input_value: Not used
            params: Dictionary with 'liminf', 'limsup' keys

        Returns:
            MethodResult with limit value if exists, else average
        """
        liminf = params.get("liminf", 1.0)
        limsup = params.get("limsup", 1.0)

        if abs(liminf - limsup) < 1e-9:
            # Limit exists
            result = liminf
            description = f"liminf = limsup = {result:.6f} → limit exists = {result:.6f}"
        else:
            # Limit doesn't exist, return average
            result = (liminf + limsup) / 2
            description = f"liminf = {liminf:.6f} < {limsup:.6f} = limsup → no limit (avg = {result:.6f})"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name], "limit_exists": abs(liminf - limsup) < 1e-9}
        )

    def can_invert(self):
        return False


@register_technique
class LiminfConclusion(MethodBlock):
    """Compute liminf of a sequence."""

    def __init__(self):
        super().__init__()
        self.name = "liminf_conclusion"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "sequences", "liminf"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence."""
        length = random.randint(5, 15)
        return {"sequence": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Compute liminf of sequence.

        For finite sequences: liminf = min(tail values).
        Simplified: returns minimum of sequence.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with liminf value
        """
        sequence = input_value if input_value is not None else params.get("sequence", [1, 2, 3])

        if not sequence:
            raise ValueError("Empty sequence")

        # For finite sequence, liminf is the minimum
        result = min(sequence)

        return MethodResult(
            value=result,
            description=f"liminf of {sequence[:5]}{'...' if len(sequence) > 5 else ''} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LimitOfProduct(MethodBlock):
    """Compute limit of product: lim f·g."""

    def __init__(self):
        super().__init__()
        self.name = "limit_of_product"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "product"]

    def generate_parameters(self, input_value=None):
        """Generate random function values."""
        return {
            "f_limit": random.uniform(1, 10),
            "g_limit": random.uniform(1, 10)
        }

    def compute(self, input_value, params):
        """
        Compute limit of product: lim f·g = (lim f)·(lim g).

        Args:
            input_value: Not used
            params: Dictionary with 'f_limit', 'g_limit' keys

        Returns:
            MethodResult with product of limits
        """
        f_limit = params.get("f_limit", 2.0)
        g_limit = params.get("g_limit", 3.0)

        result = f_limit * g_limit

        return MethodResult(
            value=result,
            description=f"lim(f·g) = (lim f)·(lim g) = {f_limit:.6f} × {g_limit:.6f} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

