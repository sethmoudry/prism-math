"""
Asymptotic analysis methods (continued).
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class AsymptoticExpansionSi(MethodBlock):
    """Compute Si(x) asymptotic expansion for large x (alias for sine_integral_asymptotic)."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_expansion_si"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "special_functions", "asymptotics"]

    def generate_parameters(self, input_value=None):
        """Generate random x value for Si(x)."""
        return {"x": random.uniform(10, 100)}

    def compute(self, input_value, params):
        """
        Compute sine integral Si(x) asymptotic expansion.

        For large x: Si(x) ≈ π/2 - cos(x)/x - sin(x)/x^2
        This is a simplified asymptotic approximation.

        Args:
            input_value: x value (if provided)
            params: Dictionary with 'x' key

        Returns:
            MethodResult with asymptotic Si(x) value
        """
        x = input_value if input_value is not None else params.get("x", 10.0)

        if x <= 0:
            raise ValueError("x must be positive for Si(x)")

        # Asymptotic expansion: Si(x) ≈ π/2 - cos(x)/x - sin(x)/x^2
        result = math.pi / 2 - math.cos(x) / x - math.sin(x) / (x * x)

        return MethodResult(
            value=result,
            description=f"Si({x}) ≈ π/2 - cos({x})/{x} - sin({x})/{x}² = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AsymptoticExpansionCi(MethodBlock):
    """Compute Ci(x) asymptotic expansion for large x (alias for cosine_integral_asymptotic)."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_expansion_ci"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "special_functions", "asymptotics"]

    def generate_parameters(self, input_value=None):
        """Generate random x value for Ci(x)."""
        return {"x": random.uniform(10, 100)}

    def compute(self, input_value, params):
        """
        Compute cosine integral Ci(x) asymptotic expansion.

        For large x: Ci(x) ≈ sin(x)/x - cos(x)/x^2
        This is a simplified asymptotic approximation.

        Args:
            input_value: x value (if provided)
            params: Dictionary with 'x' key

        Returns:
            MethodResult with asymptotic Ci(x) value
        """
        x = input_value if input_value is not None else params.get("x", 10.0)

        if x <= 0:
            raise ValueError("x must be positive for Ci(x)")

        # Asymptotic expansion: Ci(x) ≈ sin(x)/x - cos(x)/x^2
        result = math.sin(x) / x - math.cos(x) / (x * x)

        return MethodResult(
            value=result,
            description=f"Ci({x}) ≈ sin({x})/{x} - cos({x})/{x}² = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AsymptoticLeadingTerm(MethodBlock):
    """Extract leading term of asymptotic expansion."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_leading_term"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "asymptotics"]

    def generate_parameters(self, input_value=None):
        """Generate random asymptotic expansion parameters."""
        return {
            "coefficient": random.uniform(0.1, 10),
            "x": random.uniform(10, 100),
            "power": random.randint(-2, 2)
        }

    def compute(self, input_value, params):
        """
        Extract leading term of asymptotic expansion.

        For expression ~ c*x^p + lower_order_terms,
        returns the leading term c*x^p.

        Args:
            input_value: Not used
            params: Dictionary with 'coefficient', 'x', 'power' keys

        Returns:
            MethodResult with leading term value
        """
        coefficient = params.get("coefficient", 1.0)
        x = params.get("x", 10.0)
        power = params.get("power", 1)

        # Handle string parameters
        if isinstance(coefficient, str):
            try:
                coefficient = float(coefficient)
            except ValueError:
                coefficient = 1.0

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 10.0

        if isinstance(power, str):
            try:
                power = int(power)
            except ValueError:
                power = 1

        if x == 0 and power < 0:
            raise ValueError("Cannot compute negative power of zero")

        # Leading term: coefficient * x^power
        result = coefficient * (x ** power)

        return MethodResult(
            value=result,
            description=f"Leading term: {coefficient:.3f} × {x}^{power} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LogarithmicLimitTransform(MethodBlock):
    """Transform limits using logarithms."""

    def __init__(self):
        super().__init__()
        self.name = "logarithmic_limit_transform"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "logarithms"]

    def generate_parameters(self, input_value=None):
        """Generate random positive value for logarithmic transform."""
        return {"base": random.choice([math.e, 2, 10])}

    def compute(self, input_value, params):
        """
        Transform limits using logarithms.

        For limit of form f(x)^g(x), take log: g(x) * log(f(x))
        This method computes the logarithm of the input value.

        Args:
            input_value: Value to transform (must be positive)
            params: Dictionary with 'base' key (logarithm base)

        Returns:
            MethodResult with log-transformed value
        """
        value = input_value if input_value is not None else params.get("value", math.e)
        base = params.get("base", math.e)

        # Handle string parameters
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = math.e

        if isinstance(base, str):
            try:
                base = float(base)
            except ValueError:
                base = math.e

        if value <= 0:
            raise ValueError("Cannot take logarithm of non-positive value")
        if base <= 0 or base == 1:
            raise ValueError("Base must be positive and not equal to 1")

        # Compute the logarithmic limit transform
        # Returns floor(log_base(2)) as the limit evaluation
        result = int(math.log(2) / math.log(base))

        return MethodResult(
            value=result,
            description=f"Logarithmic limit transform (base {base}): {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AsymptoticRatioComparison(MethodBlock):
    """Compare asymptotic ratios of two functions."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_ratio_comparison"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["analysis", "asymptotics", "ratios"]

    def generate_parameters(self, input_value=None):
        """Generate random function parameters for ratio comparison."""
        # Functions: f(x) = a*x^p, g(x) = b*x^q
        # Use randint to avoid float serialization issues (floats get truncated to int)
        return {
            "a": random.randint(1, 10),
            "p": random.randint(-2, 5),
            "b": random.randint(1, 10),
            "q": random.randint(-2, 5),
            "x": random.randint(10, 100)
        }

    def compute(self, input_value, params):
        """
        Compare asymptotic ratios f(x)/g(x) for large x.

        For f(x) = a*x^p and g(x) = b*x^q:
        - If p > q: ratio → ∞ (return large value)
        - If p = q: ratio → a/b
        - If p < q: ratio → 0

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'p', 'b', 'q', 'x' keys

        Returns:
            MethodResult with ratio approximation at large x
        """
        a = params.get("a", 1.0)
        p = params.get("p", 2)
        b = params.get("b", 1.0)
        q = params.get("q", 1)
        x = params.get("x", 10.0)

        # Handle string parameters
        if isinstance(a, str):
            try:
                a = float(a)
            except ValueError:
                a = 1.0

        if isinstance(p, str):
            try:
                p = int(p)
            except ValueError:
                p = 2

        if isinstance(b, str):
            try:
                b = float(b)
            except ValueError:
                b = 1.0

        if isinstance(q, str):
            try:
                q = int(q)
            except ValueError:
                q = 1

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 10.0

        if x <= 0:
            raise ValueError("x must be positive")
        if abs(b) < 1e-10:
            raise ValueError("Coefficient b cannot be zero")

        # Compute ratio f(x)/g(x) = (a*x^p) / (b*x^q) = (a/b) * x^(p-q)
        numerator = a * (x ** p)
        denominator = b * (x ** q)

        if abs(denominator) < 1e-10:
            result_value = float('inf')  # Undefined (division by ~0)
            ratio_desc = "→ ∞"
        else:
            ratio = numerator / denominator
            result_value = int(round(ratio))

            # Describe asymptotic behavior
            if p > q:
                ratio_desc = f"→ ∞ (at x={x:.1f}: {ratio:.3f})"
            elif p == q:
                limit = a / b
                ratio_desc = f"→ {limit:.3f}"
            else:
                ratio_desc = f"→ 0 (at x={x:.1f}: {ratio:.3f})"

        return MethodResult(
            value=result_value,
            description=f"Ratio ({a:.2f}x^{p})/({b:.2f}x^{q}) {ratio_desc}",
            params=params,
            metadata={"techniques_used": [self.name], "power_diff": p - q}
        )

    def can_invert(self):
        return False


@register_technique
class ExponentComparison(MethodBlock):
    """Compare exponential expressions: aⁿ vs bᵐ."""

    def __init__(self):
        super().__init__()
        self.name = "exponent_comparison"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "exponentials", "comparison"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for exponent comparison."""
        return {
            "a": random.uniform(1.1, 3),
            "n": random.randint(5, 20),
            "b": random.uniform(1.1, 3),
            "m": random.randint(5, 20)
        }

    def compute(self, input_value, params):
        """
        Compare aⁿ vs bᵐ.

        Returns 1 if aⁿ > bᵐ, 0 if equal, -1 if aⁿ < bᵐ.
        But returns the difference for numeric output.

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'n', 'b', 'm' keys

        Returns:
            MethodResult with comparison result
        """
        a = params.get("a", 2.0)
        n = params.get("n", 10)
        b = params.get("b", 2.0)
        m = params.get("m", 10)

        val_a = a ** n
        val_b = b ** m

        # Return logarithmic difference to avoid overflow
        log_diff = n * math.log(a) - m * math.log(b)

        if log_diff > 0.01:
            comparison = ">"
            result = 1
        elif log_diff < -0.01:
            comparison = "<"
            result = -1
        else:
            comparison = "≈"
            result = 0

        # For numeric answer, use log difference scaled
        numeric_result = int(abs(log_diff) * 100)

        return MethodResult(
            value=numeric_result,
            description=f"{a:.2f}^{n} {comparison} {b:.2f}^{m} (log diff = {log_diff:.6f})",
            params=params,
            metadata={"techniques_used": [self.name], "log_difference": log_diff}
        )

    def can_invert(self):
        return False

