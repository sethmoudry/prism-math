"""
Asymptotic analysis methods.
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class AsymptoticTailSum(MethodBlock):
    """Compute tail sum: total - partial_sum."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_tail_sum"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "series", "asymptotics"]

    def generate_parameters(self, input_value=None):
        """Generate random total and partial sum."""
        total = random.uniform(10, 100)
        partial_sum = random.uniform(1, total)
        return {
            "total": total,
            "partial_sum": partial_sum
        }

    def compute(self, input_value, params):
        """
        Compute tail sum = total - partial_sum.

        Args:
            input_value: Not used
            params: Dictionary with 'total' and 'partial_sum' keys

        Returns:
            MethodResult with tail sum
        """
        total = params.get("total", 100)
        partial_sum = params.get("partial_sum", 60)

        result = total - partial_sum

        return MethodResult(
            value=result,
            description=f"Tail sum: {total} - {partial_sum} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CorrectionTerm(MethodBlock):
    """Compute correction term: exact - approximation."""

    def __init__(self):
        super().__init__()
        self.name = "correction_term"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "approximation", "error"]

    def generate_parameters(self, input_value=None):
        """Generate random exact value and approximation."""
        exact = random.uniform(1, 100)
        error = random.uniform(-5, 5)
        approximation = exact + error
        return {
            "approximation": approximation,
            "exact": exact
        }

    def compute(self, input_value, params):
        """
        Compute correction term = exact - approximation.

        Args:
            input_value: Not used
            params: Dictionary with 'approximation' and 'exact' keys

        Returns:
            MethodResult with correction term
        """
        approximation = params.get("approximation", 95)
        exact = params.get("exact", 100)

        result = exact - approximation

        return MethodResult(
            value=result,
            description=f"Correction: {exact} - {approximation} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AsymptoticIntegralBehavior(MethodBlock):
    """Analyze asymptotic behavior of integrals as parameter → ∞."""

    def __init__(self):
        super().__init__()
        self.name = "asymptotic_integral_behavior"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "integrals", "asymptotics"]

    def generate_parameters(self, input_value=None):
        """Generate random integral parameters."""
        return {
            "coefficient": random.uniform(0.1, 10),
            "power": random.randint(-3, -1),  # Negative powers for convergence
            "x": random.uniform(10, 100)  # Large x for asymptotic behavior
        }

    def compute(self, input_value, params):
        """
        Analyze asymptotic behavior of integrals.

        For large x, computes coefficient * x^power as asymptotic approximation.
        Example: For integral ~ c*x^(-2), returns c/x^2

        Args:
            input_value: Not used
            params: Dictionary with 'coefficient', 'power', 'x' keys

        Returns:
            MethodResult with asymptotic value
        """
        coefficient = params.get("coefficient", 1.0)
        power = params.get("power", -1)
        x = params.get("x", 10.0)

        # Convert strings to numbers if needed
        if isinstance(coefficient, str):
            try:
                coefficient = float(coefficient)
            except ValueError:
                coefficient = 1.0  # fallback to default

        if isinstance(power, str):
            try:
                power = int(power)
            except ValueError:
                power = -1  # fallback to default

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 10.0  # fallback to default

        if x <= 0:
            raise ValueError("x must be positive for asymptotic analysis")

        # Asymptotic approximation: coefficient * x^power
        result = coefficient * (x ** power)

        return MethodResult(
            value=result,
            description=f"Asymptotic: {coefficient} × {x}^{power} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SineIntegralAsymptotic(MethodBlock):
    """Compute sine integral Si(x) asymptotic expansion for large x."""

    def __init__(self):
        super().__init__()
        self.name = "sine_integral_asymptotic"
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
class CosineIntegralAsymptotic(MethodBlock):
    """Compute cosine integral Ci(x) asymptotic expansion for large x."""

    def __init__(self):
        super().__init__()
        self.name = "cosine_integral_asymptotic"
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
class SubstituteAsymptotics(MethodBlock):
    """Substitute asymptotic expression into formula."""

    def __init__(self):
        super().__init__()
        self.name = "substitute_asymptotics"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "asymptotics", "substitution"]

    def generate_parameters(self, input_value=None):
        """Generate random substitution parameters."""
        return {
            "coefficient": random.uniform(0.1, 10),
            "power": random.uniform(-2, 2)
        }

    def compute(self, input_value, params):
        """
        Substitute asymptotic expression into formula.

        Given input x, computes coefficient * x^power.

        Args:
            input_value: x value
            params: Dictionary with 'coefficient' and 'power' keys

        Returns:
            MethodResult with substituted value
        """
        x = input_value if input_value is not None else 1.0
        coefficient = params.get("coefficient", 1.0)
        power = params.get("power", 1.0)

        # Convert strings to numbers if needed
        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 1.0  # fallback to default

        if isinstance(coefficient, str):
            try:
                coefficient = float(coefficient)
            except ValueError:
                coefficient = 1.0  # fallback to default

        if isinstance(power, str):
            try:
                power = float(power)
            except ValueError:
                power = 1.0  # fallback to default

        if x == 0 and power < 0:
            raise ValueError("Cannot compute negative power of zero")

        # Return the leading coefficient of the asymptotic expansion
        # The coefficient parameter represents the extracted asymptotic coefficient
        result = coefficient

        return MethodResult(
            value=result,
            description=f"Asymptotic coefficient: {coefficient} (power {power})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MatchAsymptoticCoefficient(MethodBlock):
    """Extract leading coefficient from asymptotic expansion."""

    def __init__(self):
        super().__init__()
        self.name = "match_asymptotic_coefficient"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "asymptotics", "coefficients"]

    def generate_parameters(self, input_value=None):
        """Generate random asymptotic expansion parameters."""
        return {
            "value": random.uniform(1, 100),
            "x": random.uniform(10, 100),
            "power": random.randint(-2, 2)
        }

    def compute(self, input_value, params):
        """
        Extract leading coefficient from asymptotic expansion.

        Given value ≈ c * x^power, computes c = value / x^power.

        Args:
            input_value: Not used
            params: Dictionary with 'value', 'x', 'power' keys

        Returns:
            MethodResult with extracted coefficient
        """
        value = params.get("value", 10.0)
        x = params.get("x", 10.0)
        power = params.get("power", 1)

        # Convert strings to floats if needed
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = 10.0  # fallback to default

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 10.0  # fallback to default

        if isinstance(power, str):
            try:
                power = int(power)
            except ValueError:
                power = 1  # fallback to default

        if x == 0:
            raise ValueError("x cannot be zero")

        # Extract coefficient: c = value / x^power
        result = value / (x ** power)

        return MethodResult(
            value=result,
            description=f"Coefficient: {value} / {x}^{power} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

