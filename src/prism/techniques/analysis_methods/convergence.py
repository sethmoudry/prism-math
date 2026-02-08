"""
Convergence analysis methods.
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class SupremumLimit(MethodBlock):
    """Compute supremum (max) of a finite sequence."""

    def __init__(self):
        super().__init__()
        self.name = "supremum_limit"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "supremum"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence."""
        length = random.randint(3, 10)
        return {"sequence": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Compute supremum (max) of finite sequence.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with supremum value
        """
        sequence = input_value or params.get("sequence", [1, 2, 3, 4, 5])

        # Handle int/float input (from chained methods)
        if isinstance(sequence, (int, float)):
            result = sequence  # Single value is its own supremum
        else:
            # Handle string input (JSON-encoded list)
            if isinstance(sequence, str):
                try:
                    import json
                    sequence = json.loads(sequence)
                except (json.JSONDecodeError, ValueError):
                    # Try converting to float as fallback
                    try:
                        result = float(sequence)
                    except ValueError:
                        raise ValueError(f"Cannot parse sequence from string: {sequence}")
                    return MethodResult(
                        value=result,
                        description=f"Supremum of value {sequence} = {result:.6f}",
                        params=params,
                        metadata={"techniques_used": [self.name]}
                    )

            # Ensure sequence is a list
            if not isinstance(sequence, list):
                sequence = [sequence]

            if not sequence:
                raise ValueError("Empty sequence")

            result = max(sequence)

        return MethodResult(
            value=result,
            description=f"Supremum of {sequence} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class DominatedConvergenceLimit(MethodBlock):
    """Compute limit using dominated convergence (last value for finite sequence)."""

    def __init__(self):
        super().__init__()
        self.name = "dominated_convergence_limit"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "convergence"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence of values."""
        length = random.randint(3, 10)
        return {"values": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Compute limit using dominated convergence.

        For finite sequences, returns the last value as the limit.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'values' key

        Returns:
            MethodResult with limit value
        """
        values = input_value or params.get("values", [1, 2, 3, 4, 5])

        if not values:
            result = 0
            description = "Empty sequence → 0"
        else:
            result = values[-1]
            description = f"Limit of sequence {values} = {result:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class PointwiseLimitAnalysis(MethodBlock):
    """Analyze pointwise limit from sequence of function values."""

    def __init__(self):
        super().__init__()
        self.name = "pointwise_limit_analysis"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "pointwise"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence of function values."""
        length = random.randint(3, 10)
        return {"fn_values": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Analyze pointwise limit from sequence of function values.

        Returns the converged value (last value in finite sequence).

        Args:
            input_value: Function values (if provided)
            params: Dictionary with 'fn_values' key

        Returns:
            MethodResult with pointwise limit
        """
        fn_values = input_value or params.get("fn_values", [1, 2, 3, 4, 5])

        if not fn_values:
            raise ValueError("Empty function values")

        # For finite sequence, the pointwise limit is the last value
        result = fn_values[-1]

        # Format result appropriately based on type
        if isinstance(result, (int, float)):
            result_str = f"{result:.6f}" if isinstance(result, float) else str(result)
        else:
            result_str = str(result)

        return MethodResult(
            value=result,
            description=f"Pointwise limit of {fn_values} = {result_str}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IntegrandDominatedBound(MethodBlock):
    """Compute integrand dominated bound: max_val * interval_length."""

    def __init__(self):
        super().__init__()
        self.name = "integrand_dominated_bound"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "integrals", "bounds"]

    def generate_parameters(self, input_value=None):
        """Generate random max value and interval length."""
        return {
            "max_val": random.uniform(1, 10),
            "interval_length": random.uniform(1, 10)
        }

    def compute(self, input_value, params):
        """
        Compute integrand dominated bound.

        Returns max_val * interval_length (simple domination bound).

        Args:
            input_value: Not used
            params: Dictionary with 'max_val' and 'interval_length' keys

        Returns:
            MethodResult with bound value
        """
        max_val = params.get("max_val", 5)
        interval_length = params.get("interval_length", 10)

        result = max_val * interval_length

        return MethodResult(
            value=result,
            description=f"Dominated bound: {max_val} × {interval_length} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConcludeNormLimit(MethodBlock):
    """Compute limit of norm sequence."""

    def __init__(self):
        super().__init__()
        self.name = "conclude_norm_limit"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "limits", "norms"]

    def generate_parameters(self, input_value=None):
        """Generate random norm sequence."""
        length = random.randint(3, 10)
        return {"norm_sequence": [random.uniform(0, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Compute limit of norm sequence.

        Returns the last value of the norm sequence.

        Args:
            input_value: Norm sequence (if provided)
            params: Dictionary with 'norm_sequence' key

        Returns:
            MethodResult with norm limit
        """
        norm_sequence = input_value or params.get("norm_sequence", [5, 4, 3, 2, 1])

        if not norm_sequence:
            result = 0
            description = "Empty norm sequence → 0"
        else:
            result = norm_sequence[-1]
            description = f"Norm limit of {norm_sequence} = {result:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MutualSingularityLimit(MethodBlock):
    """Analyze limits with mutual singularities."""

    def __init__(self):
        super().__init__()
        self.name = "mutual_singularity_limit"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "singularities"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for functions with singularities."""
        return {
            "numerator_power": random.randint(1, 3),
            "denominator_power": random.randint(1, 3),
            "point": random.uniform(0.01, 0.1)  # Near singularity at 0
        }

    def compute(self, input_value, params):
        """
        Analyze limit near mutual singularity: lim_{x→0} x^p / x^q.

        Args:
            input_value: Not used
            params: Dictionary with 'numerator_power', 'denominator_power', 'point' keys

        Returns:
            MethodResult with limit near singularity
        """
        p = params.get("numerator_power", 2)
        q = params.get("denominator_power", 1)
        x = params.get("point", 0.01)

        # Compute at point near 0
        if p >= q:
            result = x ** (p - q)
            if p == q:
                limit_desc = "→ 1"
            else:
                limit_desc = "→ 0"
        else:
            result = x ** (p - q)
            limit_desc = "→ ∞"

        return MethodResult(
            value=result,
            description=f"lim_{{x→0}} x^{p}/x^{q} = x^{p-q} {limit_desc} (at x={x:.4f}: {result:.6f})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IntegralConvergenceTest(MethodBlock):
    """Test if improper integral ∫_1^∞ 1/x^p dx converges (p > 1)."""

    def __init__(self):
        super().__init__()
        self.name = "integral_convergence_test"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "integrals", "convergence"]

    def generate_parameters(self, input_value=None):
        """Generate random p for convergence test."""
        return {"p": random.uniform(0.5, 3.0)}

    def compute(self, input_value, params):
        """
        Test convergence of ∫_1^∞ 1/x^p dx.

        Converges iff p > 1. When p > 1, integral = 1/(p-1).
        When p ≤ 1, integral diverges (we return 0 as indicator).

        Args:
            input_value: p value (if provided)
            params: Dictionary with 'p' key

        Returns:
            MethodResult with integral value if convergent, else 0
        """
        p = input_value if input_value is not None else params.get("p", 1.5)

        if isinstance(p, str):
            try:
                p = float(p)
            except ValueError:
                p = 1.5

        if p <= 0:
            raise ValueError("p must be positive")

        # Convergence test
        if p > 1:
            # Converges: ∫_1^∞ 1/x^p dx = 1/(p-1)
            result = 1.0 / (p - 1)
            converges = True
            description = f"∫_1^∞ 1/x^{p:.3f} dx = 1/({p:.3f}-1) = {result:.6f} (converges)"
        else:
            # Diverges
            result = 0
            converges = False
            description = f"∫_1^∞ 1/x^{p:.3f} dx diverges (p ≤ 1)"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "techniques_used": [self.name],
                "converges": converges,
                "p": p
            }
        )

    def can_invert(self):
        return False


@register_technique
class SupremumNormBound(MethodBlock):
    """Compute supremum norm bound ||f||_∞ = sup|f(x)| over domain."""

    def __init__(self):
        super().__init__()
        self.name = "supremum_norm_bound"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "norms", "supremum", "bounds"]

    def generate_parameters(self, input_value=None):
        """Generate random function values."""
        n = random.randint(5, 20)
        return {"values": [random.uniform(-10, 10) for _ in range(n)]}

    def compute(self, input_value, params):
        """
        Compute supremum norm ||f||_∞ = sup|f(x)| over domain.

        For discrete samples, returns max(|f(x_i)|).

        Args:
            input_value: List of function values (if provided)
            params: Dictionary with 'values' key

        Returns:
            MethodResult with supremum norm
        """
        values = input_value if input_value is not None else params.get("values", [1, -2, 3, -4, 5])

        if not values:
            raise ValueError("Empty values list")

        # Convert to floats if needed
        if isinstance(values, list):
            values = [float(v) if isinstance(v, str) else v for v in values]

        # Supremum norm: max of absolute values
        result = max(abs(v) for v in values)

        # Format values for description (limit to 5 for readability)
        if len(values) <= 5:
            values_str = ", ".join(f"{v:.2f}" for v in values)
        else:
            values_str = ", ".join(f"{v:.2f}" for v in values[:5]) + ", ..."

        return MethodResult(
            value=result,
            description=f"||f||_∞ = sup|f| over [{values_str}] = {result:.6f}",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "num_samples": len(values)
            }
        )

    def can_invert(self):
        return False

