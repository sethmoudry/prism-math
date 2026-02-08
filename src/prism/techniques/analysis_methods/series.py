"""
Series and summation methods.
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class CombineTerms(MethodBlock):
    """Sum a list of terms."""

    def __init__(self):
        super().__init__()
        self.name = "combine_terms"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["analysis", "arithmetic", "summation"]

    def generate_parameters(self, input_value=None):
        """Generate random list of terms."""
        length = random.randint(2, 10)
        return {"terms": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Sum a list of terms.

        Args:
            input_value: List of terms (if provided)
            params: Dictionary with 'terms' key

        Returns:
            MethodResult with sum of terms
        """
        terms = input_value or params.get("terms", [1.0, 2.0, 3.0])

        result = sum(terms)

        return MethodResult(
            value=result,
            description=f"Sum of terms {terms} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class HarmonicSumMinimal(MethodBlock):
    """Compute minimal harmonic sum H_n = 1 + 1/2 + ... + 1/n."""

    def __init__(self):
        super().__init__()
        self.name = "harmonic_sum_minimal"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "series", "harmonic"]

    def generate_parameters(self, input_value=None):
        """Generate random n for harmonic sum."""
        return {"n": random.randint(1, 100)}

    def compute(self, input_value, params):
        """
        Compute harmonic sum H_n = 1 + 1/2 + 1/3 + ... + 1/n.

        Args:
            input_value: n value (if provided)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with H_n value
        """
        n = input_value if input_value is not None else params.get("n", 10)

        # Handle string parameters
        if isinstance(n, str):
            try:
                n = int(n)
            except ValueError:
                n = 10

        if n < 1:
            raise ValueError("n must be at least 1")

        # Compute harmonic sum
        result = sum(1.0 / k for k in range(1, n + 1))

        return MethodResult(
            value=result,
            description=f"H_{n} = 1 + 1/2 + ... + 1/{n} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AlternatingHarmonicSeriesSum(MethodBlock):
    """Sum of alternating harmonic series."""

    def __init__(self):
        super().__init__()
        self.name = "alternating_harmonic_series_sum"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "series", "harmonic", "alternating"]

    def generate_parameters(self, input_value=None):
        """Generate random n for alternating harmonic sum."""
        return {"n": random.randint(1, 100)}

    def compute(self, input_value, params):
        """
        Compute alternating harmonic sum: 1 - 1/2 + 1/3 - 1/4 + ... ± 1/n.

        Args:
            input_value: n value (if provided)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with alternating H_n value
        """
        n = input_value if input_value is not None else params.get("n", 10)

        # Handle string parameters
        if isinstance(n, str):
            try:
                n = int(n)
            except ValueError:
                n = 10

        if n < 1:
            raise ValueError("n must be at least 1")

        # Compute alternating harmonic sum
        result = sum((-1) ** (k + 1) / k for k in range(1, n + 1))

        # For reference: as n→∞, this converges to ln(2) ≈ 0.693147
        return MethodResult(
            value=result,
            description=f"Alternating H_{n} = 1 - 1/2 + 1/3 - ... ± 1/{n} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name], "converges_to": math.log(2)}
        )

    def can_invert(self):
        return False


@register_technique
class PowerSeriesExpansionLog(MethodBlock):
    """Taylor series expansion of log(1+x)."""

    def __init__(self):
        super().__init__()
        self.name = "power_series_expansion_log"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "series", "logarithms", "taylor"]

    def generate_parameters(self, input_value=None):
        """Generate random parameters for log(1+x) expansion."""
        return {
            "x": random.uniform(-0.9, 0.9),  # |x| < 1 for convergence
            "terms": random.randint(5, 20)
        }

    def compute(self, input_value, params):
        """
        Compute Taylor series expansion of log(1+x).

        log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ... = Σ(-1)^(n+1) * x^n / n

        Args:
            input_value: x value (if provided)
            params: Dictionary with 'x' and 'terms' keys

        Returns:
            MethodResult with series approximation
        """
        x = params.get("x", 0.5)
        if isinstance(input_value, (int, float)) and abs(input_value) < 1:
            x = input_value
        terms = params.get("terms", 10)

        # Handle string parameters
        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 0.5

        if isinstance(terms, str):
            try:
                terms = int(terms)
            except ValueError:
                terms = 10

        if abs(x) >= 1:
            raise ValueError("x must satisfy |x| < 1 for convergence")
        if terms < 1:
            raise ValueError("Number of terms must be at least 1")

        # Compute Taylor series: Σ(-1)^(n+1) * x^n / n for n=1 to terms
        result = sum((-1) ** (n + 1) * (x ** n) / n for n in range(1, terms + 1))

        # Compare with exact value
        exact = math.log(1 + x)

        return MethodResult(
            value=result,
            description=f"log(1+{x:.3f}) ≈ Σ(n=1 to {terms}) (-1)^(n+1) * {x:.3f}^n / n = {result:.6f} (exact: {exact:.6f})",
            params=params,
            metadata={"techniques_used": [self.name], "exact_value": exact, "error": abs(result - exact)}
        )

    def validate_params(self, params, prev_value=None):
        """Validate that |x| < 1 for convergence."""
        x = params.get("x", prev_value) if prev_value is not None else params.get("x")
        if x is None:
            return True  # Will use default
        try:
            return abs(float(x)) < 1
        except (TypeError, ValueError):
            return False

    def can_invert(self):
        return False


@register_technique
class HarmonicSeriesDivergence(MethodBlock):
    """Prove/check harmonic series divergence using approximation H_n ≈ ln(n) + γ."""

    def __init__(self):
        super().__init__()
        self.name = "harmonic_series_divergence"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "series", "harmonic", "divergence"]

    def generate_parameters(self, input_value=None):
        """Generate random n for harmonic series analysis."""
        return {"n": random.randint(10, 1000)}

    def compute(self, input_value, params):
        """
        Check harmonic series divergence using approximation.

        H_n ≈ ln(n) + γ where γ ≈ 0.5772156649 (Euler-Mascheroni constant)
        As n→∞, H_n diverges. We return the approximation value.

        Args:
            input_value: n value (if provided)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with H_n approximation showing divergence
        """
        n = input_value if input_value is not None else params.get("n", 100)

        if isinstance(n, str):
            try:
                n = int(n)
            except ValueError:
                n = 100

        if n < 1:
            raise ValueError("n must be at least 1")

        # Euler-Mascheroni constant
        gamma = 0.5772156649

        # Approximation: H_n ≈ ln(n) + γ
        result = math.log(n) + gamma

        # For reference, compute exact sum for small n
        if n <= 1000:
            exact = sum(1.0 / k for k in range(1, n + 1))
            error = abs(result - exact)
        else:
            exact = None
            error = None

        description = f"H_{n} ≈ ln({n}) + γ = {result:.6f}"
        if exact is not None:
            description += f" (exact: {exact:.6f}, error: {error:.6f})"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "techniques_used": [self.name],
                "diverges": True,
                "exact_value": exact,
                "error": error
            }
        )

    def can_invert(self):
        return False


@register_technique
class LogarithmExpansion(MethodBlock):
    """Taylor expansion of log(1+x) around x=0."""

    def __init__(self):
        super().__init__()
        self.name = "logarithm_expansion"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "logarithms", "taylor", "series"]

    def generate_parameters(self, input_value=None):
        """Generate random x for log(1+x) expansion."""
        return {
            "x": random.uniform(-0.9, 0.9),
            "n_terms": random.randint(5, 15)
        }

    def compute(self, input_value, params):
        """
        Taylor expansion of log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...

        Valid for |x| < 1. Returns partial sum up to n_terms.

        Args:
            input_value: x value (if provided)
            params: Dictionary with 'x' and 'n_terms' keys

        Returns:
            MethodResult with Taylor series approximation
        """
        x = params.get("x", 0.5)
        if isinstance(input_value, (int, float)) and abs(input_value) < 1:
            x = input_value
        n_terms = params.get("n_terms", 10)

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 0.5

        if isinstance(n_terms, str):
            try:
                n_terms = int(n_terms)
            except ValueError:
                n_terms = 10

        if abs(x) >= 1:
            raise ValueError("x must satisfy |x| < 1 for convergence")

        # Taylor series: log(1+x) = Σ(-1)^(n+1) * x^n / n for n=1,2,3,...
        result = sum((-1) ** (n + 1) * (x ** n) / n for n in range(1, n_terms + 1))

        exact = math.log(1 + x)
        error = abs(result - exact)

        return MethodResult(
            value=result,
            description=f"log(1+{x:.4f}) ≈ Σ(n=1 to {n_terms}) = {result:.6f} (exact: {exact:.6f})",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "exact": exact,
                "error": error
            }
        )

    def validate_params(self, params, prev_value=None):
        """Validate that |x| < 1 for convergence."""
        x = params.get("x", prev_value) if prev_value is not None else params.get("x")
        if x is None:
            return True  # Will use default
        try:
            return abs(float(x)) < 1
        except (TypeError, ValueError):
            return False

    def can_invert(self):
        return False

