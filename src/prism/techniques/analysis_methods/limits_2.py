"""
Limit computation methods (continued).
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class LimitAsNGoesToInfinity(MethodBlock):
    """Compute limit as n→∞ for a sequence."""

    def __init__(self):
        super().__init__()
        self.name = "limit_as_n_goes_to_infinity"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "sequences", "limits"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence."""
        length = random.randint(10, 50)
        # Generate convergent sequence
        limit_val = random.uniform(1, 10)
        return {
            "sequence": [limit_val + random.uniform(-1, 1) / (i + 1) for i in range(length)]
        }

    def compute(self, input_value, params):
        """
        Compute limit as n→∞ for a sequence.

        For finite sequences, returns the last value as approximation.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with limit value
        """
        sequence = input_value if input_value is not None else params.get("sequence", [1, 2, 3])

        if not sequence:
            raise ValueError("Empty sequence")

        # For finite sequence, take last value as limit approximation
        result = sequence[-1]

        return MethodResult(
            value=result,
            description=f"lim_{{n→∞}} aₙ ≈ {result:.6f} (from sequence of length {len(sequence)})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class NthRootLimitBounds(MethodBlock):
    """Compute bounds for lim ⁿ√aₙ."""

    def __init__(self):
        super().__init__()
        self.name = "nth_root_limit_bounds"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "root_test"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence for nth root test."""
        length = random.randint(5, 20)
        # Generate positive sequence
        return {"sequence": [random.uniform(0.1, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Compute bounds for lim ⁿ√aₙ.

        Uses: liminf ⁿ√aₙ ≤ limsup ⁿ√aₙ
        Returns the geometric mean as approximation.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with nth root limit bound
        """
        sequence = input_value if input_value is not None else params.get("sequence", [1, 2, 3])

        if not sequence:
            raise ValueError("Empty sequence")

        # Compute nth roots
        n_vals = list(range(1, len(sequence) + 1))
        nth_roots = [abs(a) ** (1.0 / n) for n, a in zip(n_vals, sequence) if a != 0]

        if not nth_roots:
            result = 0
        else:
            # Return limsup approximation (max of later values)
            result = max(nth_roots[-min(5, len(nth_roots)):])

        return MethodResult(
            value=result,
            description=f"lim sup ⁿ√|aₙ| ≈ {result:.6f} (root test bound)",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class NestedSqrtLimitSequence(MethodBlock):
    """Compute limit of nested square root sequence."""

    def __init__(self):
        super().__init__()
        self.name = "nested_sqrt_limit_sequence"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "nested_radicals"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for nested sqrt."""
        return {
            "a": random.uniform(1, 10),
            "iterations": random.randint(5, 15)
        }

    def compute(self, input_value, params):
        """
        Compute nested square root: √(a + √(a + √(a + ...))).

        This converges to the positive root of x² - x - a = 0,
        which is (1 + √(1 + 4a)) / 2.

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'iterations' keys

        Returns:
            MethodResult with nested sqrt limit
        """
        a = params.get("a", 2.0)
        iterations = params.get("iterations", 10)

        # Iteratively compute nested sqrt
        x = 0
        for _ in range(iterations):
            x = math.sqrt(a + x)

        # Theoretical limit
        limit_exact = (1 + math.sqrt(1 + 4*a)) / 2

        result = x

        return MethodResult(
            value=result,
            description=f"Nested √(a + √(a + ...)) with a={a:.2f} → {result:.6f} (exact: {limit_exact:.6f})",
            params=params,
            metadata={"techniques_used": [self.name], "exact_limit": limit_exact}
        )

    def can_invert(self):
        return False


@register_technique
class LimitPowerSeries(MethodBlock):
    """Compute limit of power series at a point."""

    def __init__(self):
        super().__init__()
        self.name = "limit_power_series"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "power_series"]

    def generate_parameters(self, input_value=None):
        """Generate power series parameters."""
        n_terms = random.randint(5, 15)
        return {
            "coefficients": [random.uniform(-2, 2) for _ in range(n_terms)],
            "point": random.uniform(-0.5, 0.5)  # Within radius of convergence
        }

    def compute(self, input_value, params):
        """
        Compute power series limit: Σ aₙ·xⁿ at point x.

        Args:
            input_value: Not used
            params: Dictionary with 'coefficients', 'point' keys

        Returns:
            MethodResult with power series value
        """
        coefficients = params.get("coefficients", [1, 1, 1])
        point = params.get("point", 0.5)

        # Compute power series
        result = sum(coeff * (point ** n) for n, coeff in enumerate(coefficients))

        return MethodResult(
            value=result,
            description=f"Power series Σaₙx^n at x={point:.3f} with {len(coefficients)} terms = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class RatioAnalysisLimit(MethodBlock):
    """Analyze limit of a ratio (ratio test for series convergence)."""

    def __init__(self):
        super().__init__()
        self.name = "ratio_analysis_limit"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "limits", "ratio_test"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence for ratio test."""
        length = random.randint(5, 20)
        # Generate sequence with geometric-like growth
        ratio = random.uniform(0.1, 2.0)
        return {"sequence": [random.uniform(1, 5) * (ratio ** n) for n in range(length)]}

    def compute(self, input_value, params):
        """
        Analyze limit using ratio test: lim aₙ₊₁/aₙ.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with ratio limit
        """
        sequence = input_value if input_value is not None else params.get("sequence", [1, 2, 4, 8])

        if len(sequence) < 2:
            raise ValueError("Sequence too short for ratio test")

        # Compute ratios
        ratios = []
        for i in range(len(sequence) - 1):
            if abs(sequence[i]) > 1e-10:
                ratios.append(abs(sequence[i+1] / sequence[i]))

        if not ratios:
            result = 0
        else:
            # Take average of last few ratios as limit approximation
            result = sum(ratios[-min(5, len(ratios)):]) / min(5, len(ratios))

        if result < 1:
            convergence = "converges"
        elif result > 1:
            convergence = "diverges"
        else:
            convergence = "inconclusive"

        return MethodResult(
            value=result,
            description=f"Ratio test: lim |aₙ₊₁/aₙ| ≈ {result:.6f} → series {convergence}",
            params=params,
            metadata={"techniques_used": [self.name], "convergence": convergence}
        )

    def can_invert(self):
        return False


@register_technique
class SequenceExtraction(MethodBlock):
    """Extract subsequence by indices."""

    def __init__(self):
        super().__init__()
        self.name = "sequence_extraction"
        self.input_type = "list"
        self.output_type = "list"
        self.difficulty = 2
        self.tags = ["analysis", "sequences", "subsequence"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence and indices."""
        length = random.randint(10, 20)
        num_indices = random.randint(3, min(8, length))
        indices = sorted(random.sample(range(length), num_indices))
        return {
            "sequence": [random.uniform(-10, 10) for _ in range(length)],
            "indices": indices
        }

    def compute(self, input_value, params):
        """
        Extract subsequence by indices.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence', 'indices' keys

        Returns:
            MethodResult with extracted subsequence
        """
        sequence = input_value if input_value is not None else params.get("sequence", list(range(10)))
        indices = params.get("indices", [0, 2, 4])

        if not sequence:
            raise ValueError("Empty sequence")

        # Extract subsequence
        result = [sequence[i] for i in indices if 0 <= i < len(sequence)]

        # For numeric output, return sum or last element
        numeric_result = sum(result) if result else 0

        return MethodResult(
            value=numeric_result,
            description=f"Extract subsequence at indices {indices[:5]}{'...' if len(indices) > 5 else ''} → sum = {numeric_result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name], "subsequence": result[:10]}
        )

    def can_invert(self):
        return False


@register_technique
class DominantTermSelection(MethodBlock):
    """Select dominant term from a list of terms (largest magnitude)."""

    def __init__(self):
        super().__init__()
        self.name = "dominant_term_selection"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "asymptotics", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate random list of terms."""
        n = random.randint(3, 10)
        return {"terms": [random.uniform(-100, 100) for _ in range(n)]}

    def compute(self, input_value, params):
        """
        Select dominant term (largest absolute value).

        For asymptotic analysis, the dominant term is the one with largest magnitude.

        Args:
            input_value: List of terms (if provided)
            params: Dictionary with 'terms' key

        Returns:
            MethodResult with dominant term value
        """
        terms = input_value or params.get("terms", [1.0, -5.0, 3.0, 2.0])

        # Handle string parameters
        if isinstance(terms, list):
            terms = [float(t) if isinstance(t, str) else t for t in terms]

        if not terms:
            raise ValueError("Empty terms list")

        # Find term with largest absolute value
        dominant = max(terms, key=abs)

        # For result, return the absolute value (integer) to fit answer range
        result = int(abs(dominant))

        return MethodResult(
            value=result,
            description=f"Dominant term from {terms}: {dominant} (magnitude={abs(dominant):.3f})",
            params=params,
            metadata={"techniques_used": [self.name], "dominant_value": dominant}
        )

    def can_invert(self):
        return False

