"""
Convergence and bounds methods (continued).
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class InfimumOfConvergenceRegion(MethodBlock):
    """Compute infimum of convergence region."""

    def __init__(self):
        super().__init__()
        self.name = "infimum_of_convergence_region"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "convergence", "infimum"]

    def generate_parameters(self, input_value=None):
        """Generate convergence region bounds."""
        length = random.randint(5, 15)
        return {"region": sorted([random.uniform(-5, 5) for _ in range(length)])}

    def compute(self, input_value, params):
        """
        Compute infimum of convergence region.

        Args:
            input_value: Region points (if provided)
            params: Dictionary with 'region' key

        Returns:
            MethodResult with infimum
        """
        region = input_value if input_value is not None else params.get("region", [1, 2, 3, 4])

        if not region:
            raise ValueError("Empty region")

        result = min(region)

        return MethodResult(
            value=result,
            description=f"inf(convergence region) = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SupremumOfBallIntegrals(MethodBlock):
    """Compute supremum of integrals over balls."""

    def __init__(self):
        super().__init__()
        self.name = "supremum_of_ball_integrals"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "integrals", "supremum"]

    def generate_parameters(self, input_value=None):
        """Generate ball integral values."""
        num_balls = random.randint(5, 15)
        return {"integrals": [random.uniform(0, 10) for _ in range(num_balls)]}

    def compute(self, input_value, params):
        """
        Compute supremum of integrals over different balls.

        Args:
            input_value: Integral values (if provided)
            params: Dictionary with 'integrals' key

        Returns:
            MethodResult with supremum
        """
        integrals = input_value if input_value is not None else params.get("integrals", [1, 2, 3])

        if not integrals:
            raise ValueError("Empty integrals list")

        result = max(integrals)

        return MethodResult(
            value=result,
            description=f"sup{{∫_B f}} over {len(integrals)} balls = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FunctionSupremum(MethodBlock):
    """Compute supremum of function over domain."""

    def __init__(self):
        super().__init__()
        self.name = "function_supremum"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "supremum", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate function values over domain."""
        num_points = random.randint(10, 30)
        return {"values": [random.uniform(-10, 10) for _ in range(num_points)]}

    def compute(self, input_value, params):
        """
        Compute supremum of function f over domain D.

        Args:
            input_value: Function values (if provided)
            params: Dictionary with 'values' key

        Returns:
            MethodResult with supremum
        """
        values = input_value if input_value is not None else params.get("values", [1, 2, 3])

        if not values:
            raise ValueError("Empty values list")

        result = max(values)

        return MethodResult(
            value=result,
            description=f"sup_{{x∈D}} f(x) = {result:.6f} (from {len(values)} sample points)",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class WeakStarNullConvergence(MethodBlock):
    """Check weak* convergence to null."""

    def __init__(self):
        super().__init__()
        self.name = "weak_star_null_convergence"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "functional_analysis", "weak_convergence"]

    def generate_parameters(self, input_value=None):
        """Generate sequence for weak* convergence test."""
        length = random.randint(10, 30)
        # Generate sequence converging to 0 in weak* sense
        return {"sequence": [random.uniform(-1, 1) / (i + 1) for i in range(length)]}

    def compute(self, input_value, params):
        """
        Check if sequence converges to 0 in weak* topology.

        Returns 1 if converges to 0, else 0.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' key

        Returns:
            MethodResult with convergence indicator
        """
        sequence = input_value if input_value is not None else params.get("sequence", [1, 0.5, 0.25])

        if not sequence:
            raise ValueError("Empty sequence")

        # Check if later terms are small (converging to 0)
        tail_size = min(5, len(sequence))
        tail_max = max(abs(x) for x in sequence[-tail_size:])

        converges = tail_max < 0.1
        result = 1 if converges else 0

        return MethodResult(
            value=result,
            description=f"Weak* convergence to 0: {'yes' if converges else 'no'} (tail max = {tail_max:.6f})",
            params=params,
            metadata={"techniques_used": [self.name], "tail_max": tail_max}
        )

    def can_invert(self):
        return False

