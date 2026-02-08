"""
Validation and checking primitives.
"""

import math
import random
from typing import List, Tuple, Union, Optional
from sympy import (
    symbols, simplify, expand, sin, cos, tan, exp, log, sqrt, sympify
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class IsOpenSet(MethodBlock):
    """Check if a mathematical set is open (simplified: always returns 1 for basic sets)."""

    def __init__(self):
        super().__init__()
        self.name = "is_open_set"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["topology", "sets"]

    def generate_parameters(self, input_value=None):
        """Generate random set description."""
        return {
            "set_type": random.choice(["interval", "ball", "union", "intersection"]),
            "radius": random.uniform(0.1, 10)
        }

    def compute(self, input_value, params):
        """
        Check if mathematical set is open.

        Simplified implementation: open intervals (a,b), open balls B(x,r) are open.
        Returns 1 for open sets, 0 otherwise.

        Args:
            input_value: Not used
            params: Dictionary with 'set_type' and 'radius' keys

        Returns:
            MethodResult with 1 (open) or 0 (not open)
        """
        set_type = params.get("set_type", "interval")
        radius = params.get("radius", 1.0)

        # Simplified logic: open intervals and balls are open
        is_open = set_type in ["interval", "ball", "union"]
        result = 1 if is_open else 0

        return MethodResult(
            value=result,
            description=f"Is {set_type} set open? {is_open}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IsConnectedSpace(MethodBlock):
    """Check if a topological space is connected."""

    def __init__(self):
        super().__init__()
        self.name = "is_connected_space"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["topology", "connectivity"]

    def generate_parameters(self, input_value=None):
        """Generate random space parameters."""
        return {
            "space_type": random.choice(["interval", "union_intervals", "single_point", "disconnected"]),
            "num_components": random.randint(1, 3)
        }

    def compute(self, input_value, params):
        """
        Check if topological space is connected.

        A space is connected if it cannot be written as a union of two disjoint open sets.
        Simplified: intervals and single points are connected, unions may not be.

        Args:
            input_value: Not used
            params: Dictionary with 'space_type' and 'num_components' keys

        Returns:
            MethodResult with 1 (connected) or 0 (disconnected)
        """
        space_type = params.get("space_type", "interval")
        num_components = params.get("num_components", 1)

        # Simplified logic
        if space_type == "disconnected":
            is_connected = False
        elif space_type == "union_intervals" and num_components > 1:
            is_connected = False
        else:
            is_connected = True

        result = 1 if is_connected else 0

        return MethodResult(
            value=result,
            description=f"Is {space_type} space connected? {is_connected}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CandidateTest(MethodBlock):
    """Test if a candidate value satisfies a given condition."""

    def __init__(self):
        super().__init__()
        self.name = "candidate_test"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["verification", "testing"]

    def generate_parameters(self, input_value=None):
        """Generate random candidate and condition."""
        candidate = random.randint(-100, 100)
        threshold = random.randint(-50, 50)
        return {
            "candidate": candidate,
            "condition_type": random.choice(["greater", "less", "equal", "divisible"]),
            "threshold": threshold
        }

    def compute(self, input_value, params):
        """
        Test if candidate satisfies condition.

        Conditions:
        - "greater": candidate > threshold
        - "less": candidate < threshold
        - "equal": candidate == threshold
        - "divisible": candidate % threshold == 0

        Args:
            input_value: Not used
            params: Dictionary with 'candidate', 'condition_type', 'threshold' keys

        Returns:
            MethodResult with 1 (satisfies) or 0 (doesn't satisfy)
        """
        candidate = params.get("candidate", 10)
        condition_type = params.get("condition_type", "greater")
        threshold = params.get("threshold", 5)

        # Apply condition
        if condition_type == "greater":
            satisfies = candidate > threshold
            desc = f"{candidate} > {threshold}"
        elif condition_type == "less":
            satisfies = candidate < threshold
            desc = f"{candidate} < {threshold}"
        elif condition_type == "equal":
            satisfies = candidate == threshold
            desc = f"{candidate} == {threshold}"
        elif condition_type == "divisible":
            if threshold == 0:
                satisfies = False
            else:
                satisfies = (candidate % threshold == 0)
            desc = f"{candidate} divisible by {threshold}"
        else:
            satisfies = False
            desc = f"unknown condition"

        result = 1 if satisfies else 0

        return MethodResult(
            value=result,
            description=f"Candidate test: {desc}? {satisfies}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CheckConvergenceInProbability(MethodBlock):
    """Check convergence in probability (simplified: check if sequence stabilizes)."""

    def __init__(self):
        super().__init__()
        self.name = "check_convergence_in_probability"
        self.input_type = "list"
        self.output_type = "boolean"
        self.difficulty = 3
        self.tags = ["probability", "convergence", "analysis"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence with convergence behavior."""
        n = random.randint(5, 20)
        # Generate converging sequence with noise
        target = random.uniform(-10, 10)
        sequence = [target + random.uniform(-1, 1) / (i + 1) for i in range(n)]
        return {"sequence": sequence, "tolerance": 0.1}

    def compute(self, input_value, params):
        """
        Check convergence in probability.

        Simplified: check if last few values are within tolerance.

        Args:
            input_value: Sequence (if provided)
            params: Dictionary with 'sequence' and 'tolerance' keys

        Returns:
            MethodResult with 1 (converges) or 0 (doesn't converge)
        """
        sequence = input_value or params.get("sequence", [1, 2, 3, 4, 5])
        tolerance = params.get("tolerance", 0.1)

        if len(sequence) < 3:
            # Need at least 3 values to check convergence
            result = 0
        else:
            # Check if last 3 values are within tolerance
            last_three = sequence[-3:]
            mean = sum(last_three) / len(last_three)
            converges = all(abs(x - mean) < tolerance for x in last_three)
            result = 1 if converges else 0

        return MethodResult(
            value=result,
            description=f"Sequence converges in probability? {bool(result)} (tolerance={tolerance:.3f})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CheckIdenticalDistribution(MethodBlock):
    """Check if two distributions are identical (simplified: compare means and variances)."""

    def __init__(self):
        super().__init__()
        self.name = "check_identical_distribution"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["probability", "distributions", "statistics"]

    def generate_parameters(self, input_value=None):
        """Generate two distributions (as sample lists)."""
        # 50% chance they're identical
        if random.random() < 0.5:
            # Same distribution
            mean = random.uniform(-10, 10)
            std = random.uniform(1, 5)
            dist1 = [mean + random.gauss(0, std) for _ in range(10)]
            dist2 = [mean + random.gauss(0, std) for _ in range(10)]
        else:
            # Different distributions
            mean1 = random.uniform(-10, 10)
            mean2 = mean1 + random.uniform(5, 10)
            dist1 = [mean1 + random.gauss(0, 2) for _ in range(10)]
            dist2 = [mean2 + random.gauss(0, 2) for _ in range(10)]

        return {"dist1": dist1, "dist2": dist2, "tolerance": 1.0}

    def compute(self, input_value, params):
        """
        Check if two distributions are identical.

        Simplified: compare sample means and variances within tolerance.

        Args:
            input_value: Not used
            params: Dictionary with 'dist1', 'dist2', 'tolerance' keys

        Returns:
            MethodResult with 1 (identical) or 0 (different)
        """
        dist1 = params.get("dist1", [1, 2, 3, 4, 5])
        dist2 = params.get("dist2", [1, 2, 3, 4, 5])
        tolerance = params.get("tolerance", 1.0)

        # Compute statistics
        mean1 = sum(dist1) / len(dist1)
        mean2 = sum(dist2) / len(dist2)

        var1 = sum((x - mean1) ** 2 for x in dist1) / len(dist1)
        var2 = sum((x - mean2) ** 2 for x in dist2) / len(dist2)

        # Check if means and variances are close
        means_close = abs(mean1 - mean2) < tolerance
        vars_close = abs(var1 - var2) < tolerance

        identical = means_close and vars_close
        result = 1 if identical else 0

        return MethodResult(
            value=result,
            description=f"Distributions identical? {identical} (μ1={mean1:.2f}, μ2={mean2:.2f})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CheckIndependenceProduct(MethodBlock):
    """Check independence via product: P(A∩B) = P(A)×P(B)."""

    def __init__(self):
        super().__init__()
        self.name = "check_independence_product"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["probability", "independence"]

    def generate_parameters(self, input_value=None):
        """Generate random probabilities."""
        # 50% chance they're independent
        if random.random() < 0.5:
            # Independent
            p_a = random.uniform(0.1, 0.9)
            p_b = random.uniform(0.1, 0.9)
            p_ab = p_a * p_b
        else:
            # Dependent
            p_a = random.uniform(0.1, 0.9)
            p_b = random.uniform(0.1, 0.9)
            p_ab = random.uniform(0.1, min(p_a, p_b))

        return {"p_a": p_a, "p_b": p_b, "p_ab": p_ab, "tolerance": 0.01}

    def compute(self, input_value, params):
        """
        Check independence via product rule.

        Events A and B are independent if P(A∩B) = P(A)×P(B).

        Args:
            input_value: Not used
            params: Dictionary with 'p_a', 'p_b', 'p_ab', 'tolerance' keys

        Returns:
            MethodResult with 1 (independent) or 0 (dependent)
        """
        p_a = params.get("p_a", 0.5)
        p_b = params.get("p_b", 0.5)
        p_ab = params.get("p_ab", 0.25)
        tolerance = params.get("tolerance", 0.01)

        # Check if P(A∩B) = P(A)×P(B)
        expected_product = p_a * p_b
        independent = abs(p_ab - expected_product) < tolerance
        result = 1 if independent else 0

        return MethodResult(
            value=result,
            description=f"Independent? {independent} (P(A∩B)={p_ab:.3f}, P(A)×P(B)={expected_product:.3f})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ValidateTripleCondition(MethodBlock):
    """Validate that a triple (a, b, c) satisfies a condition."""

    def __init__(self):
        super().__init__()
        self.name = "validate_triple_condition"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["verification", "algebra"]

    def generate_parameters(self, input_value=None):
        """Generate random triple and condition."""
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)
        condition = random.choice(["sum_zero", "product_positive", "triangle", "pythagorean"])
        return {"triple": (a, b, c), "condition": condition}

    def compute(self, input_value, params):
        """
        Validate triple condition.

        Conditions:
        - "sum_zero": a + b + c = 0
        - "product_positive": a×b×c > 0
        - "triangle": |a| + |b| > |c| (and permutations)
        - "pythagorean": a² + b² = c²

        Args:
            input_value: Not used
            params: Dictionary with 'triple' and 'condition' keys

        Returns:
            MethodResult with 1 (satisfies) or 0 (doesn't satisfy)
        """
        triple = params.get("triple", (1, 2, 3))
        condition = params.get("condition", "sum_zero")

        a, b, c = triple

        # Check condition
        if condition == "sum_zero":
            satisfies = (a + b + c == 0)
            desc = f"{a} + {b} + {c} = 0"
        elif condition == "product_positive":
            satisfies = (a * b * c > 0)
            desc = f"{a}×{b}×{c} > 0"
        elif condition == "triangle":
            satisfies = (abs(a) + abs(b) > abs(c)) and (abs(b) + abs(c) > abs(a)) and (abs(c) + abs(a) > abs(b))
            desc = f"triangle inequality for ({a}, {b}, {c})"
        elif condition == "pythagorean":
            satisfies = (a*a + b*b == c*c) or (b*b + c*c == a*a) or (c*c + a*a == b*b)
            desc = f"Pythagorean: {a}² + {b}² = {c}²"
        else:
            satisfies = False
            desc = "unknown condition"

        result = 1 if satisfies else 0

        return MethodResult(
            value=result,
            description=f"Triple {triple} satisfies {desc}? {satisfies}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CheckConstructionExists(MethodBlock):
    """Check if a mathematical construction exists (simplified: always returns 1)."""

    def __init__(self):
        super().__init__()
        self.name = "check_construction_exists"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["logic", "existence", "verification"]

    def generate_parameters(self, input_value=None):
        """Generate random construction parameters."""
        return {
            "construction_type": random.choice(["triangle", "perpendicular", "parallel", "circle"]),
            "constraints": random.randint(1, 3)
        }

    def compute(self, input_value, params):
        """
        Check if construction exists.

        Simplified: most basic constructions exist (returns 1).

        Args:
            input_value: Not used
            params: Dictionary with 'construction_type' and 'constraints' keys

        Returns:
            MethodResult with 1 (exists) or 0 (doesn't exist)
        """
        construction_type = params.get("construction_type", "triangle")
        constraints = params.get("constraints", 1)

        # Simplified logic: most constructions exist
        # Too many constraints might make it impossible
        exists = constraints <= 3
        result = 1 if exists else 0

        return MethodResult(
            value=result,
            description=f"Construction {construction_type} with {constraints} constraints exists? {exists}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

