"""
Sequence primitives.
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
class TriangularNumber(MethodBlock):
    """Compute nth triangular number T(n) = n(n+1)/2."""

    def __init__(self):
        super().__init__()
        self.name = "triangular_number"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["sequences", "number_theory"]

    def validate_params(self, params, prev_value=None):
        """Triangular number requires n >= 0."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 0

    def generate_parameters(self, input_value=None):
        """Generate random index."""
        return {"n": random.randint(1, 100)}

    def compute(self, input_value, params):
        """Compute nth triangular number: T(n) = 1+2+...+n = n(n+1)/2."""
        n = input_value or params.get("n", 5)
        if n < 0:
            raise ValueError("n must be non-negative")
        result = n * (n + 1) // 2
        return MethodResult(
            value=result,
            description=f"T({n}) = {n}({n}+1)/2 = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ArithmeticSequence(MethodBlock):
    """Generate arithmetic sequence: a, a+d, a+2d, ..., a+(n-1)d."""

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_sequence"
        self.input_type = "none"
        self.output_type = "sequence"  # Changed from "list" to connect to sequence consumers
        self.difficulty = 1
        self.tags = ["sequences", "algebra"]

    def generate_parameters(self, input_value=None):
        """Generate random sequence parameters."""
        return {
            "start": random.randint(1, 100),
            "step": random.randint(1, 10),
            "n": random.randint(3, 10)
        }

    def compute(self, input_value, params):
        """Generate arithmetic sequence: a, a+d, a+2d, ..., a+(n-1)d."""
        start = params.get("start", 1)
        step = params.get("step", 2)
        n = params.get("n", 5)
        result = [start + i * step for i in range(n)]
        return MethodResult(
            value=result,
            description=f"Arithmetic sequence: first={start}, step={step}, n={n}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CountIntegerRange(MethodBlock):
    """Count integers in a range [a, b]."""

    def __init__(self):
        super().__init__()
        self.name = "count_integer_range"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "counting"]

    def generate_parameters(self, input_value=None):
        """Generate random range."""
        a = random.randint(1, 100)
        b = random.randint(a, a + 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        """Count integers in range [a, b]."""
        a = params.get("a", 1)
        b = params.get("b", 10)
        result = b - a + 1
        return MethodResult(
            value=result,
            description=f"Number of integers in [{a}, {b}] = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CountSolutionsInInterval(MethodBlock):
    """Count integer solutions in an interval."""

    def __init__(self):
        super().__init__()
        self.name = "count_solutions_in_interval"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["counting", "intervals"]

    def generate_parameters(self, input_value=None):
        """Generate random interval and step."""
        a = random.randint(0, 100)
        b = random.randint(a, a + 100)
        step = random.randint(1, 10)
        return {"a": a, "b": b, "step": step}

    def compute(self, input_value, params):
        """Count solutions of form a + k*step in [a, b]."""
        a = params.get("a", 0)
        b = params.get("b", 100)
        step = params.get("step", 5)

        if step == 0:
            raise ValueError("step cannot be zero")

        # Count integers of form a + k*step in [a, b]
        count = 0
        val = a
        while val <= b:
            count += 1
            val += step

        return MethodResult(
            value=count,
            description=f"Count of values {a}, {a}+{step}, {a}+2·{step}, ... in [{a}, {b}] = {count}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Tribonacci(MethodBlock):
    """Compute nth tribonacci number: T(n) = T(n-1) + T(n-2) + T(n-3)."""

    def __init__(self):
        super().__init__()
        self.name = "tribonacci"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["sequences", "tribonacci"]

    def generate_parameters(self, input_value=None):
        """Generate random index."""
        n = input_value if input_value else random.randint(1, 25)
        return {"n": min(n, 30)}

    def compute(self, input_value, params):
        """Compute nth tribonacci number.

        Standard tribonacci: T(0)=0, T(1)=1, T(2)=1,
        T(n) = T(n-1) + T(n-2) + T(n-3) for n >= 3.
        """
        n = input_value if input_value is not None else params.get("n", 5)
        n = min(abs(n) if n else 5, 30)

        if n == 0:
            result = 0
        elif n == 1:
            result = 1
        elif n == 2:
            result = 1
        else:
            a, b, c = 0, 1, 1
            for _ in range(3, n + 1):
                a, b, c = b, c, a + b + c
            result = c

        return MethodResult(
            value=result,
            description=f"T({n}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ArithmeticProgression(MethodBlock):
    """Compute properties of arithmetic progressions.

    Supports:
    1. AP sum: S = n/2 * (2a + (n-1)d)
    2. AP nth term: a_n = a + (n-1)d
    3. GP sum: S = a(r^n - 1)/(r - 1)
    """

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_progression"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["sequences", "arithmetic_progression"]

    def generate_parameters(self, input_value=None):
        """Generate AP parameters."""
        problem_type = random.choice(["ap_sum", "ap_nth_term", "gp_sum"])

        if problem_type == "ap_sum":
            a = random.randint(1, 50)
            d = random.randint(1, 10)
            n = random.randint(3, 50)
            return {"problem_type": problem_type, "a": a, "d": d, "n": n}
        elif problem_type == "ap_nth_term":
            a = random.randint(1, 50)
            d = random.randint(1, 10)
            n = random.randint(1, 100)
            return {"problem_type": problem_type, "a": a, "d": d, "n": n}
        else:
            a = random.randint(1, 10)
            r = random.randint(2, 5)
            n = random.randint(2, 10)
            return {"problem_type": problem_type, "a": a, "r": r, "n": n}

    def compute(self, input_value, params):
        """Compute arithmetic/geometric progression result."""
        problem_type = params.get("problem_type", "ap_sum")

        if problem_type == "ap_sum":
            a = params.get("a", 1)
            d = params.get("d", 1)
            n = params.get("n", 5)
            result = n * (2 * a + (n - 1) * d) // 2
            desc = f"AP sum: a={a}, d={d}, n={n}, S={result}"
        elif problem_type == "ap_nth_term":
            a = params.get("a", 1)
            d = params.get("d", 1)
            n = params.get("n", 5)
            result = a + (n - 1) * d
            desc = f"AP nth term: a={a}, d={d}, n={n}, a_n={result}"
        else:
            a = params.get("a", 1)
            r = params.get("r", 2)
            n = params.get("n", 5)
            if r == 1:
                result = a * n
            else:
                result = a * (r ** n - 1) // (r - 1)
            desc = f"GP sum: a={a}, r={r}, n={n}, S={result}"

        return MethodResult(
            value=result,
            description=desc,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CollatzExtremal(MethodBlock):
    """Find input in range that maximizes Collatz iterations."""

    def __init__(self):
        super().__init__()
        self.name = "collatz_extremal"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["iterative_process", "collatz", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate search range parameters."""
        max_n = min(input_value, 10000) if input_value else random.randint(100, 1000)
        min_n = max(1, max_n // 10)
        return {"min_n": min_n, "max_n": max_n}

    def compute(self, input_value, params):
        """Find n in range that maximizes Collatz iterations."""
        min_n = params.get("min_n", 10)
        max_n = params.get("max_n", input_value) if input_value else params.get("max_n", 100)
        max_n = min(abs(max_n) if max_n else 100, 10000)
        min_n = max(1, min_n)

        def collatz_count(start):
            n, count = start, 0
            while n != 1 and count < 10000:
                n = n // 2 if n % 2 == 0 else 3 * n + 1
                count += 1
            return count

        best_n = min_n
        best_count = 0
        for n in range(min_n, max_n + 1):
            c = collatz_count(n)
            if c > best_count:
                best_count = c
                best_n = n

        return MethodResult(
            value=best_n,
            description=f"In [{min_n}, {max_n}], n={best_n} maximizes Collatz steps ({best_count})",
            params=params,
            metadata={"best_n": best_n, "max_iterations": best_count}
        )

    def can_invert(self):
        return False


@register_technique
class DigitSumExtremal(MethodBlock):
    """Find input in range that maximizes digit-sum iterations."""

    def __init__(self):
        super().__init__()
        self.name = "digit_sum_extremal"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["iterative_process", "digit_sum", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate search range parameters."""
        max_n = min(input_value, 100000) if input_value else random.randint(1000, 100000)
        min_n = max(10, max_n // 100)
        return {"min_n": min_n, "max_n": max_n}

    def compute(self, input_value, params):
        """Find n in range that maximizes digit-sum iterations to single digit."""
        min_n = params.get("min_n", 10)
        max_n = params.get("max_n", input_value) if input_value else params.get("max_n", 10)
        max_n = min(abs(max_n) if max_n else 10, 100000)

        def digit_sum_step(n):
            return sum(int(d) for d in str(abs(n)))

        def count_iterations(initial_value, max_iter=100):
            value = initial_value
            count = 0
            while not (value < 10) and count < max_iter:
                value = digit_sum_step(value)
                count += 1
            return count

        best_n = min_n
        best_count = 0

        # Sample at different magnitudes (matches old system exactly)
        sample_points = []
        magnitude = min_n
        while magnitude <= max_n:
            sample_points.extend(range(magnitude, min(magnitude * 10, max_n + 1), max(1, (magnitude * 9) // 10)))
            magnitude *= 10

        # Limit sample size
        if len(sample_points) > 200:
            sample_points = random.sample(sample_points, 200)

        for n in sample_points:
            if n < min_n or n > max_n:
                continue

            c = count_iterations(n)
            if c > best_count:
                best_count = c
                best_n = n

        return MethodResult(
            value=int(best_n),
            description=f"In range [{min_n}, {max_n}], n={best_n} maximizes digit-sum iterations ({best_count} steps)",
            params=params,
            metadata={"min_n": min_n, "max_n": max_n, "best_n": best_n, "max_iterations": best_count}
        )

    def can_invert(self):
        return False

