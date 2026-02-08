"""
Additional arithmetic and utility primitives.
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
class CompareValues(MethodBlock):
    """Compare two values and return -1, 0, or 1."""

    def __init__(self):
        super().__init__()
        self.name = "compare_values"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "comparison", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random values to compare."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compare a and b: return -1 if a < b, 0 if a == b, 1 if a > b."""
        a = params.get("a", 5)
        b = params.get("b", 3)

        if a is None or b is None:
            raise ValueError("Both a and b must be provided")

        if a < b:
            result = -1
        elif a == b:
            result = 0
        else:
            result = 1

        return MethodResult(
            value=result,
            description=f"compare({a}, {b}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FilterOdd(MethodBlock):
    """Filter a list to keep only odd numbers."""

    def __init__(self):
        super().__init__()
        self.name = "filter_odd"
        self.input_type = "list"
        self.output_type = "list"
        self.difficulty = 1
        self.tags = ["arithmetic", "parity", "filtering"]

    def generate_parameters(self, input_value=None):
        """Generate random list of integers."""
        length = random.randint(5, 15)
        return {"numbers": [random.randint(-50, 50) for _ in range(length)]}

    def compute(self, input_value, params):
        """Filter list to keep only odd numbers."""
        numbers = input_value or params.get("numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        if numbers is None:
            raise ValueError("numbers list must be provided")

        odd_numbers = [n for n in numbers if n % 2 != 0]

        return MethodResult(
            value=odd_numbers,
            description=f"Odd numbers from {numbers} = {odd_numbers}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FilterEven(MethodBlock):
    """Filter a list to keep only even numbers."""

    def __init__(self):
        super().__init__()
        self.name = "filter_even"
        self.input_type = "list"
        self.output_type = "list"
        self.difficulty = 1
        self.tags = ["arithmetic", "parity", "filtering"]

    def generate_parameters(self, input_value=None):
        """Generate random list of integers."""
        length = random.randint(5, 15)
        return {"numbers": [random.randint(-50, 50) for _ in range(length)]}

    def compute(self, input_value, params):
        """Filter list to keep only even numbers."""
        numbers = input_value or params.get("numbers", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        if numbers is None:
            raise ValueError("numbers list must be provided")

        even_numbers = [n for n in numbers if n % 2 == 0]

        return MethodResult(
            value=even_numbers,
            description=f"Even numbers from {numbers} = {even_numbers}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class RangeIntegers(MethodBlock):
    """Generate range of integers from start to end (inclusive)."""

    def __init__(self):
        super().__init__()
        self.name = "range_integers"
        self.input_type = "none"
        self.output_type = "list"
        self.difficulty = 1
        self.tags = ["arithmetic", "sequences", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random range."""
        start = random.randint(1, 50)
        end = random.randint(start, start + 20)
        return {"start": start, "end": end}

    def compute(self, input_value, params):
        """Generate list of integers from start to end (inclusive)."""
        start = params.get("start", 1)
        end = params.get("end", 10)

        if start is None or end is None:
            raise ValueError("Both start and end must be provided")

        result = list(range(start, end + 1))

        return MethodResult(
            value=result,
            description=f"Integers from {start} to {end} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ProductRange(MethodBlock):
    """Compute product of integers from start to end (inclusive).

    Similar to sum_range but for products. Useful for computing:
    - Factorial-like products: product_range(1, n) = n!
    - Partial products: product_range(a, b) = a * (a+1) * ... * b
    - Falling factorials: product_range(n-k+1, n)
    """

    def __init__(self):
        super().__init__()
        self.name = "product_range"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic", "product"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random start and end for product range."""
        start = random.randint(1, 10)
        end = random.randint(start, start + 10)
        return {"start": start, "end": end}

    def compute(self, input_value, params):
        """Compute product of all integers from start to end inclusive.

        Args:
            input_value: Not used
            params: Dictionary with 'start' and 'end' keys

        Returns:
            MethodResult with product value
        """
        start = params.get("start", 1)
        end = params.get("end", 5)

        if start > end:
            # Empty product is 1
            result = 1
            desc = f"product_range({start}, {end}) = 1 (empty product)"
        else:
            result = 1
            for i in range(start, end + 1):
                result *= i
            desc = f"product_range({start}, {end}) = {start} * ... * {end} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Mod(MethodBlock):
    """Simple modulo operation: a % b.

    This is an alias for the commonly requested 'mod' operation.
    Different from power_mod which computes (a^b) % m.
    """

    def __init__(self):
        super().__init__()
        self.name = "mod"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic", "modular"]
        self.is_primitive = True

    def validate_params(self, params, prev_value=None):
        """Modulo requires non-zero divisor."""
        b = params.get("b")
        return b is not None and b != 0

    def generate_parameters(self, input_value=None):
        """Generate random a and b for a % b."""
        return {
            "a": random.randint(1, 1000),
            "b": random.randint(2, 100)
        }

    def compute(self, input_value, params):
        """Compute a % b (remainder of a divided by b).

        Args:
            input_value: Not used
            params: Dictionary with 'a' and 'b' keys

        Returns:
            MethodResult with remainder value
        """
        a = params.get("a", 17)
        b = params.get("b", 5)

        if b == 0:
            raise ValueError("Modulo by zero is undefined")

        result = a % b

        return MethodResult(
            value=result,
            description=f"mod({a}, {b}) = {a} % {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Calculate100aPlusB(MethodBlock):
    """Calculate 100*a + b."""

    def __init__(self):
        super().__init__()
        self.name = "calculate_100a_plus_b"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random a and b."""
        return {
            "a": random.randint(0, 99),
            "b": random.randint(0, 99)
        }

    def compute(self, input_value, params):
        """Calculate 100*a + b."""
        a = params.get("a", 1)
        b = params.get("b", 23)

        if a is None or b is None:
            raise ValueError("Both a and b must be provided")

        result = 100 * a + b

        return MethodResult(
            value=result,
            description=f"100 × {a} + {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AddNumeratorDenominator(MethodBlock):
    """Add numerator and denominator: p + q."""

    def __init__(self):
        super().__init__()
        self.name = "add_numerator_denominator"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "fractions", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random numerator and denominator."""
        return {
            "p": random.randint(1, 100),
            "q": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Add numerator and denominator: p + q."""
        p = params.get("p", 3)
        q = params.get("q", 5)

        if p is None or q is None:
            raise ValueError("Both p and q must be provided")

        result = p + q

        return MethodResult(
            value=result,
            description=f"{p} + {q} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ExtractNumeratorDenominator(MethodBlock):
    """Extract numerator and denominator from a fraction."""

    def __init__(self):
        super().__init__()
        self.name = "extract_numerator_denominator"
        self.input_type = "tuple"
        self.output_type = "tuple"
        self.difficulty = 1
        self.tags = ["arithmetic", "fractions", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random fraction as tuple."""
        return {
            "fraction": (random.randint(1, 100), random.randint(1, 100))
        }

    def compute(self, input_value, params):
        """Extract numerator (p) and denominator (q) from fraction.

        Input can be:
        - A tuple (p, q)
        - A Fraction object with .numerator and .denominator attributes
        """
        fraction = input_value or params.get("fraction", (3, 5))

        if fraction is None:
            raise ValueError("fraction must be provided")

        # Handle different fraction representations
        if isinstance(fraction, tuple) and len(fraction) == 2:
            p, q = fraction
        elif hasattr(fraction, 'numerator') and hasattr(fraction, 'denominator'):
            p = fraction.numerator
            q = fraction.denominator
        else:
            raise ValueError("Invalid fraction format")

        return MethodResult(
            value=(p, q),
            description=f"Extract from fraction: numerator={p}, denominator={q}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SumNumeratorDenominator(MethodBlock):
    """Sum numerator and denominator: p + q (same as add_numerator_denominator)."""

    def __init__(self):
        super().__init__()
        self.name = "sum_numerator_denominator"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "fractions", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random numerator and denominator."""
        return {
            "p": random.randint(1, 100),
            "q": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Sum numerator and denominator: p + q."""
        p = params.get("p", 3)
        q = params.get("q", 5)

        if p is None or q is None:
            raise ValueError("Both p and q must be provided")

        result = p + q

        return MethodResult(
            value=result,
            description=f"Sum of numerator and denominator: {p} + {q} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConstantZero(MethodBlock):
    """Return the constant value 0.

    Useful as a terminal/base case in compositions, or as a zero element
    in algebraic structures. No parameters needed.
    """

    def __init__(self):
        super().__init__()
        self.name = "constant_zero"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 0
        self.tags = ["constants", "basic", "terminal"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """No parameters needed for constant."""
        return {}

    def compute(self, input_value, params):
        """Return 0."""
        result = 0
        return MethodResult(
            value=result,
            description="Constant value 0",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConstantOne(MethodBlock):
    """Return the constant value 1.

    Useful as an identity/multiplicative element in compositions, or as
    a base case in sequences. No parameters needed.
    """

    def __init__(self):
        super().__init__()
        self.name = "constant_one"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 0
        self.tags = ["constants", "basic", "identity"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """No parameters needed for constant."""
        return {}

    def compute(self, input_value, params):
        """Return 1."""
        result = 1
        return MethodResult(
            value=result,
            description="Constant value 1",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConstantResult(MethodBlock):
    """Wrap any numeric value as a constant result.

    Use this for ALL numeric constants in method_steps.
    This is the standard way to inject parameter values into computation chains.

    Example:
        n = constant_result(5)  # Wrap the value 5
        result = catalan(n)     # Use wrapped value
    """

    def __init__(self):
        super().__init__()
        self.name = "constant_result"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 0
        self.tags = ["constants", "basic", "wrapper"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate parameters with the input value."""
        if input_value is not None:
            return {"value": int(input_value)}
        return {"value": 0}

    def compute(self, input_value, params):
        """Return the constant value."""
        # Accept value from params or direct input
        if isinstance(input_value, (int, float)):
            result = int(input_value)
        elif isinstance(params, dict) and "value" in params:
            result = int(params["value"])
        elif isinstance(params, (int, float)):
            result = int(params)
        else:
            result = int(input_value) if input_value is not None else 0

        return MethodResult(
            value=result,
            description=f"Constant value {result}",
            params=params if isinstance(params, dict) else {"value": result},
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

