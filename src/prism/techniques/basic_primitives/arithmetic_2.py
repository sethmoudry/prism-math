"""
Basic arithmetic primitives (continued).
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
class Negate(MethodBlock):
    """Negate a number: -n."""

    def __init__(self):
        super().__init__()
        self.name = "negate"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random number to negate."""
        return {"n": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Return the negation of n."""
        n = input_value if input_value is not None else params.get("n", 5)
        result = -n
        return MethodResult(
            value=result,
            description=f"-({n}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Negate again to get original value."""
        return -output_value


@register_technique
class RoundToNearest(MethodBlock):
    """Round a number to the nearest integer or specified precision."""

    def __init__(self):
        super().__init__()
        self.name = "round_to_nearest"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic", "rounding"]

    def generate_parameters(self, input_value=None):
        """Generate random number to round."""
        return {
            "value": random.uniform(-100, 100),
            "precision": random.choice([1, 5, 10, 0.1, 0.01])
        }

    def compute(self, input_value, params):
        """Round value to nearest multiple of precision."""
        value = input_value if input_value is not None else params.get("value", 3.7)
        precision = params.get("precision", 1)

        if precision == 1:
            # Round to nearest integer
            result = round(value)
            desc = f"Rounding {value} to nearest integer = {result}"
        else:
            # Round to nearest multiple of precision
            result = round(value / precision) * precision
            desc = f"Rounding {value} to nearest {precision} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Add(MethodBlock):
    """Addition: a + b."""

    def __init__(self):
        super().__init__()
        self.name = "add"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random numbers to add."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compute a + b."""
        a = params.get("a", 10)
        b = params.get("b", 3)
        if a is None or b is None:
            raise ValueError(f"add: input values cannot be None (a={a}, b={b})")
        result = a + b
        return MethodResult(
            value=result,
            description=f"{a} + {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Subtract(MethodBlock):
    """Subtraction: a - b."""

    def __init__(self):
        super().__init__()
        self.name = "subtract"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random numbers to subtract."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compute a - b."""
        a = params.get("a", 10)
        b = params.get("b", 3)
        if a is None or b is None:
            raise ValueError(f"subtract: input values cannot be None (a={a}, b={b})")
        result = a - b
        return MethodResult(
            value=result,
            description=f"{a} - {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Multiply(MethodBlock):
    """Multiplication: a * b."""

    def __init__(self):
        super().__init__()
        self.name = "multiply"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random numbers to multiply."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compute a * b."""
        a = params.get("a", 10)
        b = params.get("b", 3)
        if a is None or b is None:
            raise ValueError(f"multiply: input values cannot be None (a={a}, b={b})")
        result = a * b
        return MethodResult(
            value=result,
            description=f"{a} * {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AddOne(MethodBlock):
    """Increment a number by 1: n + 1."""

    def __init__(self):
        super().__init__()
        self.name = "add_one"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(0, 100)}

    def compute(self, input_value, params):
        """Add one to n."""
        n = input_value or params.get("n", 10)
        result = n + 1
        return MethodResult(
            value=result,
            description=f"{n} + 1 = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Subtract one to get original value."""
        return output_value - 1


@register_technique
class SubtractOne(MethodBlock):
    """Subtract one from a number: n - 1."""

    def __init__(self):
        super().__init__()
        self.name = "subtract_one"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(1, 100)}

    def compute(self, input_value, params):
        """Subtract one from n."""
        n = input_value or params.get("n", 10)
        result = n - 1
        return MethodResult(
            value=result,
            description=f"{n} - 1 = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Add one to get original value."""
        return output_value + 1


@register_technique
class Divide(MethodBlock):
    """Division: a / b (or a // b for integer division if both are ints)."""

    def __init__(self):
        super().__init__()
        self.name = "divide"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "division"]
        self.is_primitive = True

    def validate_params(self, params, prev_value=None):
        """Division requires non-zero divisor."""
        b = params.get("b")
        return b is not None and b != 0

    def generate_parameters(self, input_value=None):
        """Generate random dividend and divisor."""
        return {
            "a": random.randint(1, 1000),
            "b": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Compute a / b (or a // b for integer division if both are ints).

        Args:
            input_value: Not used
            params: Dictionary with 'a' and 'b' keys

        Returns:
            MethodResult with division result

        Examples:
            divide(10, 2) -> 5 (integer division)
            divide(10, 3) -> 3 (integer division)
            divide(10.0, 3) -> 3.333... (float division)
        """
        a = params.get("a", 10)
        b = params.get("b", 2)

        if a is None or b is None:
            raise ValueError(f"divide: input values cannot be None (a={a}, b={b})")
        if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
            raise ValueError(f"divide: expected numeric values, got a={type(a).__name__}, b={type(b).__name__}")
        if b == 0:
            raise ValueError("Division by zero")

        # Use integer division if both are integers, otherwise float division
        if isinstance(a, int) and isinstance(b, int):
            result = a // b
            description = f"{a} // {b} = {result}"
        else:
            result = a / b
            description = f"{a} / {b} = {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Difference(MethodBlock):
    """Compute difference: a - b."""

    def __init__(self):
        super().__init__()
        self.name = "difference"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random numbers."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compute a - b."""
        a = params.get("a", 10)
        b = params.get("b", 3)

        if a is None or b is None:
            raise ValueError("Both a and b must be provided")

        result = a - b

        return MethodResult(
            value=result,
            description=f"{a} - {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class HalfValue(MethodBlock):
    """Compute half of a value: x / 2."""

    def __init__(self):
        super().__init__()
        self.name = "half_value"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random value."""
        return {"x": random.randint(1, 100)}

    def compute(self, input_value, params):
        """Compute x / 2."""
        x = input_value or params.get("x", 10)

        if x is None:
            raise ValueError("x must be provided")

        result = x / 2

        return MethodResult(
            value=result,
            description=f"{x} / 2 = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Multiply by 2 to get original value."""
        return output_value * 2


@register_technique
class AbsoluteDifference(MethodBlock):
    """Compute absolute difference: |a - b|."""

    def __init__(self):
        super().__init__()
        self.name = "absolute_difference"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic", "absolute_value"]

    def generate_parameters(self, input_value=None):
        """Generate random numbers."""
        return {
            "a": random.randint(-100, 100),
            "b": random.randint(-100, 100)
        }

    def compute(self, input_value, params):
        """Compute |a - b|."""
        a = params.get("a", 10)
        b = params.get("b", 3)

        if a is None or b is None:
            raise ValueError("Both a and b must be provided")

        result = abs(a - b)

        return MethodResult(
            value=result,
            description=f"|{a} - {b}| = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AbsoluteValue(MethodBlock):
    """Compute absolute value: |x| = abs(x).

    Simple absolute value operation for a single number.
    Different from absolute_difference which computes |a - b|.
    """

    def __init__(self):
        super().__init__()
        self.name = "absolute_value"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        """Generate random number for absolute value."""
        return {"x": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Compute |x| = abs(x).

        Args:
            input_value: Optional input number (takes precedence)
            params: Dictionary with 'x' key

        Returns:
            MethodResult with absolute value
        """
        x = params.get("x", input_value if input_value is not None else -5)

        result = abs(x)

        return MethodResult(
            value=result,
            description=f"|{x}| = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MinValue(MethodBlock):
    """Compute minimum of two values: min(a, b).

    Simple minimum operation for comparing two numbers.
    """

    def __init__(self):
        super().__init__()
        self.name = "min_value"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "comparison", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random a and b for min(a, b)."""
        return {
            "a": random.randint(1, 100),
            "b": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Compute min(a, b).

        Args:
            input_value: Not used
            params: Dictionary with 'a' and 'b' keys

        Returns:
            MethodResult with minimum value
        """
        a = params.get("a", 5)
        b = params.get("b", 10)

        result = min(a, b)

        return MethodResult(
            value=result,
            description=f"min({a}, {b}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MaxValue(MethodBlock):
    """Compute maximum of two values: max(a, b).

    Simple maximum operation for comparing two numbers.
    """

    def __init__(self):
        super().__init__()
        self.name = "max_value"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "comparison", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random a and b for max(a, b)."""
        return {
            "a": random.randint(1, 100),
            "b": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Compute max(a, b).

        Args:
            input_value: Not used
            params: Dictionary with 'a' and 'b' keys

        Returns:
            MethodResult with maximum value
        """
        a = params.get("a", 5)
        b = params.get("b", 10)

        result = max(a, b)

        return MethodResult(
            value=result,
            description=f"max({a}, {b}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

