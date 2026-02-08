"""
Basic arithmetic primitives.
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
class Sum(MethodBlock):
    """Sum of array/list elements."""

    def __init__(self):
        super().__init__()
        self.name = "sum"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random list to sum."""
        length = random.randint(2, 10)
        return {"arr": [random.randint(1, 100) for _ in range(length)]}

    def compute(self, input_value, params):
        """Sum all elements in array."""
        arr = input_value or params.get("arr", [1, 2, 3])
        result = sum(arr)
        return MethodResult(
            value=result,
            description=f"Sum of {arr} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Factorial(MethodBlock):
    """Factorial computation n! = n × (n-1) × ... × 2 × 1."""

    def __init__(self):
        super().__init__()
        self.name = "factorial"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "combinatorics"]
        self.is_primitive = True

    def validate_params(self, params, prev_value=None):
        """Factorial requires n >= 0."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 0

    def generate_parameters(self, input_value=None):
        """Generate random n for factorial."""
        return {"n": random.randint(0, 20)}

    def compute(self, input_value, params):
        """Compute n! = n × (n-1) × ... × 2 × 1."""
        n = input_value or params.get("n", 5)
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        # Bound n to prevent slow factorial computation
        # 20! ~ 2.4e18, still fast; 100! would be slow
        n = min(abs(n), 20)
        result = math.factorial(n)
        try:
            description = f"{n}! = {result}"
        except ValueError:
            description = f"{n}! = <large number>"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class CeilDiv(MethodBlock):
    """Ceiling division: ceil(a/b) without floating point."""

    def __init__(self):
        super().__init__()
        self.name = "ceil_div"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "division"]

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
        """Compute ceil(a/b) without floating point."""
        a = params.get("a", 10)
        b = params.get("b", 3)
        if b == 0:
            raise ValueError("Division by zero")
        # Equivalent to math.ceil(a/b) but avoids float
        result = (a + b - 1) // b if a * b > 0 else a // b
        return MethodResult(
            value=result,
            description=f"⌈{a}/{b}⌉ = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class DigitCount(MethodBlock):
    """Count number of digits in an integer."""

    def __init__(self):
        super().__init__()
        self.name = "digit_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "digits"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {
            "n": random.randint(1, 10**9),
            "base": 10
        }

    def compute(self, input_value, params):
        """Count number of digits in n."""
        n = input_value or params.get("n", 12345)
        base = params.get("base", 10)
        if n == 0:
            return MethodResult(
                value=1,
                description=f"Number of digits in 0 = 1",
                params=params,
                metadata={"techniques_used": [self.name]}
            )
        result = len(str(abs(n)))
        return MethodResult(
            value=result,
            description=f"Number of digits in {n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Log2(MethodBlock):
    """Binary logarithm log₂(x)."""

    def __init__(self):
        super().__init__()
        self.name = "log2"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["arithmetic", "logarithms"]

    def generate_parameters(self, input_value=None):
        """Generate random positive number."""
        return {"x": random.randint(1, 1000)}

    def compute(self, input_value, params):
        """Compute log base 2 of x."""
        x = input_value or params.get("x", 8)
        if x <= 0:
            raise ValueError("log2 undefined for non-positive numbers")
        result = math.log2(x)
        return MethodResult(
            value=result,
            description=f"log₂({x}) = {result:.4f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Compute 2^output_value."""
        return 2 ** output_value


@register_technique
class BinaryRepresentation(MethodBlock):
    """Convert integer to binary representation."""

    def __init__(self):
        super().__init__()
        self.name = "binary_representation"
        self.input_type = "integer"
        self.output_type = "string"
        self.difficulty = 1
        self.tags = ["arithmetic", "binary", "representation"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(0, 1000)}

    def compute(self, input_value, params):
        """Convert integer to binary representation."""
        n = input_value or params.get("n", 10)
        result = bin(n)
        return MethodResult(
            value=result,
            description=f"Binary representation of {n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Convert binary string to integer."""
        return int(output_value, 2)


@register_technique
class BinaryPopcount(MethodBlock):
    """Count number of 1s in binary representation."""

    def __init__(self):
        super().__init__()
        self.name = "binary_popcount"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "binary", "counting"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(0, 1000)}

    def compute(self, input_value, params):
        """Count number of 1s in binary representation."""
        n = input_value or params.get("n", 7)
        result = bin(n).count('1')
        return MethodResult(
            value=result,
            description=f"Number of 1s in binary({n}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ClosestInteger(MethodBlock):
    """Round to nearest integer."""

    def __init__(self):
        super().__init__()
        self.name = "closest_integer"
        self.input_type = "number"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "rounding"]

    def generate_parameters(self, input_value=None):
        """Generate random float."""
        return {"x": random.uniform(0, 100)}

    def compute(self, input_value, params):
        """Round to nearest integer (0.5 rounds to even)."""
        x = input_value or params.get("x", 3.7)
        result = round(x)
        return MethodResult(
            value=result,
            description=f"Round({x:.2f}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FloorTechnique(MethodBlock):
    """Floor function - greatest integer ≤ x."""

    def __init__(self):
        super().__init__()
        self.name = "floor"
        self.input_type = "number"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "rounding"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        import random
        return {"x": random.uniform(0, 100)}

    def compute(self, input_value, params):
        import math
        x = params.get("x", input_value)
        result = math.floor(x)
        return MethodResult(
            value=result,
            description=f"⌊{x}⌋ = {result}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class Power(MethodBlock):
    """Power function: base^exp."""

    def __init__(self):
        super().__init__()
        self.name = "power"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "exponentiation"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random base and exponent."""
        return {
            "base": random.randint(1, 10),
            "exp": random.randint(0, 5)
        }

    def compute(self, input_value, params):
        """Compute base ** exp with safeguards for large exponents."""
        base = params.get("base", 2)
        exp = params.get("exp", 3)

        # Cap exponent to prevent runaway computation (e.g., 3^(2025!) would hang)
        MAX_EXP = 10000
        if isinstance(exp, int) and abs(exp) > MAX_EXP:
            # Return symbolic result for extremely large exponents
            return MethodResult(
                value=f"{base}^{exp}",
                description=f"{base}^{exp} (symbolic - exponent too large)",
                params=params,
                metadata={"techniques_used": [self.name], "symbolic": True}
            )

        result = base ** exp
        try:
            description = f"{base}^{exp} = {result}"
        except ValueError:
            description = f"{base}^{exp} = <large number>"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Product(MethodBlock):
    """Product of array/list elements."""

    def __init__(self):
        super().__init__()
        self.name = "product"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random list to multiply."""
        length = random.randint(2, 6)
        return {"arr": [random.randint(1, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Multiply all elements in array or multiple scalar arguments.

        Supports two use cases:
        1. product([2, 3, 4]) -> multiplies list elements
        2. product(2, 3, 4) -> multiplies multiple arguments
           (the dna_schema.py executor automatically converts product(2, 3, 4)
            into arr=[2, 3, 4] when 'arr' is the first param name)
        """
        # Get the array from input_value or params
        arr = input_value or params.get("arr", [2, 3, 4])

        # Ensure arr is iterable
        if not isinstance(arr, (list, tuple)):
            # If it's a single value, wrap it
            arr = [arr]

        result = 1
        for val in arr:
            result *= val
        try:
            description = f"Product of {arr} = {result}"
        except ValueError:
            description = f"Product of {arr} = <large number>"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FloorDivision(MethodBlock):
    """Floor division: a // b."""

    def __init__(self):
        super().__init__()
        self.name = "floor_division"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "division"]

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
        """Compute a // b (integer division)."""
        a = params.get("a", 17)
        b = params.get("b", 5)
        if b == 0:
            raise ValueError("Division by zero")
        result = a // b
        return MethodResult(
            value=result,
            description=f"{a} // {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Modulo(MethodBlock):
    """Modulo operation: a % b."""

    def __init__(self):
        super().__init__()
        self.name = "modulo"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "modular"]
        self.is_primitive = True

    def validate_params(self, params, prev_value=None):
        """Modulo requires non-zero divisor."""
        b = params.get("b")
        return b is not None and b != 0

    def generate_parameters(self, input_value=None):
        """Generate random dividend and divisor."""
        return {
            "a": random.randint(1, 1000),
            "b": random.randint(2, 100)
        }

    def compute(self, input_value, params):
        """Compute a % b (remainder)."""
        a = params.get("a", 17)
        b = params.get("b", 5)
        if b == 0:
            raise ValueError("Modulo by zero")
        result = a % b
        return MethodResult(
            value=result,
            description=f"{a} mod {b} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class PowerOfTwo(MethodBlock):
    """Compute 2^n for a given exponent n."""

    def __init__(self):
        super().__init__()
        self.name = "power_of_two"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "exponentiation"]

    def generate_parameters(self, input_value=None):
        """Generate a random exponent n."""
        n = input_value if input_value else random.randint(1, 20)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 3)
        n = min(abs(n) if n else 3, 60)
        result = 2 ** n
        return MethodResult(
            value=result,
            description=f"2^{n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True


@register_technique
class PowerOfBase(MethodBlock):
    """Compute base^exp for given base and exponent."""

    def __init__(self):
        super().__init__()
        self.name = "power_of_base"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "exponentiation"]

    def generate_parameters(self, input_value=None):
        """Generate base and exponent."""
        base = random.randint(2, 10)
        exp = random.randint(1, 10)
        return {"base": base, "exp": exp}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        exp = params.get("exp", params.get("n", 3))

        result = int(base ** exp)

        return MethodResult(
            value=result,
            description=f"{base}^{exp} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class NearestMultiple(MethodBlock):
    """Round n to nearest multiple of base."""

    def __init__(self):
        super().__init__()
        self.name = "nearest_multiple"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "rounding"]

    def generate_parameters(self, input_value=None):
        """Generate random number and base."""
        return {"n": random.randint(1, 1000), "base": random.randint(2, 50)}

    def compute(self, input_value, params):
        """Round n to nearest multiple of base."""
        n = input_value if input_value is not None else params.get("n", 17)
        base = params.get("base", 5)
        if base == 0:
            base = 1
        result = round(n / base) * base
        return MethodResult(
            value=result,
            description=f"Nearest multiple of {base} to {n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

