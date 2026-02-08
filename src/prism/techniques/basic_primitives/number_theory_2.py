"""
Number theory primitives (continued).
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
class RationalReduce(MethodBlock):
    """Simplify fraction using GCD (alias for ReduceFraction)."""

    def __init__(self):
        super().__init__()
        self.name = "rational_reduce"
        self.input_type = "none"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["number_theory", "fractions", "gcd"]

    def generate_parameters(self, input_value=None):
        """Generate random numerator and denominator."""
        gcd_val = random.randint(1, 10)
        p = gcd_val * random.randint(1, 100)
        q = gcd_val * random.randint(1, 100)
        return {"p": p, "q": q}

    def compute(self, input_value, params):
        """Reduce fraction p/q to lowest terms using GCD."""
        p = params.get("p", 6)
        q = params.get("q", 9)

        if p is None or q is None:
            raise ValueError("Both p and q must be provided")

        if q == 0:
            raise ValueError("Denominator cannot be zero")

        # Handle negative fractions
        sign = 1
        if p < 0:
            sign *= -1
            p = -p
        if q < 0:
            sign *= -1
            q = -q

        # Compute GCD and reduce
        gcd_val = math.gcd(p, q)
        reduced_p = (p // gcd_val) * sign
        reduced_q = q // gcd_val

        return MethodResult(
            value=(reduced_p, reduced_q),
            description=f"Reduce {params.get('p')}/{params.get('q')} = {reduced_p}/{reduced_q} (GCD={gcd_val})",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "gcd": gcd_val,
                "reduced": (reduced_p, reduced_q)
            }
        )

    def can_invert(self):
        return False


@register_technique
class DivisorsOf(MethodBlock):
    """List all divisors of n (alias for DivisorList)."""

    def __init__(self):
        super().__init__()
        self.name = "divisors_of"
        self.input_type = "integer"
        self.output_type = "list"
        self.difficulty = 2
        self.tags = ["number_theory", "divisors"]

    def generate_parameters(self, input_value=None):
        """Generate random positive integer."""
        return {"n": random.randint(1, 1000)}

    def compute(self, input_value, params):
        """Find all divisors of n."""
        n = input_value or params.get("n", 12)

        if n is None:
            raise ValueError("n must be provided")

        if n <= 0:
            raise ValueError("n must be positive")

        divisors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)

        divisors = sorted(divisors)
        return MethodResult(
            value=divisors,
            description=f"Divisors of {n} = {divisors}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FilterPrimes(MethodBlock):
    """Filter a list to keep only prime numbers."""

    def __init__(self):
        super().__init__()
        self.name = "filter_primes"
        self.input_type = "list"
        self.output_type = "list"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "filtering"]

    def generate_parameters(self, input_value=None):
        """Generate random list of integers."""
        length = random.randint(5, 15)
        return {"numbers": [random.randint(1, 100) for _ in range(length)]}

    def compute(self, input_value, params):
        """Filter list to keep only prime numbers."""
        numbers = input_value or params.get("numbers", [2, 3, 4, 5, 6, 7, 8, 9, 10])

        if numbers is None:
            raise ValueError("numbers list must be provided")

        def is_prime(n):
            """Check if n is prime."""
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            i = 3
            while i * i <= n:
                if n % i == 0:
                    return False
                i += 2
            return True

        primes = [n for n in numbers if is_prime(n)]

        return MethodResult(
            value=primes,
            description=f"Primes from {numbers} = {primes}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class BinaryExponentiation(MethodBlock):
    """Fast exponentiation using binary method (no modulus)."""

    def __init__(self):
        super().__init__()
        self.name = "binary_exponentiation"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "exponentiation", "algorithms"]

    def generate_parameters(self, input_value=None):
        import random
        base = random.randint(2, 10)
        exp = random.randint(1, 20)
        return {"base": base, "exp": exp}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        exp = params.get("exp", 1)

        # Binary exponentiation algorithm
        result = 1
        current = base
        remaining = exp

        while remaining > 0:
            if remaining % 2 == 1:
                result *= current
            current *= current
            remaining //= 2

        return MethodResult(
            value=result,
            description=f"{base}^{exp} = {result} (binary method)",
            params=params,
            metadata={"algorithm": "binary_exponentiation"}
        )

    def can_invert(self):
        return False


@register_technique
class FloorLog2(MethodBlock):
    """Floor of binary logarithm: ⌊log₂(n)⌋."""

    def __init__(self):
        super().__init__()
        self.name = "floor_log2"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["arithmetic", "logarithms", "floor"]

    def validate_params(self, params, prev_value=None):
        """Log requires n > 0."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n > 0

    def generate_parameters(self, input_value=None):
        """Generate random positive integer."""
        return {"n": random.randint(1, 10000)}

    def compute(self, input_value, params):
        """Compute ⌊log₂(n)⌋."""
        n = input_value or params.get("n", 10)
        if n <= 0:
            raise ValueError("log2 undefined for non-positive numbers")
        result = int(math.floor(math.log2(n)))
        return MethodResult(
            value=result,
            description=f"⌊log₂({n})⌋ = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FloorSqrt(MethodBlock):
    """Floor of square root: ⌊√n⌋."""

    def __init__(self):
        super().__init__()
        self.name = "floor_sqrt"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "sqrt", "floor"]

    def validate_params(self, params, prev_value=None):
        """Sqrt requires non-negative input."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 0

    def generate_parameters(self, input_value=None):
        """Generate random non-negative integer."""
        return {"n": random.randint(0, 10000)}

    def compute(self, input_value, params):
        """Compute ⌊√n⌋."""
        n = input_value or params.get("n", 10)
        if n < 0:
            raise ValueError("sqrt undefined for negative numbers")
        result = int(math.floor(math.sqrt(n)))
        return MethodResult(
            value=result,
            description=f"⌊√{n}⌋ = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SqrtComputation(MethodBlock):
    """Square root computation: √n."""

    def __init__(self):
        super().__init__()
        self.name = "sqrt_computation"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "sqrt"]

    def generate_parameters(self, input_value=None):
        """Generate random non-negative number."""
        return {"n": random.randint(0, 10000)}

    def compute(self, input_value, params):
        """Compute √n."""
        n = input_value or params.get("n", 16)
        if n < 0:
            raise ValueError("sqrt undefined for negative numbers")
        result = math.sqrt(n)
        return MethodResult(
            value=result,
            description=f"√{n} = {result:.6f}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Compute n^2 to get original value."""
        return output_value ** 2


@register_technique
class CeilSqrt(MethodBlock):
    """Ceiling of square root: ⌈√n⌉."""

    def __init__(self):
        super().__init__()
        self.name = "ceil_sqrt"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["arithmetic", "sqrt", "ceil"]

    def validate_params(self, params, prev_value=None):
        """Sqrt requires non-negative input."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 0

    def generate_parameters(self, input_value=None):
        """Generate random non-negative integer."""
        return {"n": random.randint(0, 10000)}

    def compute(self, input_value, params):
        """Compute ⌈√n⌉."""
        n = input_value or params.get("n", 10)
        if n < 0:
            raise ValueError("sqrt undefined for negative numbers")
        result = int(math.ceil(math.sqrt(n)))
        return MethodResult(
            value=result,
            description=f"⌈√{n}⌉ = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FloorCbrt(MethodBlock):
    """Floor of cube root: ⌊∛n⌋."""

    def __init__(self):
        super().__init__()
        self.name = "floor_cbrt"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["arithmetic", "cbrt", "floor"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(-1000, 10000)}

    def compute(self, input_value, params):
        """Compute ⌊∛n⌋."""
        n = input_value or params.get("n", 27)
        # Handle negative numbers properly
        if n >= 0:
            result = int(math.floor(n ** (1/3)))
        else:
            result = -int(math.floor(abs(n) ** (1/3)))
        return MethodResult(
            value=result,
            description=f"⌊∛{n}⌋ = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

