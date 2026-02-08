"""
Number theory primitives.
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
class DivisorList(MethodBlock):
    """Find all divisors of a number."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_list"
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
class DivisorPairs(MethodBlock):
    """Find all pairs (d, n/d) where d divides n."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_pairs"
        self.input_type = "integer"
        self.output_type = "list"
        self.difficulty = 2
        self.tags = ["number_theory", "divisors"]

    def generate_parameters(self, input_value=None):
        """Generate random positive integer."""
        return {"n": random.randint(1, 1000)}

    def compute(self, input_value, params):
        """Find all pairs (d, n/d) where d divides n."""
        n = input_value or params.get("n", 12)
        if n <= 0:
            raise ValueError("n must be positive")

        pairs = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                pairs.append((i, n // i))

        return MethodResult(
            value=pairs,
            description=f"Divisor pairs of {n} = {pairs}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class PerfectSquareCheck(MethodBlock):
    """Check if a number is a perfect square."""

    def __init__(self):
        super().__init__()
        self.name = "perfect_square_check"
        self.input_type = "integer"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "perfect_squares"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(0, 1000)}

    def compute(self, input_value, params):
        """Check if n is a perfect square."""
        n = input_value or params.get("n", 16)

        # Type checking to handle edge cases
        if not isinstance(n, (int, float)):
            return MethodResult(
                value=0,  # Not a perfect square if not a number
                description=f"Is {n} a perfect square? False (invalid type: {type(n).__name__})",
                params=params,
                metadata={"techniques_used": [self.name]}
            )

        # Ensure we're working with an integer
        n = int(n)

        if n < 0:
            result = False
        else:
            sqrt_n = int(math.sqrt(n))
            result = sqrt_n * sqrt_n == n

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} a perfect square? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class DivisibilityCheck(MethodBlock):
    """Check if one number is divisible by another."""

    def __init__(self):
        super().__init__()
        self.name = "divisibility_check"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "divisibility"]

    def generate_parameters(self, input_value=None):
        """Generate random number and divisor."""
        return {
            "n": random.randint(1, 1000),
            "d": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Check if n is divisible by d (n % d == 0)."""
        n = params.get("n", 12)
        d = params.get("d", 3)

        # Handle edge case: division by zero
        if d == 0:
            result = False
        else:
            result = (n % d == 0)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} divisible by {d}? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IsEven(MethodBlock):
    """Check if a number is even."""

    def __init__(self):
        super().__init__()
        self.name = "is_even"
        self.input_type = "integer"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "parity"]

    def generate_parameters(self, input_value=None):
        """Generate random integer to test for evenness."""
        return {"n": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Check if n is even.

        Returns 1 for True (even), 0 for False (odd).

        Args:
            input_value: Integer to check (optional)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with value 1 if even, 0 if odd

        Examples:
            is_even(4) -> 1
            is_even(3) -> 0
            is_even(0) -> 1
            is_even(-2) -> 1
        """
        n = input_value if input_value is not None else params.get("n", 4)

        # Check if even
        result = (n % 2 == 0)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} even? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IsOdd(MethodBlock):
    """Check if a number is odd."""

    def __init__(self):
        super().__init__()
        self.name = "is_odd"
        self.input_type = "integer"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "parity"]

    def generate_parameters(self, input_value=None):
        """Generate random integer to test for oddness."""
        return {"n": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Check if n is odd.

        Returns 1 for True (odd), 0 for False (even).

        Args:
            input_value: Integer to check (optional)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with value 1 if odd, 0 if even

        Examples:
            is_odd(3) -> 1
            is_odd(4) -> 0
            is_odd(1) -> 1
            is_odd(-3) -> 1
        """
        n = input_value if input_value is not None else params.get("n", 3)

        # Check if odd
        result = (n % 2 != 0)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} odd? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IsPrime(MethodBlock):
    """Check if a number is prime."""

    def __init__(self):
        super().__init__()
        self.name = "is_prime"
        self.input_type = "integer"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "primes"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        """Generate random integer to test for primality."""
        return {"n": random.randint(0, 1000)}

    def compute(self, input_value, params):
        """Check if n is prime using trial division.

        Returns 1 for True (prime), 0 for False (not prime).

        Edge cases:
        - n < 2: returns False (0)
        - n = 2: returns True (1)
        - Even numbers > 2: returns False (0)
        """
        n = input_value if input_value is not None else params.get("n", 7)

        # Use efficient primality test
        if n < 2:
            result = False
        elif n == 2:
            result = True
        elif n % 2 == 0:
            result = False
        else:
            result = True
            i = 3
            while i * i <= n:
                if n % i == 0:
                    result = False
                    break
                i += 2

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} prime? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ReduceFraction(MethodBlock):
    """Reduce a fraction to lowest terms by dividing out the GCD."""

    def __init__(self):
        super().__init__()
        self.name = "reduce_fraction"
        self.input_type = "none"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["number_theory", "fractions", "gcd"]

    def generate_parameters(self, input_value=None):
        """Generate random numerator and denominator."""
        # Generate with common factors to make reduction interesting
        gcd_val = random.randint(1, 10)
        numerator = gcd_val * random.randint(1, 100)
        denominator = gcd_val * random.randint(1, 100)
        return {
            "numerator": numerator,
            "denominator": denominator
        }

    def compute(self, input_value, params):
        """
        Reduce fraction to lowest terms.

        Can be called in two ways:
        1. reduce_fraction(numerator, denominator) - with params dict
        2. reduce_fraction(fraction_tuple) - with input_value as (num, denom) tuple

        Returns:
            MethodResult with value as (reduced_numerator, reduced_denominator) tuple

        Examples:
            reduce_fraction(6, 9) -> (2, 3)
            reduce_fraction(12, 18) -> (2, 3)
            reduce_fraction(-6, 9) -> (-2, 3)
            reduce_fraction(6, -9) -> (-2, 3)
            reduce_fraction(-6, -9) -> (2, 3)
        """
        # Handle two calling conventions:
        # 1. Input value is a tuple (numerator, denominator)
        # 2. Parameters dict with 'numerator' and 'denominator' keys
        if input_value is not None and isinstance(input_value, (tuple, list)) and len(input_value) == 2:
            numerator, denominator = input_value
        else:
            numerator = params.get("numerator", 6)
            denominator = params.get("denominator", 9)

        # Validate denominator is not zero
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")

        # Handle negative fractions: keep sign in numerator
        # This ensures consistent behavior: -a/b = a/-b = -(a/b)
        sign = 1
        if numerator < 0:
            sign *= -1
            numerator = -numerator
        if denominator < 0:
            sign *= -1
            denominator = -denominator

        # Compute GCD and reduce
        gcd_val = math.gcd(numerator, denominator)
        reduced_num = (numerator // gcd_val) * sign
        reduced_denom = denominator // gcd_val

        # Store original values for description
        orig_num = params.get("numerator", input_value[0] if input_value else 6)
        orig_denom = params.get("denominator", input_value[1] if input_value else 9)

        return MethodResult(
            value=(reduced_num, reduced_denom),
            description=f"Reduce {orig_num}/{orig_denom} = {reduced_num}/{reduced_denom} (GCD={gcd_val})",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "gcd": gcd_val,
                "original": (orig_num, orig_denom),
                "reduced": (reduced_num, reduced_denom)
            }
        )

    def can_invert(self):
        return False


@register_technique
class DivisibleBy3(MethodBlock):
    """Check if a number is divisible by 3."""

    def __init__(self):
        super().__init__()
        self.name = "divisible_by_3"
        self.input_type = "integer"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "divisibility"]

    def generate_parameters(self, input_value=None):
        """Generate random integer to test for divisibility by 3."""
        return {"n": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Check if n is divisible by 3.

        Returns 1 for True (divisible by 3), 0 for False (not divisible by 3).

        Args:
            input_value: Integer to check (optional)
            params: Dictionary with 'n' key

        Returns:
            MethodResult with value 1 if divisible by 3, 0 otherwise

        Examples:
            divisible_by_3(9) -> 1
            divisible_by_3(10) -> 0
            divisible_by_3(0) -> 1
            divisible_by_3(-6) -> 1
        """
        n = input_value if input_value is not None else params.get("n", 9)

        # Check if divisible by 3
        result = (n % 3 == 0)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Is {n} divisible by 3? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class Divides(MethodBlock):
    """Check if d divides n (i.e., n % d == 0)."""

    def __init__(self):
        super().__init__()
        self.name = "divides"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["number_theory", "divisibility"]

    def validate_params(self, params, prev_value=None):
        """Divides check requires non-zero divisor."""
        d = params.get("d")
        return d is not None and d != 0

    def generate_parameters(self, input_value=None):
        """Generate random number and divisor."""
        return {
            "n": random.randint(1, 1000),
            "d": random.randint(1, 100)
        }

    def compute(self, input_value, params):
        """Check if d divides n (n % d == 0).

        Returns 1 for True (d divides n), 0 for False (d does not divide n).

        Args:
            input_value: Not used
            params: Dictionary with 'n' and 'd' keys

        Returns:
            MethodResult with value 1 if d divides n, 0 otherwise

        Examples:
            divides(12, 3) -> 1
            divides(12, 5) -> 0
            divides(0, 5) -> 1
            divides(15, 1) -> 1
        """
        n = params.get("n", 12)
        d = params.get("d", 3)

        # Handle edge case: division by zero
        if d == 0:
            result = False
        else:
            result = (n % d == 0)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Does {d} divide {n}? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

