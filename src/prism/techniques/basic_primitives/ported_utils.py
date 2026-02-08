"""
Ported utility/primitive methods.

Methods ported (sorted by impact):
- induction (19 failing problems)
- less_than (13 failing problems)
- integer_log (12 failing problems)
- verify_equation (12 failing problems)
- square (12 failing problems)
- equation_solution_check (6 failing problems)
- equal_to (6 failing problems)
- powerset (6 failing problems)
- greater_than (2 failing problems)
- gcd (2 failing problems)
- factor_integer (3 failing problems)
- parity_check (1 failing problem)
"""

import random
import math
from typing import Any, Dict, Optional, List
from math import gcd as math_gcd

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# INDUCTION (19 failing problems)
# ============================================================================

@register_technique
class InductionSimplified(MethodBlock):
    """Mathematical induction: verify base case and inductive step."""

    def __init__(self):
        super().__init__()
        self.name = "induction"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["reasoning", "proof", "induction"]

    def generate_parameters(self, input_value=None):
        base = random.randint(0, 5)
        # Target value to prove formula for
        n = input_value if input_value else random.randint(10, 100)
        return {"base_case": base, "n": n, "formula": "sum_1_to_n"}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value else 10)
        formula = params.get("formula", "sum_1_to_n")
        base = params.get("base_case", 0)

        # Compute the result based on formula type
        if formula == "sum_1_to_n":
            result = n * (n + 1) // 2
            desc = f"By induction: sum(1..{n}) = {n}*({n}+1)/2 = {result}"
        elif formula == "sum_squares":
            result = n * (n + 1) * (2 * n + 1) // 6
            desc = f"By induction: sum(i^2, i=1..{n}) = {result}"
        else:
            result = n
            desc = f"Induction on n={n} from base {base}"

        return MethodResult(value=result, description=desc, params=params,
                            metadata={"n": n})

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value)
        return n is not None and n >= 0

    def can_invert(self):
        return False


# ============================================================================
# LESS THAN (13 failing problems)
# ============================================================================

@register_technique
class LessThan(MethodBlock):
    """Check if a < b."""

    def __init__(self):
        super().__init__()
        self.name = "less_than"
        self.input_type = "none"
        self.output_type = "bool"
        self.difficulty = 1
        self.tags = ["comparison", "primitive"]

    def generate_parameters(self, input_value=None):
        """Generate two numbers for comparison."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        """Return True if a < b, else False."""
        a = params.get("a", 5)
        b = params.get("b", 10)
        result = a < b
        return MethodResult(
            value=result,
            description=f"Compare {a} < {b}: {result}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# VERIFY EQUATION (12 failing problems)
# ============================================================================

@register_technique
class VerifyEquation(MethodBlock):
    """Verify that a value satisfies a two-sided inequality constraint.

    Checks if a and b satisfy both:
    - a >= b/2 + offset
    - b >= a/2 + offset

    Returns 1 if satisfied, 0 otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "verify_equation"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["verification", "constraints", "inequalities"]

    def generate_parameters(self, input_value=None):
        """Generate random a, b pair."""
        a = random.randint(14, 50)
        b = random.randint(14, 50)
        return {"a": a, "b": b, "offset": 7}

    def compute(self, input_value, params):
        """Check if a and b satisfy the two-sided inequality."""
        a = params.get("a", 20)
        b = params.get("b", 20)
        offset = params.get("offset", 7)

        # Check both conditions
        cond1 = a >= b / 2 + offset  # a >= b/2 + offset
        cond2 = b >= a / 2 + offset  # b >= a/2 + offset

        result = 1 if (cond1 and cond2) else 0

        return MethodResult(
            value=result,
            description=f"verify_equation({a}, {b}, {offset}): cond1={cond1}, cond2={cond2}, result={result}",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "condition1_satisfied": cond1,
                "condition2_satisfied": cond2
            }
        )

    def can_invert(self):
        return False


# ============================================================================
# SQUARE (12 failing problems)
# ============================================================================

@register_technique
class Square(MethodBlock):
    """
    Square a number: x^2

    Simple primitive that squares its input.
    """

    def __init__(self):
        super().__init__()
        self.name = "square"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["arithmetic", "basic"]

    def generate_parameters(self, input_value=None):
        x = input_value if input_value is not None else random.randint(1, 100)
        return {"x": x}

    def compute(self, input_value, params):
        x = params.get("x", input_value)
        result = x * x

        return MethodResult(
            value=result,
            description=f"{x}^2 = {result}",
            params=params,
            metadata={"x": x}
        )

    def can_invert(self):
        return False


# ============================================================================
# EQUAL TO (6 failing problems)
# ============================================================================

@register_technique
class EqualTo(MethodBlock):
    """Check if a == b."""

    def __init__(self):
        super().__init__()
        self.name = "equal_to"
        self.input_type = "none"
        self.output_type = "bool"
        self.difficulty = 1
        self.tags = ["comparison", "primitive"]

    def generate_parameters(self, input_value=None):
        """Generate two numbers for comparison."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        """Return True if a == b, else False."""
        a = params.get("a", 5)
        b = params.get("b", 10)
        result = a == b
        return MethodResult(
            value=result,
            description=f"Compare {a} == {b}: {result}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# POWERSET (6 failing problems)
# ============================================================================

@register_technique
class Powerset(MethodBlock):
    """
    Generate all subsets of a set S and count them.
    For a set of size n, returns 2^n (number of subsets).
    """

    def __init__(self):
        super().__init__()
        self.name = "powerset"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "set_theory", "powerset"]

    def generate_parameters(self, input_value=None):
        # Size of the set
        n = input_value if input_value is not None else random.randint(5, 15)
        return {"n": n}

    def validate_params(self, params, prev_value=None):
        """Validate powerset parameters: n must be non-negative."""
        n = params.get("n")
        if n is None:
            return False
        try:
            return int(n) >= 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        n = params.get("n", input_value)

        # Bound n to prevent astronomically large 2^n
        n = min(abs(n) if n else 10, 60)  # 2^60 ~ 10^18, still manageable

        # Number of subsets of a set with n elements is 2^n
        result = 2 ** n

        try:
            description = f"|P(S)| = 2^{n} = {result} where |S| = {n}"
        except ValueError:
            description = f"|P(S)| = 2^{n} = <large number> where |S| = {n}"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "formula": "2^n"}
        )

    def can_invert(self):
        return False


# ============================================================================
# GREATER THAN (2 failing problems)
# ============================================================================

@register_technique
class GreaterThan(MethodBlock):
    """Check if a > b."""

    def __init__(self):
        super().__init__()
        self.name = "greater_than"
        self.input_type = "none"
        self.output_type = "bool"
        self.difficulty = 1
        self.tags = ["comparison", "primitive"]

    def generate_parameters(self, input_value=None):
        """Generate two numbers for comparison."""
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        """Return True if a > b, else False."""
        a = params.get("a", 5)
        b = params.get("b", 10)
        result = a > b
        return MethodResult(
            value=result,
            description=f"Compare {a} > {b}: {result}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# GCD (2 failing problems)
# ============================================================================

@register_technique
class Gcd(MethodBlock):
    """
    Greatest common divisor of two integers using Euclidean algorithm.
    """

    def __init__(self):
        super().__init__()
        self.name = "gcd"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "gcd", "euclidean_algorithm"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        a = input_value if input_value is not None else random.randint(100, 10000)
        b = random.randint(100, 10000)
        return {"a": a, "b": b}

    def validate_params(self, params, prev_value=None):
        """Validate gcd parameters: a and b must be present (can be zero)."""
        a = params.get("a")
        b = params.get("b")
        if a is None or b is None:
            return False
        try:
            a_val = int(a)
            b_val = int(b)
            return not (a_val == 0 and b_val == 0)  # At least one must be non-zero
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        a = params.get("a", input_value)
        b = params.get("b", 10)

        # Compute gcd
        result = math_gcd(abs(a), abs(b))

        description = f"gcd({a}, {b}) = {result}"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"a": a, "b": b}
        )

    def can_invert(self):
        return False


# ============================================================================
# FACTOR INTEGER (3 failing problems)
# ============================================================================

@register_technique
class FactorInteger(MethodBlock):
    """
    Factorize an integer into its prime factors.

    Returns a dictionary of {prime: exponent} pairs representing the prime
    factorization. For example: 12 = 2^2 * 3^1 -> {2: 2, 3: 1}
    """

    def __init__(self):
        super().__init__()
        self.name = "factor_integer"
        self.input_type = "integer"
        self.output_type = "dict"
        self.difficulty = 2
        self.tags = ["number_theory", "factorization", "primes"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(2, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value)

        if n <= 1:
            factors = {}
            description = f"factor_integer({n}) = {{}} (no prime factors)"
            return MethodResult(
                value=1,
                description=description,
                params=params,
                metadata={"n": n, "factors": factors, "product": 1}
            )

        # Perform prime factorization
        factors = {}
        remaining = abs(n)

        # Check for factor of 2
        if remaining % 2 == 0:
            count = 0
            while remaining % 2 == 0:
                count += 1
                remaining //= 2
            factors[2] = count

        # Check for odd factors from 3 onwards
        divisor = 3
        while divisor * divisor <= remaining:
            if remaining % divisor == 0:
                count = 0
                while remaining % divisor == 0:
                    count += 1
                    remaining //= divisor
                factors[divisor] = count
            divisor += 2

        # If remaining > 1, then it's a prime factor
        if remaining > 1:
            factors[remaining] = 1

        # Create description string
        if factors:
            factor_strs = [f"{p}^{e}" if e > 1 else str(p)
                           for p, e in sorted(factors.items())]
            description = (f"factor_integer({n}) = {' * '.join(factor_strs)}"
                           f" = {dict(sorted(factors.items()))}")
        else:
            description = f"factor_integer({n}) = 1 (no prime factors)"

        # Compute the product to verify
        product = 1
        for prime, exp in factors.items():
            product *= prime ** exp

        return MethodResult(
            value=product,
            description=description,
            params=params,
            metadata={"n": n, "factors": factors, "product": product}
        )

    def can_invert(self):
        return False


# ============================================================================
# PARITY CHECK (1 failing problem)
# ============================================================================

@register_technique
class ParityCheck(MethodBlock):
    """Return parity of n: 0 if even, 1 if odd."""

    def __init__(self):
        super().__init__()
        self.name = "parity_check"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "parity"]

    def generate_parameters(self, input_value=None):
        """Generate random integer."""
        return {"n": random.randint(-100, 100)}

    def compute(self, input_value, params):
        """Return parity of n: 0 if even, 1 if odd."""
        n = input_value if input_value is not None else params.get("n", 3)

        result = n % 2

        return MethodResult(
            value=result,
            description=f"Parity of {n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# INTEGER LOG (12 failing problems)
# ============================================================================

@register_technique
class IntegerLog(MethodBlock):
    """Compute floor(log_base(x)) for given base and x."""

    def __init__(self):
        super().__init__()
        self.name = "integer_log"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["arithmetic", "logarithm", "counting"]

    def generate_parameters(self, input_value=None):
        base = random.randint(2, 10)
        x = input_value if input_value else random.randint(10, 10000)
        return {"base": base, "x": x}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        x = params.get("x", input_value) if input_value else params.get("x", 100)

        if x is None:
            x = 100
        elif x <= 0:
            x = abs(x) if x != 0 else 1
        if base is None or base <= 1:
            base = 2

        result = int(math.log(x) / math.log(base))

        return MethodResult(
            value=result,
            description=f"floor(log_{base}({x})) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# EQUATION SOLUTION CHECK (6 failing problems)
# ============================================================================

@register_technique
class EquationSolutionCheck(MethodBlock):
    """Check if two values are equal (equation solution verification)."""

    def __init__(self):
        super().__init__()
        self.name = "equation_solution_check"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["algebra", "verification"]

    def generate_parameters(self, input_value=None):
        value = random.randint(-100, 100)
        if random.random() < 0.5:
            return {"lhs": value, "rhs": value}
        else:
            return {"lhs": value, "rhs": value + random.randint(1, 10)}

    def compute(self, input_value, params):
        lhs = params.get("lhs", 10)
        rhs = params.get("rhs", 10)

        if isinstance(lhs, float) or isinstance(rhs, float):
            result = abs(lhs - rhs) < 1e-10
        else:
            result = lhs == rhs

        return MethodResult(
            value=1 if result else 0,
            description=f"Does {lhs} = {rhs}? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False
