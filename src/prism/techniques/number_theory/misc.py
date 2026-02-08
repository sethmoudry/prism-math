"""
Number theory miscellaneous techniques.

Implements techniques for:
- Digit operations
- Carmichael lambda
- Perfect squares and cubes checking
- Wilson's theorem
- Linear combinations
- Common term factoring
"""

import random
import math
from typing import Any, Dict, Optional, List
from sympy import (
    factorint, gcd, isprime,
    reduced_totient
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# DIGIT OPERATIONS
# ============================================================================

@register_technique
class DigitSumBaseB(MethodBlock):
    """Compute sum of digits in base b."""

    def __init__(self):
        super().__init__()
        self.name = "digit_sum_base_b"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "digits"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        b = random.choice([2, 3, 5, 7, 10, 16])
        return {"n": n, "b": b}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        b = params.get("b", 10)

        s = 0
        m = abs(n)
        while m > 0:
            s += m % b
            m //= b

        return MethodResult(
            value=s,
            description=f"Sum of digits of {n} in base {b} is {s}",
            metadata={"n": n, "b": b, "digit_sum": s}
        )

    def validate_params(self, params):
        return params.get("b", 10) > 1

    def can_invert(self):
        return True


@register_technique
class SumOfDigits(MethodBlock):
    """Sum of digits in base 10."""

    def __init__(self):
        super().__init__()
        self.name = "sum_of_digits"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "digits"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 100000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        s = sum(int(d) for d in str(abs(n)))

        return MethodResult(
            value=s,
            description=f"Sum of digits of {n} is {s}",
            metadata={"n": n, "digit_sum": s}
        )

    def validate_params(self, params):
        return True

    def can_invert(self):
        return True


@register_technique
class DigitSumInverse(MethodBlock):
    """Find smallest n with digit sum = s."""

    def __init__(self):
        super().__init__()
        self.name = "digit_sum_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "digits", "inverse"]

    def generate_parameters(self, input_value=None):
        s = input_value if input_value else random.randint(5, 50)
        return {"s": s}

    def compute(self, input_value, params):
        s = params.get("s", input_value) if input_value else params.get("s", 10)
        s = min(abs(s) if s else 10, 1000)

        # Smallest n with digit sum s: use as many 9s as possible
        nines = s // 9
        remainder = s % 9

        if remainder == 0:
            n = int("9" * nines) if nines > 0 else 0
        else:
            n = int(str(remainder) + "9" * nines) if nines > 0 else remainder

        return MethodResult(
            value=n,
            description=f"Smallest n with digit sum {s} is {n}",
            metadata={"s": s, "n": n}
        )

    def validate_params(self, params):
        return params.get("s", 10) >= 0

    def can_invert(self):
        return False


# ============================================================================
# PERFECT POWERS
# ============================================================================

@register_technique
class PerfectSquaresUpTo(MethodBlock):
    """Count perfect squares <= n."""

    def __init__(self):
        super().__init__()
        self.name = "perfect_squares_up_to"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "squares", "counting"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 100)

        count = int(math.sqrt(n))

        return MethodResult(
            value=count,
            description=f"Number of perfect squares <= {n} is {count} (1^2, 2^2, ..., {count}^2)",
            metadata={"n": n, "count": count, "largest_square": count**2}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return True


@register_technique
class IsPerfectSquare(MethodBlock):
    """Check if n is a perfect square."""

    def __init__(self):
        super().__init__()
        self.name = "is_perfect_square"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "squares"]

    def generate_parameters(self, input_value=None):
        if input_value:
            n = input_value
        else:
            if random.random() < 0.5:
                k = random.randint(1, 100)
                n = k * k
            else:
                n = random.randint(1, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        root = math.isqrt(abs(n))
        is_square = (root * root == n)

        return MethodResult(
            value=1 if is_square else 0,
            description=f"{n} is {'a' if is_square else 'not a'} perfect square" + (f" ({root}^2)" if is_square else ""),
            metadata={"n": n, "is_perfect_square": is_square, "sqrt": root if is_square else None}
        )

    def validate_params(self, params):
        return params.get("n", 1) >= 0

    def can_invert(self):
        return False


@register_technique
class IsPerfectCube(MethodBlock):
    """Check if n is a perfect cube."""

    def __init__(self):
        super().__init__()
        self.name = "is_perfect_cube"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "cubes"]

    def generate_parameters(self, input_value=None):
        if input_value:
            n = input_value
        else:
            if random.random() < 0.5:
                k = random.randint(1, 50)
                n = k * k * k
            else:
                n = random.randint(1, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        root = round(abs(n) ** (1/3))
        is_cube = (root ** 3 == n) or ((root + 1) ** 3 == n) or ((root - 1) ** 3 == n)

        if is_cube:
            for r in [root - 1, root, root + 1]:
                if r ** 3 == n:
                    root = r
                    break

        return MethodResult(
            value=1 if is_cube else 0,
            description=f"{n} is {'a' if is_cube else 'not a'} perfect cube" + (f" ({root}^3)" if is_cube else ""),
            metadata={"n": n, "is_perfect_cube": is_cube, "cbrt": root if is_cube else None}
        )

    def validate_params(self, params):
        return params.get("n", 1) >= 0

    def can_invert(self):
        return False


# ============================================================================
# SPECIAL FUNCTIONS
# ============================================================================

@register_technique
class CarmichaelLambda(MethodBlock):
    """Compute Carmichael's lambda function lambda(n)."""

    def __init__(self):
        super().__init__()
        self.name = "carmichael_lambda"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "carmichael"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(2, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        lam = reduced_totient(n)

        return MethodResult(
            value=lam,
            description=f"lambda({n}) = {lam}",
            metadata={"n": n, "lambda": lam}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0

    def can_invert(self):
        return True


@register_technique
class CeilLog2Large(MethodBlock):
    """Compute ceil(log2(n)) for large n."""

    def __init__(self):
        super().__init__()
        self.name = "ceil_log2_large"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "logarithm"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1000, 10**12)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 1000)

        # Handle symbolic string expressions (from large power computations)
        if isinstance(n, str):
            # Try to parse as a power expression like "10^100000"
            if '^' in n or '**' in n:
                parts = n.replace('**', '^').split('^')
                if len(parts) == 2:
                    try:
                        base, exp = int(parts[0]), int(parts[1])
                        # ceil(log2(base^exp)) = ceil(exp * log2(base))
                        import math
                        result = math.ceil(exp * math.log2(base))
                        return MethodResult(
                            value=result,
                            description=f"ceil(log2({n})) = {result}",
                            metadata={"n": n, "result": result, "symbolic": True}
                        )
                    except (ValueError, TypeError):
                        pass
            return MethodResult(
                value=n,
                description=f"ceil(log2({n})) - symbolic",
                metadata={"n": n, "symbolic": True}
            )

        if not isinstance(n, (int, float)) or n <= 0:
            return MethodResult(
                value=None,
                description=f"log2({n}) undefined",
                metadata={"error": "invalid_n"}
            )

        result = (int(n) - 1).bit_length()

        return MethodResult(
            value=result,
            description=f"ceil(log2({n})) = {result}",
            metadata={"n": n, "result": result}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False


@register_technique
class WilsonTheoremMod(MethodBlock):
    """Compute (p-1)! mod p using Wilson's theorem."""

    def __init__(self):
        super().__init__()
        self.name = "wilson_theorem_mod"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "factorial", "modular"]

    def generate_parameters(self, input_value=None):
        p = input_value if (input_value and isprime(input_value)) else random.choice([5, 7, 11, 13, 17, 19, 23])
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 7)

        if not isprime(p):
            return MethodResult(
                value=0,
                description=f"{p} is not prime",
                metadata={"error": "not_prime"}
            )

        # Wilson's theorem: (p-1)! = -1 (mod p)
        result = p - 1

        return MethodResult(
            value=result,
            description=f"({p}-1)! = -1 = {result} (mod {p}) by Wilson's theorem",
            metadata={"p": p, "result": result}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class FactorCommonTerm(MethodBlock):
    """Factor out GCD from a list of terms."""

    def __init__(self):
        super().__init__()
        self.name = "factor_common_term"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "gcd", "factoring"]

    def generate_parameters(self, input_value=None):
        if input_value and isinstance(input_value, list):
            terms = input_value
        else:
            common = random.choice([2, 3, 5, 6, 10, 12])
            count = random.randint(2, 5)
            terms = [common * random.randint(1, 20) for _ in range(count)]
        return {"terms": terms}

    def compute(self, input_value, params):
        terms = params.get("terms", [12, 18, 24])

        if not terms or len(terms) < 2:
            return MethodResult(
                value=None,
                description="Need at least 2 terms",
                metadata={"error": "insufficient_terms"}
            )

        result = terms[0]
        for t in terms[1:]:
            result = gcd(result, t)

        return MethodResult(
            value=result,
            description=f"GCD({terms}) = {result}",
            metadata={"terms": terms, "gcd": result}
        )

    def validate_params(self, params):
        terms = params.get("terms", [])
        return len(terms) >= 2

    def can_invert(self):
        return False


@register_technique
class CountLinearCombinationRange(MethodBlock):
    """Count integers in [1, n] representable as ax + by for x, y >= 0."""

    def __init__(self):
        super().__init__()
        self.name = "count_linear_combination_range"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "diophantine"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(50, 500)
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        while gcd(a, b) != 1:
            b = random.randint(2, 10)
        return {"n": n, "a": a, "b": b}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 100)
        a = params.get("a", 3)
        b = params.get("b", 5)

        n = min(abs(n) if n else 100, 10000)

        # Count representable numbers
        representable = set()
        for x in range(n // a + 1):
            for y in range((n - a * x) // b + 1):
                val = a * x + b * y
                if 1 <= val <= n:
                    representable.add(val)

        count = len(representable)

        return MethodResult(
            value=count,
            description=f"Count of integers in [1, {n}] representable as {a}x + {b}y = {count}",
            metadata={"n": n, "a": a, "b": b, "count": count}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0 and params.get("a", 1) > 0 and params.get("b", 1) > 0

    def can_invert(self):
        return False


@register_technique
class BaseRepresentation(MethodBlock):
    """Convert n to base b and return as integer formed by concatenating digits."""

    def __init__(self):
        super().__init__()
        self.name = "base_representation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "base_conversion"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        base = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 16])
        return {"n": n, "base": base}

    def compute(self, input_value, params):
        n = input_value if input_value is not None else params.get("n", 10)
        base = params.get("base", 2)
        n = abs(n) if n else 10
        base = max(2, min(base, 36))

        if n == 0:
            digits_str = "0"
        else:
            digits = []
            val = n
            while val > 0:
                digits.append(val % base)
                val //= base
            digits.reverse()
            digits_str = "".join(str(d) for d in digits)

        # Return the digit string interpreted as a base-10 integer
        result = int(digits_str)
        return MethodResult(
            value=result,
            description=f"{n} in base {base} = {digits_str} (as integer: {result})",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class BaseDigitSumInverse(MethodBlock):
    """Find smallest n with given digit sum s in base b."""

    def __init__(self):
        super().__init__()
        self.name = "base_digit_sum_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "base_conversion", "digit_sum"]

    def generate_parameters(self, input_value=None):
        s = input_value if input_value else random.randint(1, 50)
        b = random.choice([2, 3, 5, 7, 10])
        return {"s": s, "b": b}

    def compute(self, input_value, params):
        b = params.get("b", params.get("base", 10))
        s = params.get("s", input_value) if input_value else params.get("s", params.get("target_sum", 20))
        # Bound s to prevent slow O(s) loop
        s = min(abs(s) if s else 20, 1000)

        if s == 0:
            result = 0
        else:
            max_digit = b - 1
            num_max = s // max_digit
            remainder = s % max_digit

            if remainder == 0:
                result = 0
                for i in range(num_max):
                    result = result * b + max_digit
            else:
                result = remainder
                for i in range(num_max):
                    result = result * b + max_digit

        return MethodResult(
            value=int(result),
            description=f"Smallest n with digit sum {s} in base {b} is {result}",
            params={"s": s, "b": b},
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class BaseDigitCount(MethodBlock):
    """Count number of digits of n in base b."""

    def __init__(self):
        super().__init__()
        self.name = "base_digit_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "base_conversion"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1, 100000)
        base = random.choice([2, 3, 5, 7, 8, 10, 16])
        return {"n": n, "base": base}

    def compute(self, input_value, params):
        n = input_value if input_value is not None else params.get("n", 10)
        base = params.get("base", 10)
        n = abs(n) if n else 1
        base = max(2, min(base, 36))

        if n == 0:
            result = 1
        else:
            result = 0
            val = n
            while val > 0:
                val //= base
                result += 1

        return MethodResult(
            value=result,
            description=f"Number of digits of {n} in base {base} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# GCD ITERATION / BASE DIGIT SUM / DIGIT SUM
# ============================================================================

def _gcd_iteration_step(pair):
    """Single step of GCD Euclidean algorithm."""
    a, b = pair
    if b == 0:
        return (a, 0)
    return (b, a % b)


@register_technique
class GCDIteration(MethodBlock):
    """One step of GCD Euclidean algorithm: (a, b) -> (b, a % b). Returns new_b."""

    def __init__(self):
        super().__init__()
        self.name = "gcd_iteration"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["iterative_process", "gcd", "number_theory"]

    def generate_parameters(self, input_value=None):
        if input_value:
            a = input_value
            b = random.randint(1, max(2, a))
        else:
            a = random.randint(10, 1000)
            b = random.randint(1, a)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        b = params.get("b", 10)
        if b == 0:
            return MethodResult(
                value=a,
                description=f"GCD iteration: ({a}, 0) -> {a} (terminated)",
                metadata={"a": a, "b": 0, "gcd": a}
            )
        new_a, new_b = _gcd_iteration_step((a, b))
        return MethodResult(
            value=new_b,
            description=f"GCD iteration: ({a}, {b}) -> ({new_a}, {new_b})",
            metadata={"old_a": a, "old_b": b, "new_a": new_a, "new_b": new_b}
        )

    def can_invert(self):
        return False


@register_technique
class BaseDigitSum(MethodBlock):
    """Compute sum of digits of n in base b (registered as base_digit_sum)."""

    def __init__(self):
        super().__init__()
        self.name = "base_digit_sum"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "digits", "base_conversion"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        if input_value is not None and input_value > 1000:
            b = random.choice([2, 3, 5])
        else:
            b = random.choice([2, 3, 5, 7, 10, 16])
        return {"n": n, "b": b}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        b = params.get("b", 10)
        digit_sum = 0
        m = abs(n)
        while m > 0:
            digit_sum += m % b
            m //= b
        return MethodResult(
            value=digit_sum,
            description=f"Sum of digits of {n} in base {b} is {digit_sum}",
            metadata={"n": n, "b": b, "digit_sum": digit_sum}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0 and params.get("b", 10) >= 2

    def can_invert(self):
        return True


@register_technique
class DigitSum(MethodBlock):
    """Compute sum of digits in base 10 (registered as digit_sum)."""

    def __init__(self):
        super().__init__()
        self.name = "digit_sum"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "digits"]

    def generate_parameters(self, input_value=None):
        if input_value is None:
            return {"n": random.randint(100, 10000)}
        return {}

    def compute(self, input_value, params):
        n = params.get("n", input_value)
        if n is None:
            raise ValueError("digit_sum requires 'n' parameter or input_value")
        n = abs(n)
        result = sum(int(d) for d in str(n))
        return MethodResult(
            value=result,
            description=f"Sum of the digits of {n} is {result}",
            metadata={"original": n}
        )

    def can_invert(self):
        return False
