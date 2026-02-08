"""
Number theory prime-related techniques.

Implements techniques for:
- Prime factorization
- Prime counting (pi function)
- Prime divisors
- Prime power checking
- Primality-related operations
"""

import random
import math
from typing import Any, Dict, Optional
from sympy import (
    factorint, primefactors, isprime, nextprime, prevprime,
    primerange, prime, divisors
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class PrimeFactorization(MethodBlock):
    """Compute prime factorization of n."""

    def __init__(self):
        super().__init__()
        self.name = "prime_factorization"
        self.input_type = "integer"
        self.output_type = "dict"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "factorization"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 12)

        if n <= 1:
            return MethodResult(
                value=0,  # Return 0 for no prime factors, not empty dict
                description=f"{n} has no prime factorization",
                metadata={"n": n, "factors": {}, "num_distinct_primes": 0}
            )

        factors = factorint(n)
        num_distinct_primes = len(factors)

        factor_str = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items()))

        return MethodResult(
            value=num_distinct_primes,  # Return count of distinct primes, not dict
            description=f"{n} = {factor_str} ({num_distinct_primes} distinct prime factors)",
            metadata={"n": n, "factors": factors, "num_distinct_primes": num_distinct_primes}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False


@register_technique
class PrimeCountingFunction(MethodBlock):
    """Count primes <= n using pi(n) - the prime counting function."""

    def __init__(self):
        super().__init__()
        self.name = "prime_counting_function"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "counting"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 100)
        n = min(abs(n) if n else 100, 5000)

        count = len(list(primerange(2, n + 1)))

        return MethodResult(
            value=count,
            description=f"pi({n}) = {count} (number of primes <= {n})",
            metadata={"n": n, "prime_count": count}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False


@register_technique
class SmallestPrimeDivisor(MethodBlock):
    """Find the smallest prime divisor of n."""

    def __init__(self):
        super().__init__()
        self.name = "smallest_prime_divisor"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "divisors"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 10000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 12)

        if n <= 1:
            return MethodResult(
                value=None,
                description=f"{n} has no prime divisors",
                metadata={"n": n, "error": "no_prime_divisors"}
            )

        factors = primefactors(n)
        if not factors:
            return MethodResult(
                value=n,
                description=f"{n} is prime, smallest prime divisor is {n}",
                metadata={"n": n, "is_prime": True}
            )

        smallest = min(factors)

        return MethodResult(
            value=smallest,
            description=f"Smallest prime divisor of {n} is {smallest}",
            metadata={"n": n, "smallest_prime_divisor": smallest}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 1

    def can_invert(self):
        return False


@register_technique
class PrimesBelow(MethodBlock):
    """List all primes < n."""

    def __init__(self):
        super().__init__()
        self.name = "primes_below"
        self.input_type = "integer"
        self.output_type = "list"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "listing"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(20, 200)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 100)
        n = min(abs(n) if n else 100, 10000)
        if n < 2:
            n = 100

        primes = list(primerange(2, n))
        count = len(primes)

        return MethodResult(
            value=count,
            description=f"Primes below {n}: {primes[:10]}{'...' if count > 10 else ''} (total: {count})",
            metadata={"n": n, "primes": primes, "count": count}
        )

    def validate_params(self, params):
        return params.get("n", 2) >= 2

    def can_invert(self):
        return False


@register_technique
class MinPrimeSatisfyingConstraint(MethodBlock):
    """Find smallest prime p satisfying p > n."""

    def __init__(self):
        super().__init__()
        self.name = "min_prime_satisfying_constraint"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "search"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        p = nextprime(n)

        return MethodResult(
            value=p,
            description=f"Smallest prime > {n} is {p}",
            metadata={"n": n, "next_prime": p}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return True


@register_technique
class CheckPrimePowerExponent(MethodBlock):
    """Check if n = p^k for some prime p and k >= 1, return k if true."""

    def __init__(self):
        super().__init__()
        self.name = "check_prime_power_exponent"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "prime_power", "factorization"]

    def generate_parameters(self, input_value=None):
        if input_value:
            n = input_value
        else:
            p = random.choice([2, 3, 5, 7, 11])
            k = random.randint(1, 5)
            n = p ** k
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 8)

        if n <= 1:
            return MethodResult(
                value=0,
                description=f"{n} is not a prime power",
                metadata={"n": n, "is_prime_power": False}
            )

        factors = factorint(n)

        if len(factors) == 1:
            p, k = list(factors.items())[0]
            return MethodResult(
                value=k,
                description=f"{n} = {p}^{k} (prime power with exponent {k})",
                metadata={"n": n, "is_prime_power": True, "prime": p, "exponent": k}
            )
        else:
            return MethodResult(
                value=0,
                description=f"{n} is not a prime power (has {len(factors)} distinct prime factors)",
                metadata={"n": n, "is_prime_power": False, "num_primes": len(factors)}
            )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False


@register_technique
class SquareFreeDivisors(MethodBlock):
    """Find all squarefree divisors of n."""

    def __init__(self):
        super().__init__()
        self.name = "square_free_divisors"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "divisors", "squarefree"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 500)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 12)

        if n <= 0:
            return MethodResult(
                value=0,
                description=f"{n} has no squarefree divisors",
                metadata={"error": "invalid_n"}
            )

        all_divs = divisors(n)

        squarefree_divs = []
        for d in all_divs:
            factors = factorint(d)
            if all(exp == 1 for exp in factors.values()):
                squarefree_divs.append(d)

        count = len(squarefree_divs)

        return MethodResult(
            value=count,
            description=f"Squarefree divisors of {n}: {squarefree_divs} (count: {count})",
            metadata={"n": n, "squarefree_divisors": squarefree_divs, "count": count}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False


@register_technique
class FilterByPrimeFactorCount(MethodBlock):
    """Filter integers by number of distinct prime factors."""

    def __init__(self):
        super().__init__()
        self.name = "filter_by_prime_factor_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "counting"]

    def generate_parameters(self, input_value=None):
        upper = input_value if input_value else random.randint(50, 500)
        target_count = random.randint(1, 4)
        return {"upper": upper, "target_count": target_count}

    def compute(self, input_value, params):
        upper = params.get("upper", input_value) if input_value else params.get("upper", 100)
        target_count = params.get("target_count", 2)

        upper = min(abs(upper) if upper else 100, 10000)

        matches = []
        for n in range(2, upper + 1):
            if len(primefactors(n)) == target_count:
                matches.append(n)

        count = len(matches)

        return MethodResult(
            value=count,
            description=f"Numbers <= {upper} with exactly {target_count} distinct prime factors: count = {count}",
            metadata={"upper": upper, "target_count": target_count, "count": count, "examples": matches[:10]}
        )

    def validate_params(self, params):
        return params.get("upper", 1) > 1 and params.get("target_count", 1) > 0

    def can_invert(self):
        return False


@register_technique
class PrimesOfForm2kPlus1(MethodBlock):
    """Find primes of form 2k+1 (odd primes) in range."""

    def __init__(self):
        super().__init__()
        self.name = "primes_of_form_2k_plus_1"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes"]

    def generate_parameters(self, input_value=None):
        upper = input_value if input_value else random.randint(50, 500)
        return {"upper": upper}

    def compute(self, input_value, params):
        upper = params.get("upper", input_value) if input_value else params.get("upper", 100)
        upper = min(abs(upper) if upper else 100, 10000)

        primes = [p for p in primerange(3, upper + 1)]
        count = len(primes)

        return MethodResult(
            value=count,
            description=f"Odd primes <= {upper}: {primes[:10]}{'...' if count > 10 else ''} (count: {count})",
            metadata={"upper": upper, "primes": primes, "count": count}
        )

    def validate_params(self, params):
        return params.get("upper", 3) >= 3

    def can_invert(self):
        return False


@register_technique
class DistinctPrimeDivisorsEqualsB(MethodBlock):
    """Count n <= upper with exactly b distinct prime divisors."""

    def __init__(self):
        super().__init__()
        self.name = "distinct_prime_divisors_equals_b"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "counting"]

    def generate_parameters(self, input_value=None):
        upper = input_value if input_value else random.randint(50, 500)
        b = random.randint(1, 3)
        return {"upper": upper, "b": b}

    def compute(self, input_value, params):
        upper = params.get("upper", input_value) if input_value else params.get("upper", 100)
        b = params.get("b", 2)

        upper = min(abs(upper) if upper else 100, 10000)

        count = 0
        for n in range(2, upper + 1):
            if len(primefactors(n)) == b:
                count += 1

        return MethodResult(
            value=count,
            description=f"Count of n <= {upper} with omega(n) = {b}: {count}",
            metadata={"upper": upper, "b": b, "count": count}
        )

    def validate_params(self, params):
        return params.get("upper", 1) > 1 and params.get("b", 1) > 0

    def can_invert(self):
        return False


@register_technique
class AllowedPrimeDivisors(MethodBlock):
    """Count integers in range with all prime divisors from allowed set."""

    def __init__(self):
        super().__init__()
        self.name = "allowed_prime_divisors"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "primes", "counting"]

    def generate_parameters(self, input_value=None):
        upper = input_value if input_value else random.randint(50, 500)
        allowed = sorted(random.sample([2, 3, 5, 7, 11, 13], random.randint(2, 4)))
        return {"upper": upper, "allowed": allowed}

    def compute(self, input_value, params):
        upper = params.get("upper", input_value) if input_value else params.get("upper", 100)
        allowed = params.get("allowed", [2, 3, 5])

        upper = min(abs(upper) if upper else 100, 10000)
        allowed_set = set(allowed)

        count = 0
        for n in range(1, upper + 1):
            factors = set(primefactors(n))
            if factors.issubset(allowed_set):
                count += 1

        return MethodResult(
            value=count,
            description=f"Count of n <= {upper} with prime divisors in {allowed}: {count}",
            metadata={"upper": upper, "allowed": allowed, "count": count}
        )

    def validate_params(self, params):
        return params.get("upper", 1) > 0 and len(params.get("allowed", [])) > 0

    def can_invert(self):
        return False


@register_technique
class NumberFromPrimePowers(MethodBlock):
    """Construct n = p1^a1 * p2^a2 * ... from prime power specification."""

    def __init__(self):
        super().__init__()
        self.name = "number_from_prime_powers"
        self.input_type = "dict"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "primes", "construction"]

    def generate_parameters(self, input_value=None):
        num_primes = random.randint(1, 4)
        primes = sorted(random.sample([2, 3, 5, 7, 11, 13, 17, 19], num_primes))
        powers = {p: random.randint(1, 4) for p in primes}
        return {"powers": powers}

    def compute(self, input_value, params):
        powers = params.get("powers", {2: 3, 3: 2})

        n = 1
        for p, e in powers.items():
            n *= p ** e

        factor_str = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(powers.items()))

        return MethodResult(
            value=n,
            description=f"{factor_str} = {n}",
            metadata={"powers": powers, "n": n}
        )

    def validate_params(self, params):
        powers = params.get("powers", {})
        return all(isprime(p) and e > 0 for p, e in powers.items())

    def can_invert(self):
        return True


@register_technique
class Zsigmondy(MethodBlock):
    """Find primitive prime divisor of a^n - 1."""

    def __init__(self):
        super().__init__()
        self.name = "zsigmondy"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "zsigmondy"]

    def generate_parameters(self, input_value=None):
        a = random.randint(2, 10)
        n = input_value if input_value else random.randint(3, 20)
        return {"a": a, "n": n}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        # Bound n to prevent huge a^n - 1 values that are slow to factor
        n = min(abs(n) if n else 10, 25)
        a = min(abs(a) if a else 2, 10)

        # Find prime p that divides a^n - 1 but not a^k - 1 for k < n
        val = a**n - 1
        primes = primefactors(val)

        for p in primes:
            is_primitive = True
            for k in range(1, n):
                if (a**k - 1) % p == 0:
                    is_primitive = False
                    break
            if is_primitive:
                return MethodResult(
                    value=p,
                    description=f"Primitive prime divisor of {a}^{n} - 1 is {p}",
                    metadata={"a": a, "n": n, "p": p}
                )

        # No primitive divisor (exceptional cases)
        p = primes[0] if primes else 2
        return MethodResult(
            value=p,
            description=f"Using prime {p} (may not be primitive)",
            metadata={"a": a, "n": n, "p": p}
        )

    def validate_params(self, params):
        return params.get("a", 10) > 1 and params.get("n", 10) > 0

    def can_invert(self):
        return True
