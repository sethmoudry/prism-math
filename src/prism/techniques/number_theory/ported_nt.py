"""
Ported number theory techniques.

Methods ported:
- lcm_multiplicative_order_backward
- sieve_count
- carmichael_condition_primes
- fermat_condition_primes
"""

import random
import math
from typing import Any, Dict, Optional
from itertools import combinations
from sympy import (
    factorint, gcd, lcm, isprime, nextprime, primefactors,
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# LCM MULTIPLICATIVE ORDER BACKWARD
# ============================================================================

@register_technique
class LcmMultiplicativeOrderBackward(MethodBlock):
    """
    Backward generation: Find primes p, q and bases a, b such that
    lcm(ord_p(a), ord_q(b)) = target_answer.

    This solves: Find smallest n such that a^n = 1 (mod p) and b^n = 1 (mod q).
    The answer is lcm(ord_p(a), ord_q(b)).
    """

    def __init__(self):
        super().__init__()
        self.name = "lcm_multiplicative_order_backward"
        self.input_type = "integer"  # target answer
        self.output_type = "params"  # returns (a, p, b, q)
        self.difficulty = 6
        self.tags = ["number_theory", "order", "backward", "lcm"]

    def generate_parameters(self, input_value=None):
        target = input_value if input_value else random.randint(10, 500)
        return {"target": target}

    def compute(self, input_value, params):
        target = params.get("target", input_value) if input_value else params.get("target", 10)

        from sympy import divisors
        from sympy.ntheory import n_order

        # Get divisors of target
        divs = sorted(divisors(target))

        # Try to split into two factors that we can use as orders
        # We want ord1 and ord2 such that lcm(ord1, ord2) = target
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            # Pick two divisors
            if len(divs) >= 2:
                ord1 = random.choice(divs[1:])
                ord2 = target // gcd(ord1, target)
                if lcm(ord1, ord2) != target:
                    ord2 = target // ord1 if target % ord1 == 0 else target
            else:
                # Target is prime or has few divisors
                ord1 = target
                ord2 = target

            # Find prime p such that phi(p) is divisible by ord1
            # For prime p, phi(p) = p-1
            # So we need p-1 divisible by ord1, i.e., p = k*ord1 + 1 for some k
            p = None
            for k in range(1, 20):
                candidate = k * ord1 + 1
                if isprime(candidate) and candidate > ord1:
                    p = candidate
                    break

            if p is None:
                attempts += 1
                continue

            # Find prime q such that phi(q) is divisible by ord2
            q = None
            for k in range(1, 20):
                candidate = k * ord2 + 1
                if isprime(candidate) and candidate > ord2 and candidate != p:
                    q = candidate
                    break

            if q is None:
                attempts += 1
                continue

            # Find a with ord_p(a) = ord1
            a = None
            for candidate in range(2, min(p, 50)):
                if gcd(candidate, p) == 1 and n_order(candidate, p) == ord1:
                    a = candidate
                    break

            if a is None:
                attempts += 1
                continue

            # Find b with ord_q(b) = ord2
            b = None
            for candidate in range(2, min(q, 50)):
                if gcd(candidate, q) == 1 and n_order(candidate, q) == ord2:
                    b = candidate
                    break

            if b is None:
                attempts += 1
                continue

            # Success! Verify
            actual_ord1 = n_order(a, p)
            actual_ord2 = n_order(b, q)
            actual_answer = lcm(actual_ord1, actual_ord2)

            if actual_answer == target:
                return MethodResult(
                    value=target,
                    description=f"Find smallest n: {a}^n = 1 (mod {p}) and {b}^n = 1 (mod {q})",
                    params=params,
                    metadata={
                        "a": a, "p": p, "b": b, "q": q,
                        "ord_p_a": actual_ord1,
                        "ord_q_b": actual_ord2,
                        "answer": actual_answer
                    }
                )

            attempts += 1

        # Fallback: use simpler construction
        from sympy.ntheory import primitive_root

        p = nextprime(target)
        while (p - 1) % target != 0:
            p = nextprime(p)

        q = nextprime(p)
        while (q - 1) % target != 0 or q == p:
            q = nextprime(q)

        g_p = primitive_root(p)
        a = pow(g_p, (p - 1) // target, p)

        g_q = primitive_root(q)
        b = pow(g_q, (q - 1) // target, q)

        # Verify
        from sympy.ntheory import n_order
        actual_ord1 = n_order(a, p)
        actual_ord2 = n_order(b, q)
        actual_answer = lcm(actual_ord1, actual_ord2)

        return MethodResult(
            value=actual_answer,
            description=f"Find smallest n: {a}^n = 1 (mod {p}) and {b}^n = 1 (mod {q})",
            params=params,
            metadata={
                "a": a, "p": p, "b": b, "q": q,
                "ord_p_a": actual_ord1,
                "ord_q_b": actual_ord2,
                "answer": actual_answer,
                "note": "Used fallback construction"
            }
        )

    def validate_params(self, params):
        return params.get("target", 0) > 0

    def can_invert(self):
        return False


# ============================================================================
# SIEVE COUNT
# ============================================================================

@register_technique
class SieveCount(MethodBlock):
    """Count integers in [1,n] coprime to given primes using sieve/PIE."""

    def __init__(self):
        super().__init__()
        self.name = "sieve_count"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "sieve", "number_theory"]

    def generate_parameters(self, input_value=None):
        # Dynamic: use input to drive n for larger coprime counts
        if input_value is not None and input_value < 500:
            # Small input -> scale to larger range
            n = input_value * random.randint(10, 50)
        elif input_value is not None:
            n = input_value
        else:
            n = random.randint(500, 5000)
        primes = [2, 3, 5, 7][:random.randint(2, 3)]
        return {"n": n, "primes": primes}

    def compute(self, input_value, params):
        n = params.get("n", input_value)
        primes = params.get("primes", [])

        # If primes is empty, return n (all integers in [1,n] are trivially coprime)
        if not primes:
            description = f"Integers in [1,{n}] coprime to empty set: {n}"
            return MethodResult(value=n, description=description, params=params,
                                metadata={"n": n, "primes": primes})

        count = n
        for r in range(1, len(primes) + 1):
            for combo in combinations(primes, r):
                prod = 1
                for p in combo:
                    prod *= p
                multiples = n // prod
                if r % 2 == 1:
                    count -= multiples
                else:
                    count += multiples
        description = f"Integers in [1,{n}] coprime to {primes}: {count}"
        return MethodResult(value=count, description=description, params=params,
                            metadata={"n": n, "primes": primes})

    def can_invert(self):
        return False


# ============================================================================
# CARMICHAEL CONDITION PRIMES
# ============================================================================

@register_technique
class CarmichaelConditionPrimes(MethodBlock):
    """Check Carmichael condition: n is composite and p-1 | n-1 for all prime divisors p."""

    def __init__(self):
        super().__init__()
        self.name = "carmichael_condition_primes"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "carmichael", "pseudoprimes"]

    def generate_parameters(self, input_value=None):
        # Common Carmichael numbers: 561, 1105, 1729, 2465, 2821
        if input_value:
            n = input_value
        else:
            n = random.choice([561, 1105, 1729, 2465, 2821, 100, 200])
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 561)

        if n <= 1:
            return MethodResult(
                value=0,
                description=f"{n} is not a valid candidate",
                params=params,
                metadata={"n": n, "is_carmichael": False}
            )

        # Check if n is composite
        if isprime(n):
            return MethodResult(
                value=0,
                description=f"{n} is prime, not composite",
                params=params,
                metadata={"n": n, "is_carmichael": False, "is_prime": True}
            )

        # Check if p-1 | n-1 for all prime divisors p
        primes = primefactors(n)
        is_carmichael = True

        for p in primes:
            if (n - 1) % (p - 1) != 0:
                is_carmichael = False
                break

        result = 1 if is_carmichael else 0

        return MethodResult(
            value=result,
            description=f"{n} {'satisfies' if is_carmichael else 'does not satisfy'} Carmichael condition (primes: {primes})",
            params=params,
            metadata={"n": n, "is_carmichael": is_carmichael, "prime_divisors": primes}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 1

    def can_invert(self):
        return False


# ============================================================================
# FERMAT CONDITION PRIMES
# ============================================================================

@register_technique
class FermatConditionPrimes(MethodBlock):
    """Check if n is a Fermat number: F_k = 2^(2^k) + 1."""

    def __init__(self):
        super().__init__()
        self.name = "fermat_condition_primes"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "fermat_numbers", "special_forms"]

    def generate_parameters(self, input_value=None):
        if input_value:
            n = input_value
        else:
            # Fermat numbers: F_0=3, F_1=5, F_2=17, F_3=257, F_4=65537
            k = random.randint(0, 4)
            n = 2**(2**k) + 1
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 5)

        if n <= 2:
            return MethodResult(
                value=0,
                description=f"{n} is not a Fermat number",
                params=params,
                metadata={"n": n, "is_fermat": False}
            )

        # Check if n-1 is a power of 2
        m = n - 1
        if m & (m - 1) != 0:  # Not a power of 2
            return MethodResult(
                value=0,
                description=f"{n} is not a Fermat number ({n-1} is not a power of 2)",
                params=params,
                metadata={"n": n, "is_fermat": False}
            )

        # Check if m = 2^k where k is a power of 2
        k = int(math.log2(m))
        if 2**k != m:
            return MethodResult(
                value=0,
                description=f"{n} is not a Fermat number",
                params=params,
                metadata={"n": n, "is_fermat": False}
            )

        # Check if k is a power of 2
        if k > 0 and k & (k - 1) != 0:
            return MethodResult(
                value=0,
                description=f"{n} is not a Fermat number (exponent {k} is not a power of 2)",
                params=params,
                metadata={"n": n, "is_fermat": False}
            )

        # n = 2^(2^j) + 1 for some j
        j = int(math.log2(k)) if k > 0 else 0

        return MethodResult(
            value=1,
            description=f"{n} = F_{j} = 2^(2^{j}) + 1 (Fermat number)",
            params=params,
            metadata={"n": n, "is_fermat": True, "fermat_index": j}
        )

    def validate_params(self, params):
        return params.get("n", 1) > 0

    def can_invert(self):
        return False
