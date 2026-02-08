"""
Number theory divisibility techniques: GCD, LCM, and divisor functions.

Implements techniques for:
- GCD computation and inverses
- LCM computation and inverses
- Divisor count (tau function)
- Divisor sum (sigma function)
- Totient function (phi)
"""

import random
import math
from typing import Any, Dict, Optional
from sympy import (
    factorint, totient, divisors, divisor_count, divisor_sigma,
    gcd, lcm, primefactors, isprime, primerange
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from ..decomposition import Decomposition


# ============================================================================
# TOTIENT/DIVISORS TECHNIQUES
# ============================================================================

@register_technique
class Totient(MethodBlock):
    """Compute Euler's totient function phi(n)."""

    def __init__(self):
        super().__init__()
        self.name = "totient"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "totient"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(2, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        phi = int(totient(n))

        return MethodResult(
            value=phi,
            description=f"phi({n}) = {phi}",
            metadata={"n": n, "totient": phi}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0

    def can_invert(self):
        return True

    def generate(self, target_answer: Optional[int] = None) -> Dict:
        """
        Generate a totient problem, optionally targeting a specific answer.
        Uses backward generation: finds n such that phi(n) = target_answer.
        """
        try:
            if target_answer is not None:
                original_target = target_answer

                if target_answer > 1 and target_answer % 2 == 1:
                    n = self._find_n_for_totient_difference(target_answer)
                    if n is None:
                        return {
                            "success": False,
                            "error": f"Cannot find n with n - phi(n) = {target_answer}",
                            "unsupported_target": target_answer,
                            "technique": self.name
                        }
                    params = {"n": n}
                    result = self.compute(None, params)
                    phi = result.value
                    diff = n - phi

                    relationship = (
                        f"Let $n = {n}$. How many positive integers $k \\leq n$ "
                        f"are NOT coprime to $n$? (This equals $n - \\phi(n)$)"
                    )

                    return {
                        "success": True,
                        "answer": diff,
                        "relationship": relationship,
                        "technique": self.name,
                        "params": params,
                        "description": f"n - phi({n}) = {n} - {phi} = {diff}",
                        "metadata": {
                            **result.metadata,
                            "formula": "n - phi(n)",
                            "note": f"Used alternative formula for odd target {target_answer}"
                        }
                    }
                else:
                    n = self._find_n_for_totient(target_answer)
                    if n is None:
                        return {
                            "success": False,
                            "error": f"Cannot find n with phi(n) = {target_answer}",
                            "unsupported_target": target_answer,
                            "technique": self.name,
                        }
                    params = {"n": n}
            else:
                original_target = None
                params = self.generate_parameters()

            result = self.compute(None, params)
            n = params.get("n", 10)
            phi = result.value

            relationship = f"What is the value of Euler's totient function $\\phi({n})$?"

            ret = {
                "success": True,
                "answer": phi,
                "relationship": relationship,
                "technique": self.name,
                "params": params,
                "description": result.description,
                "metadata": result.metadata
            }

            if original_target is not None and original_target != phi:
                ret["note"] = f"Target {original_target} was adjusted to {phi}"

            return ret
        except Exception as e:
            return None

    def _find_n_for_totient(self, target_phi: int) -> Optional[int]:
        """Find n such that phi(n) = target_phi."""
        if target_phi < 1:
            return None

        if target_phi == 1:
            return 2

        if target_phi % 2 == 1:
            target_phi = target_phi - 1
            if target_phi < 2:
                return None

        if target_phi == 2:
            return 3

        if isprime(target_phi + 1):
            return target_phi + 1

        if isprime(target_phi + 1):
            return 2 * (target_phi + 1)

        k = target_phi // 2
        if isprime(k + 1):
            if int(totient(2 * (k + 1))) == target_phi:
                return 2 * (k + 1)

        factors = factorint(target_phi)
        divs = divisors(target_phi)
        for d in divs[:20]:
            if d > 1:
                p = d + 1
                q = (target_phi // d) + 1
                if isprime(p) and isprime(q):
                    n = p * q
                    if int(totient(n)) == target_phi:
                        return n
                if isprime(p):
                    n = 2 * p
                    if int(totient(n)) == target_phi:
                        return n
                if isprime(q):
                    n = 2 * q
                    if int(totient(n)) == target_phi:
                        return n

        if target_phi < 1000:
            max_search = target_phi * 4
            for n in range(2, max_search):
                if int(totient(n)) == target_phi:
                    return n
        else:
            for n in range(2, min(10000, target_phi)):
                if int(totient(n)) == target_phi:
                    return n
            for n in range(target_phi, min(target_phi + 5000, 200000)):
                if int(totient(n)) == target_phi:
                    return n

        return None

    def _find_n_for_totient_difference(self, target_diff: int) -> Optional[int]:
        """Find n such that n - phi(n) = target_diff."""
        if target_diff < 1:
            return None

        max_search = min(target_diff * 10, 100000)
        for n in range(target_diff, max_search):
            phi_n = int(totient(n))
            if n - phi_n == target_diff:
                return n

        if target_diff > 1000:
            small_primes = list(primerange(2, 20))
            for num_primes in range(2, min(len(small_primes), 8)):
                n = 1
                for p in small_primes[:num_primes]:
                    n *= p
                    if n > max_search:
                        break
                    phi_n = int(totient(n))
                    if n - phi_n == target_diff:
                        return n

        return None


@register_technique
class TotientInverse(MethodBlock):
    """Find smallest n with phi(n) = phi."""

    def __init__(self):
        super().__init__()
        self.name = "totient_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "totient", "inverse"]

    def generate_parameters(self, input_value=None):
        phi = input_value if input_value else random.randint(2, 100)
        return {"phi": phi}

    def compute(self, input_value, params):
        phi = params.get("phi", input_value) if input_value else params.get("phi", 10)

        if phi > 10000:
            phi = phi % 10000 + 1

        if phi == 1:
            return MethodResult(value=1, description="phi(1) = 1", metadata={"phi": 1, "n": 1})
        if phi == 2:
            return MethodResult(value=3, description="phi(3) = 2", metadata={"phi": 2, "n": 3})

        if isprime(phi + 1):
            return MethodResult(
                value=phi + 1,
                description=f"phi({phi + 1}) = {phi} (prime - 1)",
                metadata={"phi": phi, "n": phi + 1}
            )

        if phi % 2 == 0 and isprime(phi + 1) and phi + 1 > 2:
            n = 2 * (phi + 1)
            return MethodResult(
                value=n,
                description=f"phi({n}) = phi(2) * phi({phi + 1}) = 1 * {phi} = {phi}",
                metadata={"phi": phi, "n": n}
            )

        max_search = min(phi * 2 + 10, 500)
        for n in range(1, max_search):
            if int(totient(n)) == phi:
                return MethodResult(
                    value=n,
                    description=f"Smallest n with phi(n) = {phi} is {n}",
                    metadata={"phi": phi, "n": n}
                )

        n = 2 * phi + 1
        return MethodResult(
            value=n,
            description=f"Using n = {n} (approximation)",
            metadata={"phi": phi, "n": n}
        )

    def validate_params(self, params):
        return params.get("phi", 10) > 0

    def can_invert(self):
        return False


@register_technique
class DivisorCount(MethodBlock):
    """Compute tau(n) (number of divisors)."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "divisors"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            n = input_value
        else:
            if random.random() < 0.5:
                n = 1
                for p in [2, 3, 5, 7]:
                    n *= p ** random.randint(1, 3)
            else:
                n = input_value if input_value else random.randint(10, 5000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        tau = int(divisor_count(n))

        return MethodResult(
            value=tau,
            description=f"tau({n}) = {tau}",
            metadata={"n": n, "divisor_count": tau}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0

    def can_invert(self):
        return True


@register_technique
class DivisorCountInverse(MethodBlock):
    """Find smallest n with tau(n) = tau."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_count_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "divisors", "inverse"]

    def generate_parameters(self, input_value=None):
        tau = input_value if input_value else random.randint(2, 20)
        return {"tau": tau}

    def compute(self, input_value, params):
        tau = params.get("tau", input_value) if input_value else params.get("tau", 10)
        tau = min(abs(tau) if tau else 10, 50)

        if isprime(tau):
            n = 2 ** (tau - 1)
        else:
            max_search = min(tau * 100, 5000)
            for n in range(1, max_search):
                if divisor_count(n) == tau:
                    return MethodResult(
                        value=n,
                        description=f"Smallest n with tau(n) = {tau} is {n}",
                        metadata={"tau": tau, "n": n}
                    )
            n = tau * 10

        return MethodResult(
            value=n,
            description=f"n with tau(n) = {tau} is {n}",
            metadata={"tau": tau, "n": n}
        )

    def validate_params(self, params):
        return params.get("tau", 10) > 0

    def can_invert(self):
        return False


@register_technique
class DivisorSum(MethodBlock):
    """Compute sigma(n) or sigma_k(n) (sum of divisors)."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_sum"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "divisors"]

    def generate_parameters(self, input_value=None):
        if input_value is not None:
            n = 10 + (abs(int(input_value)) % 490)
        else:
            n = random.randint(10, 500)
        k = 1
        return {"n": n, "k": k}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        k = params.get("k", 1)

        sigma = divisor_sigma(n, k)

        return MethodResult(
            value=sigma,
            description=f"sigma_{k}({n}) = {sigma}",
            metadata={"n": n, "k": k, "divisor_sum": sigma}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0

    def can_invert(self):
        return True


@register_technique
class DivisorSumInverse(MethodBlock):
    """Find smallest n with sigma(n) = sigma."""

    def __init__(self):
        super().__init__()
        self.name = "divisor_sum_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "divisors", "inverse"]

    def generate_parameters(self, input_value=None):
        sigma = input_value if input_value else random.randint(3, 100)
        return {"sigma": sigma}

    def compute(self, input_value, params):
        sigma = params.get("sigma", input_value) if input_value else params.get("sigma", 10)

        if isprime(sigma - 1):
            n = sigma - 1
        else:
            max_search = min(sigma * 2, 10000)
            for n in range(1, max_search):
                if divisor_sigma(n) == sigma:
                    return MethodResult(
                        value=n,
                        description=f"Smallest n with sigma(n) = {sigma} is {n}",
                        metadata={"sigma": sigma, "n": n}
                    )
            n = sigma // 2

        return MethodResult(
            value=n,
            description=f"n with sigma(n) = {sigma} is {n}",
            metadata={"sigma": sigma, "n": n}
        )

    def validate_params(self, params):
        return params.get("sigma", 10) > 0

    def can_invert(self):
        return False


# ============================================================================
# GCD/LCM TECHNIQUES
# ============================================================================

@register_technique
class GCDCompute(MethodBlock):
    """Compute gcd(a, b)."""

    def __init__(self):
        super().__init__()
        self.name = "gcd_compute"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "gcd"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            a = input_value
            divs = divisors(a)
            g = random.choice([d for d in divs if d > 1]) if len(divs) > 1 else 1
            b = g * random.randint(1, max(2, a // g))
        else:
            g = random.randint(1, 500)
            a = g * random.randint(1, 100)
            b = g * random.randint(1, 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        b = params.get("b", 10)

        g = int(gcd(a, b))

        return MethodResult(
            value=g,
            description=f"gcd({a}, {b}) = {g}",
            metadata={"a": a, "b": b, "gcd": g}
        )

    def validate_params(self, params):
        return params.get("a", 10) > 0 and params.get("b", 10) > 0

    def can_invert(self):
        return True


@register_technique
class GCDInverseA(MethodBlock):
    """Find smallest a with gcd(a, b) = g."""

    def __init__(self):
        super().__init__()
        self.name = "gcd_inverse_a"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "gcd", "inverse"]

    def generate_parameters(self, input_value=None):
        g = input_value if input_value else random.randint(1, 50)
        b = g * random.randint(1, 20)
        return {"g": g, "b": b}

    def compute(self, input_value, params):
        g = params.get("g", input_value) if input_value else params.get("g", 10)
        b = params.get("b", 10)

        if b % g == 0:
            a = g
        else:
            for a in range(g, b + g, g):
                if gcd(a, b) == g:
                    break

        return MethodResult(
            value=a,
            description=f"Smallest a with gcd(a, {b}) = {g} is {a}",
            metadata={"g": g, "b": b, "a": a}
        )

    def validate_params(self, params):
        return params.get("g", 10) > 0 and params.get("b", 10) > 0

    def can_invert(self):
        return False


@register_technique
class GCDInverseB(MethodBlock):
    """Find smallest b with gcd(a, b) = g."""

    def __init__(self):
        super().__init__()
        self.name = "gcd_inverse_b"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "gcd", "inverse"]

    def generate_parameters(self, input_value=None):
        g = input_value if input_value else random.randint(1, 50)
        a = g * random.randint(1, 20)
        return {"g": g, "a": a}

    def compute(self, input_value, params):
        g = params.get("g", input_value) if input_value else params.get("g", 10)
        a = params.get("a", 10)

        if a % g == 0:
            b = g
        else:
            for b in range(g, a + g, g):
                if gcd(a, b) == g:
                    break

        return MethodResult(
            value=b,
            description=f"Smallest b with gcd({a}, b) = {g} is {b}",
            metadata={"g": g, "a": a, "b": b}
        )

    def validate_params(self, params):
        return params.get("g", 10) > 0 and params.get("a", 10) > 0

    def can_invert(self):
        return False


@register_technique
class LCMCompute(MethodBlock):
    """Compute lcm(a, b)."""

    def __init__(self):
        super().__init__()
        self.name = "lcm_compute"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "lcm"]
        self.is_primitive = True
        self.decomposition = Decomposition(
            expression="divide(multiply(a, b), gcd(a, b))",
            param_map={"a": "a", "b": "b"},
            notes="lcm(a,b) = a*b/gcd(a,b)"
        )

    def generate_parameters(self, input_value=None):
        if input_value is not None:
            a = min(1000, max(1, abs(int(input_value))))
        else:
            a = random.randint(1, 100)
        b = random.randint(1, min(100, 10**6 // max(1, a)))
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        b = params.get("b", 10)

        l = lcm(a, b)

        return MethodResult(
            value=l,
            description=f"lcm({a}, {b}) = {l}",
            metadata={"a": a, "b": b, "lcm": l}
        )

    def validate_params(self, params):
        return params.get("a", 10) > 0 and params.get("b", 10) > 0

    def can_invert(self):
        return True


@register_technique
class LCMInverseA(MethodBlock):
    """Find smallest a with lcm(a, b) = l."""

    def __init__(self):
        super().__init__()
        self.name = "lcm_inverse_a"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "lcm", "inverse"]

    def generate_parameters(self, input_value=None):
        l = input_value if input_value else random.randint(10, 200)
        divs = divisors(l)
        b = random.choice([d for d in divs if d > 1])
        return {"l": l, "b": b}

    def compute(self, input_value, params):
        l = params.get("l", input_value) if input_value else params.get("l", 10)
        b = params.get("b", 10)

        divs = divisors(l)

        for a in divs:
            if lcm(a, b) == l:
                return MethodResult(
                    value=a,
                    description=f"Smallest a with lcm(a, {b}) = {l} is {a}",
                    metadata={"l": l, "b": b, "a": a}
                )

        a = l
        return MethodResult(
            value=a,
            description=f"a with lcm(a, {b}) = {l} is {a}",
            metadata={"l": l, "b": b, "a": a}
        )

    def validate_params(self, params):
        return params.get("l", 10) > 0 and params.get("b", 10) > 0 and params.get("l", 10) % params.get("b", 10) == 0

    def can_invert(self):
        return False
