"""
Number theory modular order and primitive root techniques.

Implements techniques for:
- Multiplicative order
- Primitive roots
- Mod formatting for answer reduction
"""

import random
import math
from typing import Any, Dict
from sympy import isprime, primerange, gcd
from sympy.ntheory import n_order, primitive_root

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# ORDER/PRIMITIVE ROOTS TECHNIQUES
# ============================================================================

@register_technique
class MultiplicativeOrder(MethodBlock):
    """Compute multiplicative order ord_n(a)."""

    def __init__(self):
        super().__init__()
        self.name = "multiplicative_order"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "order"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            n = random.choice([p for p in primerange(500, 2000)])
        else:
            n = input_value if input_value else random.choice([p for p in primerange(100, 500)])

        a = random.randint(2, n - 1)
        while gcd(a, n) != 1:
            a = random.randint(2, n - 1)
        return {"a": a, "n": n}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        order = int(n_order(a, n))

        return MethodResult(
            value=order,
            description=f"ord_{n}({a}) = {order}",
            metadata={"a": a, "n": n, "order": order}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 1 and gcd(params.get("a", 10), params.get("n", 10)) == 1

    def can_invert(self):
        return True


@register_technique
class MultiplicativeOrderInverseA(MethodBlock):
    """Find a with ord_n(a) = ord."""

    def __init__(self):
        super().__init__()
        self.name = "multiplicative_order_inverse_a"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "order", "inverse"]

    def generate_parameters(self, input_value=None):
        ord_val = input_value if isinstance(input_value, int) and 2 <= input_value <= 20 else random.randint(2, 20)
        candidates = [p for p in primerange(ord_val + 1, ord_val + 200) if (p - 1) % ord_val == 0]
        if not candidates:
            ord_val = random.choice([2, 3, 4, 5, 6])
            candidates = [p for p in primerange(ord_val + 1, ord_val + 200) if (p - 1) % ord_val == 0]
        n = random.choice(candidates)
        return {"ord": ord_val, "n": n}

    def compute(self, input_value, params):
        ord_val = params.get("ord", input_value) if input_value else params.get("ord", 10)
        n = params.get("n", 10)

        for a in range(2, n):
            if gcd(a, n) == 1:
                if n_order(a, n) == ord_val:
                    return MethodResult(
                        value=a,
                        description=f"ord_{n}({a}) = {ord_val}",
                        metadata={"ord": ord_val, "n": n, "a": a}
                    )

        a = 2
        return MethodResult(
            value=a,
            description=f"Using a = {a} (fallback)",
            metadata={"ord": ord_val, "n": n, "a": a}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 1 and params.get("ord", 10) > 0

    def can_invert(self):
        return False


@register_technique
class PrimitiveRoot(MethodBlock):
    """Find smallest primitive root mod p."""

    def __init__(self):
        super().__init__()
        self.name = "primitive_root"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "primitive_root"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            p = random.choice([pr for pr in primerange(1000, 5000)])
        else:
            p = input_value if input_value else random.choice([pr for pr in primerange(100, 1000)])
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 5)

        try:
            g = int(primitive_root(p))
        except Exception:
            g = 2

        return MethodResult(
            value=g,
            description=f"Primitive root mod {p} = {g}",
            metadata={"p": p, "primitive_root": g}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class MultiplicativeOrderModPrime(MethodBlock):
    """Compute multiplicative order ord_p(a) - smallest k such that a^k = 1 (mod p)."""

    def __init__(self):
        super().__init__()
        self.name = "multiplicative_order_mod_prime"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "modular_arithmetic", "order"]

    def generate_parameters(self, input_value=None):
        p = random.choice([7, 11, 13, 17, 19, 23])
        a = random.randint(2, p - 1)
        return {"a": a, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", 2)
        p = params.get("p", 7)

        if not isprime(p):
            return MethodResult(
                value=None,
                description=f"{p} is not prime",
                metadata={"error": "not_prime"}
            )

        if math.gcd(a, p) != 1:
            return MethodResult(
                value=None,
                description=f"gcd({a}, {p}) != 1",
                metadata={"error": "not_coprime"}
            )

        order = n_order(a, p)

        return MethodResult(
            value=order,
            description=f"ord_{p}({a}) = {order} (smallest k where {a}^k = 1 (mod {p}))",
            metadata={"a": a, "p": p, "order": order}
        )

    def validate_params(self, params):
        p = params.get("p", 7)
        a = params.get("a", 2)
        return isprime(p) and 1 < a < p and math.gcd(a, p) == 1

    def can_invert(self):
        return False


# ============================================================================
# MOD TECHNIQUES FOR ANSWER FORMATTING
# ============================================================================

@register_technique
class ModByPowerOfTen(MethodBlock):
    """Reduce integer modulo a power of 10 for answer formatting."""

    def __init__(self):
        super().__init__()
        self.name = "mod_100000"
        self.input_type = "int"
        self.output_type = "int"
        self.difficulty = 1
        self.tags = ["modular", "answer_format"]

    def validate_params(self, params, prev_value=None):
        modulus = params.get("modulus", 100000)
        return modulus > 0

    def generate_parameters(self, input_value=None):
        modulus = random.choice([100000, 10000, 1000])
        return {"n": input_value if input_value is not None else 0, "modulus": modulus}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value is not None else 0)
        modulus = params.get("modulus", 100000)

        result = abs(n) % modulus

        return MethodResult(
            value=result,
            description=f"{n} mod {modulus} = {result}",
            metadata={"n": n, "modulus": modulus, "result": result}
        )

    def can_invert(self):
        return False


@register_technique
class ModByPrime(MethodBlock):
    """Reduce integer modulo a prime for answer formatting."""

    def __init__(self):
        super().__init__()
        self.name = "mod_99991"
        self.input_type = "int"
        self.output_type = "int"
        self.difficulty = 1
        self.tags = ["modular", "answer_format", "prime"]

    def validate_params(self, params, prev_value=None):
        prime = params.get("prime", 99991)
        return prime > 0

    def generate_parameters(self, input_value=None):
        prime = random.choice([99991, 97, 101, 1009, 10007])
        return {"n": input_value if input_value is not None else 0, "prime": prime}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value is not None else 0)
        prime = params.get("prime", 99991)

        result = abs(n) % prime

        return MethodResult(
            value=result,
            description=f"{n} mod {prime} = {result}",
            metadata={"n": n, "prime": prime, "result": result}
        )

    def can_invert(self):
        return False


@register_technique
class ModularPowerResult(MethodBlock):
    """Compute a^b mod m with specified parameters."""

    def __init__(self):
        super().__init__()
        self.name = "modular_power_result"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(2, 50)
        b = random.randint(10, 100)
        m = random.randint(100, 10000)
        return {"a": a, "b": b, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 2)
        b = params.get("b", 10)
        m = params.get("m", 1000)

        result = pow(a, b, m)

        return MethodResult(
            value=result,
            description=f"{a}^{b} = {result} (mod {m})",
            metadata={"a": a, "b": b, "m": m, "result": result}
        )

    def validate_params(self, params):
        return params.get("m", 1) > 0

    def can_invert(self):
        return False


@register_technique
class ModularArithmeticAnalysis(MethodBlock):
    """Analyze modular patterns and cycles (e.g., find cycle length of a^n mod m)."""

    def __init__(self):
        super().__init__()
        self.name = "modular_arithmetic_analysis"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular", "cycles"]

    def generate_parameters(self, input_value=None):
        # Base for the cycle
        a = input_value if input_value else random.randint(2, 20)

        # Modulus (small for tractable cycle detection)
        m = random.choice([p for p in primerange(10, 100)])

        # Ensure gcd(a, m) = 1 for well-defined cycle
        while gcd(a, m) != 1:
            if input_value is not None:
                a = (a % m) + 1
                if a == m:
                    a = 1
            else:
                a = random.randint(2, 20)

        return {"a": a, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        m = params.get("m", 10)

        # Find cycle length (multiplicative order of a mod m)
        cycle_length = 1
        current = a % m

        while current != 1:
            current = (current * a) % m
            cycle_length += 1
            if cycle_length > m:  # Safety check
                break

        return MethodResult(
            value=cycle_length,
            description=f"The multiplicative order of {a} modulo {m} is {cycle_length}",
            params=params,
            metadata={"a": a, "m": m, "order": cycle_length}
        )

    def can_invert(self):
        return False
