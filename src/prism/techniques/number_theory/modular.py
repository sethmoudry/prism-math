"""
Number theory modular arithmetic techniques.

Implements techniques for:
- Modular exponentiation
- Modular inverse
- Chinese Remainder Theorem
- Fermat's little theorem
- Euler's theorem
"""

import random
from typing import Any, Dict, Optional
from sympy import (
    gcd, isprime, primerange, totient, mod_inverse,
)
from sympy.ntheory.modular import crt

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class ModularExponentiation(MethodBlock):
    """Compute a^n mod m."""

    def __init__(self):
        super().__init__()
        self.name = "modular_exponentiation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(2, 100)
        n = random.randint(1, 100)
        m = random.randint(10, 1000)
        return {"a": a, "n": n, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        n = params.get("n", 10)
        m = params.get("m", 10)

        result = pow(a, n, m)

        return MethodResult(
            value=result,
            description=f"{a}^{n} = {result} (mod {m})",
            metadata={"a": a, "n": n, "m": m, "result": result}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 0

    def can_invert(self):
        return True


@register_technique
class ModExpInverseBase(MethodBlock):
    """Find a where a^n = r (mod m) - discrete root."""

    def __init__(self):
        super().__init__()
        self.name = "mod_exp_inverse_base"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "modular", "inverse"]

    def generate_parameters(self, input_value=None):
        r = input_value if input_value else random.randint(1, 50)
        n = random.choice([2, 3, 5])

        if input_value is not None and input_value > 1000:
            m = random.choice([p for p in primerange(500, 5000)])
        else:
            m = random.choice([p for p in primerange(10, 100)])
        return {"r": r, "n": n, "m": m}

    def compute(self, input_value, params):
        r = params.get("r", input_value) if input_value else params.get("r", 5)
        n = params.get("n", 10)
        m = params.get("m", 10)

        result = 1
        for a in range(1, m):
            if pow(a, n, m) == r % m:
                result = a
                break

        return MethodResult(
            value=result,
            description=f"{result}^{n} = {r} (mod {m})",
            metadata={"r": r, "n": n, "m": m, "a": result}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 1 and params.get("n", 10) > 0

    def can_invert(self):
        return False


@register_technique
class ModExpInverseExp(MethodBlock):
    """Find n where a^n = r (mod m) - discrete log."""

    def __init__(self):
        super().__init__()
        self.name = "mod_exp_inverse_exp"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "modular", "discrete_log"]

    def generate_parameters(self, input_value=None):
        r = input_value if input_value else random.randint(1, 50)
        a = random.randint(2, 10)

        if input_value is not None and input_value > 1000:
            m = random.choice([p for p in primerange(500, 5000)])
        else:
            m = random.choice([p for p in primerange(10, 100)])
        return {"r": r, "a": a, "m": m}

    def compute(self, input_value, params):
        r = params.get("r", input_value) if input_value else params.get("r", 5)
        a = params.get("a", 10)
        m = params.get("m", 10)

        result = 1
        for n in range(1, m):
            if pow(a, n, m) == r % m:
                result = n
                break

        return MethodResult(
            value=result,
            description=f"{a}^{result} = {r} (mod {m})",
            metadata={"r": r, "a": a, "m": m, "n": result}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 1 and params.get("a", 10) > 0

    def can_invert(self):
        return False


@register_technique
class ModInverse(MethodBlock):
    """Compute a^(-1) mod m."""

    def __init__(self):
        super().__init__()
        self.name = "mod_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(2, 100)

        if input_value is not None and input_value > 1000:
            m = random.choice([p for p in primerange(1000, 10000)])
        else:
            m = random.choice([p for p in primerange(100, 5000)])

        while gcd(a, m) != 1:
            if input_value is not None:
                a = (a + 1) % m
                if a == 0:
                    a = 1
            else:
                a = random.randint(2, 100)
        return {"a": a, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        m = params.get("m", 10)

        inv = mod_inverse(a, m)

        return MethodResult(
            value=inv,
            description=f"{a}^(-1) = {inv} (mod {m})",
            metadata={"a": a, "m": m, "inverse": inv}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 1 and gcd(params.get("a", 10), params.get("m", 10)) == 1

    def _find_params_for_answer(self, target_answer: int) -> Optional[Dict[str, Any]]:
        """Find (a, m) such that a^(-1) = target_answer (mod m)."""
        if target_answer <= 0:
            return None

        min_prime = max(100, target_answer + 10)
        max_prime = min(10000, target_answer + 5000)

        primes = list(primerange(min_prime, max_prime))
        if not primes:
            primes = list(primerange(100, 5000))

        random.shuffle(primes)

        for m in primes[:20]:
            try:
                if gcd(target_answer, m) != 1:
                    continue

                a = mod_inverse(target_answer, m)
                inv = mod_inverse(a, m)
                if inv == target_answer:
                    return {"a": a, "m": m}
            except (ValueError, ZeroDivisionError):
                continue

        return None

    def can_invert(self):
        return True


@register_technique
class ChineseRemainderSolve(MethodBlock):
    """Solve Chinese Remainder Theorem system."""

    def __init__(self):
        super().__init__()
        self.name = "chinese_remainder_solve"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "modular", "crt"]

    def generate_parameters(self, input_value=None):
        primes = list(primerange(5, 50))
        random.shuffle(primes)
        moduli = primes[:random.randint(2, 3)]
        residues = [random.randint(0, m - 1) for m in moduli]
        return {"moduli": moduli, "residues": residues}

    def compute(self, input_value, params):
        moduli = params.get("moduli", [7, 11])
        residues = params.get("residues", [3, 5])

        if isinstance(moduli, int):
            moduli = [moduli]
        if isinstance(residues, int):
            residues = [residues]

        if len(moduli) < 2:
            primes = list(primerange(5, 50))
            random.shuffle(primes)
            while len(moduli) < 2:
                p = primes.pop()
                if all(gcd(p, m) == 1 for m in moduli):
                    moduli.append(p)
                    residues.append(random.randint(0, p - 1))

        result, mod = crt(moduli, residues)

        return MethodResult(
            value=result,
            description=f"CRT solution: x = {result} (mod {mod})",
            metadata={"moduli": moduli, "residues": residues, "solution": result, "mod": mod}
        )

    def validate_params(self, params):
        moduli = params.get("moduli", [7, 11])
        if isinstance(moduli, int):
            moduli = [moduli]
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                if gcd(moduli[i], moduli[j]) != 1:
                    return False
        return True

    def can_invert(self):
        return True


@register_technique
class ChineseRemainderInverseResidue(MethodBlock):
    """Find one residue given x and other constraints."""

    def __init__(self):
        super().__init__()
        self.name = "chinese_remainder_inverse_residue"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "modular", "crt", "inverse"]

    def generate_parameters(self, input_value=None):
        x = input_value if input_value else random.randint(10, 500)

        if input_value is not None and input_value > 1000:
            primes = list(primerange(100, 1000))
        else:
            primes = list(primerange(5, 100))

        random.shuffle(primes)
        moduli = primes[:2]
        return {"x": x, "moduli": moduli}

    def compute(self, input_value, params):
        x = params.get("x", input_value) if input_value else params.get("x", 5)
        moduli = params.get("moduli", 10)

        residues = [x % m for m in moduli]
        result = residues[0]

        return MethodResult(
            value=result,
            description=f"{x} = {result} (mod {moduli[0]})",
            metadata={"x": x, "moduli": moduli, "residues": residues, "result": result}
        )

    def validate_params(self, params):
        return params.get("x", 5) > 0 and len(params.get("moduli", 10)) > 0

    def can_invert(self):
        return False


@register_technique
class FermatReduce(MethodBlock):
    """Reduce a^n mod p using Fermat's little theorem."""

    def __init__(self):
        super().__init__()
        self.name = "fermat_reduce"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular", "fermat"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(2, 100)
        p = random.choice([p for p in primerange(10, 100)])
        n = random.randint(p, p * 10)
        return {"a": a, "n": n, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        n = params.get("n", 10)
        p = params.get("p", 5)

        if gcd(a, p) == 1:
            reduced_exp = n % (p - 1)
            result = pow(a, reduced_exp, p)
        else:
            result = 0

        return MethodResult(
            value=result,
            description=f"{a}^{n} = {result} (mod {p}) by Fermat",
            metadata={"a": a, "n": n, "p": p, "result": result}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class EulerReduce(MethodBlock):
    """Reduce a^n mod m using Euler's theorem."""

    def __init__(self):
        super().__init__()
        self.name = "euler_reduce"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "modular", "euler"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(2, 100)

        if input_value is not None and input_value > 1000:
            m = random.randint(1000, 10000)
        else:
            m = random.randint(100, 5000)

        while gcd(a, m) != 1:
            if input_value is not None:
                a = (a + 1) % m
                if a == 0:
                    a = 1
            else:
                a = random.randint(2, 100)

        phi_m = int(totient(m))
        n = random.randint(phi_m, phi_m * 10)
        return {"a": a, "n": n, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        n = params.get("n", 10)
        m = params.get("m", 10)

        phi_m = int(totient(m))
        reduced_exp = n % phi_m
        result = pow(a, reduced_exp, m)

        return MethodResult(
            value=result,
            description=f"{a}^{n} = {result} (mod {m}) by Euler",
            metadata={"a": a, "n": n, "m": m, "phi_m": phi_m, "result": result}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 1 and gcd(params.get("a", 10), params.get("m", 10)) == 1

    def can_invert(self):
        return True


@register_technique
class SolveLinearCongruence(MethodBlock):
    """Solve ax = b (mod m) for x."""

    def __init__(self):
        super().__init__()
        self.name = "solve_linear_congruence"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "modular", "congruence"]

    def validate_params(self, params, prev_value=None):
        """Modular inverse requires gcd(a, m) == 1."""
        a = params.get("a", prev_value) if prev_value is not None else params.get("a")
        m = params.get("m")
        if a is None or m is None:
            return False
        return gcd(int(a), int(m)) == 1

    def generate_parameters(self, input_value=None):
        m = random.choice([p for p in primerange(100, 500)])
        a = random.randint(2, m - 1)
        while gcd(a, m) != 1:
            a = random.randint(2, m - 1)
        b = input_value if input_value else random.randint(1, m - 1)
        return {"a": a, "b": b, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        b = params.get("b", input_value) if input_value else params.get("b", 10)
        m = params.get("m", 10)

        a_inv = mod_inverse(a, m)
        x = (b * a_inv) % m

        return MethodResult(
            value=x,
            description=f"Solution to {a}x = {b} (mod {m}) is x = {x} (mod {m})",
            metadata={"a": a, "b": b, "m": m, "solution": x}
        )

    def can_invert(self):
        return True
