"""
Number theory quadratic residue techniques.

Implements techniques for:
- Legendre symbol
- Jacobi symbol
- Square roots modulo p
- Quadratic reciprocity
- Two squares decomposition
"""

import random
import math
from typing import Any, Dict, Optional
from sympy import (
    isprime, legendre_symbol, jacobi_symbol, sqrt_mod,
    primerange, primefactors, mod_inverse
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class LegendreSymbol(MethodBlock):
    """Compute Legendre symbol (a/p)."""

    def __init__(self):
        super().__init__()
        self.name = "legendre_symbol"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "quadratic_residues"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(1, 10000)

        if input_value is not None and input_value > 1000:
            p = random.choice([pr for pr in primerange(100, 1000)])
        else:
            p = random.choice([pr for pr in primerange(5, 200)])
        return {"a": a, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        p = params.get("p", 5)

        leg = int(legendre_symbol(a, p))
        result = leg

        return MethodResult(
            value=result,
            description=f"Legendre symbol ({a}/{p}) = {result}",
            metadata={"a": a, "p": p, "legendre": result}
        )

    def validate_params(self, params, prev_value=None):
        p = params.get("p")
        return p is not None and isprime(p) and p > 2

    def can_invert(self):
        return True


@register_technique
class SqrtModP(MethodBlock):
    """Find x where x^2 = a (mod p)."""

    def __init__(self):
        super().__init__()
        self.name = "sqrt_mod_p"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "quadratic_residues"]

    def generate_parameters(self, input_value=None):
        p = random.choice([pr for pr in primerange(5, 100)])
        a = input_value if input_value else random.randint(1, p - 1)
        while legendre_symbol(a, p) != 1:
            a = random.randint(1, p - 1)
        return {"a": a, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        p = params.get("p", 5)

        try:
            x = sqrt_mod(a, p)
            if x is None:
                x = 0
        except Exception:
            x = 0

        return MethodResult(
            value=x,
            description=f"x^2 = {a} (mod {p}), x = {x}",
            metadata={"a": a, "p": p, "x": x}
        )

    def validate_params(self, params, prev_value=None):
        p = params.get("p", 5)
        a = params.get("a", prev_value) if prev_value is not None else params.get("a", 0)
        return isprime(p) and p > 2 and a is not None and a >= 0

    def can_invert(self):
        return True


@register_technique
class JacobiSymbol(MethodBlock):
    """Compute Jacobi symbol (a/n)."""

    def __init__(self):
        super().__init__()
        self.name = "jacobi_symbol"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "quadratic_residues"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value else random.randint(1, 10000)

        if input_value is not None and input_value > 1000:
            n = random.choice([2 * k + 1 for k in range(100, 500)])
        else:
            n = random.choice([2 * k + 1 for k in range(5, 100)])
        return {"a": a, "n": n}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        n = params.get("n", 10)

        jac = int(jacobi_symbol(a, n))
        result = jac

        return MethodResult(
            value=result,
            description=f"Jacobi symbol ({a}/{n}) = {result}",
            metadata={"a": a, "n": n, "jacobi": result}
        )

    def validate_params(self, params):
        return params.get("n", 10) % 2 == 1 and params.get("n", 10) > 0

    def can_invert(self):
        return True


@register_technique
class CountQuadResidues(MethodBlock):
    """Count quadratic residues mod p."""

    def __init__(self):
        super().__init__()
        self.name = "count_quad_residues"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "quadratic_residues"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            p = random.choice([pr for pr in primerange(1000, 5000)])
        else:
            p = input_value if input_value else random.choice([pr for pr in primerange(100, 1000)])
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 5)
        count = (p - 1) // 2

        return MethodResult(
            value=count,
            description=f"Number of QRs mod {p} = {count}",
            metadata={"p": p, "count": count}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5)) and params.get("p", 5) > 2

    def can_invert(self):
        return True


@register_technique
class QuadraticResidueProductModP(MethodBlock):
    """Compute the product of all quadratic residues modulo p."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_residue_product_mod_p"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "quadratic_residues", "modular_arithmetic"]

    def generate_parameters(self, input_value=None):
        p = input_value if (input_value and isprime(input_value)) else random.choice([7, 11, 13, 17, 19, 23])
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 7)

        if not isprime(p) or p == 2:
            return MethodResult(
                value=0,
                description=f"{p} is not an odd prime",
                metadata={"error": "not_odd_prime"}
            )

        residues = []
        for a in range(1, p):
            if legendre_symbol(a, p) == 1:
                residues.append(a)

        product = 1
        for r in residues:
            product = (product * r) % p

        return MethodResult(
            value=product,
            description=f"Product of quadratic residues mod {p} = {product} (residues: {residues})",
            metadata={"p": p, "residues": residues, "product": product}
        )

    def validate_params(self, params):
        p = params.get("p", 7)
        return isprime(p) and p > 2

    def can_invert(self):
        return False


@register_technique
class CubicResidueSet(MethodBlock):
    """Find the set of cubic residues modulo p."""

    def __init__(self):
        super().__init__()
        self.name = "cubic_residue_set"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "cubic_residues", "modular_arithmetic"]

    def generate_parameters(self, input_value=None):
        candidates = [p for p in [7, 13, 19, 31, 37, 43] if p % 3 == 1]
        p = input_value if (input_value and isprime(input_value)) else random.choice(candidates)
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 7)

        if not isprime(p):
            return MethodResult(
                value=0,
                description=f"{p} is not prime",
                metadata={"error": "not_prime"}
            )

        residues = set()
        for a in range(p):
            residues.add(pow(a, 3, p))

        residues = sorted(list(residues))
        count = len(residues)

        return MethodResult(
            value=count,
            description=f"Cubic residues mod {p}: {residues[:10]}{'...' if count > 10 else ''} (count: {count})",
            metadata={"p": p, "residues": residues, "count": count}
        )

    def validate_params(self, params):
        p = params.get("p", 7)
        return isprime(p)

    def can_invert(self):
        return False


@register_technique
class QuadraticReciprocity(MethodBlock):
    """Verify quadratic reciprocity law for two odd primes."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_reciprocity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "quadratic_residues"]

    def generate_parameters(self, input_value=None):
        primes = [p for p in primerange(3, 100)]
        p = random.choice(primes)
        q = random.choice([pr for pr in primes if pr != p])
        return {"p": p, "q": q}

    def compute(self, input_value, params):
        p = params.get("p", 5)
        q = params.get("q", 7)

        p_over_q = int(legendre_symbol(p, q))
        q_over_p = int(legendre_symbol(q, p))

        exponent = ((p - 1) // 2) * ((q - 1) // 2)
        reciprocity_sign = (-1) ** exponent

        law_holds = (p_over_q * q_over_p == reciprocity_sign)

        # Return (p/q) shifted to [0,2] for valid range: -1->0, 0->1, 1->2
        result = p_over_q + 1

        return MethodResult(
            value=result,
            description=f"({p}/{q}) * ({q}/{p}) = {p_over_q * q_over_p}, (-1)^((p-1)/2 * (q-1)/2) = {reciprocity_sign}",
            metadata={
                "p": p,
                "q": q,
                "p_over_q": p_over_q,
                "q_over_p": q_over_p,
                "exponent": exponent,
                "reciprocity_sign": reciprocity_sign,
                "law_holds": law_holds
            }
        )

    def validate_params(self, params):
        p = params.get("p", 5)
        q = params.get("q", 7)
        return isprime(p) and isprime(q) and p > 2 and q > 2 and p != q

    def can_invert(self):
        return False


@register_technique
class FermatTwoSquares(MethodBlock):
    """Decompose p = 1 (mod 4) as a^2 + b^2."""

    _PRIMES_1MOD4_SMALL = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    _PRIMES_1MOD4_LARGE = [1009, 1013, 1021, 1033, 1049, 1061, 1069, 1093, 1097, 1109, 1117, 1129, 1153, 1181, 1193, 1201, 1213, 1217, 1229, 1237, 1249, 1277, 1289, 1297, 1301, 1321, 1361, 1373, 1381, 1409, 1429, 1433, 1453, 1481, 1489, 1493, 1549, 1553, 1597, 1601, 1609, 1613, 1621, 1637, 1657, 1669, 1693, 1697, 1709, 1721]

    def __init__(self):
        super().__init__()
        self.name = "fermat_two_squares"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "sums_of_squares"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            primes_1mod4 = self._PRIMES_1MOD4_LARGE
        else:
            primes_1mod4 = self._PRIMES_1MOD4_SMALL

        p = input_value if (input_value and input_value in primes_1mod4) else random.choice(primes_1mod4)
        return {"p": p}

    def compute(self, input_value, params):
        p = params.get("p", input_value) if input_value else params.get("p", 5)
        p = min(abs(p) if p else 5, 10000)

        for a in range(1, int(math.isqrt(p)) + 1):
            b_sq = p - a * a
            if b_sq < 0:
                break
            b = math.isqrt(b_sq)
            if b * b == b_sq:
                return MethodResult(
                    value=a,
                    description=f"{p} = {a}^2 + {b}^2",
                    metadata={"p": p, "a": a, "b": b}
                )

        a = math.isqrt(p)
        return MethodResult(
            value=a,
            description=f"Could not decompose {p}, using isqrt({p}) = {a}",
            metadata={"p": p, "a": a}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5)) and params.get("p", 5) % 4 == 1

    def can_invert(self):
        return True


@register_technique
class SumOfSquaresCount(MethodBlock):
    """Count ways to write n = a^2 + b^2."""

    def __init__(self):
        super().__init__()
        self.name = "sum_of_squares_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sums_of_squares"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1, 200)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 10000000)

        count = 0
        for a in range(int(n**0.5) + 1):
            b_sq = n - a * a
            if b_sq < 0:
                break
            b = int(b_sq**0.5)
            if b * b == b_sq:
                count += 1

        return MethodResult(
            value=count,
            description=f"{n} can be written as a^2 + b^2 in {count} ways (a,b >= 0)",
            metadata={"n": n, "count": count}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0

    def can_invert(self):
        return True


@register_technique
class HenselLift(MethodBlock):
    """Lift solution mod p^k to p^(k+1) using Hensel's lemma."""

    def __init__(self):
        super().__init__()
        self.name = "hensel_lift"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "hensel", "padic"]

    def generate_parameters(self, input_value=None):
        # f(x) = x^2 - a, lift solution to x^2 = a (mod p^k)

        # Dynamic parameter selection: large input -> larger p and k
        if input_value is not None and input_value > 1000:
            p = random.choice([11, 13, 17, 19, 23])
            k = random.randint(3, 6)
        else:
            p = random.choice([3, 5, 7, 11])
            k = random.randint(2, 5)

        # Choose a that is QR mod p
        a = random.randint(1, p - 1)
        while legendre_symbol(a, p) != 1:
            a = random.randint(1, p - 1)
        return {"a": a, "p": p, "k": k}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        p = params.get("p", 5)
        k = params.get("k", 5)

        # Find x such that x^2 = a (mod p^k)
        # Start with solution mod p
        x = sqrt_mod(a, p)
        if x is None:
            x = 1

        # Lift to p^k using Hensel
        pk = p
        for i in range(1, k):
            # x_{i+1} = x_i - f(x_i) / f'(x_i) mod p^{i+1}
            # For f(x) = x^2 - a, f'(x) = 2x
            pk *= p
            fx = (x * x - a) % pk
            fpx = (2 * x) % pk
            if fpx % p != 0:
                x = (x - fx * mod_inverse(fpx, pk)) % pk

        return MethodResult(
            value=x,
            description=f"x^2 = {a} (mod {p}^{k}), x = {x}",
            metadata={"a": a, "p": p, "k": k, "x": x}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5)) and params.get("k", 5) > 0

    def can_invert(self):
        return True


@register_technique
class SolveQuadraticCongruence(MethodBlock):
    """Solve x^2 = a (mod p) for prime p."""

    def __init__(self):
        super().__init__()
        self.name = "solve_quadratic_congruence"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "modular", "quadratic_residue"]

    def generate_parameters(self, input_value=None):
        p = random.choice([q for q in primerange(50, 500)])
        if input_value is not None:
            a = input_value % p
            if legendre_symbol(a, p) != 1:
                a = (a * a) % p
        else:
            x = random.randint(1, p - 1)
            a = (x * x) % p
        return {"a": a, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", input_value) if input_value else params.get("a", 10)
        p = params.get("p", 5)

        if legendre_symbol(a, p) != 1:
            a = (a * a) % p

        solutions = sqrt_mod(a, p, all_roots=True)
        x = min(solutions) if solutions else 0

        return MethodResult(
            value=x,
            description=f"Solution to x^2 = {a} (mod {p}) is x = {x} (mod {p})",
            metadata={"a": a, "p": p, "solution": x, "all_solutions": solutions}
        )

    def can_invert(self):
        return False
