"""
Number theory valuation techniques.

Implements techniques for:
- Legendre valuation (v_p(n!))
- p-adic valuation
- Kummer valuation for binomial coefficients
- Lifting the Exponent lemma
"""

import random
import math
from typing import Any, Dict, Optional
from sympy import isprime

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from ..decomposition import Decomposition


@register_technique
class LegendreValuation(MethodBlock):
    """Compute v_p(n!) using Legendre's formula."""

    def __init__(self):
        super().__init__()
        self.name = "legendre_valuation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "valuation", "factorial"]
        self.decomposition = Decomposition(
            expression="sum_legendre_terms(n, p)",
            param_map={"n": "n", "p": "p"},
            notes="v_p(n!) = sum(floor(n/p^k)) - iterative"
        )

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            n = input_value
            p = random.choice([7, 11, 13, 17, 19])
        else:
            n = input_value if input_value else random.randint(100, 5000)
            p = random.choice([2, 3, 5, 7])
        return {"n": n, "p": p}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        p = params.get("p", 5)

        v = 0
        pk = p
        while pk <= n:
            v += n // pk
            pk *= p

        return MethodResult(
            value=v,
            description=f"v_{p}({n}!) = {v}",
            metadata={"n": n, "p": p, "valuation": v}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class LegendreInverse(MethodBlock):
    """Find smallest n such that v_p(n!) = v."""

    def __init__(self):
        super().__init__()
        self.name = "legendre_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "valuation", "inverse"]

    def generate_parameters(self, input_value=None):
        v = input_value if input_value else random.randint(5, 100)
        p = random.choice([2, 3, 5, 7, 11])
        return {"v": v, "p": p}

    def compute(self, input_value, params):
        v = params.get("v", input_value) if input_value else params.get("v", 10)
        p = params.get("p", 5)

        left, right = 1, v * p
        result = v * p

        while left <= right:
            mid = (left + right) // 2
            val = 0
            pk = p
            while pk <= mid:
                val += mid // pk
                pk *= p

            if val >= v:
                result = mid
                right = mid - 1
            else:
                left = mid + 1

        return MethodResult(
            value=result,
            description=f"Smallest n where v_{p}(n!) >= {v} is {result}",
            metadata={"v": v, "p": p, "n": result}
        )

    def validate_params(self, params):
        return params.get("v", 10) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class PadicValuation(MethodBlock):
    """Compute v_p(n) for integer n."""

    def __init__(self):
        super().__init__()
        self.name = "padic_valuation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "valuation"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            n = input_value
            p = random.choice([2, 3, 5, 7, 11])
        else:
            p = random.choice([2, 3, 5, 7, 11])
            k = random.randint(1, 6)
            n = (p ** k) * random.randint(1, 100)
        return {"n": n, "p": p}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        p = params.get("p", 5)

        if n == 0:
            return MethodResult(
                value=float('inf'),
                description=f"v_{p}(0) = infinity",
                metadata={"n": 0, "p": p}
            )

        v = 0
        n_abs = abs(n)
        while n_abs % p == 0:
            v += 1
            n_abs //= p

        return MethodResult(
            value=v,
            description=f"v_{p}({n}) = {v}",
            metadata={"n": n, "p": p, "valuation": v}
        )

    def validate_params(self, params):
        return isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class PadicInverse(MethodBlock):
    """Find smallest n with v_p(n) = v."""

    def __init__(self):
        super().__init__()
        self.name = "padic_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "valuation", "inverse"]

    def generate_parameters(self, input_value=None):
        v = input_value if input_value else random.randint(1, 10)
        p = random.choice([2, 3, 5, 7, 11])
        return {"v": v, "p": p}

    def compute(self, input_value, params):
        v = params.get("v", input_value) if input_value else params.get("v", 10)
        p = params.get("p", 5)

        v = min(abs(v) if v else 10, 50)
        result = p ** v

        return MethodResult(
            value=result,
            description=f"Smallest n with v_{p}(n) = {v} is {result}",
            metadata={"v": v, "p": p, "n": result}
        )

    def validate_params(self, params):
        return params.get("v", 10) >= 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class KummerValuation(MethodBlock):
    """Compute v_p(C(n,k)) using Kummer's theorem."""

    def __init__(self):
        super().__init__()
        self.name = "kummer_valuation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "valuation", "binomial"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            n = input_value
            p = random.choice([5, 7, 11])
        else:
            n = input_value if input_value else random.randint(100, 5000)
            p = random.choice([2, 3, 5, 7])

        k = random.randint(1, min(n, max(100, n // 10)))
        return {"n": n, "k": k, "p": p}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        k = params.get("k", 5)
        p = params.get("p", 5)

        def digit_sum_base_p(m, p):
            s = 0
            while m > 0:
                s += m % p
                m //= p
            return s

        s_k = digit_sum_base_p(k, p)
        s_nk = digit_sum_base_p(n - k, p)
        s_n = digit_sum_base_p(n, p)

        v = (s_k + s_nk - s_n) // (p - 1)

        return MethodResult(
            value=v,
            description=f"v_{p}(C({n},{k})) = {v} by Kummer",
            metadata={"n": n, "k": k, "p": p, "valuation": v}
        )

    def validate_params(self, params):
        return 0 <= params.get("k", 5) <= params.get("n", 10) and isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class KummerInverseN(MethodBlock):
    """Find n given v_p(C(n,k)) = v (fix k, find n)."""

    def __init__(self):
        super().__init__()
        self.name = "kummer_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "valuation", "inverse"]

    def generate_parameters(self, input_value=None):
        v = input_value if input_value else random.randint(1, 5)
        k = random.randint(2, 20)
        p = random.choice([2, 3, 5])
        return {"v": v, "k": k, "p": p}

    def compute(self, input_value, params):
        v = params.get("v", input_value) if input_value else params.get("v", 10)
        k = params.get("k", 5)
        p = params.get("p", 5)

        def digit_sum_base_p(m, p):
            s = 0
            while m > 0:
                s += m % p
                m //= p
            return s

        n = k
        max_n = k + v * p * 10

        while n < max_n:
            s_k = digit_sum_base_p(k, p)
            s_nk = digit_sum_base_p(n - k, p)
            s_n = digit_sum_base_p(n, p)
            val = (s_k + s_nk - s_n) // (p - 1)

            if val >= v:
                break
            n += 1

        return MethodResult(
            value=n,
            description=f"Smallest n >= {k} where v_{p}(C(n,{k})) >= {v} is {n}",
            metadata={"v": v, "k": k, "p": p, "n": n}
        )

    def validate_params(self, params):
        return params.get("v", 10) >= 0 and params.get("k", 5) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class KummerInverseK(MethodBlock):
    """Find k given v_p(C(n,k)) = v (fix n, find k)."""

    def __init__(self):
        super().__init__()
        self.name = "kummer_inverse_k"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "valuation", "inverse"]

    def generate_parameters(self, input_value=None):
        v = input_value if input_value else random.randint(1, 5)
        n = random.randint(20, 200)
        p = random.choice([2, 3, 5])
        return {"v": v, "n": n, "p": p}

    def compute(self, input_value, params):
        v = params.get("v", input_value) if input_value else params.get("v", 10)
        n = params.get("n", 10)
        p = params.get("p", 5)

        def digit_sum_base_p(m, p):
            s = 0
            while m > 0:
                s += m % p
                m //= p
            return s

        for k in range(1, n + 1):
            s_k = digit_sum_base_p(k, p)
            s_nk = digit_sum_base_p(n - k, p)
            s_n = digit_sum_base_p(n, p)
            val = (s_k + s_nk - s_n) // (p - 1)

            if val >= v:
                return MethodResult(
                    value=k,
                    description=f"Smallest k where v_{p}(C({n},k)) >= {v} is {k}",
                    metadata={"v": v, "n": n, "p": p, "k": k}
                )

        k = n // 2
        return MethodResult(
            value=k,
            description=f"Using k = {k} for C({n},k)",
            metadata={"v": v, "n": n, "p": p, "k": k}
        )

    def validate_params(self, params):
        return params.get("v", 10) >= 0 and params.get("n", 10) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class LiftingExponentValuation(MethodBlock):
    """Compute v_p(a^n - b^n) using Lifting the Exponent lemma."""

    def __init__(self):
        super().__init__()
        self.name = "lifting_exponent_valuation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "valuation", "lte"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(2, 20)
        p = random.choice([2, 3, 5, 7])
        a = random.randint(2, 20)
        b = a + p * random.randint(1, 5)
        return {"a": a, "b": b, "n": n, "p": p}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        b = params.get("b", 10)
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        p = params.get("p", 5)

        if a == b:
            return MethodResult(
                value=float('inf'),
                description=f"v_{p}(0) = infinity",
                metadata={"a": a, "b": b, "n": n, "p": p}
            )

        diff = a**n - b**n
        if diff == 0:
            return MethodResult(
                value=float('inf'),
                description=f"v_{p}(0) = infinity",
                metadata={"a": a, "b": b, "n": n, "p": p}
            )

        v = 0
        diff_abs = abs(diff)
        while diff_abs % p == 0:
            v += 1
            diff_abs //= p

        return MethodResult(
            value=v,
            description=f"v_{p}({a}^{n} - {b}^{n}) = {v} by LTE",
            metadata={"a": a, "b": b, "n": n, "p": p, "valuation": v}
        )

    def validate_params(self, params):
        return params.get("n", 10) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return True


@register_technique
class LiftingExponentInverseN(MethodBlock):
    """Find n in v_p(a^n - b^n) = v."""

    def __init__(self):
        super().__init__()
        self.name = "lifting_exponent_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "valuation", "lte", "inverse"]

    def generate_parameters(self, input_value=None):
        v = input_value if input_value else random.randint(1, 10)
        p = random.choice([2, 3, 5, 7])
        a = random.randint(2, 20)
        b = a + p * random.randint(1, 5)
        return {"v": v, "a": a, "b": b, "p": p}

    def compute(self, input_value, params):
        v = params.get("v", input_value) if input_value else params.get("v", 10)
        a = params.get("a", 10)
        b = params.get("b", 10)
        p = params.get("p", 5)

        for n in range(1, 100):
            diff = a**n - b**n
            if diff == 0:
                continue
            val = 0
            diff_abs = abs(diff)
            while diff_abs % p == 0:
                val += 1
                diff_abs //= p
            if val >= v:
                return MethodResult(
                    value=n,
                    description=f"v_{p}({a}^{n} - {b}^{n}) >= {v}",
                    metadata={"v": v, "a": a, "b": b, "p": p, "n": n}
                )

        n = v
        return MethodResult(
            value=n,
            description=f"Using n = {n} (fallback)",
            metadata={"v": v, "a": a, "b": b, "p": p, "n": n}
        )

    def validate_params(self, params):
        return params.get("v", 10) > 0 and isprime(params.get("p", 5))

    def can_invert(self):
        return False


@register_technique
class LTEEvenPower(MethodBlock):
    """Apply Lifting the Exponent lemma for even powers.

    For odd prime p and even n, computes v_2((p^(nk) - 1)/(p^k - 1)).
    Using LTE: v_2((p^(nk) - 1)/(p^k - 1)) = v_2(p^k + 1) + v_2(n) - 1.
    """

    def __init__(self):
        super().__init__()
        self.name = "lte_even_power"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["number_theory", "valuation", "lte"]

    def generate_parameters(self, input_value=None):
        p = random.choice([3, 5, 7, 11, 13, 17, 19, 23])
        n = 16
        k = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        return {"p": p, "n": n, "k": k}

    def compute(self, input_value, params):
        p = params.get("p", 5)
        n = params.get("n", 10)
        k = params.get("k", 5)

        # For odd p: v_2((p^(nk) - 1)/(p^k - 1)) = v_2(p^k + 1) + v_2(n) - 1

        # Compute v_2(p^k + 1)
        # Since p is odd, p^k is odd, so p^k + 1 is even
        pk_plus_1 = pow(p, k, 8)  # Compute p^k mod 8 to determine v_2(p^k + 1)
        pk_plus_1 = (pk_plus_1 + 1) % 8

        # Count trailing zeros in p^k + 1
        v2_pk_plus_1 = 0
        temp = pk_plus_1
        while temp > 0 and temp % 2 == 0:
            v2_pk_plus_1 += 1
            temp //= 2

        # If p^k + 1 ≡ 0 (mod 8), need actual value
        if pk_plus_1 == 0:
            actual_pk_plus_1 = pow(p, k) + 1
            temp = actual_pk_plus_1
            v2_pk_plus_1 = 0
            while temp % 2 == 0:
                v2_pk_plus_1 += 1
                temp //= 2

        # Compute v_2(n)
        v2_n = 0
        temp = n
        while temp % 2 == 0:
            v2_n += 1
            temp //= 2

        # LTE result: v_2((p^(nk) - 1)/(p^k - 1))
        result = v2_pk_plus_1 + v2_n - 1

        return MethodResult(
            value=result,
            description=f"v_2(({p}^({n}*{k}) - 1)/({p}^{k} - 1)) = {result}",
            metadata={"p": p, "n": n, "k": k, "valuation": result}
        )

    def validate_params(self, params):
        return (isprime(params.get("p", 5)) and params.get("p", 5) > 2
                and params.get("n", 10) > 0 and params.get("k", 5) > 0)

    def can_invert(self):
        return False


@register_technique
class MaxValuationUnderConstraint(MethodBlock):
    """Find max v_p(n) for n in a constrained range."""

    def __init__(self):
        super().__init__()
        self.name = "max_valuation_under_constraint"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "valuation", "optimization"]

    def generate_parameters(self, input_value=None):
        upper = input_value if input_value else random.randint(100, 10000)
        p = random.choice([2, 3, 5, 7])
        return {"upper": upper, "p": p}

    def compute(self, input_value, params):
        upper = params.get("upper", input_value) if input_value else params.get("upper", 1000)
        p = params.get("p", 2)

        upper = min(abs(upper) if upper else 1000, 1000000)

        max_v = 0
        pk = p
        while pk <= upper:
            max_v += 1
            pk *= p

        best_n = p ** max_v

        return MethodResult(
            value=max_v,
            description=f"Max v_{p}(n) for n <= {upper} is {max_v} (achieved at n = {best_n})",
            metadata={"upper": upper, "p": p, "max_valuation": max_v, "best_n": best_n}
        )

    def validate_params(self, params):
        return params.get("upper", 1) > 0 and isprime(params.get("p", 2))

    def can_invert(self):
        return False
