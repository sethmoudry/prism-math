"""
Number theory special sequences techniques.

Implements techniques for:
- Fibonacci numbers
- Lucas numbers
- Fermat numbers
- Golden ratio powers
- Large value generation for mod reduction
"""

import random
import math
from typing import Any, Dict, Optional
from sympy import fibonacci, lucas, gcd

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class Fibonacci(MethodBlock):
    """Compute F_n (Fibonacci number).

    Also supports sub-technique dispatch: fibonacci('fibonacci_gcd', m, n)
    routes to the fibonacci_gcd technique with positional args.
    """

    # Sub-techniques that can be dispatched via string first argument
    _SUB_TECHNIQUES = {"fibonacci_gcd", "fibonacci_inverse", "large_fibonacci"}

    def __init__(self):
        super().__init__()
        self.name = "fibonacci"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "sequences", "fibonacci"]
        self.is_primitive = True

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1, 30)
        n = min(n, 50)
        return {"n": n}

    def compute(self, input_value, params):
        # String dispatch: fibonacci('fibonacci_gcd', m, n) -> delegate
        n_val = params.get("n", input_value)
        if isinstance(n_val, str) and n_val in self._SUB_TECHNIQUES:
            return self._dispatch(n_val, input_value, params)
        if isinstance(input_value, str) and input_value in self._SUB_TECHNIQUES:
            return self._dispatch(input_value, None, params)

        n = n_val if n_val else params.get("n", 10)
        n = min(n, 50)

        fib = fibonacci(n)

        return MethodResult(
            value=fib,
            description=f"F_{n} = {fib}",
            metadata={"n": n, "fibonacci": fib}
        )

    def _dispatch(self, sub_name, input_value, params):
        """Dispatch to a sub-technique by name."""
        from ..registry import MethodRegistry
        try:
            sub_method = MethodRegistry.get(sub_name)
        except KeyError:
            raise ValueError(f"Unknown fibonacci sub-technique: {sub_name}")
        # Remaining positional args were mapped to params by the executor.
        # For fibonacci('fibonacci_gcd', 64, 24), the executor maps:
        #   params["n"] = 'fibonacci_gcd' (the dispatch key, consumed above)
        # But the actual numeric args (64, 24) end up as extra positional
        # args that get lost in the current param mapping.
        # Instead, we re-invoke the sub-method's compute with the params
        # that are available (excluding the string dispatch key).
        clean_params = {k: v for k, v in params.items()
                        if not isinstance(v, str) or k != "n"}
        return sub_method.compute(input_value, clean_params)

    def validate_params(self, params):
        n = params.get("n", 10)
        if isinstance(n, str):
            return n in self._SUB_TECHNIQUES
        return n >= 0 and n <= 50

    def can_invert(self):
        return True


@register_technique
class FibonacciInverse(MethodBlock):
    """Find n where F_n = F."""

    def __init__(self):
        super().__init__()
        self.name = "fibonacci_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "fibonacci", "inverse"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            return {"F": input_value}
        else:
            F = input_value if input_value else fibonacci(random.randint(5, 20))
            return {"F": F}

    def compute(self, input_value, params):
        F = params.get("F", input_value) if input_value else params.get("F", 10)

        phi = (1 + 5**0.5) / 2

        if F < 10000:
            for n in range(1, 100):
                if fibonacci(n) == F:
                    return MethodResult(
                        value=n,
                        description=f"F_{n} = {F}",
                        metadata={"F": F, "n": n}
                    )

        result = int(math.log(max(1, F) * 5**0.5) / math.log(phi))

        return MethodResult(
            value=result,
            description=f"F_{result} ~= {F}",
            metadata={"F": F, "n": result}
        )

    def validate_params(self, params):
        return params.get("F", 10) > 0

    def can_invert(self):
        return False


@register_technique
class Lucas(MethodBlock):
    """Compute L_n (Lucas number)."""

    def __init__(self):
        super().__init__()
        self.name = "lucas"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "sequences", "lucas"]

    def generate_parameters(self, input_value=None):
        if input_value is not None:
            n = min(35, max(1, abs(int(input_value)) % 36))
        else:
            n = random.randint(1, 30)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 50)

        luc = lucas(n)

        return MethodResult(
            value=luc,
            description=f"L_{n} = {luc}",
            metadata={"n": n, "lucas": luc}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0

    def can_invert(self):
        return True


@register_technique
class LucasInverse(MethodBlock):
    """Find n where L_n = L."""

    def __init__(self):
        super().__init__()
        self.name = "lucas_inverse"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "lucas", "inverse"]

    def generate_parameters(self, input_value=None):
        L = input_value if input_value else lucas(random.randint(5, 20))
        return {"L": L}

    def compute(self, input_value, params):
        L = params.get("L", input_value) if input_value else params.get("L", 10)

        for n in range(1, 100):
            if lucas(n) == L:
                return MethodResult(
                    value=n,
                    description=f"L_{n} = {L}",
                    metadata={"L": L, "n": n}
                )

        phi = (1 + 5**0.5) / 2
        n = int(math.log(L) / math.log(phi))

        return MethodResult(
            value=n,
            description=f"Approximate: L_{n} ~= {L}",
            metadata={"L": L, "n": n}
        )

    def validate_params(self, params):
        return params.get("L", 10) > 0

    def can_invert(self):
        return False


@register_technique
class FibonacciGCD(MethodBlock):
    """Compute gcd(F_m, F_n) = F_gcd(m,n)."""

    def __init__(self):
        super().__init__()
        self.name = "fibonacci_gcd"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "fibonacci", "gcd"]

    def generate_parameters(self, input_value=None):
        if input_value is not None and input_value > 1000:
            m = random.randint(20, 50)
        else:
            m = input_value if input_value else random.randint(10, 40)

        m = min(m, 50)
        n = random.randint(10, 40)
        n = min(n, 50)
        return {"m": m, "n": n}

    def compute(self, input_value, params):
        m = params.get("m", input_value) if input_value else params.get("m", 10)
        n = params.get("n", 10)

        m = min(abs(m) if m else 10, 200)
        n = min(abs(n) if n else 10, 200)

        g = int(gcd(m, n))
        result = int(fibonacci(g))

        return MethodResult(
            value=result,
            description=f"gcd(F_{m}, F_{n}) = F_{g} = {result}",
            metadata={"m": m, "n": n, "gcd_mn": g, "result": result}
        )

    def validate_params(self, params):
        return params.get("m", 10) > 0 and params.get("n", 10) > 0

    def can_invert(self):
        return True


@register_technique
class FermatNumber(MethodBlock):
    """Compute F_n = 2^(2^n) + 1 (Fermat number)."""

    def __init__(self):
        super().__init__()
        self.name = "fermat_number"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "fermat", "exponential"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(0, 4)
        n = min(n, 4)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(n, 4)

        fermat = 2**(2**n) + 1

        return MethodResult(
            value=fermat,
            description=f"F_{n} = 2^(2^{n}) + 1 = {fermat}",
            metadata={"n": n, "fermat": fermat}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0 and params.get("n", 10) <= 4

    def can_invert(self):
        return True


@register_technique
class GoldenRatioPower(MethodBlock):
    """Compute phi^n where phi is the golden ratio."""

    def __init__(self):
        super().__init__()
        self.name = "golden_ratio_power"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "golden_ratio"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(5, 30)
        n = min(n, 50)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 50)

        # phi^n = F_n * phi + F_{n-1}
        # For integer part: phi^n ≈ F_n * phi
        phi = (1 + 5**0.5) / 2
        result = int(round(phi ** n / 5**0.5))

        return MethodResult(
            value=result,
            description=f"phi^{n} ~= {result} (nearest integer)",
            metadata={"n": n, "result": result}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0

    def can_invert(self):
        return False


@register_technique
class ExtractPQFromGoldenPower(MethodBlock):
    """Extract p, q from phi^n = (p + q*sqrt(5))/2."""

    def __init__(self):
        super().__init__()
        self.name = "extract_pq_from_golden_power"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "sequences", "golden_ratio"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(5, 30)
        n = min(n, 50)
        extract = random.choice(["p", "q"])
        return {"n": n, "extract": extract}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 50)
        extract = params.get("extract", "p")

        # phi^n = (L_n + F_n * sqrt(5)) / 2
        # So p = L_n, q = F_n
        L_n = int(lucas(n))
        F_n = int(fibonacci(n))

        if extract == "p":
            result = L_n
            desc = f"phi^{n} = ({L_n} + {F_n}*sqrt(5))/2, p = {L_n}"
        else:
            result = F_n
            desc = f"phi^{n} = ({L_n} + {F_n}*sqrt(5))/2, q = {F_n}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"n": n, "L_n": L_n, "F_n": F_n, "extract": extract}
        )

    def validate_params(self, params):
        return params.get("n", 10) >= 0

    def can_invert(self):
        return False


# ============================================================================
# EXPLOSIVE GROWTH TECHNIQUES - Large intermediate values
# ============================================================================

@register_technique
class LargePower(MethodBlock):
    """Compute a^b with guaranteed large result."""

    def __init__(self):
        super().__init__()
        self.name = "large_power"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "exponentiation", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        base = params.get("base")
        exp = params.get("exp")
        if base is None or exp is None:
            return False
        return 2 <= base <= 10 and 10 <= exp <= 35

    def generate_parameters(self, input_value=None):
        base = random.randint(2, 7)
        if base <= 3:
            exp = random.randint(18, 30)
        elif base <= 5:
            exp = random.randint(15, 25)
        else:
            exp = random.randint(12, 20)
        return {"base": base, "exp": exp}

    def compute(self, input_value, params):
        base = params.get("base", 3)
        exp = params.get("exp", 20)

        base = max(2, min(base, 10))
        exp = max(10, min(exp, 35))

        result = base ** exp

        return MethodResult(
            value=result,
            description=f"{base}^{exp} = {result}",
            metadata={
                "base": base,
                "exp": exp,
                "magnitude": len(str(result)),
                "formula": "base^exp",
                "growth_type": "explosive_power"
            }
        )

    def can_invert(self):
        return False


@register_technique
class LargeFibonacci(MethodBlock):
    """Compute Fibonacci number F_n for n in range 50-80."""

    def __init__(self):
        super().__init__()
        self.name = "large_fibonacci"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "sequences", "fibonacci", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return 50 <= n <= 80

    def generate_parameters(self, input_value=None):
        n = random.randint(50, 75)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 60)
        n = max(1, int(n))

        result = int(fibonacci(n))

        return MethodResult(
            value=result,
            description=f"F_{n} = {result}",
            metadata={
                "n": n,
                "magnitude": len(str(result)),
                "formula": "Fibonacci(n)",
                "growth_type": "explosive_fibonacci"
            }
        )

    def can_invert(self):
        return False


@register_technique
class LargePowerTower(MethodBlock):
    """Compute power tower (tetration) with controlled large output."""

    def __init__(self):
        super().__init__()
        self.name = "large_power_tower"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "exponentiation", "tetration", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        base = params.get("base")
        height_exp = params.get("height_exp")
        if base is None or height_exp is None:
            return False
        return 2 <= base <= 3 and 2 <= height_exp <= 5

    def generate_parameters(self, input_value=None):
        base = random.choice([2, 3])
        if base == 2:
            height_exp = random.randint(4, 5)
        else:
            height_exp = random.randint(2, 3)
        return {"base": base, "height_exp": height_exp}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        height_exp = params.get("height_exp", 4)

        base = max(2, min(base, 3))
        height_exp = max(2, min(height_exp, 5))

        inner = base ** height_exp
        result = base ** inner

        return MethodResult(
            value=result,
            description=f"{base}^({base}^{height_exp}) = {base}^{inner} = {result}",
            metadata={
                "base": base,
                "height_exp": height_exp,
                "inner_value": inner,
                "magnitude": len(str(result)),
                "formula": "base^(base^height_exp)",
                "growth_type": "explosive_tower"
            }
        )

    def can_invert(self):
        return False
