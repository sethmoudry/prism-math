"""
Equation-solving algebra techniques.

Contains:
- Recurrences (5)
- Functional equations (4)
"""

import random
import math
from typing import Any, Dict, Optional, List
import numpy as np

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# RECURRENCES (5 techniques)
# ============================================================================

@register_technique
class LinearRecurrence(MethodBlock):
    """Solve linear recurrence a_n = c1*a_(n-1) + c2*a_(n-2) + ... """

    def __init__(self):
        super().__init__()
        self.name = "linear_recurrence"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "recurrence", "sequences"]

    def validate_params(self, params, prev_value=None):
        """Validate that coefficients and initial values are properly specified."""
        coeffs = params.get("coeffs")
        initial = params.get("initial")
        n = params.get("n")
        if coeffs is None or initial is None or n is None:
            return False
        if not (isinstance(coeffs, (list, tuple)) and isinstance(initial, (list, tuple))):
            return False
        return len(coeffs) > 0 and len(initial) >= len(coeffs) and n >= 0

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate recurrence coefficients and initial conditions."""
        c1 = random.randint(1, 3)
        c2 = random.randint(-2, 2)
        a0 = random.randint(0, 5)
        a1 = random.randint(0, 5)
        n = random.randint(5, 10)
        return {"coeffs": [c1, c2], "initial": [a0, a1], "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute a_n using iteration."""
        coeffs = params.get("coeffs", [1, 1])
        initial = params.get("initial", [0, 1])
        n = params.get("n", 10)

        if not isinstance(coeffs, (list, tuple)):
            coeffs = [1, 1]
        if not isinstance(initial, (list, tuple)):
            initial = [0, 1]

        order = len(coeffs)
        if len(initial) < order:
            raise ValueError(
                f"linear_recurrence needs at least {order} initial values for order-{order} recurrence. "
                f"Got {len(initial)} initial values: {initial}"
            )

        seq = list(initial)
        for i in range(len(initial), n + 1):
            next_val = sum(coeffs[j] * seq[i - 1 - j] for j in range(order))
            seq.append(next_val)

        result = seq[n]

        if order == 2:
            desc = f"a_{n} = {result} for recurrence a_n = {coeffs[0]}*a_(n-1) + {coeffs[1]}*a_(n-2)"
        else:
            terms = " + ".join(f"{coeffs[j]}*a_(n-{j+1})" for j in range(order))
            desc = f"a_{n} = {result} for order-{order} recurrence: a_n = {terms}"

        return MethodResult(
            value=result,
            description=desc,
            params=params, metadata={"coeffs": coeffs, "initial": initial, "n": n, "sequence": seq, "order": order}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        """Find recurrence parameters that produce target answer."""
        if target < 0:
            return None

        if target == 0:
            return {"coeffs": [1, 1], "initial": [0, 0], "n": 5}

        max_n = 25 if target > 100 else 20

        for c1 in [1, 2, 3]:
            for c2 in [-2, -1, 0, 1, 2]:
                for a0 in range(0, 6):
                    for a1 in range(0, 6):
                        seq = [a0, a1]
                        for n in range(2, max_n):
                            next_val = c1 * seq[-1] + c2 * seq[-2]
                            seq.append(next_val)

                            if next_val == target:
                                return {"coeffs": [c1, c2], "initial": [a0, a1], "n": n}

                            if abs(next_val) > max(target * 20, 10000):
                                break

        return None

    def generate(self, target_answer: Optional[int] = None):
        """Generate a linear recurrence problem, optionally targeting a specific answer."""
        try:
            if target_answer is not None:
                params = self._find_params_for_answer(target_answer)
                if params is None:
                    return None
            else:
                params = self.generate_parameters()
        except Exception:
            return None

        result = self.compute(None, params)

        return {
            "success": True,
            "answer": result.value,
            "description": result.description,
            "params": params,
            "metadata": result.metadata,
            "technique": self.name
        }

    def can_invert(self) -> bool:
        return False


@register_technique
class LinearRecurrenceInverseN(MethodBlock):
    """Find n where a_n = target for linear recurrence."""

    def __init__(self):
        super().__init__()
        self.name = "linear_recurrence_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "recurrence", "sequences", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate recurrence and target value."""
        if input_value is not None and input_value > 1000:
            c1 = random.randint(1, 2)
            c2 = random.randint(0, 1)
            a0 = random.randint(0, 3)
            a1 = random.randint(1, 5)
            target = random.randint(1000, 5000)
        else:
            c1 = random.randint(1, 2)
            c2 = random.randint(0, 1)
            a0 = random.randint(0, 3)
            a1 = random.randint(1, 5)
            target = random.randint(20, 100)
        return {"coeffs": [c1, c2], "initial": [a0, a1], "target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find smallest n where a_n >= target."""
        coeffs = params.get("coeffs", 10)
        initial = params.get("initial", 10)
        target = params.get("target", 10)

        seq = list(initial)
        n = 1
        while seq[-1] < target and n < 1000:
            next_val = coeffs[0] * seq[-1] + coeffs[1] * seq[-2]
            seq.append(next_val)
            n += 1

        description = f"Smallest n where a_n >= {target} is n={n} (a_n={seq[-1]})"

        return MethodResult(
            value=n,
            description=description,
            params=params, metadata={"coeffs": coeffs, "initial": initial, "target": target}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CharPolynomial(MethodBlock):
    """Find characteristic polynomial roots for recurrence."""

    def __init__(self):
        super().__init__()
        self.name = "characteristic_polynomial"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "recurrence", "polynomials", "roots"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate recurrence coefficients with controlled ranges."""
        c1 = random.randint(1, 3)
        c2 = random.randint(-2, 2)
        if c2 == 0:
            c2 = 1
        return {"coeffs": [c1, c2]}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find roots of characteristic polynomial."""
        coeffs = params.get("coeffs", [1, 1])

        if input_value is not None:
            if not isinstance(input_value, (list, tuple)):
                raise ValueError(f"characteristic_polynomial expects sequence input, got {type(input_value)}")
            if len(input_value) < 2:
                raise ValueError(f"characteristic_polynomial needs at least 2 sequence elements, got {len(input_value)}")

        if len(coeffs) == 2:
            c1, c2 = coeffs
            discriminant = c1**2 + 4*c2

            if discriminant >= 0:
                sqrt_d = math.sqrt(discriminant)
                r1 = (c1 + sqrt_d) / 2
                r2 = (c1 - sqrt_d) / 2
                roots = [r1, r2]
                root_sum = abs(r1) + abs(r2)
            else:
                if c2 > 0:
                    root_sum = 2 * math.sqrt(abs(c2))
                else:
                    real_part = c1 / 2
                    root_sum = abs(real_part) * 2

                roots = [complex(c1/2, math.sqrt(-discriminant)/2),
                        complex(c1/2, -math.sqrt(-discriminant)/2)]

            description = f"Characteristic roots for a_n={c1}*a_(n-1)+{c2}*a_(n-2): {roots}"
            metadata = {"coeffs": coeffs, "discriminant": discriminant, "roots": roots}
        else:
            poly_coeffs = [1] + [-c for c in coeffs]

            try:
                roots_arr = np.roots(poly_coeffs)
                root_sum = sum(abs(r.real) for r in roots_arr)
                roots = roots_arr.tolist()
            except ImportError:
                raise ValueError(
                    f"characteristic_polynomial currently only supports 2nd-order recurrences without numpy. "
                    f"Got {len(coeffs)} coefficients: {coeffs}. "
                    f"Install numpy or use only 2 coefficients."
                )

            description = f"Characteristic roots for order-{len(coeffs)} recurrence: {roots}"
            metadata = {"coeffs": coeffs, "order": len(coeffs), "roots": roots}

        result = int(root_sum)

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata=metadata
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class BinetFormula(MethodBlock):
    """Compute Fibonacci-like sequence using Binet's formula."""

    def __init__(self):
        super().__init__()
        self.name = "binet_formula"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "recurrence", "fibonacci", "closed_form"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate Fibonacci-like parameters."""
        f0 = random.randint(0, 2)
        f1 = random.randint(1, 3)
        n = random.randint(5, 15)
        return {"f0": f0, "f1": f1, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute using Binet's formula for generalized Fibonacci."""
        f0, f1 = params.get("f0", 1), params.get("f1", 5)
        n = params.get("n", 10)

        phi = (1 + math.sqrt(5)) / 2
        psi = (1 - math.sqrt(5)) / 2

        A = (f1 - f0*psi) / (phi - psi)
        B = (f0*phi - f1) / (phi - psi)

        result = A * phi**n + B * psi**n
        result = int(round(result))

        try:
            description = f"F_{n} = {result} using Binet's formula with F_0={f0}, F_1={f1}"
        except ValueError:
            description = f"F_{n} = <large number> using Binet's formula with F_0={f0}, F_1={f1}"

        return MethodResult(
            value=result,
            description=description,
            params=params, metadata={"f0": f0, "f1": f1, "n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Recurrence(MethodBlock):
    """Solve linear recurrence relations."""

    def __init__(self):
        super().__init__()
        self.name = "recurrence"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "recurrence", "sequences"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for recurrence solving."""
        recurrence_type = random.choice([
            "first_order", "second_order", "fibonacci", "lucas", "geometric"
        ])

        n = random.randint(5, 20)

        if recurrence_type == "first_order":
            c = random.randint(1, 3)
            d = random.randint(0, 10)
            a0 = random.randint(1, 5)
            return {"recurrence_type": recurrence_type, "c": c, "d": d, "a0": a0, "n": n}

        elif recurrence_type == "second_order":
            p = random.randint(1, 3)
            q = random.randint(-2, 2)
            a0 = random.randint(1, 3)
            a1 = random.randint(1, 5)
            n = min(n, 15)
            return {"recurrence_type": recurrence_type, "p": p, "q": q, "a0": a0, "a1": a1, "n": n}

        elif recurrence_type == "fibonacci":
            a0 = 1
            a1 = 1
            return {"recurrence_type": recurrence_type, "a0": a0, "a1": a1, "n": n}

        elif recurrence_type == "lucas":
            a0 = 2
            a1 = 1
            return {"recurrence_type": recurrence_type, "a0": a0, "a1": a1, "n": n}

        else:
            r = random.randint(2, 3)
            a0 = random.randint(1, 5)
            n = min(n, 12)
            return {"recurrence_type": recurrence_type, "r": r, "a0": a0, "n": n}

    def compute(self, input_value, params):
        """Solve the recurrence and compute the n-th term."""
        recurrence_type = params.get("recurrence_type", "first_order")
        n = params.get("n", 10)

        if recurrence_type == "first_order":
            c = params.get("c", 2)
            d = params.get("d", 1)
            a0 = params.get("a0", 1)

            if c == 1:
                result = a0 + n * d
                desc = f"a_n = a_{{n-1}} + {d}, a_0 = {a0}: a_{n} = {a0} + {n}*{d} = {result}"
            else:
                a = a0
                for _ in range(n):
                    a = c * a + d
                result = a
                desc = f"a_n = {c}*a_{{n-1}} + {d}, a_0 = {a0}: a_{n} = {result}"

        elif recurrence_type == "second_order":
            p = params.get("p", 1)
            q = params.get("q", 1)
            a0 = params.get("a0", 1)
            a1 = params.get("a1", 1)

            if n == 0:
                result = a0
            elif n == 1:
                result = a1
            else:
                prev2, prev1 = a0, a1
                for _ in range(2, n + 1):
                    curr = p * prev1 + q * prev2
                    prev2, prev1 = prev1, curr
                result = prev1

            desc = f"a_n = {p}*a_{{n-1}} + {q}*a_{{n-2}}, a_0={a0}, a_1={a1}: a_{n} = {result}"

        elif recurrence_type == "fibonacci":
            a0 = params.get("a0", 1)
            a1 = params.get("a1", 1)

            if n == 0:
                result = a0
            elif n == 1:
                result = a1
            else:
                prev2, prev1 = a0, a1
                for _ in range(2, n + 1):
                    curr = prev1 + prev2
                    prev2, prev1 = prev1, curr
                result = prev1

            desc = f"Fibonacci recurrence a_n = a_{{n-1}} + a_{{n-2}}: F_{n} = {result}"

        elif recurrence_type == "lucas":
            a0 = params.get("a0", 2)
            a1 = params.get("a1", 1)

            if n == 0:
                result = a0
            elif n == 1:
                result = a1
            else:
                prev2, prev1 = a0, a1
                for _ in range(2, n + 1):
                    curr = prev1 + prev2
                    prev2, prev1 = prev1, curr
                result = prev1

            desc = f"Lucas recurrence L_n = L_{{n-1}} + L_{{n-2}}: L_{n} = {result}"

        else:
            r = params.get("r", 2)
            a0 = params.get("a0", 1)
            result = a0 * (r ** n)
            desc = f"Geometric sequence a_n = {r}*a_{{n-1}}, a_0 = {a0}: a_{n} = {a0}*{r}^{n} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"recurrence_type": recurrence_type, "n": n}
        )

    def can_invert(self):
        return False


# ============================================================================
# FUNCTIONAL EQUATIONS (4 techniques)
# ============================================================================

@register_technique
class FunctionalEqCauchy(MethodBlock):
    """Solve Cauchy functional equation f(x+y) = f(x) + f(y)."""

    def __init__(self):
        super().__init__()
        self.name = "functional_eq_cauchy"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "functional_equations", "cauchy"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate Cauchy equation instance."""
        if input_value is not None and input_value > 1000:
            c = random.randint(10, 50)
            x = random.randint(20, 100)
        else:
            c = random.randint(1, 5)
            x = random.randint(5, 15)
        return {"c": c, "x": x}

    def validate_params(self, params, prev_value=None):
        """Validate Cauchy functional equation parameters."""
        c = params.get("c")
        x = params.get("x")
        if c is None or x is None:
            return False
        try:
            float(c) if isinstance(c, str) else c
            float(x) if isinstance(x, str) else x
            return True
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solution: f(x) = cx."""
        c, x = params.get("c", 10), params.get("x", 5)
        result = c * x

        description = f"For f(x+y)=f(x)+f(y), f(x)=cx, so f({x})={result}"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FunctionalEqMultiplicative(MethodBlock):
    """Solve multiplicative functional equation f(mn) = f(m) + f(n)."""

    def __init__(self):
        super().__init__()
        self.name = "functional_eq_multiplicative"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "functional_equations", "multiplicative"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate multiplicative equation."""
        if input_value is not None and input_value > 1000:
            n = random.choice([64, 128, 256, 512, 1024, 81, 243, 729])
        else:
            n = random.choice([4, 8, 9, 16, 27])
        return {"n": n, "f_p": 1}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solution related to logarithm."""
        n = params.get("n", 10)

        p = 2
        k = 0
        temp = n
        while temp % p == 0:
            k += 1
            temp //= p

        if temp == 1:
            result = k
        else:
            p = 3
            k = 0
            temp = n
            while temp % p == 0:
                k += 1
                temp //= p
            result = k if temp == 1 else 0

        description = f"For f(mn)=f(m)+f(n), f({n})={result}"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FunctionalEqPower(MethodBlock):
    """Solve f(x^n) = n*f(x)."""

    def __init__(self):
        super().__init__()
        self.name = "functional_eq_power"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "functional_equations", "power"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate power equation."""
        if input_value is not None and input_value > 1000:
            base = random.randint(2, 5)
            n = random.randint(5, 10)
            c = random.randint(10, 50)
        else:
            base = random.randint(2, 5)
            n = random.randint(2, 4)
            c = random.randint(1, 3)
        return {"base": base, "n": n, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solution: f(x) = c*log(x)."""
        base, n, c = params.get("base", 2), params.get("n", 10), params.get("c", 10)

        x = base ** n
        f_base = c
        result = n * f_base

        try:
            description = f"For f(x^n)=n*f(x), f({x})=f({base}^{n})={n}*f({base})={result}"
        except ValueError:
            description = f"For f(x^n)=n*f(x), f({x})=f({base}^{n})={n}*f({base})=<large number>"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FunctionalEqAddMult(MethodBlock):
    """Solve f(x+y) = f(x)*f(y) (exponential equation)."""

    def __init__(self):
        super().__init__()
        self.name = "functional_eq_add_mult"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "functional_equations", "exponential"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate exponential equation."""
        if input_value is not None and input_value > 1000:
            a = random.randint(2, 4)
            x = random.randint(8, 12)
        else:
            a = random.randint(2, 4)
            x = random.randint(3, 6)
        return {"a": a, "x": x}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solution: f(x) = a^x."""
        a, x = params.get("a", 10), params.get("x", 5)
        result = a ** x

        try:
            description = f"For f(x+y)=f(x)*f(y), f(x)={a}^x, so f({x})={result}"
        except ValueError:
            description = f"For f(x+y)=f(x)*f(y), f(x)={a}^x, so f({x})=<large number>"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False
