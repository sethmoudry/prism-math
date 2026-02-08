"""
Polynomial-related algebra techniques.

Contains:
- Polynomial operations (evaluation, division, difference, factorization, roots)
- Cyclotomic polynomials (3)
- Laurent polynomials (1)
"""

import random
import math
from typing import Any, Dict, Optional, List
import numpy as np
import sympy as sp

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# POLYNOMIAL OPERATIONS (6 techniques)
# ============================================================================

@register_technique
class PolynomialEvaluation(MethodBlock):
    """Evaluate polynomial P(x) at a point."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_evaluation"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "polynomials", "evaluation"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate polynomial and evaluation point."""
        degree = random.randint(2, 4)
        coeffs = [random.randint(-5, 5) for _ in range(degree + 1)]
        x = random.randint(-3, 3)
        return {"coeffs": coeffs, "x": x}

    def validate_params(self, params, prev_value=None):
        """Validate polynomial evaluation parameters: coefficients must be a non-empty list."""
        coeffs = params.get("coeffs")
        if coeffs is None:
            return False
        if isinstance(coeffs, str):
            try:
                import json
                coeffs = json.loads(coeffs)
            except Exception:
                return False
        if not isinstance(coeffs, (list, tuple)) or len(coeffs) == 0:
            return False
        return True

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Evaluate P(x)."""
        coeffs = params.get("coeffs", [1, 0])
        x = params.get("x", 0)

        if isinstance(coeffs, str):
            try:
                import json
                coeffs = json.loads(coeffs)
            except Exception:
                coeffs = [1, 0]
        coeffs = [float(c) if isinstance(c, str) else c for c in coeffs]

        if isinstance(x, str):
            try:
                x = float(x)
            except ValueError:
                x = 0

        result = coeffs[0]
        for c in coeffs[1:]:
            result = result * x + c

        return MethodResult(
            value=result,
            description=f"P({x}) = {result}",
            params=params, metadata={"coeffs": coeffs, "x": x}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolynomialEvaluationInverseX(MethodBlock):
    """Find x where P(x) = y (root finding)."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_evaluation_inverse_x"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "polynomials", "root_finding", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate polynomial and target value."""
        if input_value is not None and input_value > 1000:
            a = random.randint(1, 5)
            b = random.randint(-100, 100)
            y = random.randint(200, 1000)
        else:
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            y = random.randint(-20, 20)
        return {"coeffs": [a, b], "y": y}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solve ax + b = y for x (currently supports only linear polynomials)."""
        coeffs = params.get("coeffs", [1, 0])
        y = params.get("y", 0)

        if isinstance(coeffs, str):
            try:
                import json
                coeffs = json.loads(coeffs)
            except Exception:
                coeffs = [1, 0]
        coeffs = [float(c) if isinstance(c, str) else c for c in coeffs]

        if isinstance(y, str):
            try:
                y = float(y)
            except ValueError:
                y = 0

        if len(coeffs) != 2:
            raise ValueError(
                f"polynomial_evaluation_inverse_x currently only supports linear polynomials (degree 1). "
                f"Expected 2 coefficients [a, b], got {len(coeffs)}: {coeffs}"
            )

        a, b = coeffs
        x = (y - b) / a
        if x == int(x):
            x = int(x)

        description = f"x = {x} where {a}x + {b} = {y}"

        return MethodResult(
            value=x,
            description=description,
            params=params, metadata={"coeffs": coeffs, "y": y}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolynomialDivision(MethodBlock):
    """Polynomial division p / q using sympy."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_division"
        self.input_type = "polynomial"
        self.output_type = "polynomial"
        self.difficulty = 3
        self.tags = ["algebra", "polynomials", "division"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two polynomials p and q where p is divisible by q."""
        x = sp.Symbol('x')

        if random.choice([True, False]):
            a = random.randint(1, 3)
            b = random.randint(-5, 5)
            q = a * x + b
        else:
            a = random.randint(1, 2)
            b = random.randint(-3, 3)
            c = random.randint(-5, 5)
            q = a * x**2 + b * x + c

        deg_quot = random.randint(1, 2)
        quot_coeffs = [random.randint(-3, 3) for _ in range(deg_quot + 1)]
        quot = sum(c * x**i for i, c in enumerate(quot_coeffs))

        p = sp.expand(q * quot)

        return {
            "p_coeffs": [int(c) for c in sp.Poly(p, x).all_coeffs()],
            "q_coeffs": [int(c) for c in sp.Poly(q, x).all_coeffs()],
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute polynomial division p / q."""
        x = sp.Symbol('x')
        p_coeffs = params.get("p_coeffs", [1, 0])
        q_coeffs = params.get("q_coeffs", [1])

        p = sum(c * x**(len(p_coeffs) - 1 - i) for i, c in enumerate(p_coeffs))
        q = sum(c * x**(len(q_coeffs) - 1 - i) for i, c in enumerate(q_coeffs))

        quotient, remainder = sp.div(p, q, domain='ZZ')

        result_coeffs = [int(c) for c in sp.Poly(quotient, x).all_coeffs()]

        return MethodResult(
            value=result_coeffs,
            description=f"Polynomial division: ({p}) / ({q}) = {quotient} (remainder: {remainder})",
            params=params,
            metadata={"quotient": str(quotient), "remainder": str(remainder)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolynomialDifference(MethodBlock):
    """Compute p(x) - q(x) for two polynomials."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_difference"
        self.input_type = "polynomial"
        self.output_type = "polynomial"
        self.difficulty = 2
        self.tags = ["algebra", "polynomials", "subtraction"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two polynomials p and q."""
        deg_p = random.randint(2, 4)
        deg_q = random.randint(2, 4)

        p_coeffs = [random.randint(-5, 5) for _ in range(deg_p + 1)]
        q_coeffs = [random.randint(-5, 5) for _ in range(deg_q + 1)]

        return {"p_coeffs": p_coeffs, "q_coeffs": q_coeffs}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute p(x) - q(x)."""
        x = sp.Symbol('x')
        p_coeffs = params.get("p_coeffs", [1, 0])
        q_coeffs = params.get("q_coeffs", [1])

        p = sum(c * x**(len(p_coeffs) - 1 - i) for i, c in enumerate(p_coeffs))
        q = sum(c * x**(len(q_coeffs) - 1 - i) for i, c in enumerate(q_coeffs))

        diff = sp.expand(p - q)
        result_coeffs = [int(c) for c in sp.Poly(diff, x).all_coeffs()] if diff != 0 else [0]

        return MethodResult(
            value=result_coeffs,
            description=f"Polynomial difference: ({p}) - ({q}) = {diff}",
            params=params,
            metadata={"difference": str(diff)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolynomialFactorization(MethodBlock):
    """Factor a polynomial using sympy."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_factorization"
        self.input_type = "polynomial"
        self.output_type = "factored_form"
        self.difficulty = 3
        self.tags = ["algebra", "polynomials", "factorization"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a factorable polynomial."""
        x = sp.Symbol('x')

        num_factors = random.randint(2, 3)
        factors = []

        for _ in range(num_factors):
            if random.choice([True, False]):
                a = random.randint(1, 3)
                b = random.randint(-5, 5)
                factors.append(a * x + b)
            else:
                r = random.randint(-3, 3)
                factors.append(x - r)

        poly = sp.expand(sp.prod(factors))
        coeffs = [int(c) for c in sp.Poly(poly, x).all_coeffs()]

        return {"coeffs": coeffs}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Factor the polynomial."""
        x = sp.Symbol('x')
        coeffs = params.get("coeffs", [1, 0])

        poly = sum(c * x**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs))
        factored = sp.factor(poly)

        return MethodResult(
            value=str(factored),
            description=f"Factorization of {poly}: {factored}",
            params=params,
            metadata={"factored_form": str(factored)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolynomialRoots(MethodBlock):
    """Find roots or count roots of polynomials."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_roots"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "polynomials", "roots"]

    def validate_params(self, params, prev_value=None):
        """Validate polynomial root parameters."""
        coeffs = params.get("coeffs")
        if coeffs is None or not isinstance(coeffs, (list, tuple)):
            return False
        if len(coeffs) < 2:
            return False
        return coeffs[0] != 0

    def generate_parameters(self, input_value=None):
        """Generate polynomial parameters for root finding."""
        operation = random.choice(["find_roots", "count_real_roots", "sum_of_roots", "product_of_roots"])
        degree = random.choice([2, 3])

        if degree == 2:
            if operation in ["find_roots", "sum_of_roots", "product_of_roots"]:
                r1 = random.randint(-10, 10)
                r2 = random.randint(-10, 10)
                a = random.randint(1, 3)
                b = -a * (r1 + r2)
                c = a * r1 * r2
                return {"coeffs": [a, b, c], "degree": 2, "operation": operation, "roots": [r1, r2]}
            else:
                a = random.randint(1, 5)
                b = random.randint(-20, 20)
                c = random.randint(-30, 30)
                return {"coeffs": [a, b, c], "degree": 2, "operation": operation}
        else:
            r1 = random.randint(-5, 5)
            r2 = random.randint(-5, 5)
            r3 = random.randint(-5, 5)
            a = 1
            b = -(r1 + r2 + r3)
            c = r1*r2 + r1*r3 + r2*r3
            d = -r1*r2*r3
            return {"coeffs": [a, b, c, d], "degree": 3, "operation": operation, "roots": [r1, r2, r3]}

    def compute(self, input_value, params):
        """Compute polynomial roots or root-related quantities."""
        coeffs = params.get("coeffs", [1, 0, -1])
        degree = params.get("degree", 2)
        operation = params.get("operation", "count_real_roots")

        if degree == 2:
            a, b, c = coeffs[0], coeffs[1], coeffs[2]
            discriminant = b**2 - 4*a*c

            if operation == "count_real_roots":
                if discriminant > 0:
                    result = 2
                    desc = f"Quadratic {a}x^2 + {b}x + {c} has discriminant {discriminant} > 0, so 2 real roots"
                elif discriminant == 0:
                    result = 1
                    desc = f"Quadratic {a}x^2 + {b}x + {c} has discriminant 0, so 1 real root (repeated)"
                else:
                    result = 0
                    desc = f"Quadratic {a}x^2 + {b}x + {c} has discriminant {discriminant} < 0, so 0 real roots"

            elif operation == "sum_of_roots":
                result = -b // a if b % a == 0 else round(-b / a)
                desc = f"Sum of roots of {a}x^2 + {b}x + {c} = -{b}/{a} = {result}"

            elif operation == "product_of_roots":
                result = c // a if c % a == 0 else round(c / a)
                desc = f"Product of roots of {a}x^2 + {b}x + {c} = {c}/{a} = {result}"

            else:
                if discriminant >= 0:
                    sqrt_disc = math.sqrt(discriminant)
                    r1 = (-b + sqrt_disc) / (2*a)
                    r2 = (-b - sqrt_disc) / (2*a)
                    if r1 == int(r1) and r2 == int(r2):
                        result = int(r1) + int(r2)
                    else:
                        result = int(r1 + r2 + 0.5)
                    desc = f"Roots of {a}x^2 + {b}x + {c}: {r1}, {r2}"
                else:
                    result = 0
                    desc = f"No real roots for {a}x^2 + {b}x + {c}"
        else:
            roots = params.get("roots", [0, 0, 0])
            if operation == "sum_of_roots":
                result = sum(roots)
                desc = f"Sum of roots = {result}"
            elif operation == "product_of_roots":
                result = roots[0] * roots[1] * roots[2]
                desc = f"Product of roots = {result}"
            elif operation == "count_real_roots":
                result = 3
                desc = f"Cubic has 3 real roots"
            else:
                result = sum(roots)
                desc = f"Roots: {roots}, sum = {result}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"coeffs": coeffs, "degree": degree, "operation": operation}
        )

    def can_invert(self):
        return False


# ============================================================================
# CYCLOTOMIC POLYNOMIALS (3 techniques)
# ============================================================================

@register_technique
class CyclotomicEval(MethodBlock):
    """Evaluate cyclotomic polynomial Phi_n(x)."""

    def __init__(self):
        super().__init__()
        self.name = "cyclotomic_eval"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "polynomials", "cyclotomic"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate n and evaluation point."""
        n = random.choice([1, 2, 3, 4, 5, 6])
        x = random.randint(2, 5)
        return {"n": n, "x": x}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Evaluate Phi_n(x)."""
        n = params.get("n", 10)
        x = params.get("x", 5)

        cyclotomic_values = {
            1: lambda x: x - 1,
            2: lambda x: x + 1,
            3: lambda x: x**2 + x + 1,
            4: lambda x: x**2 + 1,
            5: lambda x: x**4 + x**3 + x**2 + x + 1,
            6: lambda x: x**2 - x + 1,
        }

        if n in cyclotomic_values:
            result = cyclotomic_values[n](x)
        else:
            result = 0

        return MethodResult(
            value=result,
            description=f"Phi_{n}({x}) = {result}",
            params=params, metadata={"n": n, "x": x}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CyclotomicInverseN(MethodBlock):
    """Find n where Phi_n(x) = value."""

    def __init__(self):
        super().__init__()
        self.name = "cyclotomic_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "polynomials", "cyclotomic", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate evaluation point and target value."""
        x = random.randint(2, 4)
        n = random.choice([1, 2, 3, 4, 5, 6])
        return {"x": x, "target_n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find n by evaluating Phi_k(x) for small k."""
        x = params.get("x", 5)
        target_n = params.get("target_n", 10)

        cyclotomic_values = {
            1: lambda x: x - 1,
            2: lambda x: x + 1,
            3: lambda x: x**2 + x + 1,
            4: lambda x: x**2 + 1,
            5: lambda x: x**4 + x**3 + x**2 + x + 1,
            6: lambda x: x**2 - x + 1,
        }

        target_value = cyclotomic_values[target_n](x)

        for n in range(1, 10):
            if n in cyclotomic_values:
                if cyclotomic_values[n](x) == target_value:
                    result = n
                    description = f"Phi_{result}({x}) = {target_value}"
                    return MethodResult(
                        value=result,
                        description=description,
                        params=params, metadata={"x": x, "target_value": target_value}
                    )

        return MethodResult(
            value=-1,
            description="No match found",
            params=params, metadata={"x": x}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CyclotomicFactorization(MethodBlock):
    """Factor x^n - 1 into cyclotomic polynomials."""

    def __init__(self):
        super().__init__()
        self.name = "cyclotomic_factorization"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "polynomials", "cyclotomic", "factorization"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate n for x^n - 1 with controlled range."""
        n = random.randint(3, 8)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Factor x^n - 1 = Product_{d|n} Phi_d(x)."""
        n = params.get("n", 6)

        if n < 1:
            n = 3
        elif n > 15:
            n = 12

        divisors = [d for d in range(1, n + 1) if n % d == 0]

        if not divisors:
            divisors = [1, n]

        return MethodResult(
            value=len(divisors),
            description=f"x^{n} - 1 = Product Phi_d(x) for d in {divisors}",
            params=params, metadata={"n": n, "divisors": divisors, "num_divisors": len(divisors)}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# LAURENT POLYNOMIALS (1 technique)
# ============================================================================

@register_technique
class LaurentPolynomial(MethodBlock):
    """Operations on Laurent polynomials (polynomials with negative powers)."""

    def __init__(self):
        super().__init__()
        self.name = "laurent_polynomial"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "polynomials", "laurent"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate Laurent polynomial coefficients."""
        coeffs = {}
        min_pow = random.randint(-2, -1)
        max_pow = random.randint(1, 3)
        for p in range(min_pow, max_pow + 1):
            if random.random() > 0.3:
                coeffs[p] = random.randint(-5, 5)
        x = random.randint(2, 3)
        return {"coeffs": coeffs, "x": x}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Evaluate Laurent polynomial."""
        coeffs = params.get("coeffs", 10)
        x = params.get("x", 5)

        result = sum(c * x**p for p, c in coeffs.items())

        if result == int(result):
            result = int(result)

        return MethodResult(
            value=result,
            description=f"P({x}) = {result} for Laurent polynomial",
            params=params, metadata={"coeffs": coeffs, "x": x}
        )

    def can_invert(self) -> bool:
        return False
