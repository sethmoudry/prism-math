"""
Quadratic function techniques and exponential/diophantine equations.

Contains:
- Quadratic functions (5)
- Exponential equations (1)
- Diophantine equations (1)
"""

import random
import math
from typing import Any, Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# QUADRATIC FUNCTIONS (5 techniques)
# ============================================================================

@register_technique
class QuadraticVertex(MethodBlock):
    """Find vertex (h, k) of quadratic function ax^2 + bx + c."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_vertex"
        self.input_type = "quadratic"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["algebra", "quadratic", "vertex", "optimization"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b, c."""
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute vertex (h, k) of quadratic ax^2 + bx + c."""
        a = params.get("a", 10)
        b = params.get("b", 10)
        c = params.get("c", 10)

        h = -b / (2 * a)
        k = c - (b ** 2) / (4 * a)

        h_int = int(h) if h == int(h) else h
        k_int = int(k) if k == int(k) else k

        vertex = (h_int, k_int)
        quad_str = f"{a}x^2 + {b}x + {c}"

        return MethodResult(
            value=vertex,
            description=f"Vertex of {quad_str} is ({h_int}, {k_int}). h = -b/(2a) = {h_int}, k = c - b^2/(4a) = {k_int}",
            params=params,
            metadata={"a": a, "b": b, "c": c, "h": h_int, "k": k_int}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuadraticAxisOfSymmetry(MethodBlock):
    """Find axis of symmetry x = -b/(2a) for quadratic ax^2 + bx + c."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_axis_of_symmetry"
        self.input_type = "quadratic"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["algebra", "quadratic", "symmetry", "axis"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b."""
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute axis of symmetry x = -b/(2a)."""
        a = params.get("a", 10)
        b = params.get("b", 10)

        x_axis = -b / (2 * a)
        x_axis_value = int(x_axis) if x_axis == int(x_axis) else x_axis

        return MethodResult(
            value=x_axis_value,
            description=f"Axis of symmetry for ax^2 + bx + c: x = -b/(2a) = -{b}/(2*{a}) = {x_axis_value}",
            params=params,
            metadata={"a": a, "b": b, "x_axis": x_axis_value}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuadraticDiscriminant(MethodBlock):
    """Compute discriminant D = b^2 - 4ac for quadratic ax^2 + bx + c."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_discriminant"
        self.input_type = "quadratic"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "quadratic", "discriminant", "roots"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b, c."""
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute discriminant b^2 - 4ac."""
        a = params.get("a", 10)
        b = params.get("b", 10)
        c = params.get("c", 10)

        discriminant = b ** 2 - 4 * a * c

        return MethodResult(
            value=int(discriminant),
            description=f"Discriminant of {a}x^2 + {b}x + {c}: D = b^2 - 4ac = {b}^2 - 4*{a}*{c} = {discriminant}",
            params=params,
            metadata={"a": a, "b": b, "c": c, "discriminant": discriminant}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuadraticRootCount(MethodBlock):
    """Count real roots of quadratic ax^2 + bx + c based on discriminant."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_root_count"
        self.input_type = "quadratic"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "quadratic", "roots", "discriminant"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b, c."""
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Determine number of real roots based on discriminant."""
        a = params.get("a", 10)
        b = params.get("b", 10)
        c = params.get("c", 10)

        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            root_count = 2
            root_description = "two distinct real roots"
        elif discriminant == 0:
            root_count = 1
            root_description = "one repeated real root"
        else:
            root_count = 0
            root_description = "no real roots"

        quad_str = f"{a}x^2 + {b}x + {c}"

        return MethodResult(
            value=int(root_count),
            description=f"Discriminant of {quad_str}: D = {discriminant}. Since D {'>' if discriminant > 0 else '=' if discriminant == 0 else '<'} 0, there are {root_count} {root_description}",
            params=params,
            metadata={"a": a, "b": b, "c": c, "discriminant": discriminant, "root_count": root_count}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuadraticFactoring(MethodBlock):
    """Factor a quadratic polynomial and return number of real factors (0, 1, or 2)."""

    def __init__(self):
        super().__init__()
        self.name = "factoring"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "factoring"]

    def generate_parameters(self, input_value=None):
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        return {"a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate factoring parameters."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        try:
            a_val = float(a)
            return a_val != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        a = params.get("a", 10)
        b = params.get("b", 10)
        c = params.get("c", 10)
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            num_factors = 0
        elif discriminant == 0:
            num_factors = 1
        else:
            num_factors = 2
        return MethodResult(
            value=num_factors,
            description=f"Quadratic {a}x^2 + {b}x + {c} has {num_factors} real factors",
            metadata={"a": a, "b": b, "c": c, "discriminant": discriminant}
        )

    def can_invert(self):
        return False


# ============================================================================
# EXPONENTIAL AND DIOPHANTINE EQUATIONS
# ============================================================================

@register_technique
class ExponentialEquation(MethodBlock):
    """Solve exponential equations of various forms."""

    def __init__(self):
        super().__init__()
        self.name = "exponential_equation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "exponential", "equations"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for exponential equation problems."""
        equation_type = random.choice([
            "find_exponent", "find_base", "evaluate_power", "compare_powers", "power_difference", "nested_power"
        ])

        if equation_type == "find_exponent":
            base = random.randint(2, 5)
            exponent = random.randint(2, 8)
            result = base ** exponent
            return {"equation_type": equation_type, "base": base, "result": result, "answer": exponent}

        elif equation_type == "find_base":
            exponent = random.randint(2, 5)
            base = random.randint(2, 6)
            result = base ** exponent
            return {"equation_type": equation_type, "exponent": exponent, "result": result, "answer": base}

        elif equation_type == "evaluate_power":
            base = random.randint(2, 10)
            exponent = random.randint(2, 6)
            return {"equation_type": equation_type, "base": base, "exponent": exponent}

        elif equation_type == "compare_powers":
            a, b = random.randint(2, 5), random.randint(3, 8)
            c, d = random.randint(2, 5), random.randint(3, 8)
            return {"equation_type": equation_type, "a": a, "b": b, "c": c, "d": d}

        elif equation_type == "power_difference":
            base = random.randint(2, 4)
            x = random.randint(3, 8)
            y = random.randint(1, x - 1)
            return {"equation_type": equation_type, "base": base, "x": x, "y": y}

        else:
            a = random.randint(2, 3)
            b = random.randint(2, 4)
            c = random.randint(2, 3)
            return {"equation_type": equation_type, "a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        """Solve the exponential equation."""
        equation_type = params.get("equation_type", "find_exponent")

        if equation_type == "find_exponent":
            base = params.get("base", 2)
            target = params.get("result", 8)
            answer = params.get("answer", 3)

            computed = 0
            power = 1
            while power < target and computed < 100:
                power *= base
                computed += 1
                if power == target:
                    break

            result = answer
            desc = f"Solving {base}^x = {target}: x = {result}"

        elif equation_type == "find_base":
            exponent = params.get("exponent", 2)
            target = params.get("result", 16)
            answer = params.get("answer", 4)

            result = answer
            desc = f"Solving a^{exponent} = {target}: a = {result}"

        elif equation_type == "evaluate_power":
            base = params.get("base", 2)
            exponent = params.get("exponent", 3)
            result = base ** exponent
            desc = f"Evaluating {base}^{exponent} = {result}"

        elif equation_type == "compare_powers":
            a = params.get("a", 2)
            b = params.get("b", 5)
            c = params.get("c", 3)
            d = params.get("d", 3)

            val1 = a ** b
            val2 = c ** d

            if val1 > val2:
                result = 1
                desc = f"Comparing {a}^{b} = {val1} vs {c}^{d} = {val2}: first is larger (result=1)"
            elif val1 < val2:
                result = 2
                desc = f"Comparing {a}^{b} = {val1} vs {c}^{d} = {val2}: second is larger (result=2)"
            else:
                result = 0
                desc = f"Comparing {a}^{b} = {val1} vs {c}^{d} = {val2}: equal (result=0)"

        elif equation_type == "power_difference":
            base = params.get("base", 2)
            x = params.get("x", 5)
            y = params.get("y", 3)

            result = base ** x - base ** y
            desc = f"Computing {base}^{x} - {base}^{y} = {base**x} - {base**y} = {result}"

        else:
            a = params.get("a", 2)
            b = params.get("b", 3)
            c = params.get("c", 2)

            inner = a ** b
            result = inner ** c
            equivalent_exp = b * c
            desc = f"Computing ({a}^{b})^{c} = {inner}^{c} = {result} = {a}^{equivalent_exp}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"equation_type": equation_type, **params}
        )

    def can_invert(self):
        return False


@register_technique
class DiophantineEquations(MethodBlock):
    """Linear Diophantine equation ax + by = c with integer solutions."""

    def __init__(self):
        super().__init__()
        self.name = "diophantine_equations"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "number_theory"]

    def generate_parameters(self, input_value=None):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        x_sol = random.randint(1, 10)
        y_sol = random.randint(1, 10)
        c = a * x_sol + b * y_sol
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        a = params.get("a", 2)
        b = params.get("b", 3)
        c = params.get("c", 10)
        count = 0
        for x in range(c // a + 1):
            remainder = c - a * x
            if remainder >= 0 and remainder % b == 0:
                count += 1
        return MethodResult(
            value=count,
            description=f"Number of non-negative solutions to {a}x + {b}y = {c}: {count}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# NEW TIER 1 ALGEBRA METHODS
# ============================================================================

@register_technique
class CompleteSquare(MethodBlock):
    """Transform quadratic ax^2 + bx + c into vertex form a(x - h)^2 + k."""

    def __init__(self):
        super().__init__()
        self.name = "complete_square"
        self.input_type = "quadratic"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["algebra", "quadratic", "completing_the_square", "transformation"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b, c."""
        a = random.randint(1, 5)
        b = random.randint(-20, 20)
        c = random.randint(-30, 30)
        return {"a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate that a is non-zero."""
        a = params.get("a")
        return a is not None and a != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Complete the square: ax^2 + bx + c -> a(x - h)^2 + k."""
        a = params.get("a", 1)
        b = params.get("b", 0)
        c = params.get("c", 0)

        # h = -b/(2a), k = c - b^2/(4a)
        h = -b / (2 * a)
        k = c - (b ** 2) / (4 * a)

        # Convert to int if whole number
        h_val = int(h) if h == int(h) else h
        k_val = int(k) if k == int(k) else k

        quad_str = f"{a}x^2 + {b}x + {c}"
        vertex_str = f"{a}(x - {h_val})^2 + {k_val}" if h_val >= 0 else f"{a}(x + {-h_val})^2 + {k_val}"

        return MethodResult(
            value=(a, h_val, k_val),
            description=f"Complete the square: {quad_str} = {vertex_str}. Vertex at ({h_val}, {k_val})",
            params=params,
            metadata={"a": a, "b": b, "c": c, "h": h_val, "k": k_val}
        )

    def can_invert(self) -> bool:
        return True


@register_technique
class QuadraticFormula(MethodBlock):
    """Find roots of quadratic ax^2 + bx + c = 0 using the quadratic formula."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_formula"
        self.input_type = "quadratic"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["algebra", "quadratic", "roots", "formula"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate quadratic coefficients a, b, c with real roots."""
        a = random.randint(1, 5)
        # Ensure real roots by making discriminant non-negative
        b = random.randint(-20, 20)
        # Choose c such that b^2 - 4ac >= 0
        max_c = (b ** 2) // (4 * a)
        c = random.randint(-max_c, max_c) if max_c > 0 else 0
        return {"a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate that a is non-zero."""
        a = params.get("a")
        return a is not None and a != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply quadratic formula: x = (-b ± sqrt(b^2 - 4ac)) / (2a)."""
        a = params.get("a", 1)
        b = params.get("b", 0)
        c = params.get("c", 0)

        discriminant = b ** 2 - 4 * a * c
        quad_str = f"{a}x^2 + {b}x + {c} = 0"

        if discriminant < 0:
            return MethodResult(
                value=(),
                description=f"Solve {quad_str}: Discriminant = {discriminant} < 0, no real roots",
                params=params,
                metadata={"discriminant": discriminant, "root_count": 0}
            )
        elif discriminant == 0:
            x = -b / (2 * a)
            x_val = int(x) if x == int(x) else x
            return MethodResult(
                value=(x_val,),
                description=f"Solve {quad_str}: x = -b/(2a) = {x_val} (repeated root)",
                params=params,
                metadata={"discriminant": 0, "root_count": 1, "x1": x_val}
            )
        else:
            sqrt_d = math.sqrt(discriminant)
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)

            # Check if roots are integers
            x1_val = int(x1) if x1 == int(x1) else round(x1, 6)
            x2_val = int(x2) if x2 == int(x2) else round(x2, 6)

            return MethodResult(
                value=(x1_val, x2_val),
                description=f"Solve {quad_str}: x = (-{b} ± √{discriminant})/(2·{a}) = {x1_val}, {x2_val}",
                params=params,
                metadata={"discriminant": discriminant, "root_count": 2, "x1": x1_val, "x2": x2_val}
            )

    def can_invert(self) -> bool:
        return False


@register_technique
class RationalizeDenominator(MethodBlock):
    """Eliminate radicals from denominator by multiplying by conjugate."""

    def __init__(self):
        super().__init__()
        self.name = "rationalize_denominator"
        self.input_type = "expression"
        self.output_type = "expression"
        self.difficulty = 2
        self.tags = ["algebra", "radicals", "rationalization", "simplification"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a fraction with radical denominator."""
        # Simple case: a / sqrt(b) -> a*sqrt(b) / b
        numerator = random.randint(1, 10)
        radicand = random.choice([2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
        return {"numerator": numerator, "radicand": radicand, "denom_type": "simple"}

    def validate_params(self, params, prev_value=None):
        """Validate parameters."""
        radicand = params.get("radicand")
        return radicand is not None and radicand > 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Rationalize the denominator."""
        numerator = params.get("numerator", 1)
        radicand = params.get("radicand", 2)
        denom_type = params.get("denom_type", "simple")

        if denom_type == "simple":
            # a / sqrt(b) -> a*sqrt(b) / b
            # Multiply by sqrt(b)/sqrt(b)
            new_numer = numerator  # coefficient of sqrt(radicand)
            new_denom = radicand

            # Simplify by GCD
            g = math.gcd(new_numer, new_denom)
            new_numer //= g
            new_denom //= g

            original = f"{numerator}/√{radicand}"
            if new_denom == 1:
                result_str = f"{new_numer}√{radicand}" if new_numer != 1 else f"√{radicand}"
            else:
                result_str = f"{new_numer}√{radicand}/{new_denom}" if new_numer != 1 else f"√{radicand}/{new_denom}"

            return MethodResult(
                value={"coeff": new_numer, "radicand": radicand, "denom": new_denom},
                description=f"Rationalize {original}: multiply by √{radicand}/√{radicand} = {result_str}",
                params=params,
                metadata={"original_numer": numerator, "radicand": radicand, "new_numer": new_numer, "new_denom": new_denom}
            )
        else:
            # Conjugate case: a / (b + sqrt(c)) -> a(b - sqrt(c)) / (b^2 - c)
            b = params.get("denom_rational", 1)
            c = radicand

            conjugate_denom = b ** 2 - c
            if conjugate_denom == 0:
                return MethodResult(
                    value=None,
                    description=f"Cannot rationalize: conjugate gives zero denominator",
                    params=params,
                    metadata={"error": "zero_denominator"}
                )

            return MethodResult(
                value={"numer_rational": numerator * b, "numer_radical": -numerator, "radicand": c, "denom": conjugate_denom},
                description=f"Rationalize {numerator}/({b} + √{c}): multiply by conjugate ({b} - √{c})",
                params=params,
                metadata={"b": b, "c": c, "conjugate_denom": conjugate_denom}
            )

    def can_invert(self) -> bool:
        return False


@register_technique
class SolveEquation(MethodBlock):
    """Solve algebraic equations (linear, quadratic, polynomial)."""

    def __init__(self):
        super().__init__()
        self.name = "solve_equation"
        self.input_type = "equation"
        self.output_type = "solution_set"
        self.difficulty = 4
        self.tags = ["algebra", "equations", "solving", "roots"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate equation parameters."""
        eq_type = random.choice(["linear", "quadratic"])

        if eq_type == "linear":
            # ax + b = 0 -> x = -b/a
            a = random.randint(1, 10)
            b = random.randint(-50, 50)
            return {"equation_type": "linear", "a": a, "b": b}
        else:
            # ax^2 + bx + c = 0
            a = random.randint(1, 5)
            b = random.randint(-20, 20)
            # Ensure real roots
            max_c = (b ** 2) // (4 * a)
            c = random.randint(-max_c, max_c) if max_c > 0 else 0
            return {"equation_type": "quadratic", "a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate equation parameters."""
        eq_type = params.get("equation_type")
        if eq_type not in ("linear", "quadratic", "polynomial"):
            return False

        if eq_type == "linear":
            a = params.get("a")
            return a is not None and a != 0
        elif eq_type == "quadratic":
            a = params.get("a")
            return a is not None and a != 0
        else:
            coeffs = params.get("coefficients")
            return coeffs is not None and len(coeffs) >= 2 and coeffs[0] != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solve the equation."""
        eq_type = params.get("equation_type", "linear")

        if eq_type == "linear":
            return self._solve_linear(params)
        elif eq_type == "quadratic":
            return self._solve_quadratic(params)
        else:
            return self._solve_polynomial(params)

    def _solve_linear(self, params: Dict[str, Any]) -> MethodResult:
        """Solve linear equation ax + b = 0."""
        a = params.get("a", 1)
        b = params.get("b", 0)

        x = -b / a
        x_val = int(x) if x == int(x) else x

        eq_str = f"{a}x + {b} = 0" if b >= 0 else f"{a}x - {-b} = 0"

        return MethodResult(
            value=[x_val],
            description=f"Solve {eq_str}: x = -{b}/{a} = {x_val}",
            params=params,
            metadata={"equation_type": "linear", "solution": x_val}
        )

    def _solve_quadratic(self, params: Dict[str, Any]) -> MethodResult:
        """Solve quadratic equation ax^2 + bx + c = 0."""
        a = params.get("a", 1)
        b = params.get("b", 0)
        c = params.get("c", 0)

        discriminant = b ** 2 - 4 * a * c
        eq_str = f"{a}x² + {b}x + {c} = 0"

        if discriminant < 0:
            return MethodResult(
                value=[],
                description=f"Solve {eq_str}: D = {discriminant} < 0, no real solutions",
                params=params,
                metadata={"equation_type": "quadratic", "discriminant": discriminant}
            )
        elif discriminant == 0:
            x = -b / (2 * a)
            x_val = int(x) if x == int(x) else x
            return MethodResult(
                value=[x_val],
                description=f"Solve {eq_str}: x = {x_val} (double root)",
                params=params,
                metadata={"equation_type": "quadratic", "discriminant": 0}
            )
        else:
            sqrt_d = math.sqrt(discriminant)
            x1 = (-b + sqrt_d) / (2 * a)
            x2 = (-b - sqrt_d) / (2 * a)

            x1_val = int(x1) if x1 == int(x1) else round(x1, 6)
            x2_val = int(x2) if x2 == int(x2) else round(x2, 6)

            return MethodResult(
                value=[x1_val, x2_val],
                description=f"Solve {eq_str}: x = {x1_val} or x = {x2_val}",
                params=params,
                metadata={"equation_type": "quadratic", "discriminant": discriminant}
            )

    def _solve_polynomial(self, params: Dict[str, Any]) -> MethodResult:
        """Solve polynomial equation using rational root theorem for low degrees."""
        coeffs = params.get("coefficients", [1, 0])

        # For now, handle linear and quadratic via delegation
        degree = len(coeffs) - 1

        if degree == 1:
            return self._solve_linear({"a": coeffs[0], "b": coeffs[1]})
        elif degree == 2:
            return self._solve_quadratic({"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]})
        else:
            # Higher degree: try rational root theorem
            # p/q where p divides constant term and q divides leading coefficient
            leading = abs(coeffs[0])
            constant = abs(coeffs[-1])

            if constant == 0:
                # x is a factor
                return MethodResult(
                    value=[0],
                    description=f"Polynomial has x as factor, so x=0 is a root",
                    params=params,
                    metadata={"equation_type": "polynomial", "degree": degree, "partial": True}
                )

            # Try small rational roots
            roots = []
            for p in range(1, min(constant + 1, 20)):
                if constant % p == 0:
                    for q in range(1, min(leading + 1, 10)):
                        if leading % q == 0:
                            for sign in [1, -1]:
                                candidate = sign * p / q
                                # Evaluate polynomial
                                val = sum(c * (candidate ** (degree - i)) for i, c in enumerate(coeffs))
                                if abs(val) < 1e-9:
                                    if candidate not in roots:
                                        roots.append(int(candidate) if candidate == int(candidate) else candidate)

            if roots:
                return MethodResult(
                    value=roots,
                    description=f"Found rational roots: {roots}",
                    params=params,
                    metadata={"equation_type": "polynomial", "degree": degree}
                )
            else:
                return MethodResult(
                    value=[],
                    description=f"No simple rational roots found for degree-{degree} polynomial",
                    params=params,
                    metadata={"equation_type": "polynomial", "degree": degree, "needs_numerical": True}
                )

    def can_invert(self) -> bool:
        return False
