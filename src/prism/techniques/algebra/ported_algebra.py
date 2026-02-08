"""
Ported algebra and function methods.

High-impact methods covering:
- Algebraic expansion
- Function composition and iteration
- Vieta's formulas (advanced)
- Complex numbers
- Proof techniques (contradiction)
- Floor system solutions
- Functional conditions
- Piecewise functions
- Expression forms
- Sequences and sums
- Trigonometry (sine, arccosine)
- Absolute value analysis
- Random integer seed
- Power/factorial towers
- Inequality constraints
"""

import math
import random
from typing import Any, Dict, List, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# ALGEBRAIC EXPANSION
# ============================================================================

@register_technique
class AlgebraicExpand(MethodBlock):
    """Algebraic expansion of (a + b)^n expressions."""

    def __init__(self):
        super().__init__()
        self.name = "algebraic_expand"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "expansion", "polynomials"]

    def generate_parameters(self, input_value=None):
        return {
            "a": random.randint(1, 5),
            "b": random.randint(1, 5),
            "power": random.randint(2, 3)
        }

    def compute(self, input_value, params):
        a = params.get("a", 1)
        b = params.get("b", 1)
        power = params.get("power", 2)
        if isinstance(b, str):
            try:
                b = int(b)
            except ValueError:
                b = 1
        if isinstance(power, str):
            try:
                power = int(power)
            except ValueError:
                power = 2

        if power == 2:
            result = a ** 2 + 2 * a * b + b ** 2
            description = f"({a} + {b})^2 = {result}"
        elif power == 3:
            result = a ** 3 + 3 * a ** 2 * b + 3 * a * b ** 2 + b ** 3
            description = f"({a} + {b})^3 = {result}"
        else:
            result = (a + b) ** power
            description = f"({a} + {b})^{power} = {result}"

        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# FUNCTION METHODS
# ============================================================================

@register_technique
class EvaluateFunctionComposition(MethodBlock):
    """Evaluate f(g(x)) composition where f and g are linear functions."""

    def __init__(self):
        super().__init__()
        self.name = "evaluate_function_composition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "functions", "composition"]

    def generate_parameters(self, input_value=None):
        return {
            "x": input_value if input_value is not None else random.randint(-10, 10),
            "a": random.randint(-5, 5),
            "b": random.randint(-10, 10),
            "c": random.randint(-5, 5),
            "d": random.randint(-10, 10)
        }

    def compute(self, input_value, params):
        x = input_value if input_value is not None else params.get("x", 2)
        a, b = params.get("a", 2), params.get("b", 3)
        c, d = params.get("c", 1), params.get("d", 4)
        gx = c * x + d
        result = a * gx + b
        return MethodResult(
            value=result,
            description=f"f(x)={a}x+{b}, g(x)={c}x+{d}. f(g({x}))=f({gx})={result}",
            params=params, metadata={"techniques_used": [self.name], "gx": gx}
        )

    def can_invert(self):
        return False


@register_technique
class ComputeFunctionIterate(MethodBlock):
    """Compute f^n(x) = f(f(...f(x)...)) for n iterations."""

    def __init__(self):
        super().__init__()
        self.name = "compute_function_iterate"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "functions", "iteration"]

    def generate_parameters(self, input_value=None):
        return {
            "x": input_value if input_value is not None else random.randint(-5, 5),
            "a": random.choice([2, 3, -1, -2]),
            "b": random.randint(-5, 5),
            "n": random.randint(1, 5)
        }

    def compute(self, input_value, params):
        x = input_value if input_value is not None else params.get("x", 1)
        a, b = params.get("a", 2), params.get("b", 1)
        n = params.get("n", 3)
        result = x
        for _ in range(n):
            result = a * result + b
        return MethodResult(
            value=result,
            description=f"f(x)={a}x+{b}. f^{n}({x}) = {result}",
            params=params, metadata={"techniques_used": [self.name], "iterations": n}
        )

    def can_invert(self):
        return False


@register_technique
class ApplyFunctionalCondition(MethodBlock):
    """Apply condition to functional equation f(x+y) = f(x) + f(y)."""

    def __init__(self):
        super().__init__()
        self.name = "apply_functional_condition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "functional_equations", "substitution"]

    def generate_parameters(self, input_value=None):
        k = random.randint(1, 10)
        x = random.randint(1, 20)
        y = random.randint(1, 20)
        return {"k": k, "x": x, "y": y}

    def compute(self, input_value, params):
        k, x, y = params.get("k", 3), params.get("x", 5), params.get("y", 7)
        fx, fy = k * x, k * y
        result = k * (x + y)
        return MethodResult(
            value=result,
            description=f"f(x)={k}x. f({x}+{y}) = {fx}+{fy} = {result}",
            params=params, metadata={"techniques_used": [self.name], "fx": fx, "fy": fy}
        )

    def can_invert(self):
        return False


@register_technique
class DefinePiecewiseFunction(MethodBlock):
    """Define a piecewise function with conditions and evaluate at a point."""

    def __init__(self):
        super().__init__()
        self.name = "define_piecewise_function"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "functions", "piecewise"]

    def generate_parameters(self, input_value=None):
        threshold = random.randint(0, 10)
        a, b = random.randint(-5, 5), random.randint(-10, 10)
        c, d = random.randint(-5, 5), random.randint(-10, 10)
        x = input_value if input_value is not None else random.randint(-10, 20)
        return {"x": x, "threshold": threshold, "a": a, "b": b, "c": c, "d": d}

    def compute(self, input_value, params):
        x = input_value if input_value is not None else params.get("x", 5)
        threshold = params.get("threshold", 5)
        a, b = params.get("a", 2), params.get("b", 1)
        c, d = params.get("c", -1), params.get("d", 10)
        if x < threshold:
            result = a * x + b
        else:
            result = c * x + d
        return MethodResult(
            value=result,
            description=f"f(x) = {{{a}x+{b} if x<{threshold}, {c}x+{d} if x>={threshold}}}. f({x})={result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ExpressInForm(MethodBlock):
    """Express a/b + c/d in form p/q with gcd(p,q)=1, return p+q."""

    def __init__(self):
        super().__init__()
        self.name = "express_in_form"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "simplification", "forms"]

    def generate_parameters(self, input_value=None):
        return {
            "a": random.randint(1, 20), "b": random.randint(1, 20),
            "c": random.randint(1, 20), "d": random.randint(1, 20)
        }

    def compute(self, input_value, params):
        a, b = params.get("a", 1), params.get("b", 2)
        c, d = params.get("c", 1), params.get("d", 3)
        numerator = a * d + b * c
        denominator = b * d
        g = math.gcd(numerator, denominator)
        p, q = numerator // g, denominator // g
        result = p + q
        return MethodResult(
            value=result,
            description=f"{a}/{b} + {c}/{d} = {p}/{q}. p+q = {result}",
            params=params, metadata={"techniques_used": [self.name], "p": p, "q": q}
        )

    def can_invert(self):
        return False


# ============================================================================
# VIETA (ADVANCED)
# ============================================================================

@register_technique
class VietaAdvanced(MethodBlock):
    """Advanced Vieta's formulas: power sums, symmetric functions, reciprocal roots."""

    QUESTION_TYPES = [
        "power_sum", "symmetric_function", "reciprocal_roots",
        "root_products", "root_transformation"
    ]

    def __init__(self):
        super().__init__()
        self.name = "vieta"
        self.input_type = "none"
        self.output_type = "answer"
        self.difficulty = 4
        self.tags = ["algebra", "DEEP_INSIGHT", "vieta", "polynomials"]

    def generate_parameters(self, input_value=None):
        qt = random.choice(self.QUESTION_TYPES)
        if qt == "power_sum":
            return {"question_type": qt, "sum": random.randint(2, 15),
                    "product": random.randint(1, 10), "power": random.randint(2, 5)}
        elif qt == "symmetric_function":
            return {"question_type": qt, "a": random.randint(1, 5),
                    "b": random.randint(1, 10), "c": random.randint(1, 10)}
        elif qt == "reciprocal_roots":
            return {"question_type": qt, "p": random.randint(3, 15),
                    "q": random.randint(1, 10)}
        elif qt == "root_products":
            return {"question_type": qt, "p": random.randint(3, 15),
                    "q": random.randint(1, 10)}
        else:
            return {"question_type": qt, "p": random.randint(3, 10),
                    "q": random.randint(1, 8)}

    def compute(self, input_value, params):
        qt = params.get("question_type", "power_sum")
        if qt == "power_sum":
            s, p, k = params.get("sum", 10), params.get("product", 20), params.get("power", 2)
            P = [0, s]
            for i in range(2, k + 1):
                P.append(s * P[i - 1] - p * P[i - 2])
            answer = P[k]
            desc = f"Roots of x^2-{s}x+{p}=0, find r^{k}+s^{k}"
        elif qt == "symmetric_function":
            a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)
            answer = a * a - 2 * b
            desc = f"Roots of x^3-{a}x^2+{b}x-{c}=0, find r^2+s^2+t^2"
        elif qt == "reciprocal_roots":
            p, q = params.get("p", 5), params.get("q", 10)
            answer = (p * 1000) // q
            desc = f"Roots of x^2-{p}x+{q}=0, find 1/r+1/s (x1000)"
        elif qt == "root_products":
            p, q = params.get("p", 5), params.get("q", 10)
            answer = q + p + 1
            desc = f"Roots of x^2-{p}x+{q}=0, find (r+1)(s+1)"
        else:
            p, q = params.get("p", 5), params.get("q", 10)
            answer = p * p - 2 * q
            desc = f"Roots of x^2-{p}x+{q}=0, find r^2+s^2"
        return MethodResult(
            value=answer, description=desc, params=params,
            metadata={"question_type": qt}
        )

    def can_invert(self):
        return False


# ============================================================================
# COMPLEX NUMBERS
# ============================================================================

@register_technique
class ComplexNumbersMethod(MethodBlock):
    """Basic complex number arithmetic operations."""

    def __init__(self):
        super().__init__()
        self.name = "complex_numbers"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["complex_numbers", "arithmetic"]

    def generate_parameters(self, input_value=None):
        a, b = random.randint(1, 20), random.randint(1, 20)
        return {
            "z1": complex(a, b), "z2": complex(1, 0),
            "operation": "modulus_squared"
        }

    def compute(self, input_value, params):
        import cmath
        z1 = params.get("z1", input_value if input_value is not None else 3 + 4j)
        z2 = params.get("z2", 1 + 2j)
        operation = params.get("operation", "modulus_squared")

        if isinstance(z1, tuple) and len(z1) == 2:
            z1 = complex(z1[0], z1[1])
        if isinstance(z2, tuple) and len(z2) == 2:
            z2 = complex(z2[0], z2[1])

        if operation == "add":
            result = z1 + z2
            description = f"({z1}) + ({z2}) = {result}"
        elif operation == "multiply":
            result = z1 * z2
            description = f"({z1}) * ({z2}) = {result}"
        elif operation == "modulus":
            result = abs(z1)
            description = f"|{z1}| = {result}"
        elif operation == "modulus_squared":
            result = int(z1.real ** 2 + z1.imag ** 2)
            description = f"|{z1}|^2 = {int(z1.real)}^2 + {int(z1.imag)}^2 = {result}"
        elif operation == "conjugate":
            result = z1.conjugate()
            description = f"conjugate({z1}) = {result}"
        else:
            result = int(z1.real ** 2 + z1.imag ** 2)
            description = f"|{z1}|^2 = {result}"

        if isinstance(result, complex) and result.imag == 0:
            result = result.real
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name], "operation": operation}
        )

    def can_invert(self):
        return False


# ============================================================================
# PROOF TECHNIQUES
# ============================================================================

@register_technique
class ContradictionArgument(MethodBlock):
    """Proof by contradiction: assume false, derive impossibility."""

    def __init__(self):
        super().__init__()
        self.name = "contradiction_argument"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["reasoning", "proof", "contradiction"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(2, 50)
        return {"n": n, "claim": "uniqueness"}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value else 10)
        claim = params.get("claim", "uniqueness")
        if claim == "uniqueness":
            result = 1
            desc = f"By contradiction: exactly 1 solution exists for n={n}"
        elif claim == "impossibility":
            result = 0
            desc = f"By contradiction: no solution exists for n={n}"
        else:
            result = n
            desc = f"Proved by contradiction for n={n}"
        return MethodResult(
            value=result, description=desc, params=params,
            metadata={"n": n, "claim": claim}
        )

    def can_invert(self):
        return False


# ============================================================================
# FLOOR SYSTEM SOLUTIONS
# ============================================================================

@register_technique
class FloorSystemSolutions(MethodBlock):
    """Count solutions to floor-based systems like floor(x/a)*floor(y/b)=N."""

    def __init__(self):
        super().__init__()
        self.name = "floor_system_solutions"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["floor_function", "counting", "deep_insight", "factorization"]

    def generate_parameters(self, input_value=None):
        a = random.randint(2, 10)
        b = random.randint(2, 10)
        N = random.randint(10, 100)
        x_bound = random.randint(50, 200)
        y_bound = random.randint(50, 200)
        return {
            "system_type": "product_floor",
            "a": a, "b": b, "N": N,
            "x_bound": x_bound, "y_bound": y_bound
        }

    def compute(self, input_value, params):
        system_type = params.get("system_type", "product_floor")
        a = params.get("a", 3)
        b = params.get("b", 5)
        N = params.get("N", 10)
        x_bound = params.get("x_bound", 100)
        y_bound = params.get("y_bound", 100)

        if system_type == "product_floor":
            count = 0
            for x in range(1, x_bound + 1):
                fx = x // a
                if fx == 0:
                    continue
                for y in range(1, y_bound + 1):
                    fy = y // b
                    if fx * fy == N:
                        count += 1
            result = count
        else:
            result = 0

        return MethodResult(
            value=result,
            description=f"Count solutions: floor(x/{a})*floor(y/{b})={N} in [1,{x_bound}]x[1,{y_bound}]: {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# SEQUENCES
# ============================================================================

@register_technique
class SequenceSum(MethodBlock):
    """Sum elements of a sequence."""

    def __init__(self):
        super().__init__()
        self.name = "sequence_sum"
        self.difficulty = 1
        self.input_type = "sequence"
        self.output_type = "integer"
        self.tags = ["sequences", "algebra"]

    def generate_parameters(self, input_value=None):
        return {}

    def compute(self, input_value, params):
        if input_value is None:
            input_value = [random.randint(1, 100) for _ in range(random.randint(5, 10))]
        result = sum(input_value)
        return MethodResult(
            value=result,
            description=f"Sum the first {len(input_value)} terms of the sequence",
            params=params, metadata={"sequence_length": len(input_value)}
        )

    def can_invert(self):
        return False


# ============================================================================
# TRIGONOMETRY
# ============================================================================

@register_technique
class SineMethod(MethodBlock):
    """Compute sine of an angle."""

    def __init__(self):
        super().__init__()
        self.name = "sine"
        self.input_type = "angle"
        self.output_type = "trig_value"
        self.difficulty = 1
        self.tags = ["trigonometry", "basic"]

    def generate_parameters(self, input_value=None):
        common_angles = [0, 30, 45, 60, 90, 120, 135, 150, 180, 270, 360]
        angle_deg = random.choice(common_angles)
        use_degrees = random.choice([True, False])
        angle = angle_deg if use_degrees else math.radians(angle_deg)
        return {"angle": angle, "use_degrees": use_degrees}

    def compute(self, input_value, params):
        angle = params.get("angle", input_value)
        use_degrees = params.get("use_degrees", True)
        if use_degrees:
            angle_rad = math.radians(angle)
        else:
            angle_rad = angle
        value = round(math.sin(angle_rad), 10)
        return MethodResult(
            value=value,
            description=f"sin({angle}) = {value}",
            params=params, metadata={"angle": angle, "use_degrees": use_degrees}
        )

    def can_invert(self):
        return False


@register_technique
class ArccosineMethod(MethodBlock):
    """Compute arccosine (inverse cosine) of a value."""

    def __init__(self):
        super().__init__()
        self.name = "arccosine"
        self.input_type = "trig_value"
        self.output_type = "angle"
        self.difficulty = 2
        self.tags = ["trigonometry", "inverse"]

    def generate_parameters(self, input_value=None):
        common_values = [-1, -0.866, -0.707, -0.5, 0, 0.5, 0.707, 0.866, 1]
        value = random.choice(common_values)
        return {"value": value, "return_degrees": random.choice([True, False])}

    def compute(self, input_value, params):
        value = params.get("value", input_value)
        return_degrees = params.get("return_degrees", True)
        value = max(-1, min(1, value))
        angle_rad = math.acos(value)
        if return_degrees:
            angle = round(math.degrees(angle_rad), 10)
        else:
            angle = round(angle_rad, 10)
        return MethodResult(
            value=angle,
            description=f"arccos({value}) = {angle}",
            params=params, metadata={"input": value, "return_degrees": return_degrees}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        return_degrees = params.get("return_degrees", True)
        angle_rad = math.radians(output_value) if return_degrees else output_value
        return math.cos(angle_rad)


# ============================================================================
# MISCELLANEOUS
# ============================================================================

@register_technique
class AbsoluteValueAnalysis(MethodBlock):
    """Analyze absolute value expressions."""

    def __init__(self):
        super().__init__()
        self.name = "absolute_value_analysis"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra"]

    def generate_parameters(self, input_value=None):
        return {"value": random.randint(-100, 100)}

    def compute(self, input_value, params):
        value = params.get("value", input_value if input_value is not None else 0)
        if isinstance(value, (int, float)):
            result = abs(value)
            description = f"|{value}| = {result}"
        else:
            result = 0
            description = f"Absolute value analysis"
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name], "original_value": value}
        )

    def can_invert(self):
        return False


@register_technique
class RandomInteger(MethodBlock):
    """Generate a random integer in a range."""

    def __init__(self):
        super().__init__()
        self.name = "random_integer"
        self.difficulty = 1
        self.input_type = "none"
        self.output_type = "integer"
        self.tags = ["number_theory", "seed"]

    def generate_parameters(self, input_value=None):
        min_val = random.randint(1000, 50000)
        max_val = random.randint(min_val + 1000, min_val + 40000)
        value = random.randint(min_val, max_val)
        return {"value": value, "min": min_val, "max": max_val}

    def compute(self, input_value, params):
        value = params.get("value", 10)
        return MethodResult(
            value=value, description=f"Consider the number {value}",
            params=params, metadata={"range": (params.get("min", 10), params.get("max", 10))}
        )

    def can_invert(self):
        return False


@register_technique
class PowerTower(MethodBlock):
    """Tetration: base^(base^(base^...)) height times."""

    def __init__(self):
        super().__init__()
        self.name = "power_tower"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "exponentiation", "tetration"]

    def generate_parameters(self, input_value=None):
        return {"base": random.randint(2, 3), "height": random.randint(1, 3)}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        height = params.get("height", 2)
        if height < 1:
            raise ValueError("Height must be at least 1")
        result = base
        for _ in range(height - 1):
            if result > 100:
                raise ValueError(f"Power tower too large: {base}^{result}")
            result = base ** result
        return MethodResult(
            value=result,
            description=f"Power tower: {base}^(...) height {height} = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class FactorialTower(MethodBlock):
    """Iterated factorial: ((n!)!)!... height times."""

    def __init__(self):
        super().__init__()
        self.name = "factorial_tower"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "factorial", "iteration"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(1, 5), "height": random.randint(1, 2)}

    def compute(self, input_value, params):
        n = params.get("n", 3)
        height = params.get("height", 1)
        if height < 1:
            raise ValueError("Height must be at least 1")

        result = n
        for i in range(height):
            if result > 12:
                return MethodResult(
                    value=10 ** 15,
                    description=f"Factorial tower overflow at iteration {i + 1}",
                    params=params,
                    metadata={"techniques_used": [self.name], "overflow": True}
                )
            result = math.factorial(result)

        return MethodResult(
            value=result,
            description=f"Factorial tower: (({n}!)!...)! height {height} = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class InequalityConstraints(MethodBlock):
    """Determine valid ranges for variables using inequality constraints."""

    def __init__(self):
        super().__init__()
        self.name = "inequality_constraints"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["inequalities"]

    def generate_parameters(self, input_value=None):
        return {"a": random.randint(1, 10), "b": random.randint(1, 10)}

    def compute(self, input_value, params):
        a, b = params.get("a", 10), params.get("b", 10)
        disc = a ** 2 + 4 * b
        x_max = (a + math.sqrt(disc)) / 2
        result = int(x_max)
        return MethodResult(
            value=result,
            description=f"Max integer x satisfying x^2 < {a}x + {b}: x = {result}",
            params=params, metadata={"a": a, "b": b}
        )

    def can_invert(self):
        return False
