"""
Algebra primitives.
"""

import math
import random
from typing import List, Tuple, Union, Optional
from sympy import (
    symbols, simplify, expand, sin, cos, tan, exp, log, sqrt, sympify
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class PolynomialDiscriminant(MethodBlock):
    """Compute discriminant of quadratic polynomial."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_discriminant"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["algebra", "polynomials"]

    def generate_parameters(self, input_value=None):
        """Generate random quadratic coefficients."""
        return {
            "a": random.randint(1, 10),
            "b": random.randint(-10, 10),
            "c": random.randint(-10, 10)
        }

    def validate_params(self, params, prev_value=None):
        """Validate polynomial discriminant parameters: a, b, c must exist."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        try:
            # Ensure they can be used as numbers
            float(a)
            float(b)
            float(c)
            return True
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        """Compute discriminant b² - 4ac of quadratic ax² + bx + c."""
        a = params.get("a", 1)
        b = params.get("b", -3)
        c = params.get("c", 2)

        result = b * b - 4 * a * c

        return MethodResult(
            value=result,
            description=f"Discriminant of {a}x² + {b}x + {c} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class PolynomialDegree(MethodBlock):
    """Find degree of polynomial (highest power with non-zero coefficient)."""

    def __init__(self):
        super().__init__()
        self.name = "polynomial_degree"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "polynomials"]

    def generate_parameters(self, input_value=None):
        """Generate random polynomial coefficients."""
        degree = random.randint(1, 5)
        coeffs = [random.randint(-10, 10) for _ in range(degree + 1)]
        # Ensure leading coefficient is non-zero
        coeffs[-1] = random.randint(1, 10)
        return {"coeffs": coeffs}

    def compute(self, input_value, params):
        """Find degree of polynomial (highest power with non-zero coefficient)."""
        coeffs = params.get("coeffs", [1, 2, 3])

        # Remove trailing zeros
        while coeffs and abs(coeffs[-1]) < 1e-10:
            coeffs.pop()

        if not coeffs:
            result = -1  # Zero polynomial
        else:
            result = len(coeffs) - 1

        return MethodResult(
            value=result,
            description=f"Degree of polynomial with coefficients {params.get('coeffs')} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SubstitutionTransform(MethodBlock):
    """
    Apply variable substitution to transform algebraic expressions.

    This technique takes a SymPy expression and performs a variable substitution
    to create a transformed expression. Supports various substitution types:
    - Linear: x -> u + v
    - Polynomial: x -> u^2 + v
    - Trigonometric: x -> sin(u) + cos(v)
    - Exponential: x -> exp(u) + log(v)

    Example:
        Input: x^2 + 2*x + 1
        Substitution: x -> u + v
        Output: (u + v)^2 + 2*(u + v) + 1

    This is useful for creating complex nested problems where the final
    answer requires expanding and simplifying the substituted form.
    """

    def __init__(self):
        super().__init__()
        self.name = "substitution_transform"
        self.input_type = "expression"
        self.output_type = "expression"
        self.difficulty = 4
        self.tags = ["algebra", "transformation", "symbolic", "calculus"]

    def generate_parameters(self, input_value=None):
        """
        Generate substitution parameters.

        Args:
            input_value: Optional SymPy expression (used to determine variables)

        Returns:
            Dictionary with substitution type and mapping
        """
        # Choose substitution type
        sub_type = random.choice([
            "linear",
            "polynomial",
            "trigonometric",
            "exponential",
            "mixed"
        ])

        # Generate substitution based on type
        # We'll use symbolic variables for the substitution
        u, v, w = symbols('u v w', real=True, positive=True)

        if sub_type == "linear":
            # x -> a*u + b*v + c
            a = random.randint(1, 5)
            b = random.randint(1, 5)
            c = random.randint(-3, 3)
            substitution = a*u + b*v + c

        elif sub_type == "polynomial":
            # x -> u^n + v^m
            n = random.randint(2, 3)
            m = random.randint(2, 3)
            substitution = u**n + v**m

        elif sub_type == "trigonometric":
            # x -> sin(u) + cos(v)
            choice = random.choice([
                sin(u) + cos(v),
                sin(u)**2 + cos(v)**2,
                tan(u) + sin(v)
            ])
            substitution = choice

        elif sub_type == "exponential":
            # x -> exp(u) + log(v)
            choice = random.choice([
                exp(u) + v,
                u + log(v + 1),  # log(v+1) to ensure v+1 > 0
                exp(u) - exp(v)
            ])
            substitution = choice

        else:  # mixed
            # Combination of different types
            choice = random.choice([
                u**2 + sin(v),
                exp(u) + v**2,
                sqrt(u) + v**2
            ])
            substitution = choice

        # Generate a default test expression if no input_value provided
        x = symbols('x', real=True)
        test_expressions = [
            x**2 + 2*x + 1,
            x**3 - 3*x + 1,
            2*x**2 + x - 1,
            x**2 - 4,
            x + 5
        ]
        default_expression = random.choice(test_expressions)

        return {
            "substitution_type": sub_type,
            "substitution": substitution,
            "new_vars": [u, v, w],
            "expression": default_expression
        }

    def compute(self, input_value, params):
        """
        Apply the substitution transformation.

        Args:
            input_value: SymPy expression to transform
            params: Dictionary containing 'substitution' (Dict mapping symbols to expressions)
                   or 'substitution_type' and generated substitution expression

        Returns:
            MethodResult with transformed expression
        """
        # Use default expression from params if no input_value provided
        if input_value is None:
            input_value = params.get('expression')
            if input_value is None:
                raise ValueError("SubstitutionTransform requires a SymPy expression as input")

        # Handle two cases:
        # 1. User provides explicit substitution dict (for testing)
        # 2. Use generated parameters with single variable substitution

        if 'substitution' in params and isinstance(params['substitution'], dict):
            # Direct substitution dict provided
            sub_dict = params['substitution']
            sub_expr = input_value.subs(sub_dict)

            # Extract variable info for description
            old_vars = list(sub_dict.keys())
            new_expr_str = str(list(sub_dict.values())[0])
            sub_type = "custom"

        else:
            # Generated parameters case
            sub_type = params.get("substitution_type", "linear")
            new_expr = params.get("substitution")

            if new_expr is None:
                raise ValueError("Substitution expression not found in params")

            # Determine which variable to substitute
            # Get all symbols from input expression
            expr_vars = list(input_value.free_symbols)

            if not expr_vars:
                # No variables to substitute
                return MethodResult(
                    value=input_value,
                    description="No variables to substitute (constant expression)",
                    params=params,
                    metadata={"substitution_type": sub_type}
                )

            # Use the first variable (or a specific one if multiple)
            old_var = expr_vars[0]

            # Build substitution dict
            sub_dict = {old_var: new_expr}

            # Apply substitution
            sub_expr = input_value.subs(sub_dict)

            old_vars = [old_var]
            new_expr_str = str(new_expr)

        # Try to expand and simplify
        try:
            expanded = expand(sub_expr)
            simplified = simplify(expanded)
            result_expr = simplified
        except Exception:
            # If simplification fails, use the substituted expression
            result_expr = sub_expr

        # Create description
        old_var_str = ", ".join(str(v) for v in old_vars)
        description = (
            f"Substitution: {old_var_str} -> {new_expr_str}\n"
            f"Type: {sub_type}\n"
            f"Result: {result_expr}"
        )

        return MethodResult(
            value=result_expr,
            description=description,
            params=params,
            metadata={
                "substitution_type": sub_type,
                "original_expression": str(input_value),
                "substituted_expression": str(sub_expr),
                "simplified_expression": str(result_expr),
                "variables_substituted": old_var_str
            }
        )

    def can_invert(self):
        """
        Substitution can sometimes be inverted by solving for the original variable.
        However, this is not always straightforward, so we mark as non-invertible.
        """
        return False


@register_technique
class EvaluateAtZero(MethodBlock):
    """Evaluate polynomial at x=0 (returns constant term)."""

    def __init__(self):
        super().__init__()
        self.name = "evaluate_at_zero"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["algebra", "polynomials", "evaluation"]

    def generate_parameters(self, input_value=None):
        """Generate random polynomial coefficients."""
        degree = random.randint(1, 5)
        coeffs = [random.randint(-10, 10) for _ in range(degree + 1)]
        return {"coefficients": coeffs}

    def compute(self, input_value, params):
        """
        Evaluate polynomial at x=0.

        Polynomial is represented as [c0, c1, c2, ...] for c0 + c1*x + c2*x^2 + ...
        At x=0, the result is simply c0 (the constant term).

        Args:
            input_value: List of coefficients (if provided)
            params: Dictionary with 'coefficients' key

        Returns:
            MethodResult with constant term (coefficient at index 0)
        """
        # Use input_value if explicitly provided (even if empty), else use params
        if input_value is not None:
            coefficients = input_value
        else:
            coefficients = params.get("coefficients", [5, 2, 1])

        if not coefficients:
            raise ValueError("Empty coefficient list")

        # The value at x=0 is just the constant term (first coefficient)
        result = coefficients[0]

        # Build polynomial string for description
        poly_str = self._format_polynomial(coefficients)

        return MethodResult(
            value=result,
            description=f"P(0) where P(x) = {poly_str} → {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def _format_polynomial(self, coeffs):
        """Format polynomial coefficients as a readable string."""
        if not coeffs:
            return "0"

        terms = []
        for i, c in enumerate(coeffs):
            if c == 0:
                continue

            if i == 0:
                terms.append(str(c))
            elif i == 1:
                if c == 1:
                    terms.append("x")
                elif c == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{c}x")
            else:
                if c == 1:
                    terms.append(f"x^{i}")
                elif c == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{c}x^{i}")

        if not terms:
            return "0"

        # Join terms with + or -
        result = terms[0]
        for term in terms[1:]:
            if term.startswith("-"):
                result += f" - {term[1:]}"
            else:
                result += f" + {term}"

        return result

    def can_invert(self):
        return False


@register_technique
class TestConstantPolynomial(MethodBlock):
    """Test if a polynomial is constant (degree 0)."""

    def __init__(self):
        super().__init__()
        self.name = "test_constant_polynomial"
        self.input_type = "list"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["algebra", "polynomials"]

    def generate_parameters(self, input_value=None):
        """Generate random polynomial coefficients."""
        # 50% chance of constant polynomial
        if random.random() < 0.5:
            return {"coeffs": [random.randint(-10, 10)]}
        else:
            degree = random.randint(1, 4)
            coeffs = [random.randint(-10, 10) for _ in range(degree + 1)]
            # Ensure leading coeff is non-zero
            coeffs[-1] = random.randint(1, 10)
            return {"coeffs": coeffs}

    def compute(self, input_value, params):
        """
        Test if polynomial is constant.

        Polynomial [c0, c1, c2, ...] is constant if all coeffs except c0 are zero.

        Args:
            input_value: Polynomial coefficients (if provided)
            params: Dictionary with 'coeffs' key

        Returns:
            MethodResult with 1 (constant) or 0 (not constant)
        """
        coeffs = input_value or params.get("coeffs", [5])

        if not coeffs:
            # Empty polynomial is considered constant (zero)
            result = 1
        else:
            # Check if all coefficients except the first are zero
            is_constant = all(abs(c) < 1e-10 for c in coeffs[1:]) if len(coeffs) > 1 else True
            result = 1 if is_constant else 0

        return MethodResult(
            value=result,
            description=f"Is polynomial {coeffs} constant? {bool(result)}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

