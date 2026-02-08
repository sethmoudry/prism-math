"""
Miscellaneous analysis methods.
"""

import math
import random
from typing import List, Tuple, Union, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class MaxValueFromCases(MethodBlock):
    """Find maximum value from a list of cases/candidates."""

    def __init__(self):
        super().__init__()
        self.name = "max_value_from_cases"
        self.input_type = "list"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["analysis", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate random list of cases."""
        length = random.randint(2, 10)
        return {"cases": [random.uniform(-10, 10) for _ in range(length)]}

    def compute(self, input_value, params):
        """
        Find maximum value from a list of cases/candidates.

        Args:
            input_value: List of cases (if provided), can be string expression
            params: Dictionary with 'cases' key

        Returns:
            MethodResult with maximum value
        """
        cases = input_value or params.get("cases", [1.0, 2.0, 3.0])

        # Handle string expressions (symbolic/descriptive input)
        if isinstance(cases, str):
            return MethodResult(
                value=cases,
                description=f"Max value analysis for: {cases}",
                params=params,
                metadata={"techniques_used": [self.name], "symbolic": True}
            )

        if not cases:
            raise ValueError("Empty cases list")

        result = max(cases)

        # Format result appropriately based on type
        if isinstance(result, (int, float)):
            desc = f"Maximum from cases = {result:.6f}"
        else:
            desc = f"Maximum from cases = {result}"

        return MethodResult(
            value=result,
            description=desc,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MaxRatioOverConfigurations(MethodBlock):
    """Find maximum ratio over different configurations/cases."""

    def __init__(self):
        super().__init__()
        self.name = "max_ratio_over_configurations"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "optimization", "ratios"]

    def generate_parameters(self, input_value=None):
        """Generate random configurations with numerators and denominators."""
        num_configs = random.randint(2, 10)
        return {
            "numerators": [random.uniform(1, 100) for _ in range(num_configs)],
            "denominators": [random.uniform(1, 50) for _ in range(num_configs)]
        }

    def compute(self, input_value, params):
        """
        Find maximum ratio over different configurations/cases.

        Computes ratios for each configuration and returns the maximum.
        Similar to max_value_from_cases but for ratios.

        Args:
            input_value: Not used
            params: Dictionary with 'numerators' and 'denominators' keys

        Returns:
            MethodResult with maximum ratio

        Examples:
            numerators = [10, 20, 15], denominators = [5, 8, 3]
            ratios = [2.0, 2.5, 5.0] → max = 5.0
        """
        numerators = params.get("numerators", [10.0, 20.0, 15.0])
        denominators = params.get("denominators", [5.0, 8.0, 3.0])

        # Convert strings to floats if needed
        if isinstance(numerators, list):
            numerators = [float(n) if isinstance(n, str) else n for n in numerators]
        if isinstance(denominators, list):
            denominators = [float(d) if isinstance(d, str) else d for d in denominators]

        if len(numerators) != len(denominators):
            raise ValueError("Numerators and denominators must have the same length")

        if not numerators or not denominators:
            raise ValueError("Empty configurations list")

        # Compute ratios
        ratios = []
        for num, denom in zip(numerators, denominators):
            if abs(denom) < 1e-10:
                # Skip zero denominators or use large value
                ratios.append(float('inf'))
            else:
                ratios.append(num / denom)

        # Find maximum ratio
        result = max(ratios)

        # Build description
        if len(ratios) <= 5:
            ratios_str = ", ".join(f"{r:.4f}" for r in ratios)
            description = f"Ratios from configurations: [{ratios_str}]. Maximum ratio = {result:.6f}"
        else:
            ratios_str = ", ".join(f"{r:.4f}" for r in ratios[:5]) + ", ..."
            description = f"Ratios from {len(ratios)} configurations: [{ratios_str}]. Maximum ratio = {result:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "techniques_used": [self.name],
                "ratios": ratios,
                "max_ratio": result,
                "num_configurations": len(ratios)
            }
        )

    def can_invert(self):
        return False


@register_technique
class CharacteristicEquationRoots(MethodBlock):
    """Find roots of characteristic equation for ODEs."""

    def __init__(self):
        super().__init__()
        self.name = "characteristic_equation_roots"
        self.input_type = "none"
        self.output_type = "list"
        self.difficulty = 3
        self.tags = ["analysis", "ode", "characteristic_equation"]

    def generate_parameters(self, input_value=None):
        """Generate random characteristic equation coefficients."""
        # For characteristic equation: r^2 + a*r + b = 0
        return {
            "a": random.randint(-10, 10),
            "b": random.randint(-10, 10)
        }

    def compute(self, input_value, params):
        """
        Find roots of characteristic equation r^2 + a*r + b = 0.

        Uses quadratic formula: r = (-a ± sqrt(a^2 - 4b)) / 2

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'b' keys (coefficients)

        Returns:
            MethodResult with list of roots (real or complex)
        """
        a = params.get("a", 1)
        b = params.get("b", 0)

        # Discriminant
        discriminant = a * a - 4 * b

        if discriminant >= 0:
            # Real roots
            sqrt_disc = math.sqrt(discriminant)
            r1 = (-a + sqrt_disc) / 2
            r2 = (-a - sqrt_disc) / 2
            result = [r1, r2]
            description = f"Characteristic eq r² + {a}r + {b} = 0: roots = [{r1:.6f}, {r2:.6f}]"
        else:
            # Complex roots: real_part ± i*imag_part
            real_part = -a / 2
            imag_part = math.sqrt(-discriminant) / 2
            # Return as list of real parts (simplified for numeric computation)
            result = [real_part, real_part]
            description = f"Characteristic eq r² + {a}r + {b} = 0: roots = {real_part:.6f} ± {imag_part:.6f}i"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name], "discriminant": discriminant}
        )

    def can_invert(self):
        return False


@register_technique
class ParticularSolutionUndeterminedCoeffs(MethodBlock):
    """Find particular solution using undetermined coefficients for ODEs."""

    def __init__(self):
        super().__init__()
        self.name = "particular_solution_undetermined_coeffs"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "ode", "particular_solution"]

    def generate_parameters(self, input_value=None):
        """Generate random ODE parameters."""
        # For y'' + a*y' + b*y = c (constant RHS)
        # Particular solution: y_p = c/b (if b != 0)
        return {
            "a": random.randint(-10, 10),
            "b": random.randint(1, 10),  # Avoid zero
            "c": random.randint(-20, 20)
        }

    def compute(self, input_value, params):
        """
        Find particular solution for y'' + a*y' + b*y = c.

        For constant RHS c, particular solution is y_p = c/b (if b != 0).

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'b', 'c' keys

        Returns:
            MethodResult with particular solution value
        """
        a = params.get("a", 0)
        b = params.get("b", 1)
        c = params.get("c", 10)

        if abs(b) < 1e-10:
            # If b = 0, try polynomial solution (simplified: just return c)
            result = float(c)
            description = f"Particular solution for y'' + {a}y' = {c}: y_p ≈ {result:.6f}"
        else:
            # Particular solution: y_p = c/b
            result = c / b
            description = f"Particular solution for y'' + {a}y' + {b}y = {c}: y_p = {c}/{b} = {result:.6f}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConvexFunctionProperties(MethodBlock):
    """Analyze properties of convex functions (min, max, inflection)."""

    def __init__(self):
        super().__init__()
        self.name = "convex_function_properties"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 3
        self.tags = ["analysis", "convexity", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate random quadratic function parameters."""
        # For f(x) = a*x^2 + b*x + c (convex if a > 0)
        return {
            "a": random.uniform(0.1, 10),  # Positive for convexity
            "b": random.uniform(-10, 10),
            "c": random.uniform(-10, 10)
        }

    def compute(self, input_value, params):
        """
        Analyze convex function f(x) = a*x^2 + b*x + c.

        For convex quadratic (a > 0):
        - Minimum at x = -b/(2a)
        - Minimum value = c - b^2/(4a)

        Args:
            input_value: Not used
            params: Dictionary with 'a', 'b', 'c' keys

        Returns:
            MethodResult with minimum value
        """
        a = params.get("a", 1.0)
        b = params.get("b", 0.0)
        c = params.get("c", 0.0)

        if a <= 0:
            raise ValueError("a must be positive for convex function")

        # Minimum at x = -b/(2a)
        x_min = -b / (2 * a)

        # Minimum value = f(x_min) = a*x_min^2 + b*x_min + c
        # Simplifies to: c - b^2/(4a)
        min_value = c - (b * b) / (4 * a)

        result = min_value

        return MethodResult(
            value=result,
            description=f"Convex f(x) = {a:.3f}x² + {b:.3f}x + {c:.3f}: min at x = {x_min:.6f}, min_value = {min_value:.6f}",
            params=params,
            metadata={"techniques_used": [self.name], "x_min": x_min}
        )

    def can_invert(self):
        return False


@register_technique
class IsZeroRing(MethodBlock):
    """Check if a ring element is zero."""

    def __init__(self):
        super().__init__()
        self.name = "is_zero_ring"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["analysis", "algebra", "ring_theory"]

    def generate_parameters(self, input_value=None):
        """Generate random value to check."""
        return {"value": random.choice([0, 0.0, random.uniform(-10, 10)])}

    def compute(self, input_value, params):
        """
        Check if a ring element is zero.

        Returns 1 if zero, 0 otherwise (as integer for answer format).

        Args:
            input_value: Value to check (if provided)
            params: Dictionary with 'value' key

        Returns:
            MethodResult with 1 (is zero) or 0 (not zero)
        """
        value = input_value if input_value is not None else params.get("value", 0)

        # Handle string parameters
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = 0

        # Check if zero (with tolerance for floats)
        if isinstance(value, (int, float)):
            is_zero = abs(value) < 1e-10
        else:
            is_zero = value == 0

        result = 1 if is_zero else 0

        return MethodResult(
            value=result,
            description=f"Is {value} zero? {result} (1=yes, 0=no)",
            params=params,
            metadata={"techniques_used": [self.name], "input_value": value}
        )

    def can_invert(self):
        return False


@register_technique
class ItoIntegralExpectation(MethodBlock):
    """Expectation of Ito integral (E[∫f dW] = 0 for adapted f)."""

    def __init__(self):
        super().__init__()
        self.name = "ito_integral_expectation"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["analysis", "stochastic", "ito", "expectation"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for Ito integral (mostly for description)."""
        return {
            "process_name": random.choice(["f", "g", "h", "sigma"]),
            "time_interval": random.uniform(0.1, 10.0)
        }

    def compute(self, input_value, params):
        """
        Expectation of Ito integral E[∫f dW].

        For adapted integrands f, the Ito integral has zero expectation:
        E[∫_0^t f(s) dW_s] = 0

        This is a fundamental property of Ito calculus.

        Args:
            input_value: Not used
            params: Dictionary with 'process_name' and 'time_interval' keys

        Returns:
            MethodResult with value 0 (zero expectation)
        """
        process_name = params.get("process_name", "f")
        time_interval = params.get("time_interval", 1.0)

        if isinstance(time_interval, str):
            try:
                time_interval = float(time_interval)
            except ValueError:
                time_interval = 1.0

        # Ito integral expectation is always 0 for adapted integrands
        result = 0

        return MethodResult(
            value=result,
            description=f"E[∫_0^{time_interval:.2f} {process_name}(s) dW_s] = 0 (Ito integral property)",
            params=params,
            metadata={
                "techniques_used": [self.name],
                "property": "ito_martingale"
            }
        )

    def can_invert(self):
        return False


# ============================================================================
# ASYMPTOTIC AND FEASIBILITY METHODS (3 methods)
# ============================================================================


@register_technique
class TestFeasibilityK(MethodBlock):
    """Test if value k is feasible given constraints."""

    def __init__(self):
        super().__init__()
        self.name = "test_feasibility_k"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 2
        self.tags = ["analysis", "feasibility", "optimization"]

    def generate_parameters(self, input_value=None):
        """Generate random k and constraints."""
        k = random.randint(1, 100)
        min_val = random.randint(1, 50)
        max_val = random.randint(min_val + 10, 200)
        return {"k": k, "min_val": min_val, "max_val": max_val}

    def compute(self, input_value, params):
        """
        Test if k is feasible (within constraints).

        Checks if min_val <= k <= max_val.

        Args:
            input_value: k value (if provided)
            params: Dictionary with 'k', 'min_val', 'max_val' keys

        Returns:
            MethodResult with 1 (feasible) or 0 (not feasible)
        """
        k = input_value if input_value is not None else params.get("k", 50)
        min_val = params.get("min_val", 10)
        max_val = params.get("max_val", 100)

        # Handle string parameters
        if isinstance(k, str):
            try:
                k = float(k)
            except ValueError:
                k = 50

        if isinstance(min_val, str):
            try:
                min_val = float(min_val)
            except ValueError:
                min_val = 10

        if isinstance(max_val, str):
            try:
                max_val = float(max_val)
            except ValueError:
                max_val = 100

        # Check feasibility
        feasible = min_val <= k <= max_val
        result = 1 if feasible else 0

        return MethodResult(
            value=result,
            description=f"Is k={k} feasible in [{min_val}, {max_val}]? {feasible}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

