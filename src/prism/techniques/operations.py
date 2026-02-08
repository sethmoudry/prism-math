"""
Core operational methods for PRISM pipeline (Part 1 of 2).

These 5 core methods handle structure recognition, mapping, transformation,
computation, and verification tasks.

Methods:
    - recognize_structure: Pattern match against known mathematical forms
    - establish_mapping: Setup problem translation (bijection, recurrence, etc.)
    - transform_expression: Algebraic manipulation using SymPy
    - compute: Numerical computation (arithmetic, gcd, lcm, mod, evaluate)
    - verify: Check mathematical conditions

See operations_2.py for:
    - control_flow, extract_coefficients, extract_result
    - All alias classes (30 total)
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from fractions import Fraction

from sympy import (
    symbols, Symbol, Expr,
    simplify as sympy_simplify,
    factor as sympy_factor,
    expand as sympy_expand,
    solve, gcd as sympy_gcd, lcm as sympy_lcm,
    Poly, degree, sympify,
    fibonacci as sympy_fib, catalan as sympy_catalan,
    binomial as sympy_binomial,
    Integer, Rational,
)

from .base import MethodBlock, MethodResult
from .registry import register_technique


# =============================================================================
# RECOGNIZE_STRUCTURE - Pattern matching for mathematical structures
# =============================================================================

@register_technique
class RecognizeStructure(MethodBlock):
    """Recognize mathematical structure patterns in inputs.

    Identifies common mathematical forms like Catalan numbers, Fibonacci,
    Diophantine equations, telescoping series, invariants, etc.
    """

    name = "recognize_structure"
    input_type = "any"
    output_type = "dict"
    difficulty = 2
    tags = ["operational", "pattern_recognition"]

    SUPPORTED_TYPES = {
        "catalan", "fibonacci", "diophantine", "telescoping",
        "invariant", "monovariant", "arithmetic_sequence",
        "geometric_sequence", "binomial", "combinatorial"
    }

    # Known sequence values for pattern matching
    CATALAN_NUMBERS = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796]
    FIBONACCI_NUMBERS = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    TRIANGULAR_NUMBERS = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91]

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {"structure_type": random.choice(list(self.SUPPORTED_TYPES))}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Recognize structure pattern in input.

        Args:
            input_value: Value to analyze (can be int, list, expression)
            params: Must contain 'structure_type'

        Returns:
            MethodResult with simple value (passthrough input or extracted index)
        """
        structure_type = params.get("structure_type", "unknown")

        if structure_type not in self.SUPPORTED_TYPES:
            return MethodResult(
                value=input_value,  # Passthrough on failure
                description=f"Unknown structure type: {structure_type}",
                params=params,
                metadata={"success": False, "matched": False}
            )

        # Attempt pattern matching based on structure type
        match_result = self._match_structure(input_value, structure_type)

        # Return simple value: the input (passthrough) or extracted index if matched
        if match_result.get("matched"):
            # For indexed structures, return the index; otherwise passthrough
            output_value = match_result.get("index", input_value)
        else:
            output_value = input_value

        return MethodResult(
            value=output_value,  # Simple value, not dict
            description=f"Structure recognition for {structure_type}: {'matched' if match_result.get('matched') else 'no match'}",
            params=params,
            metadata={"structure_type": structure_type, "matched": match_result.get("matched", False), **match_result}
        )

    def _match_structure(self, value: Any, structure_type: str) -> Dict[str, Any]:
        """Internal method to match specific structure types."""
        result = {"matched": False, "type": structure_type, "input": value}

        if structure_type == "catalan":
            if isinstance(value, int) and value in self.CATALAN_NUMBERS:
                idx = self.CATALAN_NUMBERS.index(value)
                result.update({"matched": True, "index": idx, "formula": f"C_{idx}"})

        elif structure_type == "fibonacci":
            if isinstance(value, int) and value in self.FIBONACCI_NUMBERS:
                idx = self.FIBONACCI_NUMBERS.index(value)
                result.update({"matched": True, "index": idx, "formula": f"F_{idx}"})

        elif structure_type == "arithmetic_sequence":
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                diffs = [value[i+1] - value[i] for i in range(len(value)-1)]
                if len(set(diffs)) == 1:
                    result.update({"matched": True, "common_difference": diffs[0], "first_term": value[0]})

        elif structure_type == "geometric_sequence":
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    ratios = [value[i+1] / value[i] for i in range(len(value)-1) if value[i] != 0]
                    if ratios and len(set(round(r, 10) for r in ratios)) == 1:
                        result.update({"matched": True, "common_ratio": ratios[0], "first_term": value[0]})
                except (ZeroDivisionError, TypeError):
                    pass

        elif structure_type == "binomial":
            # Check if value matches binomial coefficient pattern
            if isinstance(value, int) and value > 0:
                for n in range(1, min(20, value + 1)):
                    for k in range(n + 1):
                        if int(sympy_binomial(n, k)) == value:
                            result.update({"matched": True, "n": n, "k": k, "formula": f"C({n},{k})"})
                            return result

        elif structure_type == "telescoping":
            # Telescoping: consecutive terms cancel
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                result.update({"matched": True, "sum": value[0] - value[-1] if len(value) > 1 else value[0]})

        elif structure_type in ("invariant", "monovariant"):
            # These require external context; mark as potentially matched
            result.update({"matched": True, "note": f"Requires problem context for {structure_type} verification"})

        elif structure_type == "diophantine":
            # Check if input looks like a Diophantine equation context
            result.update({"matched": True, "note": "Diophantine analysis requested"})

        elif structure_type == "combinatorial":
            # General combinatorial structure
            result.update({"matched": True, "note": "Combinatorial structure recognized"})

        return result

    def can_invert(self) -> bool:
        return False


# =============================================================================
# ESTABLISH_MAPPING - Setup problem translations
# =============================================================================

@register_technique
class EstablishMapping(MethodBlock):
    """Establish mathematical mappings for problem translation.

    Supports bijections, recurrence relations, coordinate systems,
    and normalization transformations.
    """

    name = "establish_mapping"
    input_type = "any"
    output_type = "dict"
    difficulty = 3
    tags = ["operational", "mapping", "transformation"]

    SUPPORTED_MAPPINGS = {"bijection", "recurrence", "coordinates", "normalization"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {"mapping_type": random.choice(list(self.SUPPORTED_MAPPINGS))}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Establish the requested mapping type.

        Args:
            input_value: Source data for mapping
            params: Must contain 'mapping_type'

        Returns:
            MethodResult with simple value: passthrough input or normalized value
        """
        mapping_type = params.get("mapping_type", "bijection")

        if mapping_type not in self.SUPPORTED_MAPPINGS:
            return MethodResult(
                value=input_value,  # Passthrough on error
                description=f"Failed to establish mapping: unknown type {mapping_type}",
                params=params,
                metadata={"success": False, "error": f"Unknown mapping type"}
            )

        result = self._establish(input_value, mapping_type, params)

        # Return simple value: normalized value if available, otherwise passthrough
        if "normalized" in result:
            output_value = result["normalized"]
        else:
            output_value = input_value  # Passthrough

        return MethodResult(
            value=output_value,  # Simple value
            description=f"Established {mapping_type} mapping",
            params=params,
            metadata={"mapping_type": mapping_type, **result}
        )

    def _establish(self, value: Any, mapping_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to establish specific mapping types."""
        result = {"success": True, "type": mapping_type, "input": value}

        if mapping_type == "bijection":
            # Create identity or specified bijection
            domain = params.get("domain", list(range(10)) if value is None else value)
            codomain = params.get("codomain", domain)
            result.update({
                "domain": domain,
                "codomain": codomain,
                "map": {i: i for i in (domain if isinstance(domain, (list, tuple)) else range(domain))}
            })

        elif mapping_type == "recurrence":
            # Setup recurrence relation framework
            order = params.get("order", 2)
            initial = params.get("initial_values", [0, 1][:order])
            result.update({
                "order": order,
                "initial_values": initial,
                "relation": params.get("relation", "a_n = a_{n-1} + a_{n-2}")
            })

        elif mapping_type == "coordinates":
            # Coordinate system transformation
            from_system = params.get("from_system", "cartesian")
            to_system = params.get("to_system", "polar")
            result.update({
                "from_system": from_system,
                "to_system": to_system,
                "transform": f"{from_system} -> {to_system}"
            })

        elif mapping_type == "normalization":
            # Normalization transformation
            if isinstance(value, (int, float)) and value != 0:
                normalized = value / abs(value)  # Sign normalization
                result.update({"normalized": normalized, "scale_factor": abs(value)})
            elif isinstance(value, (list, tuple)) and value:
                max_val = max(abs(v) for v in value if isinstance(v, (int, float)))
                if max_val > 0:
                    normalized = [v / max_val for v in value]
                    result.update({"normalized": normalized, "scale_factor": max_val})
                else:
                    result.update({"normalized": value, "scale_factor": 1})
            else:
                result.update({"normalized": value, "scale_factor": 1})

        return result

    def can_invert(self) -> bool:
        return False


# =============================================================================
# TRANSFORM_EXPRESSION - Algebraic manipulation using SymPy
# =============================================================================

@register_technique
class TransformExpression(MethodBlock):
    """Transform algebraic expressions using SymPy.

    Supports substitution, simplification, factoring, expansion,
    solving, and variable elimination.
    """

    name = "transform_expression"
    input_type = "any"
    output_type = "any"
    difficulty = 2
    tags = ["operational", "algebra", "sympy"]

    SUPPORTED_OPERATIONS = {"substitute", "simplify", "factor", "expand", "solve_for", "eliminate"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {
            "operation": random.choice(list(self.SUPPORTED_OPERATIONS)),
            "variable": "x"
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply algebraic transformation to expression.

        Args:
            input_value: Expression to transform (string, sympy expr, or numeric)
            params: Must contain 'operation', may contain 'variable', 'substitutions'

        Returns:
            MethodResult with transformed expression
        """
        operation = params.get("operation", "simplify")

        if operation not in self.SUPPORTED_OPERATIONS:
            return MethodResult(
                value=input_value,
                description=f"Unknown operation: {operation}",
                params=params,
                metadata={"success": False, "error": "unknown operation"}
            )

        try:
            result = self._transform(input_value, operation, params)
            return MethodResult(
                value=result["value"],
                description=f"Applied {operation}: {result.get('description', '')}",
                params=params,
                metadata={"operation": operation, "success": True}
            )
        except Exception as e:
            return MethodResult(
                value=input_value,
                description=f"Transform failed: {str(e)}",
                params=params,
                metadata={"success": False, "error": str(e)}
            )

    def _transform(self, value: Any, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to apply transformations."""
        # Convert to sympy expression if needed
        if isinstance(value, str):
            try:
                expr = sympify(value)
            except Exception:
                expr = value
        elif isinstance(value, (int, float)):
            expr = sympify(value)
        else:
            expr = value

        result = {"value": expr, "description": ""}

        if operation == "simplify":
            if hasattr(expr, 'simplify') or isinstance(expr, Expr):
                simplified = sympy_simplify(expr)
                result = {"value": simplified, "description": f"{expr} -> {simplified}"}
            else:
                result = {"value": expr, "description": "No simplification needed"}

        elif operation == "factor":
            if hasattr(expr, 'factor') or isinstance(expr, Expr):
                factored = sympy_factor(expr)
                result = {"value": factored, "description": f"Factored: {factored}"}
            else:
                result = {"value": expr, "description": "Cannot factor"}

        elif operation == "expand":
            if hasattr(expr, 'expand') or isinstance(expr, Expr):
                expanded = sympy_expand(expr)
                result = {"value": expanded, "description": f"Expanded: {expanded}"}
            else:
                result = {"value": expr, "description": "Cannot expand"}

        elif operation == "substitute":
            substitutions = params.get("substitutions", {})
            if substitutions and isinstance(expr, Expr):
                # Convert string keys to symbols
                sub_dict = {}
                for k, v in substitutions.items():
                    if isinstance(k, str):
                        sub_dict[Symbol(k)] = sympify(v)
                    else:
                        sub_dict[k] = sympify(v)
                substituted = expr.subs(sub_dict)
                result = {"value": substituted, "description": f"Substituted: {substituted}"}
            else:
                result = {"value": expr, "description": "No substitutions applied"}

        elif operation == "solve_for":
            variable = params.get("variable", "x")
            if isinstance(variable, str):
                variable = Symbol(variable)
            if isinstance(expr, Expr):
                solutions = solve(expr, variable)
                result = {"value": solutions, "description": f"Solutions for {variable}: {solutions}"}
            else:
                result = {"value": [], "description": "Cannot solve"}

        elif operation == "eliminate":
            variables = params.get("eliminate_vars", ["y"])
            if isinstance(expr, (list, tuple)) and len(expr) >= 2:
                # Eliminate variable from system of equations
                var_syms = [Symbol(v) if isinstance(v, str) else v for v in variables]
                # Simple elimination attempt
                result = {"value": expr, "description": f"Elimination of {variables} requested"}
            else:
                result = {"value": expr, "description": "Need system of equations for elimination"}

        return result

    def can_invert(self) -> bool:
        return False


# =============================================================================
# COMPUTE - Numerical computation
# =============================================================================

@register_technique
class Compute(MethodBlock):
    """Perform numerical computations.

    Supports arithmetic operations, GCD, LCM, modular arithmetic,
    and expression evaluation.
    """

    name = "compute"
    input_type = "any"
    output_type = "integer"
    difficulty = 1
    tags = ["operational", "arithmetic", "computation"]

    SUPPORTED_OPERATIONS = {"arithmetic", "gcd", "lcm", "mod", "evaluate"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {
            "operation": random.choice(list(self.SUPPORTED_OPERATIONS)),
            "a": random.randint(1, 100),
            "b": random.randint(1, 100)
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Perform numerical computation.

        Args:
            input_value: Primary operand (can be overridden by params)
            params: Contains 'operation' and operands 'a', 'b', 'modulus', etc.

        Returns:
            MethodResult with computed value
        """
        operation = params.get("operation", "arithmetic")

        if operation not in self.SUPPORTED_OPERATIONS:
            return MethodResult(
                value=0,
                description=f"Unknown operation: {operation}",
                params=params,
                metadata={"success": False}
            )

        try:
            result = self._compute(input_value, operation, params)
            return MethodResult(
                value=result["value"],
                description=result["description"],
                params=params,
                metadata={"operation": operation, "success": True}
            )
        except Exception as e:
            return MethodResult(
                value=0,
                description=f"Computation failed: {str(e)}",
                params=params,
                metadata={"success": False, "error": str(e)}
            )

    def _compute(self, value: Any, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method for computations."""
        a = params.get("a", value if isinstance(value, (int, float)) else 0)
        b = params.get("b", 1)

        # Ensure integer values where needed
        a = int(a) if isinstance(a, (int, float)) else 0
        b = int(b) if isinstance(b, (int, float)) else 1

        if operation == "arithmetic":
            op_type = params.get("op_type", "add")
            if op_type == "add":
                result = a + b
                desc = f"{a} + {b} = {result}"
            elif op_type == "subtract":
                result = a - b
                desc = f"{a} - {b} = {result}"
            elif op_type == "multiply":
                result = a * b
                desc = f"{a} * {b} = {result}"
            elif op_type == "divide":
                result = a // b if b != 0 else 0
                desc = f"{a} // {b} = {result}"
            elif op_type == "power":
                exp = min(b, 20)  # Limit exponent to prevent overflow
                result = a ** exp
                desc = f"{a}^{exp} = {result}"
            else:
                result = a + b
                desc = f"{a} + {b} = {result}"
            return {"value": result, "description": desc}

        elif operation == "gcd":
            result = math.gcd(abs(a), abs(b))
            return {"value": result, "description": f"gcd({a}, {b}) = {result}"}

        elif operation == "lcm":
            result = math.lcm(abs(a), abs(b)) if a != 0 and b != 0 else 0
            return {"value": result, "description": f"lcm({a}, {b}) = {result}"}

        elif operation == "mod":
            modulus = params.get("modulus", b)
            modulus = int(modulus) if modulus else 1
            if modulus == 0:
                modulus = 1
            result = a % modulus
            return {"value": result, "description": f"{a} mod {modulus} = {result}"}

        elif operation == "evaluate":
            # Evaluate an expression at given values
            expr_str = params.get("expression", str(value))
            var_values = params.get("values", {"x": a})
            try:
                expr = sympify(expr_str)
                sub_dict = {Symbol(k): v for k, v in var_values.items()}
                result = int(expr.subs(sub_dict))
                return {"value": result, "description": f"Evaluated {expr_str} = {result}"}
            except Exception:
                return {"value": a, "description": f"Evaluation fallback: {a}"}

        return {"value": 0, "description": "No computation performed"}

    def can_invert(self) -> bool:
        return False


# =============================================================================
# VERIFY - Check mathematical conditions
# =============================================================================

@register_technique
class Verify(MethodBlock):
    """Verify mathematical conditions and constraints.

    Checks constraints, base cases, boundary conditions, and
    general mathematical assertions.
    """

    name = "verify"
    input_type = "any"
    output_type = "dict"
    difficulty = 2
    tags = ["operational", "verification", "validation"]

    SUPPORTED_CONDITIONS = {"constraint", "base_case", "boundary"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {"condition_type": random.choice(list(self.SUPPORTED_CONDITIONS))}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Verify a mathematical condition.

        Args:
            input_value: Value to verify
            params: Contains 'condition_type' and condition-specific parameters

        Returns:
            MethodResult with simple value: True/False for verification, or passthrough input
        """
        condition_type = params.get("condition_type", "constraint")

        if condition_type not in self.SUPPORTED_CONDITIONS:
            return MethodResult(
                value=input_value,  # Passthrough on error
                description=f"Unknown condition type: {condition_type}",
                params=params,
                metadata={"verified": False, "error": f"Unknown condition type"}
            )

        result = self._verify(input_value, condition_type, params)
        verified = result.get("verified", False)

        # Return the input value if verified (passthrough), else return verification status
        output_value = input_value if verified else verified

        return MethodResult(
            value=output_value,  # Simple value: input passthrough or False
            description=f"Verification ({condition_type}): {'passed' if verified else 'failed'}",
            params=params,
            metadata={"condition_type": condition_type, "verified": verified, **result}
        )

    def _verify(self, value: Any, condition_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal verification logic."""
        result = {"verified": False, "condition_type": condition_type, "input": value}

        if condition_type == "constraint":
            # Check general constraint
            constraint = params.get("constraint", "positive")
            if constraint == "positive":
                result["verified"] = isinstance(value, (int, float)) and value > 0
            elif constraint == "non_negative":
                result["verified"] = isinstance(value, (int, float)) and value >= 0
            elif constraint == "integer":
                result["verified"] = isinstance(value, int) or (isinstance(value, float) and value == int(value))
            elif constraint == "in_range":
                low = params.get("low", 0)
                high = params.get("high", 100)
                result["verified"] = isinstance(value, (int, float)) and low <= value <= high
            elif constraint == "divisible_by":
                divisor = params.get("divisor", 1)
                result["verified"] = isinstance(value, int) and divisor != 0 and value % divisor == 0
            else:
                result["verified"] = True  # Default to verified for unknown constraints
            result["constraint"] = constraint

        elif condition_type == "base_case":
            # Verify base case of induction/recursion
            n = params.get("n", 0)
            expected = params.get("expected", value)
            result["verified"] = (value == expected)
            result["n"] = n
            result["expected"] = expected

        elif condition_type == "boundary":
            # Check boundary conditions
            boundary = params.get("boundary", "lower")
            limit = params.get("limit", 0)
            if boundary == "lower":
                result["verified"] = isinstance(value, (int, float)) and value >= limit
            elif boundary == "upper":
                result["verified"] = isinstance(value, (int, float)) and value <= limit
            elif boundary == "strict_lower":
                result["verified"] = isinstance(value, (int, float)) and value > limit
            elif boundary == "strict_upper":
                result["verified"] = isinstance(value, (int, float)) and value < limit
            result["boundary"] = boundary
            result["limit"] = limit

        return result

    def can_invert(self) -> bool:
        return False


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core generic methods (5)
    "RecognizeStructure",
    "EstablishMapping",
    "TransformExpression",
    "Compute",
    "Verify",
]
