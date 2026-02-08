"""
Operational method aliases and additional core methods for PRISM pipeline.

This module contains:
    - ControlFlow: Proof structure operations (case_split, induction, etc.)
    - ExtractCoefficients: Extract polynomial/series coefficients
    - ExtractResult: Finalize answer extraction
    - 30 alias classes mapping old specific method names to generic methods
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from fractions import Fraction

from sympy import (
    symbols, Symbol, Expr,
    Poly, degree, sympify,
    Integer, Rational,
)

from .base import MethodBlock, MethodResult
from .registry import register_technique
from .operations import (
    RecognizeStructure,
    EstablishMapping,
    TransformExpression,
    Compute,
    Verify,
)


# =============================================================================
# CONTROL_FLOW - Proof structure operations
# =============================================================================

@register_technique
class ControlFlow(MethodBlock):
    """Implement proof structure and control flow operations.

    Supports case splitting, induction setup, proof by contradiction,
    extremal principle, and pigeonhole principle.
    """

    name = "control_flow"
    input_type = "any"
    output_type = "dict"
    difficulty = 3
    tags = ["operational", "proof", "logic"]

    SUPPORTED_FLOWS = {"case_split", "induction", "contradiction", "extremal", "pigeonhole"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {"flow_type": random.choice(list(self.SUPPORTED_FLOWS))}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply control flow/proof structure.

        Args:
            input_value: Input for the proof structure
            params: Contains 'flow_type' and type-specific parameters

        Returns:
            MethodResult with simple value: passthrough input or extracted extreme value
        """
        flow_type = params.get("flow_type", "case_split")

        if flow_type not in self.SUPPORTED_FLOWS:
            return MethodResult(
                value=input_value,  # Passthrough on error
                description=f"Unknown flow type: {flow_type}",
                params=params,
                metadata={"success": False, "error": f"Unknown flow type"}
            )

        result = self._apply_flow(input_value, flow_type, params)

        # Return simple value: extreme_value if available, otherwise passthrough input
        if "extreme_value" in result:
            output_value = result["extreme_value"]
        elif "guaranteed_min" in result:
            output_value = result["guaranteed_min"]
        elif "active_case" in result and isinstance(result["active_case"], int):
            output_value = result["active_case"]  # For modular case split
        else:
            output_value = input_value  # Passthrough

        return MethodResult(
            value=output_value,  # Simple value
            description=f"Applied {flow_type} structure",
            params=params,
            metadata={"flow_type": flow_type, **result}
        )

    def _apply_flow(self, value: Any, flow_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method for control flow operations."""
        result = {"success": True, "flow_type": flow_type, "input": value}

        if flow_type == "case_split":
            # Split into cases based on criterion
            criterion = params.get("criterion", "parity")
            cases = params.get("cases", None)

            if criterion == "parity" and isinstance(value, int):
                cases = ["even" if value % 2 == 0 else "odd"]
                result.update({"criterion": criterion, "active_case": cases[0], "all_cases": ["even", "odd"]})
            elif criterion == "sign" and isinstance(value, (int, float)):
                if value > 0:
                    case = "positive"
                elif value < 0:
                    case = "negative"
                else:
                    case = "zero"
                result.update({"criterion": criterion, "active_case": case, "all_cases": ["positive", "zero", "negative"]})
            elif criterion == "modular":
                modulus = params.get("modulus", 3)
                if isinstance(value, int):
                    case = value % modulus
                    result.update({"criterion": criterion, "modulus": modulus, "active_case": case})
            elif cases:
                result.update({"criterion": "custom", "cases": cases})
            else:
                result.update({"criterion": criterion, "cases": []})

        elif flow_type == "induction":
            # Setup induction structure
            base = params.get("base", 0)
            hypothesis = params.get("hypothesis", "P(n)")
            result.update({
                "base_case": base,
                "hypothesis": hypothesis,
                "inductive_step": f"P(n) => P(n+1)",
                "current_n": value if isinstance(value, int) else base
            })

        elif flow_type == "contradiction":
            # Setup proof by contradiction
            assumption = params.get("assumption", "negation of claim")
            result.update({
                "assumption": assumption,
                "seeking": "contradiction",
                "status": "in_progress"
            })

        elif flow_type == "extremal":
            # Extremal principle setup
            extremal_type = params.get("extremal_type", "minimum")
            if isinstance(value, (list, tuple)) and value:
                if extremal_type == "minimum":
                    extreme = min(value)
                elif extremal_type == "maximum":
                    extreme = max(value)
                else:
                    extreme = value[0]
                result.update({"extremal_type": extremal_type, "extreme_value": extreme})
            else:
                result.update({"extremal_type": extremal_type, "extreme_value": value})

        elif flow_type == "pigeonhole":
            # Pigeonhole principle
            pigeons = params.get("pigeons", 10)
            holes = params.get("holes", 9)
            if isinstance(value, int):
                pigeons = value

            if pigeons > holes:
                min_in_hole = (pigeons - 1) // holes + 1
                result.update({
                    "pigeons": pigeons,
                    "holes": holes,
                    "guaranteed_min": min_in_hole,
                    "applies": True
                })
            else:
                result.update({
                    "pigeons": pigeons,
                    "holes": holes,
                    "applies": False
                })

        return result

    def can_invert(self) -> bool:
        return False


# =============================================================================
# EXTRACT_COEFFICIENTS - Extract polynomial/series coefficients
# =============================================================================

@register_technique
class ExtractCoefficients(MethodBlock):
    """Extract coefficients from polynomials or series.

    Can extract specific coefficients, all coefficients, leading
    coefficient, or constant term.
    """

    name = "extract_coefficients"
    input_type = "any"
    output_type = "integer"
    difficulty = 2
    tags = ["operational", "algebra", "extraction"]

    SUPPORTED_TYPES = {"polynomial", "series", "generating_function"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {
            "expression_type": random.choice(list(self.SUPPORTED_TYPES)),
            "degree": random.randint(0, 5)
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Extract coefficients from expression.

        Args:
            input_value: Polynomial, series expression, or list of coefficients
            params: Contains 'expression_type', 'degree' for specific coefficient

        Returns:
            MethodResult with extracted coefficient(s)
        """
        expression_type = params.get("expression_type", "polynomial")
        target_degree = params.get("degree", 0)

        try:
            result = self._extract(input_value, expression_type, target_degree, params)
            return MethodResult(
                value=result["value"],
                description=result["description"],
                params=params,
                metadata={"expression_type": expression_type, "success": True}
            )
        except Exception as e:
            return MethodResult(
                value=0,
                description=f"Extraction failed: {str(e)}",
                params=params,
                metadata={"success": False, "error": str(e)}
            )

    def _extract(self, value: Any, expr_type: str, degree: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal coefficient extraction."""
        # Handle list of coefficients directly
        if isinstance(value, (list, tuple)):
            if 0 <= degree < len(value):
                coeff = value[degree]
                return {"value": int(coeff) if isinstance(coeff, (int, float)) else coeff,
                        "description": f"Coefficient at degree {degree}: {coeff}"}
            else:
                return {"value": 0, "description": f"Degree {degree} out of range"}

        # Handle sympy expression
        if isinstance(value, str):
            try:
                value = sympify(value)
            except Exception:
                return {"value": 0, "description": "Could not parse expression"}

        if isinstance(value, Expr):
            x = Symbol('x')
            try:
                poly = Poly(value, x)
                all_coeffs = poly.all_coeffs()
                deg = poly.degree()

                # Coefficient for x^degree
                target_idx = deg - degree
                if 0 <= target_idx < len(all_coeffs):
                    coeff = int(all_coeffs[target_idx])
                    return {"value": coeff, "description": f"Coefficient of x^{degree}: {coeff}"}
                else:
                    return {"value": 0, "description": f"No term of degree {degree}"}
            except Exception:
                # Fallback for non-polynomial expressions
                return {"value": 0, "description": "Could not extract as polynomial"}

        # Handle integer (constant polynomial)
        if isinstance(value, (int, float)):
            if degree == 0:
                return {"value": int(value), "description": f"Constant term: {int(value)}"}
            else:
                return {"value": 0, "description": f"No x^{degree} term in constant"}

        return {"value": 0, "description": "Unknown expression type"}

    def can_invert(self) -> bool:
        return False


# =============================================================================
# EXTRACT_RESULT - Finalize answer extraction
# =============================================================================

@register_technique
class ExtractResult(MethodBlock):
    """Extract and format final answer from computation.

    Supports integer output, modular reduction, and fraction handling.
    """

    name = "extract_result"
    input_type = "any"
    output_type = "integer"
    difficulty = 1
    tags = ["operational", "output", "finalization"]

    SUPPORTED_FORMATS = {"integer", "mod", "fraction"}

    def __init__(self):
        super().__init__()

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters."""
        return {"format": random.choice(list(self.SUPPORTED_FORMATS))}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Extract final result in requested format.

        Args:
            input_value: Raw computation result
            params: Contains 'format' and format-specific parameters

        Returns:
            MethodResult with formatted answer
        """
        output_format = params.get("format", "integer")

        if output_format not in self.SUPPORTED_FORMATS:
            output_format = "integer"

        try:
            result = self._extract_result(input_value, output_format, params)
            return MethodResult(
                value=result["value"],
                description=result["description"],
                params=params,
                metadata={"format": output_format, "success": True}
            )
        except Exception as e:
            # Fallback to 0 on error
            return MethodResult(
                value=0,
                description=f"Extraction failed: {str(e)}",
                params=params,
                metadata={"success": False, "error": str(e)}
            )

    def _extract_result(self, value: Any, fmt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal result extraction and formatting."""

        # Handle dict results (from other operational methods)
        if isinstance(value, dict):
            # Try to extract meaningful value from dict
            for key in ['value', 'result', 'answer', 'matched', 'verified']:
                if key in value:
                    value = value[key]
                    break
            else:
                # Default to first numeric value found
                for v in value.values():
                    if isinstance(v, (int, float)):
                        value = v
                        break
                else:
                    value = 0

        # Handle list/tuple (take first element or sum)
        if isinstance(value, (list, tuple)):
            if value:
                if all(isinstance(x, (int, float)) for x in value):
                    value = value[0]  # Take first element
                else:
                    value = len(value)  # Take length
            else:
                value = 0

        # Handle boolean
        if isinstance(value, bool):
            value = 1 if value else 0

        # Handle sympy expressions
        if hasattr(value, 'evalf'):
            try:
                value = float(value.evalf())
            except Exception:
                value = 0

        # Now format based on requested format
        if fmt == "integer":
            try:
                result = int(round(float(value)))
            except (ValueError, TypeError):
                result = 0
            return {"value": result, "description": f"Integer result: {result}"}

        elif fmt == "mod":
            modulus = params.get("modulus", 1000000007)
            try:
                result = int(value) % modulus
            except (ValueError, TypeError):
                result = 0
            return {"value": result, "description": f"Result mod {modulus}: {result}"}

        elif fmt == "fraction":
            # Extract numerator, denominator, or combined value
            extract_part = params.get("extract", "numerator")
            try:
                if isinstance(value, Fraction):
                    frac = value
                elif isinstance(value, (int, float)):
                    frac = Fraction(value).limit_denominator(10000)
                else:
                    frac = Fraction(0)

                if extract_part == "numerator":
                    result = frac.numerator
                elif extract_part == "denominator":
                    result = frac.denominator
                elif extract_part == "sum":
                    result = frac.numerator + frac.denominator
                else:
                    result = frac.numerator

                return {"value": result, "description": f"Fraction {extract_part}: {result}"}
            except Exception:
                return {"value": 0, "description": "Could not extract fraction"}

        return {"value": 0, "description": "Unknown format"}

    def can_invert(self) -> bool:
        return False


# =============================================================================
# ALIASES - Map old specific method names to new generic methods
# =============================================================================

# --- Pattern Recognition Aliases ---
@register_technique
class RecognizeCatalanStructure(RecognizeStructure):
    """Alias for recognize_structure(structure_type='catalan')"""
    name = "recognize_catalan_structure"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "catalan", **(params or {})})

@register_technique
class RecognizeFibonacciRecurrence(RecognizeStructure):
    """Alias for recognize_structure(structure_type='fibonacci')"""
    name = "recognize_fibonacci_recurrence"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "fibonacci", **(params or {})})

@register_technique
class RecognizeDiophantineForm(RecognizeStructure):
    """Alias for recognize_structure(structure_type='diophantine')"""
    name = "recognize_diophantine_form"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "diophantine", **(params or {})})

@register_technique
class RecognizeTelescoping(RecognizeStructure):
    """Alias for recognize_structure(structure_type='telescoping')"""
    name = "recognize_telescoping"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "telescoping", **(params or {})})

@register_technique
class IdentifyInvariant(RecognizeStructure):
    """Alias for recognize_structure(structure_type='invariant')"""
    name = "identify_invariant"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "invariant", **(params or {})})

@register_technique
class IdentifyMonovariant(RecognizeStructure):
    """Alias for recognize_structure(structure_type='monovariant')"""
    name = "identify_monovariant"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"structure_type": "monovariant", **(params or {})})

# --- Mapping Aliases ---
@register_technique
class DefineBijection(EstablishMapping):
    """Alias for establish_mapping(mapping_type='bijection')"""
    name = "define_bijection"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"mapping_type": "bijection", **(params or {})})

@register_technique
class EstablishRecurrence(EstablishMapping):
    """Alias for establish_mapping(mapping_type='recurrence')"""
    name = "establish_recurrence"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"mapping_type": "recurrence", **(params or {})})

@register_technique
class TranslateToCoordinates(EstablishMapping):
    """Alias for establish_mapping(mapping_type='coordinates')"""
    name = "translate_to_coordinates"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"mapping_type": "coordinates", **(params or {})})

@register_technique
class NormalizeExpression(EstablishMapping):
    """Alias for establish_mapping(mapping_type='normalization')"""
    name = "normalize_expression"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"mapping_type": "normalization", **(params or {})})

# --- Transform Expression Aliases ---
@register_technique
class Substitute(TransformExpression):
    """Alias for transform_expression(operation='substitute')"""
    name = "substitute"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "substitute", **(params or {})})

@register_technique
class Simplify(TransformExpression):
    """Alias for transform_expression(operation='simplify')"""
    name = "simplify"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "simplify", **(params or {})})

@register_technique
class Factor(TransformExpression):
    """Alias for transform_expression(operation='factor')"""
    name = "factor"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "factor", **(params or {})})

@register_technique
class Expand(TransformExpression):
    """Alias for transform_expression(operation='expand')"""
    name = "expand"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "expand", **(params or {})})

@register_technique
class SolveFor(TransformExpression):
    """Alias for transform_expression(operation='solve_for')"""
    name = "solve_for"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "solve_for", **(params or {})})

@register_technique
class EliminateVariable(TransformExpression):
    """Alias for transform_expression(operation='eliminate')"""
    name = "eliminate_variable"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "eliminate", **(params or {})})

# --- Compute Aliases ---
@register_technique
class ComputeArithmetic(Compute):
    """Alias for compute(operation='arithmetic')"""
    name = "compute_arithmetic"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "arithmetic", **(params or {})})

@register_technique
class ModReduce(Compute):
    """Reduce value modulo modulus."""
    name = "mod_reduce"

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        return {"value": random.randint(1, 1000), "modulus": random.randint(2, 100)}

    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        params = params or {}
        value = params.get("value", input_value)
        modulus = params.get("modulus", 1)
        if isinstance(value, (int, float)):
            value = int(value)
        else:
            value = 0
        modulus = int(modulus) if modulus and modulus != 0 else 1
        result = value % modulus
        return MethodResult(value=result, description=f"{value} mod {modulus} = {result}", params=params)

@register_technique
class EvaluateAtPoint(Compute):
    """Alias for compute(operation='evaluate')"""
    name = "evaluate_at_point"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"operation": "evaluate", **(params or {})})

@register_technique
class ComputeGcd(Compute):
    """Compute GCD of a list of values."""
    name = "compute_gcd"

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        return {"values": [random.randint(1, 100), random.randint(1, 100)]}

    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        params = params or {}
        values = params.get("values", input_value)
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            result = values[0]
            for v in values[1:]:
                result = math.gcd(abs(int(result)), abs(int(v)))
            return MethodResult(value=result, description=f"gcd({values}) = {result}", params=params)
        elif isinstance(values, (list, tuple)) and len(values) == 1:
            return MethodResult(value=int(values[0]), description=f"gcd of single value = {values[0]}", params=params)
        # Fallback to parent behavior for a, b params
        return super().compute(input_value, {"operation": "gcd", **(params or {})})

@register_technique
class ComputeLcm(Compute):
    """Compute LCM of a list of values."""
    name = "compute_lcm"

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        return {"values": [random.randint(1, 100), random.randint(1, 100)]}

    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        params = params or {}
        values = params.get("values", input_value)
        if isinstance(values, (list, tuple)) and len(values) >= 2:
            result = values[0]
            for v in values[1:]:
                result = math.lcm(abs(int(result)), abs(int(v))) if result != 0 and v != 0 else 0
            return MethodResult(value=result, description=f"lcm({values}) = {result}", params=params)
        elif isinstance(values, (list, tuple)) and len(values) == 1:
            return MethodResult(value=int(values[0]), description=f"lcm of single value = {values[0]}", params=params)
        # Fallback to parent behavior for a, b params
        return super().compute(input_value, {"operation": "lcm", **(params or {})})

# --- Verify Aliases ---
@register_technique
class VerifyConstraint(Verify):
    """Alias for verify(condition_type='constraint')"""
    name = "verify_constraint"

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Return expected params for wrapper mapping: value, constraint."""
        return {"value": input_value, "constraint": "positive"}

    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        params = params or {}
        # Handle case where 'value' was mapped instead of input_value
        if 'value' in params and input_value is None:
            input_value = params.pop('value')
        # Handle case where 'constraint' is the second positional arg
        if 'constraint' in params and 'constraint' not in {'positive', 'non_negative', 'integer', 'in_range', 'divisible_by'}:
            # It's a description string, not a constraint type - use as-is
            pass
        return super().compute(input_value, {"condition_type": "constraint", **(params or {})})

@register_technique
class VerifyBaseCase(Verify):
    """Alias for verify(condition_type='base_case')"""
    name = "verify_base_case"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"condition_type": "base_case", **(params or {})})

@register_technique
class CheckBoundary(Verify):
    """Alias for verify(condition_type='boundary')"""
    name = "check_boundary"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"condition_type": "boundary", **(params or {})})

# --- Control Flow Aliases ---
@register_technique
class CaseSplit(ControlFlow):
    """Alias for control_flow(flow_type='case_split')"""
    name = "case_split"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"flow_type": "case_split", **(params or {})})

@register_technique
class InductionStep(ControlFlow):
    """Alias for control_flow(flow_type='induction')"""
    name = "induction_step"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"flow_type": "induction", **(params or {})})

@register_technique
class ContradictionSetup(ControlFlow):
    """Alias for control_flow(flow_type='contradiction')"""
    name = "contradiction_setup"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"flow_type": "contradiction", **(params or {})})

@register_technique
class ExtremalArgument(ControlFlow):
    """Alias for control_flow(flow_type='extremal')"""
    name = "extremal_argument"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"flow_type": "extremal", **(params or {})})

@register_technique
class PigeonholeApply(ControlFlow):
    """Alias for control_flow(flow_type='pigeonhole')"""
    name = "pigeonhole_apply"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"flow_type": "pigeonhole", **(params or {})})

# --- Extract Aliases ---
@register_technique
class ExtractAnswer(ExtractResult):
    """Alias for extract_result(format='integer')"""
    name = "extract_answer"
    def compute(self, input_value: Any, params: Dict[str, Any] = None) -> MethodResult:
        return super().compute(input_value, {"format": "integer", **(params or {})})


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core generic methods (3)
    "ControlFlow",
    "ExtractCoefficients",
    "ExtractResult",
    # Pattern recognition aliases (6)
    "RecognizeCatalanStructure",
    "RecognizeFibonacciRecurrence",
    "RecognizeDiophantineForm",
    "RecognizeTelescoping",
    "IdentifyInvariant",
    "IdentifyMonovariant",
    # Mapping aliases (4)
    "DefineBijection",
    "EstablishRecurrence",
    "TranslateToCoordinates",
    "NormalizeExpression",
    # Transform expression aliases (6)
    "Substitute",
    "Simplify",
    "Factor",
    "Expand",
    "SolveFor",
    "EliminateVariable",
    # Compute aliases (5)
    "ComputeArithmetic",
    "ModReduce",
    "EvaluateAtPoint",
    "ComputeGcd",
    "ComputeLcm",
    # Verify aliases (3)
    "VerifyConstraint",
    "VerifyBaseCase",
    "CheckBoundary",
    # Control flow aliases (5)
    "CaseSplit",
    "InductionStep",
    "ContradictionSetup",
    "ExtremalArgument",
    "PigeonholeApply",
    # Extract aliases (1)
    "ExtractAnswer",
]
