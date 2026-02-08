"""
Expression manipulation utilities.

Contains:
- Square sum/difference expansion (3)
- Expression separation (1)
- MN extraction (2)
- Quotient of sums (1)
- Dot product simplification (1)
- Sqrt parameter extraction (1)
"""

import random
import math
from typing import Any, Dict, Optional
from fractions import Fraction

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class SquareSumSimplify(MethodBlock):
    """Simplify (a+b)^2 to a^2 + 2ab + b^2."""

    def __init__(self):
        super().__init__()
        self.name = "square_sum_simplify"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "expansion", "squares"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two integers a and b."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute (a+b)^2."""
        a = params.get("a", 1)
        b = params.get("b", 1)

        result = (a + b) ** 2
        expanded = a**2 + 2*a*b + b**2

        return MethodResult(
            value=int(result),
            description=f"(a+b)^2 = ({a}+{b})^2 = {a}^2 + 2*{a}*{b} + {b}^2 = {a**2} + {2*a*b} + {b**2} = {result}",
            params=params,
            metadata={"a": a, "b": b, "expanded": expanded}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExpandSquareSum(MethodBlock):
    """Expand (a+b)^2 = a^2 + 2ab + b^2."""

    def __init__(self):
        super().__init__()
        self.name = "expand_square_sum"
        self.input_type = "integer"
        self.output_type = "expression"
        self.difficulty = 1
        self.tags = ["algebra", "expansion", "squares"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two terms a and b."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Expand (a+b)^2."""
        a = params.get("a", 1)
        b = params.get("b", 1)

        a_sq = a ** 2
        two_ab = 2 * a * b
        b_sq = b ** 2

        result_sum = a_sq + two_ab + b_sq
        return MethodResult(
            value=int(result_sum),  # Return simple int, not dict
            description=f"Expand (a+b)^2: ({a}+{b})^2 = {a}^2 + 2*{a}*{b} + {b}^2 = {a_sq} + {two_ab} + {b_sq} = {result_sum}",
            params=params,
            metadata={"a_squared": a_sq, "two_ab": two_ab, "b_squared": b_sq, "sum": result_sum}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExpandDifference(MethodBlock):
    """Expand (a-b)^2 = a^2 - 2ab + b^2."""

    def __init__(self):
        super().__init__()
        self.name = "expand_difference"
        self.input_type = "integer"
        self.output_type = "expression"
        self.difficulty = 1
        self.tags = ["algebra", "expansion", "squares"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two terms a and b."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Expand (a-b)^2."""
        a = params.get("a", 1)
        b = params.get("b", 1)

        a_sq = a ** 2
        two_ab = 2 * a * b
        b_sq = b ** 2
        diff = a_sq - two_ab + b_sq

        return MethodResult(
            value=int(diff),  # Return simple int, not dict
            description=f"Expand (a-b)^2: ({a}-{b})^2 = {a}^2 - 2*{a}*{b} + {b}^2 = {a_sq} - {two_ab} + {b_sq} = {diff}",
            params=params,
            metadata={"a_squared": a_sq, "minus_two_ab": -two_ab, "b_squared": b_sq, "sum": diff}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SeparateExpressionParts(MethodBlock):
    """Separate an expression into its component parts."""

    def __init__(self):
        super().__init__()
        self.name = "separate_expression_parts"
        self.input_type = "expression"
        self.output_type = "expression_parts"
        self.difficulty = 2
        self.tags = ["algebra", "parsing", "expressions"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate an expression with multiple terms."""
        num_terms = random.randint(2, 4)
        coeffs = [random.randint(-10, 10) for _ in range(num_terms)]
        return {"coeffs": coeffs}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Separate expression into parts."""
        coeffs = params.get("coeffs", [1, 0])

        parts = []
        for i, c in enumerate(coeffs):
            power = len(coeffs) - 1 - i
            if power == 0:
                parts.append(f"{c}")
            elif power == 1:
                parts.append(f"{c}x")
            else:
                parts.append(f"{c}x^{power}")

        return MethodResult(
            value=parts,
            description=f"Expression parts: {' + '.join(parts)} has {len(parts)} terms",
            params=params,
            metadata={"parts": parts, "coefficients": coeffs}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExtractMN(MethodBlock):
    """Extract m and n from an expression of form mx + n."""

    def __init__(self):
        super().__init__()
        self.name = "extract_m_n"
        self.input_type = "linear_expression"
        self.output_type = "tuple"
        self.difficulty = 1
        self.tags = ["algebra", "extraction", "linear"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate linear expression mx + n."""
        m = random.randint(-10, 10)
        n = random.randint(-20, 20)
        return {"m": m, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Extract m and n from mx + n."""
        m = params.get("m", 1)
        n = params.get("n", 0)

        return MethodResult(
            value=(m, n),
            description=f"From expression {m}x + {n}, extract m = {m} and n = {n}",
            params=params,
            metadata={"m": m, "n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExtractMNSum(MethodBlock):
    """Extract m+n from expression mx + n."""

    def __init__(self):
        super().__init__()
        self.name = "extract_m_n_sum"
        self.input_type = "linear_expression"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "extraction", "linear"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate linear expression mx + n."""
        m = random.randint(-10, 10)
        n = random.randint(-20, 20)
        return {"m": m, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Extract m+n from mx + n."""
        m = params.get("m", 1)
        n = params.get("n", 0)
        sum_mn = m + n

        return MethodResult(
            value=int(sum_mn),
            description=f"From expression {m}x + {n}, extract m + n = {m} + {n} = {sum_mn}",
            params=params,
            metadata={"m": m, "n": n, "sum": sum_mn}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuotientOfSums(MethodBlock):
    """Compute (Sigma a) / (Sigma b) for two sequences."""

    def __init__(self):
        super().__init__()
        self.name = "quotient_of_sums"
        self.input_type = "sequences"
        self.output_type = "rational"
        self.difficulty = 2
        self.tags = ["algebra", "sequences", "sums"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two sequences."""
        length = random.randint(3, 6)
        seq_a = [random.randint(1, 10) for _ in range(length)]
        seq_b = [random.randint(1, 10) for _ in range(length)]
        return {"seq_a": seq_a, "seq_b": seq_b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute quotient of sums."""
        seq_a = params.get("seq_a", [1, 2, 3])
        seq_b = params.get("seq_b", [1, 2, 3])

        sum_a = sum(seq_a)
        sum_b = sum(seq_b)

        quotient = Fraction(sum_a, sum_b)

        return MethodResult(
            value=float(quotient),
            description=f"Quotient of sums: (Sigma {seq_a}) / (Sigma {seq_b}) = {sum_a} / {sum_b} = {quotient}",
            params=params,
            metadata={"sum_a": sum_a, "sum_b": sum_b, "quotient": str(quotient)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SimplifyDotProduct(MethodBlock):
    """Simplify dot product <v1, v2>."""

    def __init__(self):
        super().__init__()
        self.name = "simplify_dot_product"
        self.input_type = "vectors"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "vectors", "dot_product"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two vectors."""
        dim = random.randint(2, 4)
        v1 = [random.randint(-5, 5) for _ in range(dim)]
        v2 = [random.randint(-5, 5) for _ in range(dim)]
        return {"v1": v1, "v2": v2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute dot product."""
        v1 = params.get("v1", [1, 0])
        v2 = params.get("v2", [0, 1])

        dot_product = sum(v1[i] * v2[i] for i in range(len(v1)))

        terms = [f"{v1[i]}*{v2[i]}" for i in range(len(v1))]

        return MethodResult(
            value=int(dot_product),
            description=f"Dot product: <{v1}, {v2}> = {' + '.join(terms)} = {dot_product}",
            params=params,
            metadata={"v1": v1, "v2": v2, "dot_product": dot_product}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExtractSqrtABCParameters(MethodBlock):
    """Extract parameters a, b, c from expression sqrt(abc)."""

    def __init__(self):
        super().__init__()
        self.name = "extract_sqrt_abc_parameters"
        self.input_type = "sqrt_expression"
        self.output_type = "tuple"
        self.difficulty = 2
        self.tags = ["algebra", "extraction", "radicals"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters for sqrt(abc)."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 10)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Extract a, b, c from sqrt(abc)."""
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 1)

        product = a * b * c
        sqrt_val = math.sqrt(product)

        return MethodResult(
            value=(a, b, c),
            description=f"From sqrt({a}*{b}*{c}) = sqrt({product}) = {sqrt_val:.2f}, extract a = {a}, b = {b}, c = {c}",
            params=params,
            metadata={"a": a, "b": b, "c": c, "product": product, "sqrt_value": sqrt_val}
        )

    def can_invert(self) -> bool:
        return False
