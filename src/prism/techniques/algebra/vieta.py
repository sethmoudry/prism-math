"""
Vieta's formulas and Newton's identities.

Contains:
- Vieta's formulas (3)
- Newton's identities (2)
"""

import random
import math
from typing import Any, Dict, Optional, List

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# VIETA'S FORMULAS (3 techniques)
# ============================================================================

@register_technique
class VietaSum(MethodBlock):
    """Compute sum of roots from polynomial coefficients using Vieta's formulas."""

    def __init__(self):
        super().__init__()
        self.name = "vieta_sum"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "vieta", "roots", "polynomials"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a polynomial with integer coefficients."""
        degree = random.choice([2, 3])

        if degree == 2:
            a = random.randint(1, 5)
            if target_output is not None:
                b = -target_output * a
            else:
                b = random.randint(-20, 20)
            c = random.randint(-50, 50)
            return {"coeffs": [a, b, c], "degree": 2}
        else:
            a = random.randint(1, 3)
            if target_output is not None:
                b = -target_output * a
            else:
                b = random.randint(-15, 15)
            c = random.randint(-30, 30)
            d = random.randint(-50, 50)
            return {"coeffs": [a, b, c, d], "degree": 3}

    def validate_params(self, params, prev_value=None):
        """Validate Vieta sum parameters: coefficients must exist with non-zero leading coefficient."""
        coeffs = params.get("coeffs")
        if coeffs is None or not isinstance(coeffs, (list, tuple)):
            return False
        if len(coeffs) < 2:
            return False
        return coeffs[0] != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute sum of roots using Vieta's formula."""
        coeffs = params.get("coeffs", 10)
        degree = params.get("degree", 10)

        a, b = coeffs[0], coeffs[1]
        sum_roots = -b / a

        if sum_roots == int(sum_roots):
            sum_roots = int(sum_roots)

        poly_str = self._format_polynomial(coeffs)

        return MethodResult(
            value=sum_roots,
            description=f"Sum of roots of {poly_str} = -{b}/{a} = {sum_roots} (Vieta's formula)",
            params=params, metadata={"coeffs": coeffs, "degree": degree}
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        """Find polynomial coefficients whose root sum equals target."""
        degree = random.choice([2, 3])
        a = random.randint(1, 5)
        b = -target * a

        if abs(b) > 100:
            a = 1
            b = -target
            if abs(b) > 100:
                return None

        if degree == 2:
            c = random.randint(-50, 50)
            return {"coeffs": [a, b, c], "degree": 2}
        else:
            c = random.randint(-30, 30)
            d = random.randint(-50, 50)
            return {"coeffs": [a, b, c, d], "degree": 3}

    def _format_polynomial(self, coeffs: List[int]) -> str:
        """Format polynomial as string."""
        n = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = n - i
            if c == 0:
                continue
            if power == 0:
                terms.append(str(c))
            elif power == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{power}" if c != 1 else f"x^{power}")
        return " + ".join(terms).replace("+ -", "- ")


@register_technique
class VietaProduct(MethodBlock):
    """Compute product of roots from polynomial coefficients using Vieta's formulas."""

    def __init__(self):
        super().__init__()
        self.name = "vieta_product"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "vieta", "roots", "polynomials"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a polynomial with integer coefficients."""
        degree = random.choice([2, 3])

        if degree == 2:
            a = random.randint(1, 5)
            b = random.randint(-20, 20)
            if target_output is not None:
                c = target_output * a
            else:
                c = a * random.randint(-10, 10)
            return {"coeffs": [a, b, c], "degree": 2}
        else:
            a = random.randint(1, 3)
            b = random.randint(-15, 15)
            c = random.randint(-30, 30)
            if target_output is not None:
                d = -target_output * a
            else:
                d = random.randint(-50, 50)
            return {"coeffs": [a, b, c, d], "degree": 3}

    def validate_params(self, params, prev_value=None):
        """Validate Vieta product parameters: coefficients must exist with non-zero leading coefficient."""
        coeffs = params.get("coeffs")
        if coeffs is None or not isinstance(coeffs, (list, tuple)):
            return False
        if len(coeffs) < 2:
            return False
        return coeffs[0] != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute product of roots using Vieta's formula."""
        coeffs = params.get("coeffs", 10)
        degree = params.get("degree", 10)

        a = coeffs[0]
        if degree == 1:
            b = coeffs[1]
            product = -b // a
        elif degree == 2:
            c = coeffs[2]
            product = c // a
        else:
            constant = coeffs[-1]
            product = ((-1) ** degree) * constant // a

        if product == int(product):
            product = int(product)

        poly_str = self._format_polynomial(coeffs)

        return MethodResult(
            value=product,
            description=f"Product of roots of {poly_str} = {product} (Vieta's formula)",
            params=params, metadata={"coeffs": coeffs, "degree": degree}
        )

    def can_invert(self) -> bool:
        return False

    def _format_polynomial(self, coeffs: List[int]) -> str:
        """Format polynomial as string."""
        n = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            power = n - i
            if c == 0:
                continue
            if power == 0:
                terms.append(str(c))
            elif power == 1:
                terms.append(f"{c}x" if c != 1 else "x")
            else:
                terms.append(f"{c}x^{power}" if c != 1 else f"x^{power}")
        return " + ".join(terms).replace("+ -", "- ")


@register_technique
class VietaInverseCoeffs(MethodBlock):
    """Reconstruct polynomial coefficients from sum and product of roots."""

    def __init__(self):
        super().__init__()
        self.name = "vieta_inverse_coeffs"
        self.input_type = "integer"
        self.output_type = "polynomial"
        self.difficulty = 3
        self.tags = ["algebra", "vieta", "roots", "polynomials"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate sum and product of roots."""
        root_sum = random.randint(-10, 10)
        root_product = random.randint(-20, 20)
        return {"root_sum": root_sum, "root_product": root_product}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Reconstruct quadratic from root sum and product."""
        s = params.get("root_sum", 10)
        p = params.get("root_product", 10)

        coeffs = [1, -s, p]

        return MethodResult(
            value=coeffs,
            description=f"Polynomial with root sum={s}, product={p} is x^2 - ({s})x + ({p})",
            params=params, metadata={"root_sum": s, "root_product": p}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Given polynomial coeffs, find root sum and product."""
        coeffs = output_value
        root_sum = -coeffs[1]
        root_product = coeffs[2]

        return MethodResult(
            value=int(root_sum),  # Return simple int (root_sum), not dict
            description=f"From polynomial, root_sum={root_sum}, root_product={root_product}",
            params=params,
            metadata={"coeffs": coeffs, "root_sum": root_sum, "root_product": root_product}
        )


# ============================================================================
# NEWTON'S IDENTITIES (2 techniques)
# ============================================================================

@register_technique
class NewtonPowerSum(MethodBlock):
    """Compute power sum p_k = Sigma r_i^k from elementary symmetric polynomials."""

    def __init__(self):
        super().__init__()
        self.name = "newton_power_sum"
        self.input_type = "polynomial"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "newton", "symmetric", "polynomials"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate roots and power to compute."""
        if input_value is not None and input_value > 1000:
            n_roots = random.choice([3, 4])
            roots = [random.randint(-20, 20) for _ in range(n_roots)]
            k = random.randint(3, 6)
        else:
            n_roots = random.choice([2, 3])
            roots = [random.randint(-5, 5) for _ in range(n_roots)]
            k = random.randint(1, 4)
        return {"roots": roots, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute power sum directly from roots."""
        roots = params.get("roots", 10)
        k = params.get("k", 5)

        power_sum = sum(r**k for r in roots)

        description = f"Power sum p_{k} = Sigma r^{k} = {power_sum} for roots {roots}"

        return MethodResult(
            value=power_sum,
            description=description,
            params=params, metadata={"roots": roots, "k": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class NewtonInverseElemSym(MethodBlock):
    """Compute elementary symmetric from power sums using Newton's identities."""

    def __init__(self):
        super().__init__()
        self.name = "newton_inverse_elemsym"
        self.input_type = "sequence"
        self.output_type = "sequence"
        self.difficulty = 5
        self.tags = ["algebra", "newton", "symmetric", "polynomials"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate power sums."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            p1 = random.randint(-50, 50)
            p2 = random.randint(-100, 100)
        else:
            p1 = random.randint(-10, 10)
            p2 = random.randint(-20, 20)
        return {"power_sums": [p1, p2], "n": 2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute elementary symmetric polynomials from power sums."""
        ps = params.get("power_sums", 10)

        e1 = ps[0]
        e2 = (ps[0]**2 - ps[1]) // 2

        description = f"Elementary symmetric: e_1={e1}, e_2={e2} from p_1={ps[0]}, p_2={ps[1]}"

        return MethodResult(
            value=[e1, e2],
            description=description,
            params=params, metadata={"power_sums": ps}
        )

    def can_invert(self) -> bool:
        return False
