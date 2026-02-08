"""
Matrix operations and curve intersection techniques.

Contains:
- Matrix operations (6)
- Curve intersection (1)
- Named cubics (1)
"""

import random
import math
from typing import Any, Dict, Optional
import numpy as np

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# MATRIX OPERATIONS (6 techniques)
# ============================================================================

@register_technique
class MatrixPower(MethodBlock):
    """Compute M^n efficiently using exponentiation by squaring."""

    def __init__(self):
        super().__init__()
        self.name = "matrix_power"
        self.input_type = "matrix"
        self.output_type = "matrix"
        self.difficulty = 3
        self.tags = ["algebra", "matrix", "exponentiation"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate matrix and exponent."""
        matrix = [[random.randint(-2, 2) for _ in range(2)] for _ in range(2)]
        n = random.randint(3, 8)
        return {"matrix": matrix, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute M^n."""
        M = np.array(params.get("matrix", 10))
        n = params.get("n", 10)

        result = np.linalg.matrix_power(M, n)

        return MethodResult(
            value=result.tolist(),
            description=f"M^{n} computed using matrix exponentiation",
            params=params, metadata={"matrix": params.get("matrix", 10), "n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MatrixPowerInverseN(MethodBlock):
    """Find n given M and M^n."""

    def __init__(self):
        super().__init__()
        self.name = "matrix_power_inverse_n"
        self.input_type = "matrix"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "matrix", "inverse", "discrete_log"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate matrix and compute M^n."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            matrix = [[2, 0], [0, 3]]
            n = random.randint(8, 15)
        else:
            matrix = [[2, 0], [0, 3]]
            n = random.randint(3, 6)
        M_n = np.linalg.matrix_power(np.array(matrix), n)
        return {"matrix": matrix, "M_n": M_n.tolist(), "true_n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find n by checking M^k until match."""
        M = np.array(params.get("matrix", 10))
        M_n = np.array(params.get("M_n", 10))

        for k in range(1, 50):
            if np.allclose(np.linalg.matrix_power(M, k), M_n):
                result = k
                description = f"Found n={result} where M^n matches target"
                return MethodResult(
                    value=result,
                    description=description,
                    params=params, metadata={"matrix": params.get("matrix", 10)}
                )

        return MethodResult(
            value=-1,
            description="Could not find n in reasonable range",
            params=params, metadata={"matrix": params.get("matrix", 10)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MatrixDet(MethodBlock):
    """Compute determinant of a matrix."""

    def __init__(self):
        super().__init__()
        self.name = "matrix_det"
        self.input_type = "matrix"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "matrix", "determinant"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate matrix."""
        size = random.choice([2, 3])
        matrix = [[random.randint(-5, 5) for _ in range(size)] for _ in range(size)]
        return {"matrix": matrix}

    def validate_params(self, params, prev_value=None):
        """Validate matrix determinant parameters."""
        if isinstance(prev_value, (int, float)):
            return True

        matrix = params.get("matrix")
        if matrix is None or not isinstance(matrix, (list, tuple)):
            return False
        if len(matrix) == 0:
            return False
        n = len(matrix)
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) != n:
                return False
        return True

    def validate_input(self, input_value: Any) -> bool:
        """Accept None, integers, floats, or matrices as input."""
        if input_value is None:
            return True
        if isinstance(input_value, (int, float)):
            return True
        if isinstance(input_value, (list, tuple)) and len(input_value) > 0:
            return all(isinstance(row, (list, tuple)) for row in input_value)
        return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute determinant."""
        if isinstance(input_value, (int, float)) and not isinstance(params.get("matrix"), list):
            seed_val = int(input_value) % 1000
            rng = random.Random(seed_val)
            size = rng.choice([2, 3])
            matrix = [[rng.randint(-5, 5) for _ in range(size)] for _ in range(size)]
        else:
            matrix = params.get("matrix")
            if matrix is None or not isinstance(matrix, list):
                size = random.choice([2, 3])
                matrix = [[random.randint(-5, 5) for _ in range(size)] for _ in range(size)]

        M = np.array(matrix)
        det = int(round(np.linalg.det(M)))

        return MethodResult(
            value=det,
            description=f"det(M) = {det}",
            params={"matrix": matrix}, metadata={"matrix": matrix}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MatrixTrace(MethodBlock):
    """Compute trace of a matrix."""

    def __init__(self):
        super().__init__()
        self.name = "matrix_trace"
        self.input_type = "matrix"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "matrix", "trace"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate matrix."""
        size = random.choice([2, 3, 4])
        matrix = [[random.randint(-10, 10) for _ in range(size)] for _ in range(size)]
        return {"matrix": matrix}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute trace (sum of diagonal elements)."""
        if isinstance(input_value, (int, float)) and not isinstance(params.get("matrix"), list):
            seed_val = int(input_value) % 1000
            rng = random.Random(seed_val)
            size = rng.choice([2, 3, 4])
            M = [[rng.randint(-10, 10) for _ in range(size)] for _ in range(size)]
        else:
            M = params.get("matrix")
            if M is None or not isinstance(M, (list, tuple)):
                size = random.choice([2, 3, 4])
                M = [[random.randint(-10, 10) for _ in range(size)] for _ in range(size)]

        trace = sum(M[i][i] for i in range(len(M)))

        return MethodResult(
            value=trace,
            description=f"tr(M) = {trace}",
            params={"matrix": M}, metadata={"matrix": M}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Determinant2x2(MethodBlock):
    """Compute 2x2 matrix determinant ad - bc."""

    def __init__(self):
        super().__init__()
        self.name = "determinant_2x2"
        self.input_type = "matrix"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "matrix", "determinant"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate 2x2 matrix."""
        matrix = [[random.randint(-10, 10) for _ in range(2)] for _ in range(2)]
        return {"matrix": matrix}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute determinant ad - bc."""
        M = params.get("matrix", [[1, 0], [0, 1]])
        a, b = M[0][0], M[0][1]
        c, d = M[1][0], M[1][1]

        det = a * d - b * c

        return MethodResult(
            value=int(det),
            description=f"Determinant of [[{a}, {b}], [{c}, {d}]]: det = {a}*{d} - {b}*{c} = {a*d} - {b*c} = {det}",
            params=params,
            metadata={"matrix": M, "determinant": det}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VectorDotProductFromDiagonals(MethodBlock):
    """Compute dot product from matrix diagonals."""

    def __init__(self):
        super().__init__()
        self.name = "vector_dot_product_from_diagonals"
        self.input_type = "matrix"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["algebra", "matrix", "dot_product"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate matrix with two diagonals."""
        size = random.randint(2, 4)
        matrix = [[random.randint(-5, 5) for _ in range(size)] for _ in range(size)]
        return {"matrix": matrix}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute dot product of main and anti-diagonal."""
        M = params.get("matrix", [[1, 0], [0, 1]])
        n = len(M)

        main_diag = [M[i][i] for i in range(n)]
        anti_diag = [M[i][n-1-i] for i in range(n)]

        dot_product = sum(main_diag[i] * anti_diag[i] for i in range(n))

        return MethodResult(
            value=int(dot_product),
            description=f"Dot product of diagonals: main_diag = {main_diag}, anti_diag = {anti_diag}, dot product = {dot_product}",
            params=params,
            metadata={"main_diagonal": main_diag, "anti_diagonal": anti_diag, "dot_product": dot_product}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# CURVE INTERSECTION AND CUBICS
# ============================================================================

@register_technique
class CurveIntersection(MethodBlock):
    """Find intersection points of two curves."""

    def __init__(self):
        super().__init__()
        self.name = "curve_intersection"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "geometry", "intersection"]

    def generate_parameters(self, input_value=None):
        curve_type = random.choice(["two_lines", "line_parabola", "two_circles"])
        if curve_type == "two_lines":
            x_int, y_int = random.randint(-10, 10), random.randint(-10, 10)
            a1, b1 = random.randint(1, 5), random.randint(-5, 5)
            a2, b2 = random.randint(-5, -1), random.randint(-5, 5)
            c1 = a1 * x_int + b1 * y_int
            c2 = a2 * x_int + b2 * y_int
            return {"curve_type": "two_lines", "a1": a1, "b1": b1, "c1": c1,
                    "a2": a2, "b2": b2, "c2": c2, "output": "x"}
        elif curve_type == "line_parabola":
            x1, x2 = random.randint(-5, 5), random.randint(-5, 5)
            if x1 == x2:
                x2 = x1 + 1
            m = x1 + x2
            b = -x1 * x2
            return {"curve_type": "line_parabola", "m": m, "b": b, "output": "count"}
        else:
            h1, k1, r1 = 0, 0, random.randint(3, 8)
            d = random.randint(1, 2 * r1 - 1)
            h2, k2, r2 = d, 0, random.randint(2, 6)
            return {"curve_type": "two_circles", "h1": h1, "k1": k1, "r1": r1,
                    "h2": h2, "k2": k2, "r2": r2, "output": "count"}

    def compute(self, input_value, params):
        curve_type = params.get("curve_type", "two_lines")
        output_type = params.get("output", "x")

        if curve_type == "two_lines":
            a1, b1, c1 = params.get("a1", 1), params.get("b1", 0), params.get("c1", 0)
            a2, b2, c2 = params.get("a2", 0), params.get("b2", 1), params.get("c2", 0)
            det = a1 * b2 - a2 * b1
            if det == 0:
                return MethodResult(value=0, description="Lines are parallel (no intersection)",
                                    metadata={"curve_type": curve_type, "parallel": True})
            x = (c1 * b2 - c2 * b1) // det if (c1 * b2 - c2 * b1) % det == 0 else (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) // det if (a1 * c2 - a2 * c1) % det == 0 else (a1 * c2 - a2 * c1) / det
            result = int(x) if output_type == "x" else int(y) if output_type == "y" else int(x + y)
            return MethodResult(
                value=result,
                description=f"Lines {a1}x + {b1}y = {c1} and {a2}x + {b2}y = {c2} intersect at ({x}, {y})",
                metadata={"curve_type": curve_type, "x": x, "y": y}
            )

        elif curve_type == "line_parabola":
            m, b = params.get("m", 0), params.get("b", 0)
            discriminant = m * m + 4 * b
            if discriminant < 0:
                count = 0
            elif discriminant == 0:
                count = 1
            else:
                count = 2
            return MethodResult(
                value=count,
                description=f"Line y = {m}x + {b} and parabola y = x^2 have {count} intersection(s)",
                metadata={"curve_type": curve_type, "m": m, "b": b, "discriminant": discriminant}
            )

        else:
            h1, k1, r1 = params.get("h1", 0), params.get("k1", 0), params.get("r1", 5)
            h2, k2, r2 = params.get("h2", 3), params.get("k2", 0), params.get("r2", 4)
            d_sq = (h2 - h1) ** 2 + (k2 - k1) ** 2
            d = d_sq ** 0.5
            if d > r1 + r2 or d < abs(r1 - r2):
                count = 0
            elif d == r1 + r2 or d == abs(r1 - r2):
                count = 1
            else:
                count = 2
            return MethodResult(
                value=count,
                description=f"Circles centered at ({h1},{k1}) r={r1} and ({h2},{k2}) r={r2} have {count} intersection(s)",
                metadata={"curve_type": curve_type, "distance": d, "r1": r1, "r2": r2}
            )

    def can_invert(self) -> bool:
        return False


@register_technique
class NamedCubicsProperties(MethodBlock):
    """Compute properties of named cubic functions."""

    def __init__(self):
        super().__init__()
        self.name = "named_cubics_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "cubics"]

    def generate_parameters(self, input_value=None):
        a = random.randint(-5, 5)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        a = params.get("a", 10)
        b = params.get("b", 10)
        c = params.get("c", 10)
        x_inflection = -a / 3
        f_inflection = x_inflection**3 + a * x_inflection**2 + b * x_inflection + c
        result = int(round(f_inflection))
        return MethodResult(
            value=result,
            description=f"Cubic x^3 + {a}x^2 + {b}x + {c} has inflection point value = {result}",
            metadata={"a": a, "b": b, "c": c, "x_inflection": x_inflection}
        )

    def can_invert(self) -> bool:
        return False
