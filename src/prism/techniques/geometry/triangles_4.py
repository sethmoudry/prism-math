"""
Geometry Techniques - Triangles (Part 4)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class SimilarTriangles(MethodBlock):
    """
    Compute unknown side or ratio using similar triangle proportions.

    For similar triangles ABC ~ DEF:
    - Corresponding sides are proportional: AB/DE = BC/EF = CA/FD
    - Given known sides, compute the unknown corresponding side

    Supports multiple operations:
    - 'find_side': Given ratio and one side, find corresponding side
    - 'find_ratio': Given corresponding sides, find the ratio
    - 'scale_triangle': Scale all sides of a triangle by a ratio
    - 'proportion': Solve a/b = c/x for x

    Parameters:
        operation: 'find_side', 'find_ratio', 'scale_triangle', 'proportion'
        ratio: scale ratio between triangles (for find_side, scale_triangle)
        side1, side2: two corresponding sides (for find_ratio)
        known_side: known side length (for find_side)
        sides: list of [a, b, c] for original triangle (for scale_triangle)
        a, b, c, x: for proportion a/b = c/x (solve for x)
    """
    def __init__(self):
        super().__init__()
        self.name = "similar_triangles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "similarity", "proportion", "triangle"]

    def generate_parameters(self, input_value=None):
        # Generate a simple proportion problem
        a = random.randint(3, 10)
        b = random.randint(3, 10)
        c = random.randint(3, 10)
        # x = b * c / a
        return {
            "operation": "proportion",
            "a": a,
            "b": b,
            "c": c
        }

    def validate_params(self, params, prev_value=None):
        """Validate similar triangles parameters: divisors must be non-zero."""
        operation = params.get("operation", "proportion")
        if operation == "proportion":
            a = params.get("a")
            if a is None:
                return False
            try:
                return float(a) != 0  # Division by a
            except (ValueError, TypeError):
                return False
        elif operation == "find_ratio":
            side2 = params.get("side2")
            if side2 is None:
                return False
            try:
                return float(side2) != 0  # Division by side2
            except (ValueError, TypeError):
                return False
        return True

    def compute(self, input_value, params):
        operation = params.get("operation", "proportion")

        if operation == "find_side":
            # Given ratio k and known side s, find corresponding side k*s
            ratio = params.get("ratio", 2)
            known_side = params.get("known_side", 5)

            # Handle ratio as fraction [numerator, denominator] or single value
            if isinstance(ratio, (list, tuple)) and len(ratio) >= 2:
                ratio_value = ratio[0] / ratio[1]
            else:
                ratio_value = float(ratio)

            unknown_side = known_side * ratio_value
            result = int(round(unknown_side))

            description = f"Similar triangles: side {known_side} x ratio {ratio_value:.2f} = {result}"
            metadata = {"ratio": ratio_value, "known_side": known_side}

        elif operation == "find_ratio":
            # Given two corresponding sides, find the ratio
            side1 = params.get("side1", 6)
            side2 = params.get("side2", 3)

            if side2 == 0:
                side2 = 1

            ratio = side1 / side2
            # Return ratio * 100 for precision (e.g., 1.5 -> 150)
            result = int(round(ratio * 100))

            description = f"Similar triangles ratio: {side1}/{side2} = {ratio:.2f} (encoded as {result})"
            metadata = {"side1": side1, "side2": side2, "ratio": ratio}

        elif operation == "scale_triangle":
            # Scale all sides by ratio
            sides = params.get("sides", [3, 4, 5])
            ratio = params.get("ratio", 2)

            if isinstance(sides, (list, tuple)):
                sides = list(sides)
            else:
                sides = [sides]

            if isinstance(ratio, (list, tuple)) and len(ratio) >= 2:
                ratio_value = ratio[0] / ratio[1]
            else:
                ratio_value = float(ratio)

            scaled = [s * ratio_value for s in sides]
            # Return sum of scaled sides (perimeter)
            result = int(round(sum(scaled)))

            description = f"Scaled triangle {sides} by {ratio_value}: perimeter = {result}"
            metadata = {"original_sides": sides, "ratio": ratio_value, "scaled_sides": scaled}

        elif operation == "proportion":
            # Solve a/b = c/x for x => x = b*c/a
            a = params.get("a", 2)
            b = params.get("b", 3)
            c = params.get("c", 4)

            if a == 0:
                a = 1

            x = (b * c) / a
            result = int(round(x))

            description = f"Proportion: {a}/{b} = {c}/x => x = {b}*{c}/{a} = {result}"
            metadata = {"a": a, "b": b, "c": c, "exact_x": x}

        else:
            # Default: simple ratio multiplication
            ratio = params.get("ratio", 2)
            known_side = params.get("known_side", input_value if input_value else 5)
            result = int(round(known_side * ratio))
            description = f"Similar triangles scale: {known_side} x {ratio} = {result}"
            metadata = {"operation": operation}

        return MethodResult(
            value=result,
            description=description,
            metadata=metadata
        )

    def can_invert(self):
        return False


@register_technique
class SumOfAngles(MethodBlock):
    """Compute sum of interior angles in polygons."""
    def __init__(self):
        super().__init__()
        self.name = "sum_of_angles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "angles"]

    def generate_parameters(self, input_value=None):
        n = random.randint(3, 12)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 10)
        result = (n - 2) * 180
        return MethodResult(
            value=result,
            description=f"Sum of interior angles in {n}-gon: {result}°",
            metadata={"n": n}
        )

    def can_invert(self):
        return False


@register_technique
class StewartCevianLength(MethodBlock):
    """
    Compute the squared length of a cevian using Stewart's theorem.
    For an angle bisector from A to D on BC in triangle ABC:
    d² = bc[(b+c)² - a²]/(b+c)²
    where a=BC, b=CA, c=AB, d=AD
    """

    def __init__(self):
        super().__init__()
        self.name = "stewart_cevian_length"
        self.input_type = "none"
        self.output_type = "symbolic_expression"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "stewart", "cevian"]

    def generate_parameters(self, input_value=None):
        return {
            "cevian_type": "angle_bisector"
        }

    def compute(self, input_value, params):
        from sympy import symbols, simplify

        cevian_type = params.get("cevian_type", "angle_bisector")

        # Define symbolic variables for triangle sides
        a, b, c = symbols('a b c', positive=True, real=True)

        # Stewart's theorem for angle bisector:
        # d² = bc[(b+c)² - a²]/(b+c)²
        d_squared = (b * c * ((b + c)**2 - a**2)) / (b + c)**2
        d_squared = simplify(d_squared)

        return MethodResult(
            value=d_squared,
            description=f"Stewart's theorem for {cevian_type}: d² = bc[(b+c)² - a²]/(b+c)²",
            params=params,
            metadata={"cevian_type": cevian_type}
        )

    def can_invert(self):
        return False


@register_technique
class SolveLengthEqualsSide(MethodBlock):
    """
    Given a length expression d², set it equal to a side (e.g., d = c).
    Derives the constraint equation.
    For d² = bc[(b+c)² - a²]/(b+c)² and d = c:
    c² = bc[(b+c)² - a²]/(b+c)²
    Simplifies to: (a/(b+c))² = (b-c)/b
    """

    def __init__(self):
        super().__init__()
        self.name = "solve_length_equals_side"
        self.input_type = "symbolic_expression"
        self.output_type = "constraint"
        self.difficulty = 3
        self.tags = ["geometry", "algebra", "constraint"]

    def generate_parameters(self, input_value=None):
        return {
            "equal_to": "c"
        }

    def compute(self, input_value, params):
        from sympy import symbols, Eq, simplify, solve

        d_squared = input_value
        equal_to = params.get("equal_to", "c")

        a, b, c = symbols('a b c', positive=True, real=True)

        # Set d = c, so d² = c²
        # c² = bc[(b+c)² - a²]/(b+c)²
        # Multiply both sides by (b+c)²:
        # c²(b+c)² = bc[(b+c)² - a²]
        # Divide by c:
        # c(b+c)² = b[(b+c)² - a²]
        # c(b+c)² = b(b+c)² - ba²
        # c(b+c)² - b(b+c)² = -ba²
        # (c-b)(b+c)² = -ba²
        # Rearranging:
        # (a/(b+c))² = (b-c)/b

        constraint = Eq((a/(b+c))**2, (b-c)/b)

        return MethodResult(
            value=constraint,
            description=f"Setting d = {equal_to} gives constraint: (a/(b+c))² = (b-c)/b",
            params=params,
            metadata={"equal_to": equal_to}
        )

    def can_invert(self):
        return False


@register_technique
class TriangleVertexCoordinates(MethodBlock):
    """
    Extract vertex coordinates from a triangle.
    Returns sum of x-coordinates of all vertices.
    """

    def __init__(self):
        super().__init__()
        self.name = "triangle_vertex_coordinates"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "triangle", "coordinates"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate triangle vertices."""
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)

        return {"v1": (x1, y1), "v2": (x2, y2), "v3": (x3, y3)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("v1", (0, 0)), "v1", 2)
        x2, y2 = ensure_tuple(params.get("v2", (1, 0)), "v2", 2)
        x3, y3 = ensure_tuple(params.get("v3", (0, 1)), "v3", 2)

        sum_x = x1 + x2 + x3

        return MethodResult(
            value=int(sum_x),
            description=f"Triangle vertices: ({x1},{y1}), ({x2},{y2}), ({x3},{y3}). Sum of x-coordinates: {sum_x}",
            metadata={"vertices": [(x1, y1), (x2, y2), (x3, y3)], "sum_x": sum_x}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointInTriangle(MethodBlock):
    """
    Check if a point is inside a triangle using barycentric coordinates.
    Returns 1 if inside, 0 if outside.
    """

    def __init__(self):
        super().__init__()
        self.name = "point_in_triangle"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "containment"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate triangle and test point."""
        # Triangle vertices
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)

        # Test point (may be inside or outside)
        px = random.randint(-15, 15)
        py = random.randint(-15, 15)

        return {"v1": (x1, y1), "v2": (x2, y2), "v3": (x3, y3), "point": (px, py)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("v1", (0, 0)), "v1", 2)
        x2, y2 = ensure_tuple(params.get("v2", (1, 0)), "v2", 2)
        x3, y3 = ensure_tuple(params.get("v3", (0, 1)), "v3", 2)
        px, py = ensure_tuple(params.get("point", (0.3, 0.3)), "point", 2)

        # Compute barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)

        if abs(denom) < 1e-9:
            # Degenerate triangle
            inside = 0
        else:
            alpha = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
            beta = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
            gamma = 1 - alpha - beta

            # Point is inside if all barycentric coords are in [0, 1]
            inside = 1 if (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1) else 0

        return MethodResult(
            value=inside,
            description=f"Point ({px},{py}) is {'inside' if inside else 'outside'} triangle [({x1},{y1}), ({x2},{y2}), ({x3},{y3})]",
            metadata={"point": (px, py), "triangle": [(x1, y1), (x2, y2), (x3, y3)]}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SideTimesSin(MethodBlock):
    """
    Compute a·sin(θ) for a given side length and angle.

    This is a fundamental operation in triangle calculations,
    especially in the law of sines and area formulas.
    """

    def __init__(self):
        super().__init__()
        self.name = "side_times_sin"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "trigonometry", "sine"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate side length and angle."""
        side = random.randint(5, 30)
        # Angle in degrees for clarity
        angle_deg = random.randint(15, 165)
        angle_rad = math.radians(angle_deg)
        return {"side": side, "angle": angle_rad, "angle_deg": angle_deg}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        side = params.get("side", 10)
        angle = params.get("angle", math.pi / 4)
        angle_deg = params.get("angle_deg", 45)

        # a·sin(θ)
        result = side * math.sin(angle)

        return MethodResult(
            value=result,
            description=f"Side × sin(angle) = {side}·sin({angle_deg}°) = {result:.4f}",
            metadata={"side": side, "angle_rad": angle, "angle_deg": angle_deg}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SineSupplementaryAngle(MethodBlock):
    """
    Apply the supplementary angle identity for sine.

    Formula: sin(π - θ) = sin(θ)
    This is useful in triangle problems where angles are supplementary.
    """

    def __init__(self):
        super().__init__()
        self.name = "sine_supplementary_angle"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "trigonometry", "sine", "supplementary"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate an angle."""
        angle_deg = random.randint(15, 165)
        angle_rad = math.radians(angle_deg)
        return {"angle": angle_rad, "angle_deg": angle_deg}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        angle = params.get("angle", math.pi / 4)
        angle_deg = params.get("angle_deg", 45)

        # sin(π - θ) = sin(θ)
        supplementary_angle = math.pi - angle
        supplementary_deg = 180 - angle_deg
        result = math.sin(supplementary_angle)
        original_sin = math.sin(angle)

        return MethodResult(
            value=result,
            description=f"sin(180°-{angle_deg}°) = sin({supplementary_deg}°) = {result:.4f} (equals sin({angle_deg}°) = {original_sin:.4f})",
            metadata={"angle": angle, "angle_deg": angle_deg, "supplementary_angle": supplementary_angle}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleBisectorPointOnSide(MethodBlock):
    """
    Find the point where an angle bisector meets the opposite side.

    By the angle bisector theorem, if the bisector from vertex A
    meets side BC at point D, then BD/DC = AB/AC.

    Returns the distance BD.
    """

    def __init__(self):
        super().__init__()
        self.name = "angle_bisector_point_on_side"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "angle_bisector", "division"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate triangle sides."""
        AB = random.randint(5, 20)
        AC = random.randint(5, 20)
        BC = random.randint(abs(AB - AC) + 1, AB + AC - 1)
        return {"AB": AB, "AC": AC, "BC": BC}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        AB = params.get("AB", 10)
        AC = params.get("AC", 12)
        BC = params.get("BC", 15)

        # BD/DC = AB/AC and BD + DC = BC
        # BD = BC·AB/(AB+AC)
        BD = BC * AB / (AB + AC)
        DC = BC * AC / (AB + AC)

        return MethodResult(
            value=BD,
            description=f"Angle bisector divides BC={BC} into BD={BD:.4f} and DC={DC:.4f}",
            metadata={"AB": AB, "AC": AC, "BC": BC, "BD": BD, "DC": DC}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TrianglePlaceCoordinates(MethodBlock):
    """
    Place a triangle in the coordinate plane given side lengths.

    Standard placement:
    - A at origin (0, 0)
    - B at (c, 0) on x-axis
    - C calculated using distances

    Returns the coordinates as a tuple.
    """

    def __init__(self):
        super().__init__()
        self.name = "triangle_place_coordinates"
        self.input_type = "triangle"
        self.output_type = "geometric"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "coordinates", "placement"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate valid triangle sides."""
        while True:
            a = random.randint(5, 20)  # BC
            b = random.randint(5, 20)  # AC
            c = random.randint(5, 20)  # AB
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)  # side BC
        b = params.get("b", 10)  # side AC
        c = params.get("c", 10)  # side AB

        # Place A at origin, B on x-axis
        A = (0, 0)
        B = (c, 0)

        # Find C using law of cosines
        # cos(A) = (b² + c² - a²) / (2bc)
        cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
        sin_A = math.sqrt(1 - cos_A**2)

        # C is at distance b from A, angle A from x-axis
        C = (b * cos_A, b * sin_A)

        coords = {"A": A, "B": B, "C": C}

        # Return sum of x-coordinates as the value
        x_sum = A[0] + B[0] + C[0]

        return MethodResult(
            value=x_sum,
            description=f"Triangle placement: A={A}, B={B}, C=({C[0]:.4f}, {C[1]:.4f}), x-sum={x_sum:.4f}",
            metadata=coords
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TriangleVerticesPlacement(MethodBlock):
    """
    Given three points (vertices), compute triangle properties.

    Returns the area of the triangle formed by the vertices.
    Area = 0.5 * |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
    """

    def __init__(self):
        super().__init__()
        self.name = "triangle_vertices_placement"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "vertices", "area"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate three non-collinear points."""
        # Simple approach: random points
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)

        # Ensure not collinear (area > 0)
        while abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) < 1:
            x3, y3 = random.randint(-10, 10), random.randint(-10, 10)

        return {
            "v1": (x1, y1),
            "v2": (x2, y2),
            "v3": (x3, y3)
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("v1", (0, 0)), "v1", 2)
        x2, y2 = ensure_tuple(params.get("v2", (1, 0)), "v2", 2)
        x3, y3 = ensure_tuple(params.get("v3", (0, 1)), "v3", 2)

        # Area using cross product formula
        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        return MethodResult(
            value=area,
            description=f"Triangle vertices ({x1},{y1}), ({x2},{y2}), ({x3},{y3}), area={area:.4f}",
            metadata={"vertices": [(x1, y1), (x2, y2), (x3, y3)]}
        )

    def can_invert(self) -> bool:
        return False


