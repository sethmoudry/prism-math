"""
Geometry Techniques - Coordinates (Part 2)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class CoordinateGeometryBasic(MethodBlock):
    """
    Basic coordinate geometry operations.

    Supports:
    - Distance between two points
    - Midpoint of a segment
    - Slope of a line
    - Equation of a line through two points
    - Section formula (dividing segment in ratio m:n)
    """
    def __init__(self):
        super().__init__()
        self.name = "coordinate_geometry_basic"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "coordinate", "basic"]

    def generate_parameters(self, input_value=None):
        operation = random.choice(["distance", "midpoint", "slope", "section"])
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        # For section formula
        m, n = random.randint(1, 5), random.randint(1, 5)
        return {"operation": operation, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "m": m, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "distance")
        x1 = params.get("x1", 0)
        y1 = params.get("y1", 0)
        x2 = params.get("x2", 3)
        y2 = params.get("y2", 4)

        if operation == "distance":
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            result = int(round(dist))
            description = f"Distance from ({x1},{y1}) to ({x2},{y2}): {result}"
        elif operation == "midpoint":
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            # Return sum of coordinates as integer
            result = int(round(mx + my))
            description = f"Midpoint of ({x1},{y1})-({x2},{y2}): ({mx},{my}), sum = {result}"
        elif operation == "slope":
            if x2 == x1:
                result = float('inf')
                description = f"Vertical line through ({x1},{y1}) and ({x2},{y2})"
            else:
                slope = (y2 - y1) / (x2 - x1)
                result = int(round(slope * 10))  # Scale for precision
                description = f"Slope of line through ({x1},{y1}) and ({x2},{y2}): {slope:.2f}, encoded as {result}"
        else:  # section
            m = params.get("m", 1)
            n = params.get("n", 1)
            # Point dividing (x1,y1)-(x2,y2) in ratio m:n
            px = (m * x2 + n * x1) / (m + n)
            py = (m * y2 + n * y1) / (m + n)
            result = int(round(px + py))
            description = f"Section point dividing ({x1},{y1})-({x2},{y2}) in {m}:{n}: ({px:.2f},{py:.2f}), sum = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "p1": (x1, y1), "p2": (x2, y2)}
        )

    def can_invert(self):
        return False


@register_technique
class DistancePointToLine(MethodBlock):
    """
    Compute distance from a point to a line using the formula:
    d = |ax₀ + by₀ + c| / √(a² + b²)
    where line is ax + by + c = 0 and point is (x₀, y₀).
    """

    def __init__(self):
        super().__init__()
        self.name = "distance_point_to_line"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "distance", "line", "point"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate random line and point parameters."""
        # Line equation: ax + by + c = 0
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)

        # Ensure a and b are not both zero
        if a == 0 and b == 0:
            a = 1

        # Point coordinates
        x0 = random.randint(-20, 20)
        y0 = random.randint(-20, 20)

        return {"a": a, "b": b, "c": c, "point": (x0, y0)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 0)
        x0, y0 = ensure_tuple(params.get("point", (0, 0)), "point", 2)

        # Distance formula: |ax₀ + by₀ + c| / √(a² + b²)
        numerator = abs(a * x0 + b * y0 + c)
        denominator = math.sqrt(a**2 + b**2)
        distance = numerator / denominator

        return MethodResult(
            value=distance,
            description=f"Distance from point ({x0}, {y0}) to line {a}x + {b}y + {c} = 0: {distance:.4f}",
            metadata={"line": (a, b, c), "point": (x0, y0)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LineThroughTwoPoints(MethodBlock):
    """
    Find line equation ax + by + c = 0 passing through two points.
    Returns the coefficient 'a' from the line equation.
    """

    def __init__(self):
        super().__init__()
        self.name = "line_through_two_points"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "line", "point"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two distinct points."""
        x1 = random.randint(-20, 20)
        y1 = random.randint(-20, 20)
        x2 = random.randint(-20, 20)
        y2 = random.randint(-20, 20)

        # Ensure points are distinct
        if x1 == x2 and y1 == y2:
            x2 += 1

        return {"p1": (x1, y1), "p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (1, 1)), "p2", 2)

        # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        a = y2 - y1
        b = -(x2 - x1)
        c = (x2 - x1) * y1 - (y2 - y1) * x1

        return MethodResult(
            value=int(a),
            description=f"Line through ({x1}, {y1}) and ({x2}, {y2}): {a}x + {b}y + {c} = 0",
            metadata={"a": a, "b": b, "c": c, "p1": (x1, y1), "p2": (x2, y2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VectorDotProduct(MethodBlock):
    """
    Compute dot product of two 2D vectors: v1 · v2 = v1_x * v2_x + v1_y * v2_y.
    """

    def __init__(self):
        super().__init__()
        self.name = "vector_dot_product"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "vector", "dot_product"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two random vectors."""
        v1_x = random.randint(-10, 10)
        v1_y = random.randint(-10, 10)
        v2_x = random.randint(-10, 10)
        v2_y = random.randint(-10, 10)

        return {"v1": (v1_x, v1_y), "v2": (v2_x, v2_y)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        v1_x, v1_y = ensure_tuple(params.get("v1", (1, 0)), "v1", 2)
        v2_x, v2_y = ensure_tuple(params.get("v2", (0, 1)), "v2", 2)

        dot_product = v1_x * v2_x + v1_y * v2_y

        return MethodResult(
            value=int(dot_product),
            description=f"Dot product of ({v1_x}, {v1_y}) · ({v2_x}, {v2_y}) = {dot_product}",
            metadata={"v1": (v1_x, v1_y), "v2": (v2_x, v2_y)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleFromDotProduct(MethodBlock):
    """
    Compute angle between two vectors using dot product:
    θ = arccos((v1·v2) / (|v1||v2|))
    Returns angle in degrees.
    """

    def __init__(self):
        super().__init__()
        self.name = "angle_from_dot_product"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "vector", "angle", "dot_product"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two non-zero vectors."""
        v1_x = random.randint(-10, 10)
        v1_y = random.randint(-10, 10)
        if v1_x == 0 and v1_y == 0:
            v1_x = 1

        v2_x = random.randint(-10, 10)
        v2_y = random.randint(-10, 10)
        if v2_x == 0 and v2_y == 0:
            v2_y = 1

        return {"v1": (v1_x, v1_y), "v2": (v2_x, v2_y)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        v1_x, v1_y = ensure_tuple(params.get("v1", (1, 0)), "v1", 2)
        v2_x, v2_y = ensure_tuple(params.get("v2", (0, 1)), "v2", 2)

        # Compute dot product and magnitudes
        dot_product = v1_x * v2_x + v1_y * v2_y
        mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
        mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

        # Avoid division by zero
        if mag_v1 < 1e-9 or mag_v2 < 1e-9:
            angle_deg = 0
            angle_rad = 0
        else:
            cos_theta = dot_product / (mag_v1 * mag_v2)
            # Clamp to [-1, 1] to avoid numerical errors
            cos_theta = max(-1, min(1, cos_theta))
            angle_rad = math.acos(cos_theta)
            angle_deg = math.degrees(angle_rad)

        return MethodResult(
            value=int(angle_deg),
            description=f"Angle between ({v1_x},{v1_y}) and ({v2_x},{v2_y}): {angle_deg:.2f}°",
            metadata={"v1": (v1_x, v1_y), "v2": (v2_x, v2_y), "angle_radians": angle_rad}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class DistanceBetweenPoints(MethodBlock):
    """
    Compute Euclidean distance between two points:
    d = √((x2-x1)² + (y2-y1)²)
    """

    def __init__(self):
        super().__init__()
        self.name = "distance_between_points"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "distance", "point"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two random points."""
        x1 = random.randint(-20, 20)
        y1 = random.randint(-20, 20)
        x2 = random.randint(-20, 20)
        y2 = random.randint(-20, 20)

        return {"p1": (x1, y1), "p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (1, 1)), "p2", 2)

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        return MethodResult(
            value=distance,
            description=f"Distance between ({x1}, {y1}) and ({x2}, {y2}): {distance:.4f}",
            metadata={"p1": (x1, y1), "p2": (x2, y2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointTranslation(MethodBlock):
    """
    Translate a point by a vector: p' = p + v.
    Returns the x-coordinate of the translated point.
    """

    def __init__(self):
        super().__init__()
        self.name = "point_translation"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "transformation", "translation"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate point and translation vector."""
        px = random.randint(-20, 20)
        py = random.randint(-20, 20)
        vx = random.randint(-10, 10)
        vy = random.randint(-10, 10)

        return {"point": (px, py), "vector": (vx, vy)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        px, py = ensure_tuple(params.get("point", (0, 0)), "point", 2)
        vx, vy = ensure_tuple(params.get("vector", (1, 1)), "vector", 2)

        new_x = px + vx
        new_y = py + vy

        return MethodResult(
            value=int(new_x),
            description=f"Translate ({px}, {py}) by ({vx}, {vy}) = ({new_x}, {new_y})",
            metadata={"original": (px, py), "vector": (vx, vy), "result": (new_x, new_y)}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Reverse translation."""
        new_x = output_value
        vx, vy = ensure_tuple(params.get("vector", (1, 1)), "vector", 2)
        px = new_x - vx

        return MethodResult(
            value=int(px),
            description=f"Original x-coordinate before translation: {px}",
            metadata={"vector": (vx, vy)}
        )


@register_technique
class SlopeToTrig(MethodBlock):
    """
    Convert slope to angle using arctan: θ = arctan(m).
    Returns angle in degrees.
    """

    def __init__(self):
        super().__init__()
        self.name = "slope_to_trig"
        self.input_type = "numeric"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "trigonometry", "slope", "angle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a random slope."""
        # Use reasonable slope values
        slope = random.uniform(-5, 5)
        return {"slope": slope}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        slope = params.get("slope", 1)

        angle_rad = math.atan(slope)
        angle_deg = math.degrees(angle_rad)

        return MethodResult(
            value=int(angle_deg),
            description=f"Angle from slope {slope:.4f}: arctan({slope:.4f}) = {angle_deg:.2f}°",
            metadata={"slope": slope, "angle_radians": angle_rad}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Convert angle back to slope."""
        angle_deg = output_value
        angle_rad = math.radians(angle_deg)
        slope = math.tan(angle_rad)

        return MethodResult(
            value=slope,
            description=f"Slope from angle {angle_deg}°: tan({angle_deg}°) = {slope:.4f}",
            metadata={"angle_degrees": angle_deg}
        )


@register_technique
class PerpendicularToLine(MethodBlock):
    """
    Find perpendicular line to ax + by + c = 0.
    Perpendicular line has form bx - ay + k = 0.
    Returns coefficient 'b' from original line.
    """

    def __init__(self):
        super().__init__()
        self.name = "perpendicular_to_line"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "line", "perpendicular"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate line coefficients."""
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)

        if a == 0 and b == 0:
            a = 1

        return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 0)

        # Perpendicular line: bx - ay + k = 0 (for some k)
        perp_a = b
        perp_b = -a

        return MethodResult(
            value=int(b),
            description=f"Perpendicular to {a}x + {b}y + {c} = 0 has form {perp_a}x + {perp_b}y + k = 0",
            metadata={"original": (a, b, c), "perpendicular_form": (perp_a, perp_b)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PerpendicularLineEquation(MethodBlock):
    """
    Find line perpendicular to ax + by + c = 0 passing through point (x0, y0).
    Returns the constant term of the perpendicular line equation.
    """

    def __init__(self):
        super().__init__()
        self.name = "perpendicular_line_equation"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "line", "perpendicular"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate line and point."""
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        c = random.randint(-20, 20)

        if a == 0 and b == 0:
            a = 1

        x0 = random.randint(-10, 10)
        y0 = random.randint(-10, 10)

        return {"a": a, "b": b, "c": c, "point": (x0, y0)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 0)
        x0, y0 = ensure_tuple(params.get("point", (0, 0)), "point", 2)

        # Perpendicular line: bx - ay + k = 0
        # Passing through (x0, y0): k = ay0 - bx0
        perp_a = b
        perp_b = -a
        perp_c = a * y0 - b * x0

        return MethodResult(
            value=int(perp_c),
            description=f"Perpendicular to {a}x + {b}y + {c} = 0 through ({x0},{y0}): {perp_a}x + {perp_b}y + {perp_c} = 0",
            metadata={"original": (a, b, c), "point": (x0, y0), "perpendicular": (perp_a, perp_b, perp_c)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ConstructLineThroughPoints(MethodBlock):
    """
    Construct line through two points and return slope.
    slope = (y2 - y1) / (x2 - x1)
    """

    def __init__(self):
        super().__init__()
        self.name = "construct_line_through_points"
        self.input_type = "geometric"
        self.output_type = "numeric"
        self.difficulty = 1
        self.tags = ["geometry", "line", "slope"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two points with different x-coordinates."""
        x1 = random.randint(-20, 20)
        y1 = random.randint(-20, 20)
        x2 = random.randint(-20, 20)
        y2 = random.randint(-20, 20)

        # Ensure different x-coordinates (non-vertical line)
        while x1 == x2:
            x2 = random.randint(-20, 20)

        return {"p1": (x1, y1), "p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (1, 1)), "p2", 2)

        # Calculate slope
        if abs(x2 - x1) < 1e-9:
            slope = float('inf')
            description = f"Vertical line through ({x1}, {y1}) and ({x2}, {y2})"
        else:
            slope = (y2 - y1) / (x2 - x1)
            description = f"Slope of line through ({x1}, {y1}) and ({x2}, {y2}): {slope:.4f}"

        return MethodResult(
            value=slope,
            description=description,
            metadata={"p1": (x1, y1), "p2": (x2, y2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LineEquationFromSlopePoint(MethodBlock):
    """
    Find line equation using point-slope form: y - y1 = m(x - x1).
    Rearrange to standard form: mx - y + (y1 - mx1) = 0.
    Returns constant term (y1 - mx1).
    """

    def __init__(self):
        super().__init__()
        self.name = "line_equation_from_slope_point"
        self.input_type = "geometric"
        self.output_type = "numeric"
        self.difficulty = 2
        self.tags = ["geometry", "line", "slope", "point-slope"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate slope and point."""
        m = random.uniform(-5, 5)
        x1 = random.randint(-10, 10)
        y1 = random.randint(-10, 10)

        return {"slope": m, "point": (x1, y1)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        m = params.get("slope", 1)
        x1, y1 = ensure_tuple(params.get("point", (0, 0)), "point", 2)

        # Standard form: mx - y + c = 0, where c = y1 - mx1
        c = y1 - m * x1

        return MethodResult(
            value=c,
            description=f"Line with slope {m:.4f} through ({x1}, {y1}): {m:.4f}x - y + {c:.4f} = 0",
            metadata={"slope": m, "point": (x1, y1), "constant": c}
        )

    def can_invert(self) -> bool:
        return False


