"""
Geometry Techniques - Coordinates (Part 3)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class IntersectionTwoLines(MethodBlock):
    """
    Find intersection point of two lines.
    Returns x-coordinate of intersection (rounded to integer).
    """

    def __init__(self):
        super().__init__()
        self.name = "intersection_two_lines"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "intersection", "line"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two non-parallel lines."""
        # First line
        a1 = random.randint(-10, 10)
        b1 = random.randint(-10, 10)
        c1 = random.randint(-20, 20)

        if a1 == 0 and b1 == 0:
            a1 = 1

        # Second line (ensure not parallel)
        a2 = random.randint(-10, 10)
        b2 = random.randint(-10, 10)
        c2 = random.randint(-20, 20)

        # Ensure not parallel: a1*b2 != a2*b1
        while abs(a1 * b2 - a2 * b1) < 1e-9:
            a2 = random.randint(-10, 10)
            b2 = random.randint(-10, 10)

        return {"line1": (a1, b1, c1), "line2": (a2, b2, c2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a1, b1, c1 = params.get("line1", (1, 0, 0))
        a2, b2, c2 = params.get("line2", (0, 1, 0))

        # Use helper function
        intersection = line_intersection((a1, b1, c1), (a2, b2, c2))

        if intersection is None:
            # Parallel lines
            return MethodResult(
                value=0,
                description=f"Lines {a1}x+{b1}y+{c1}=0 and {a2}x+{b2}y+{c2}=0 are parallel",
                metadata={"line1": (a1, b1, c1), "line2": (a2, b2, c2)}
            )

        x, y = intersection

        return MethodResult(
            value=int(round(x)),
            description=f"Intersection of lines {a1}x+{b1}y+{c1}=0 and {a2}x+{b2}y+{c2}=0: ({x:.4f}, {y:.4f})",
            metadata={"intersection": (x, y), "line1": (a1, b1, c1), "line2": (a2, b2, c2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SegmentParameterize(MethodBlock):
    """
    Find point on line segment using parametric form: P = p1 + t(p2 - p1).
    t ∈ [0, 1] gives points on segment.
    Returns x-coordinate of point.
    """

    def __init__(self):
        super().__init__()
        self.name = "segment_parameterize"
        self.input_type = "geometric"
        self.output_type = "numeric"
        self.difficulty = 2
        self.tags = ["geometry", "parametric", "segment"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate segment endpoints and parameter t."""
        x1 = random.randint(-20, 20)
        y1 = random.randint(-20, 20)
        x2 = random.randint(-20, 20)
        y2 = random.randint(-20, 20)
        t = random.uniform(0, 1)

        return {"p1": (x1, y1), "p2": (x2, y2), "t": t}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (1, 1)), "p2", 2)
        t = params.get("t", 0.5)

        # P = p1 + t(p2 - p1)
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return MethodResult(
            value=px,
            description=f"Point at t={t:.4f} on segment from ({x1},{y1}) to ({x2},{y2}): ({px:.4f}, {py:.4f})",
            metadata={"p1": (x1, y1), "p2": (x2, y2), "t": t, "result": (px, py)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TrisectionPoint(MethodBlock):
    """
    Find trisection points on a segment.
    which=1 gives the 1/3 point, which=2 gives the 2/3 point.
    Returns x-coordinate of the trisection point.
    """

    def __init__(self):
        super().__init__()
        self.name = "trisection_point"
        self.input_type = "geometric"
        self.output_type = "numeric"
        self.difficulty = 1
        self.tags = ["geometry", "segment", "trisection"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate segment and which trisection point."""
        x1 = random.randint(-20, 20)
        y1 = random.randint(-20, 20)
        x2 = random.randint(-20, 20)
        y2 = random.randint(-20, 20)
        which = random.choice([1, 2])

        return {"p1": (x1, y1), "p2": (x2, y2), "which": which}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (3, 3)), "p2", 2)
        which = params.get("which", 1)

        # Trisection parameter
        t = which / 3.0

        # P = p1 + t(p2 - p1)
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return MethodResult(
            value=px,
            description=f"{which}/3 point on segment from ({x1},{y1}) to ({x2},{y2}): ({px:.4f}, {py:.4f})",
            metadata={"p1": (x1, y1), "p2": (x2, y2), "which": which, "result": (px, py)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ConstructPerpendicularLine(MethodBlock):
    """
    Construct a perpendicular line to a given line through a point.

    Given line in form ax + by + c = 0 and point (x₀, y₀),
    perpendicular line has form: bx - ay + d = 0
    where d is chosen so the line passes through (x₀, y₀).

    Returns the constant d.
    """

    def __init__(self):
        super().__init__()
        self.name = "construct_perpendicular_line"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "line", "perpendicular", "construction"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate line and point."""
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(-20, 20)
        x0 = random.randint(-10, 10)
        y0 = random.randint(-10, 10)
        return {"a": a, "b": b, "c": c, "point": (x0, y0)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 0)
        x0, y0 = ensure_tuple(params.get("point", (0, 0)), "point", 2)

        # Perpendicular line: bx - ay + d = 0
        # Passes through (x₀, y₀): b·x₀ - a·y₀ + d = 0
        # So d = a·y₀ - b·x₀
        d = a * y0 - b * x0

        return MethodResult(
            value=d,
            description=f"Perpendicular to {a}x+{b}y+{c}=0 through ({x0},{y0}): {b}x-{a}y+{d}=0",
            metadata={"original_line": (a, b, c), "point": (x0, y0), "perp_line": (b, -a, d)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VectorOB(MethodBlock):
    """
    Compute vector from point O to point B.

    Vector OB = B - O = (Bₓ - Oₓ, Bᵧ - Oᵧ)
    Returns the magnitude of the vector.
    """

    def __init__(self):
        super().__init__()
        self.name = "vector_ob"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "vector", "magnitude"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two points."""
        ox, oy = random.randint(-10, 10), random.randint(-10, 10)
        bx, by = random.randint(-10, 10), random.randint(-10, 10)
        return {"o": (ox, oy), "b": (bx, by)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        ox, oy = ensure_tuple(params.get("o", (0, 0)), "o", 2)
        bx, by = ensure_tuple(params.get("b", (5, 5)), "b", 2)

        # Vector OB
        vx = bx - ox
        vy = by - oy

        # Magnitude
        magnitude = math.sqrt(vx**2 + vy**2)

        return MethodResult(
            value=magnitude,
            description=f"Vector OB from O({ox},{oy}) to B({bx},{by}) = ({vx},{vy}), magnitude={magnitude:.4f}",
            metadata={"O": (ox, oy), "B": (bx, by), "vector": (vx, vy)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VectorBC(MethodBlock):
    """
    Compute vector from point B to point C.

    Vector BC = C - B = (Cₓ - Bₓ, Cᵧ - Bᵧ)
    Returns the magnitude of the vector.
    """

    def __init__(self):
        super().__init__()
        self.name = "vector_bc"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "vector", "magnitude"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two points."""
        bx, by = random.randint(-10, 10), random.randint(-10, 10)
        cx, cy = random.randint(-10, 10), random.randint(-10, 10)
        return {"b": (bx, by), "c": (cx, cy)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        bx, by = ensure_tuple(params.get("b", (0, 0)), "b", 2)
        cx, cy = ensure_tuple(params.get("c", (5, 5)), "c", 2)

        # Vector BC
        vx = cx - bx
        vy = cy - by

        # Magnitude
        magnitude = math.sqrt(vx**2 + vy**2)

        return MethodResult(
            value=magnitude,
            description=f"Vector BC from B({bx},{by}) to C({cx},{cy}) = ({vx},{vy}), magnitude={magnitude:.4f}",
            metadata={"B": (bx, by), "C": (cx, cy), "vector": (vx, vy)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointO(MethodBlock):
    """
    Create the origin point O at (0, 0).

    This is a basic geometric primitive for setting up coordinate systems.
    Returns 0 (the x-coordinate).
    """

    def __init__(self):
        super().__init__()
        self.name = "point_o"
        self.input_type = "integer"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "point", "origin"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """No parameters needed for origin."""
        return {}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Origin point
        point = (0, 0)

        return MethodResult(
            value=0,
            description=f"Origin point O = {point}",
            metadata={"point": point}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointB(MethodBlock):
    """
    Create a point B at given coordinates.

    Returns the x-coordinate of point B.
    """

    def __init__(self):
        super().__init__()
        self.name = "point_b"
        self.input_type = "integer"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "point"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate coordinates for point B."""
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        return {"coords": (x, y)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x, y = ensure_tuple(params.get("coords", (5, 3)), "coords", 2)

        return MethodResult(
            value=x,
            description=f"Point B = ({x}, {y})",
            metadata={"point": (x, y)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointC(MethodBlock):
    """
    Create a point C at given coordinates.

    Returns the y-coordinate of point C.
    """

    def __init__(self):
        super().__init__()
        self.name = "point_c"
        self.input_type = "integer"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "point"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate coordinates for point C."""
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        return {"coords": (x, y)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x, y = ensure_tuple(params.get("coords", (7, 4)), "coords", 2)

        return MethodResult(
            value=y,
            description=f"Point C = ({x}, {y})",
            metadata={"point": (x, y)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Collinear(MethodBlock):
    """
    Check if three 2D points are collinear (lie on the same line).

    Uses cross product: (y2-y1)*(x3-x2) == (y3-y2)*(x2-x1)

    Examples:
    - collinear((0,0), (1,1), (2,2)) -> True (1 - on same line y=x)
    - collinear((0,0), (1,1), (2,3)) -> False (0 - not collinear)

    Returns 1 if collinear, 0 otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "collinear"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["geometry", "collinearity", "points"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        # Generate 3 points, sometimes collinear, sometimes not
        if random.random() < 0.5:
            # Generate collinear points
            x1 = random.randint(-10, 10)
            y1 = random.randint(-10, 10)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                dx = 1
            t = random.randint(1, 5)
            x2 = x1 + dx
            y2 = y1 + dy
            x3 = x1 + t * dx
            y3 = y1 + t * dy
        else:
            # Generate random (likely non-collinear) points
            x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
            x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
            x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
        return {"p1": (x1, y1), "p2": (x2, y2), "p3": (x3, y3)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        p1 = params.get("p1", (0, 0))
        p2 = params.get("p2", (1, 1))
        p3 = params.get("p3", (2, 2))

        if p1 is None or p2 is None or p3 is None:
            raise ValueError(f"collinear: points cannot be None (p1={p1}, p2={p2}, p3={p3})")

        # Extract coordinates
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        x3, y3 = p3[0], p3[1]

        # Cross product method: (y2-y1)*(x3-x2) == (y3-y2)*(x2-x1)
        # Equivalent to checking if area of triangle is 0
        cross = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)

        is_collinear = abs(cross) < 1e-10
        result = 1 if is_collinear else 0

        return MethodResult(
            value=result,
            description=f"collinear({p1}, {p2}, {p3}) = {'True' if result else 'False'}",
            params=params,
            metadata={"p1": p1, "p2": p2, "p3": p3, "cross_product": cross}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PointOnLine(MethodBlock):
    """
    Check if a point lies on the line defined by two other points.

    Examples:
    - point_on_line((1.5,1.5), (0,0), (3,3)) -> True (1 - point is on line y=x)
    - point_on_line((1,2), (0,0), (3,3)) -> False (0 - point not on line)

    Returns 1 if point is on line, 0 otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "point_on_line"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["geometry", "collinearity", "points", "line"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        # Generate line endpoints
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        if dx == 0 and dy == 0:
            dx = 1
        x2, y2 = x1 + dx, y1 + dy

        # Generate test point - sometimes on line, sometimes off
        if random.random() < 0.5:
            # Point on line
            t = random.uniform(0, 1)
            px = x1 + t * dx
            py = y1 + t * dy
        else:
            # Random point (likely off line)
            px = random.randint(-10, 10)
            py = random.randint(-10, 10)

        return {"point": (px, py), "line_p1": (x1, y1), "line_p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        point = params.get("point", (0.5, 0.5))
        line_p1 = params.get("line_p1", (0, 0))
        line_p2 = params.get("line_p2", (1, 1))

        if point is None or line_p1 is None or line_p2 is None:
            raise ValueError(f"point_on_line: values cannot be None")

        # Check collinearity using cross product
        x1, y1 = line_p1[0], line_p1[1]
        x2, y2 = line_p2[0], line_p2[1]
        px, py = point[0], point[1]

        # Cross product: (y2-y1)*(px-x2) - (py-y2)*(x2-x1)
        cross = (y2 - y1) * (px - x2) - (py - y2) * (x2 - x1)

        is_on_line = abs(cross) < 1e-10
        result = 1 if is_on_line else 0

        return MethodResult(
            value=result,
            description=f"point_on_line({point}, {line_p1}, {line_p2}) = {'True' if result else 'False'}",
            params=params,
            metadata={"point": point, "line_p1": line_p1, "line_p2": line_p2, "cross_product": cross}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# POLES AND POLARS (Projective Geometry)
# ============================================================================


