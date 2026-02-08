"""
Geometry Techniques - Polygons (Part 2)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class ParallelogramProperties(MethodBlock):
    """
    Compute properties of parallelograms including area, diagonal lengths, and
    perimeter. For a parallelogram with sides a, b and angle θ between them:
    - Area: A = a * b * sin(θ)
    - Diagonals: d₁² = a² + b² + 2ab*cos(θ), d₂² = a² + b² - 2ab*cos(θ)
    - Perimeter: P = 2(a + b)
    The diagonals bisect each other at their midpoint.
    """
    def __init__(self):
        super().__init__()
        self.name = "parallelogram_properties"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "parallelogram", "quadrilateral", "properties"]

    def validate_params(self, params, prev_value=None):
        """Validate parallelogram parameters."""
        a = params.get("a")
        b = params.get("b")
        angle = params.get("angle")
        if a is None or b is None or angle is None:
            return False
        return a > 0 and b > 0 and 0 < angle < 180

    def generate_parameters(self, input_value=None):
        """Generate valid parallelogram parameters."""
        a = input_value if input_value is not None else random.randint(5, 20)
        b = random.randint(5, 20)
        angle = random.uniform(30, 150)  # Angle in degrees
        return {"a": a, "b": b, "angle": angle}

    def compute(self, input_value, params):
        a = params.get("a", 8)
        b = params.get("b", 12)
        angle_deg = params.get("angle", 60)

        # Convert angle to radians
        angle_rad = math.radians(angle_deg)

        # Compute area using A = a * b * sin(θ)
        area = a * b * math.sin(angle_rad)

        # Compute diagonals using law of cosines
        # d₁² = a² + b² + 2ab*cos(θ)
        d1_squared = a**2 + b**2 + 2 * a * b * math.cos(angle_rad)
        d1 = math.sqrt(d1_squared)

        # d₂² = a² + b² - 2ab*cos(θ)
        d2_squared = a**2 + b**2 - 2 * a * b * math.cos(angle_rad)
        d2 = math.sqrt(d2_squared)

        # Compute perimeter
        perimeter = 2 * (a + b)

        # Return area + diagonal ratio as integer
        result = round(area + d1 / 2)

        return MethodResult(
            value=result,
            description=f"Parallelogram ({a}×{b}, angle={angle_deg:.1f}°): Area={area:.2f}, Diagonals d₁={d1:.2f}, d₂={d2:.2f}, Perimeter={perimeter}",
            metadata={
                "a": a,
                "b": b,
                "angle": angle_deg,
                "area": area,
                "d1": d1,
                "d2": d2,
                "perimeter": perimeter
            }
        )

    def can_invert(self):
        return False


@register_technique
class ConvexHullArea(MethodBlock):
    """
    Compute the area of the convex hull of a set of points.
    Uses ConvexHull algorithm followed by shoelace formula.
    """

    def __init__(self):
        super().__init__()
        self.name = "convex_hull_area"
        self.input_type = "point_set"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "convex_hull", "area", "coordinate"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate random set of points."""
        n = random.randint(5, 12)  # At least 5 points for interesting hull
        points = []
        for _ in range(n):
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            points.append((x, y))
        return {"points": points}

    def validate_params(self, params, prev_value=None):
        """Validate convex hull area parameters: need at least 3 points."""
        points = params.get("points")
        if points is None or not isinstance(points, (list, tuple)):
            return False
        return len(points) >= 3  # Need at least 3 points for area

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        points = params.get("points", 10)

        if len(points) < 3:
            return MethodResult(
                value=0,
                description="Convex hull area: need at least 3 points",
                metadata={"points": points}
            )

        try:
            from scipy.spatial import ConvexHull

            # Convert to numpy array for ConvexHull
            points_array = np.array(points)
            hull = ConvexHull(points_array)

            # Get hull vertices in order
            hull_vertices = [tuple(points_array[i]) for i in hull.vertices]

            # Compute area using shoelace formula
            area = self._shoelace_area(hull_vertices)

            return MethodResult(
                value=area,
                description=f"Convex hull area of {len(points)} points = {area:.4f}",
                metadata={
                    "num_points": len(points),
                    "hull_vertices": len(hull_vertices),
                    "original_points": points
                }
            )
        except ImportError:
            # Fallback: use all points as if they form a convex polygon
            area = self._shoelace_area(points)
            return MethodResult(
                value=area,
                description=f"Polygon area (approx) = {area:.4f}",
                metadata={"points": points, "fallback": True}
            )

    def _shoelace_area(self, vertices: List[Tuple[float, float]]) -> float:
        """Compute polygon area using shoelace formula."""
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2.0

    def can_invert(self) -> bool:
        return False


@register_technique
class PolygonArea(MethodBlock):
    """
    Compute area of any simple polygon using shoelace formula.
    Vertices should be provided in order (clockwise or counterclockwise).
    A = (1/2)|Σ(x_i·y_{i+1} - x_{i+1}·y_i)|
    """

    def __init__(self):
        super().__init__()
        self.name = "polygon_area"
        self.input_type = "polygon"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "polygon", "area", "coordinate"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate vertices of a simple polygon."""
        n = random.randint(3, 8)
        vertices = []
        for _ in range(n):
            x = random.randint(-25, 25)
            y = random.randint(-25, 25)
            vertices.append((x, y))
        return {"vertices": vertices}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        vertices = params.get("vertices", 10)

        if len(vertices) < 3:
            return MethodResult(
                value=0,
                description="Need at least 3 vertices for polygon area",
                metadata={"vertices": vertices}
            )

        # Shoelace formula
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0

        return MethodResult(
            value=area,
            description=f"Polygon area ({n} vertices) = {area:.4f}",
            metadata={"vertices": vertices, "num_vertices": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class QuadrilateralArea(MethodBlock):
    """
    Compute area of a quadrilateral given 4 vertices.
    Splits into two triangles and sums their areas.
    Can accept vertices as 4 separate points or as a list.
    """

    def __init__(self):
        super().__init__()
        self.name = "quadrilateral_area"
        self.input_type = "polygon"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "quadrilateral", "area", "coordinate"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate 4 vertices of a quadrilateral."""
        vertices = []
        for _ in range(4):
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            vertices.append((x, y))

        return {
            "vertices": vertices,
            "p1": vertices[0],
            "p2": vertices[1],
            "p3": vertices[2],
            "p4": vertices[3]
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Accept vertices as list or as individual p1, p2, p3, p4
        if "vertices" in params and len(params.get("vertices", 10)) >= 4:
            vertices = params.get("vertices", 10)[:4]
        else:
            vertices = [
                params.get("p1", (0, 0)),
                params.get("p2", (1, 0)),
                params.get("p3", (1, 1)),
                params.get("p4", (0, 1))
            ]

        # Split quadrilateral into two triangles: (p1, p2, p3) and (p1, p3, p4)
        p1, p2, p3, p4 = vertices

        # Triangle area using cross product: 0.5 * |AB × AC|
        area1 = self._triangle_area(p1, p2, p3)
        area2 = self._triangle_area(p1, p3, p4)

        total_area = area1 + area2

        return MethodResult(
            value=total_area,
            description=f"Quadrilateral area = {total_area:.4f}",
            metadata={
                "vertices": vertices,
                "triangle1_area": area1,
                "triangle2_area": area2
            }
        )

    def _triangle_area(self, p1: Tuple[float, float], p2: Tuple[float, float],
                       p3: Tuple[float, float]) -> float:
        """Compute triangle area using cross product formula."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        return area

    def can_invert(self) -> bool:
        return False


@register_technique
class TrapezoidHeight(MethodBlock):
    """
    Calculate the height of a trapezoid given its bases and area.

    Formula: h = 2A / (b₁ + b₂)
    where A is area, b₁ and b₂ are parallel bases.
    """

    def __init__(self):
        super().__init__()
        self.name = "trapezoid_height"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "trapezoid", "height"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate trapezoid bases and area."""
        b1 = random.randint(5, 20)
        b2 = random.randint(5, 20)
        h = random.randint(3, 15)
        area = (b1 + b2) * h / 2
        return {"b1": b1, "b2": b2, "area": area, "expected_height": h}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        b1 = params.get("b1", 10)
        b2 = params.get("b2", 10)
        area = params.get("area", 50)

        # h = 2A / (b₁ + b₂)
        height = (2 * area) / (b1 + b2)

        return MethodResult(
            value=height,
            description=f"Trapezoid height h = 2·{area:.2f}/({b1}+{b2}) = {height:.2f}",
            metadata={"b1": b1, "b2": b2, "area": area}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find area given height and bases."""
        b1 = params.get("b1", 10)
        b2 = params.get("b2", 10)
        h = output_value

        area = (b1 + b2) * h / 2

        return MethodResult(
            value=area,
            description=f"Trapezoid area A = ({b1}+{b2})·{h:.2f}/2 = {area:.2f}",
            metadata={"b1": b1, "b2": b2, "height": h}
        )


@register_technique
class TrapezoidMidline(MethodBlock):
    """
    Calculate the midline (midsegment) of a trapezoid.

    Formula: m = (b₁ + b₂) / 2
    The midline is parallel to the bases and its length is the average of the two bases.
    """

    def __init__(self):
        super().__init__()
        self.name = "trapezoid_midline"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "trapezoid", "midline"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate trapezoid bases."""
        b1 = random.randint(5, 30)
        b2 = random.randint(5, 30)
        return {"b1": b1, "b2": b2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        b1 = params.get("b1", 10)
        b2 = params.get("b2", 15)

        # m = (b₁ + b₂) / 2
        midline = (b1 + b2) / 2

        return MethodResult(
            value=midline,
            description=f"Trapezoid midline m = ({b1}+{b2})/2 = {midline:.2f}",
            metadata={"b1": b1, "b2": b2}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TrapezoidArea(MethodBlock):
    """
    Calculate the area of a trapezoid.

    Formula: A = (b₁ + b₂) · h / 2
    where b₁ and b₂ are parallel bases and h is height.
    """

    def __init__(self):
        super().__init__()
        self.name = "trapezoid_area"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "trapezoid", "area"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate trapezoid dimensions."""
        b1 = random.randint(5, 25)
        b2 = random.randint(5, 25)
        h = random.randint(3, 20)
        return {"b1": b1, "b2": b2, "h": h}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        b1 = params.get("b1", 10)
        b2 = params.get("b2", 15)
        h = params.get("h", 8)

        # A = (b₁ + b₂) · h / 2
        area = (b1 + b2) * h / 2

        return MethodResult(
            value=area,
            description=f"Trapezoid area A = ({b1}+{b2})·{h}/2 = {area:.2f}",
            metadata={"b1": b1, "b2": b2, "h": h}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find height given area and bases."""
        b1 = params.get("b1", 10)
        b2 = params.get("b2", 15)
        area = output_value

        h = (2 * area) / (b1 + b2)

        return MethodResult(
            value=h,
            description=f"Trapezoid height h = 2·{area:.2f}/({b1}+{b2}) = {h:.2f}",
            metadata={"b1": b1, "b2": b2, "area": area}
        )


@register_technique
class RectangleCoordinates(MethodBlock):
    """
    Generate coordinates of rectangle vertices.

    Given width, height, and center, returns the four corner coordinates.
    Returns sum of all x-coordinates.
    """

    def __init__(self):
        super().__init__()
        self.name = "rectangle_coordinates"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "rectangle", "coordinates"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate rectangle dimensions and center."""
        width = random.randint(5, 30)
        height = random.randint(5, 30)
        cx = random.randint(-10, 10)
        cy = random.randint(-10, 10)
        return {"width": width, "height": height, "center": (cx, cy)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        width = params.get("width", 10)
        height = params.get("height", 8)
        cx, cy = ensure_tuple(params.get("center", (0, 0)), "center", 2)

        # Four corners
        half_w = width / 2
        half_h = height / 2
        vertices = [
            (cx - half_w, cy - half_h),  # bottom-left
            (cx + half_w, cy - half_h),  # bottom-right
            (cx + half_w, cy + half_h),  # top-right
            (cx - half_w, cy + half_h)   # top-left
        ]

        # Sum of x-coordinates
        x_sum = sum(v[0] for v in vertices)

        return MethodResult(
            value=x_sum,
            description=f"Rectangle {width}×{height} at center {(cx, cy)}, vertices: {vertices}, x-sum={x_sum}",
            metadata={"vertices": vertices, "width": width, "height": height, "center": (cx, cy)}
        )

    def can_invert(self) -> bool:
        return False


