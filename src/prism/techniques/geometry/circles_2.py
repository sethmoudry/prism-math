"""
Geometry Techniques - Circles (Part 2)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class NinePointProperties(MethodBlock):
    """
    Properties of the nine-point circle in triangles.

    The nine-point circle passes through:
    1. The three midpoints of the sides
    2. The three feet of the altitudes
    3. The three midpoints of segments from vertices to orthocenter

    Key properties:
    - Radius of nine-point circle = R/2 (half the circumradius)
    - Center lies on Euler line, midway between circumcenter and orthocenter
    """
    def __init__(self):
        super().__init__()
        self.name = "nine_point_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["geometry", "triangle", "nine_point", "circle"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for sides a, b, c."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, input_value=None):
        # Generate triangle sides (Pythagorean triples for nicer calculations)
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        a, b, c = random.choice(triples)
        scale = random.randint(1, 3)
        operation = random.choice(["radius", "diameter", "center_distance"])
        return {"a": a * scale, "b": b * scale, "c": c * scale, "operation": operation}

    def compute(self, input_value, params):
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)
        operation = params.get("operation", "radius")

        # Calculate circumradius R = abc / (4 * Area)
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return MethodResult(
                value=1,
                description="Invalid triangle",
                metadata={"error": "degenerate"}
            )

        area = math.sqrt(area_sq)
        R = (a * b * c) / (4 * area)

        if operation == "radius":
            # Nine-point circle radius = R/2
            nine_radius = R / 2
            result = int(round(nine_radius * 10))  # Scale for precision
            description = f"Nine-point circle radius for triangle ({a},{b},{c}): R/2 = {nine_radius:.3f}, encoded as {result}"
        elif operation == "diameter":
            # Nine-point circle diameter = R
            diameter = R
            result = int(round(diameter * 10))
            description = f"Nine-point circle diameter = {diameter:.3f}, encoded as {result}"
        else:  # center_distance
            # Distance from circumcenter to nine-point center = R/2
            dist = R / 2
            result = int(round(dist * 10))
            description = f"Distance from circumcenter to nine-point center = {dist:.3f}, encoded as {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"a": a, "b": b, "c": c, "circumradius": R, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class CircumcircleAngleBisector(MethodBlock):
    """
    Properties of angle bisectors meeting the circumcircle.

    The angle bisector from vertex A meets the circumcircle at a point M.
    Key properties:
    - M is the midpoint of arc BC (not containing A)
    - BM = CM (M is equidistant from B and C)
    - The angle bisector length from A to BC is: t_a = 2bc*cos(A/2)/(b+c)

    Arc midpoint theorem: The angle bisector from A passes through the
    midpoint of the arc BC not containing A.
    """
    def __init__(self):
        super().__init__()
        self.name = "circumcircle_angle_bisector"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "circumcircle", "angle_bisector"]

    def generate_parameters(self, input_value=None):
        # Triangle sides
        a = random.randint(5, 15)
        b = random.randint(5, 15)
        c = random.randint(max(abs(a - b) + 1, 5), min(a + b - 1, 15))
        operation = random.choice(["bisector_length", "arc_length", "chord_bm"])
        return {"a": a, "b": b, "c": c, "operation": operation}

    def compute(self, input_value, params):
        a = params.get("a", 7)
        b = params.get("b", 8)
        c = params.get("c", 9)
        operation = params.get("operation", "bisector_length")

        # Calculate area and circumradius
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return MethodResult(
                value=1,
                description="Invalid triangle",
                metadata={"error": "degenerate"}
            )

        area = math.sqrt(area_sq)
        R = (a * b * c) / (4 * area)

        # cos(A) = (b^2 + c^2 - a^2) / (2bc)
        cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_A = max(-1, min(1, cos_A))  # Clamp for numerical stability
        cos_half_A = math.sqrt((1 + cos_A) / 2)

        if operation == "bisector_length":
            # Angle bisector from A to BC: t_a = 2bc*cos(A/2)/(b+c)
            t_a = (2 * b * c * cos_half_A) / (b + c)
            result = int(round(t_a * 10))
            description = f"Angle bisector from A: t_a = {t_a:.3f}, encoded as {result}"
        elif operation == "arc_length":
            # Arc BC (not containing A) = pi*R*(180 - A)/180
            A_angle = math.acos(cos_A)
            arc_angle = math.pi - A_angle  # Angle subtended by arc BC
            arc_length = R * arc_angle
            result = int(round(arc_length * 10))
            description = f"Arc BC (not containing A): {arc_length:.3f}, encoded as {result}"
        else:  # chord_bm
            # Distance from B to arc midpoint M
            # Since M is on circumcircle and BM = CM, this equals the chord
            # Using inscribed angle: angle BAM = angle CAM
            A_angle = math.acos(cos_A)
            # BM = 2R * sin(A/2)
            bm = 2 * R * math.sin(A_angle / 2)
            result = int(round(bm * 10))
            description = f"Chord BM (to arc midpoint): {bm:.3f}, encoded as {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"a": a, "b": b, "c": c, "circumradius": R, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class CircumcircleProperties(MethodBlock):
    """
    Properties of the circumcircle of a triangle.

    Key formulas:
    - Circumradius R = abc / (4 * Area)
    - Circumcircle area = pi * R^2
    - Circumcircle circumference = 2 * pi * R
    - For right triangle: R = hypotenuse / 2
    - Extended law of sines: a/sin(A) = b/sin(B) = c/sin(C) = 2R
    """
    def __init__(self):
        super().__init__()
        self.name = "circumcircle_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "circumcircle", "circle"]

    def generate_parameters(self, input_value=None):
        # Use Pythagorean triples for nice calculations
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        a, b, c = random.choice(triples)
        scale = random.randint(1, 3)
        operation = random.choice(["radius", "diameter", "area", "circumference"])
        return {"a": a * scale, "b": b * scale, "c": c * scale, "operation": operation}

    def compute(self, input_value, params):
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)
        operation = params.get("operation", "radius")

        # Calculate area using Heron's formula
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return MethodResult(
                value=1,
                description="Invalid triangle",
                metadata={"error": "degenerate"}
            )

        area = math.sqrt(area_sq)

        # Circumradius R = abc / (4 * Area)
        R = (a * b * c) / (4 * area)

        if operation == "radius":
            result = int(round(R * 10))  # Scale for precision
            description = f"Circumradius R = abc/(4*Area) = {R:.3f}, encoded as {result}"
        elif operation == "diameter":
            diameter = 2 * R
            result = int(round(diameter * 10))
            description = f"Circumcircle diameter = 2R = {diameter:.3f}, encoded as {result}"
        elif operation == "area":
            circle_area = math.pi * R ** 2
            result = int(round(circle_area))
            description = f"Circumcircle area = pi*R^2 = {circle_area:.2f}, rounded to {result}"
        else:  # circumference
            circumference = 2 * math.pi * R
            result = int(round(circumference))
            description = f"Circumcircle circumference = 2*pi*R = {circumference:.2f}, rounded to {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"a": a, "b": b, "c": c, "circumradius": R, "triangle_area": area, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class TangentCircleProperties(MethodBlock):
    """Properties of tangent circles."""
    def __init__(self):
        super().__init__()
        self.name = "tangent_circle_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "circle"]

    def generate_parameters(self, input_value=None):
        r1 = random.randint(2, 8)
        r2 = random.randint(2, 8)
        return {"r1": r1, "r2": r2}

    def compute(self, input_value, params):
        r1 = params.get("r1", 3)
        r2 = params.get("r2", 5)
        dist_external = r1 + r2
        return MethodResult(
            value=dist_external,
            description=f"Distance between tangent circles: {r1} + {r2} = {dist_external}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class IntersectionLineCircle(MethodBlock):
    """
    Find intersection points of a line and a circle.
    Returns number of intersection points (0, 1, or 2).
    """

    def __init__(self):
        super().__init__()
        self.name = "intersection_line_circle"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "intersection", "line", "circle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate line and circle that may intersect."""
        # Circle: center and radius
        cx = random.randint(-10, 10)
        cy = random.randint(-10, 10)
        r = random.randint(3, 15)

        # Line through or near the circle
        # Line: ax + by + c = 0
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        if a == 0 and b == 0:
            a = 1

        # Choose c to create various intersection scenarios
        c = random.randint(-20, 20)

        return {"center": (cx, cy), "radius": r, "a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", (0, 0)), "center", 2)
        r = params.get("radius", 5)
        a = params.get("a", 1)
        b = params.get("b", 0)
        c = params.get("c", 0)

        # Distance from center to line
        distance = abs(a * cx + b * cy + c) / math.sqrt(a**2 + b**2)

        # Determine number of intersections
        if distance > r:
            num_intersections = 0
        elif abs(distance - r) < 1e-9:
            num_intersections = 1
        else:
            num_intersections = 2

        return MethodResult(
            value=num_intersections,
            description=f"Line {a}x + {b}y + {c} = 0 intersects circle (center ({cx}, {cy}), r={r}) at {num_intersections} point(s)",
            metadata={"distance_to_center": distance, "radius": r, "center": (cx, cy)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CircleThroughThreePoints(MethodBlock):
    """
    Find circle passing through three non-collinear points.
    Returns the radius of the circle.
    """

    def __init__(self):
        super().__init__()
        self.name = "circle_through_three_points"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "circle", "circumcircle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate three non-collinear points."""
        # Generate a valid triangle
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)

        # Ensure non-collinear by checking cross product
        while abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) < 1e-9:
            x3 = random.randint(-10, 10)
            y3 = random.randint(-10, 10)

        return {"p1": (x1, y1), "p2": (x2, y2), "p3": (x3, y3)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", (0, 0)), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", (1, 0)), "p2", 2)
        x3, y3 = ensure_tuple(params.get("p3", (0, 1)), "p3", 2)

        # Calculate circumcircle using determinants
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        if abs(d) < 1e-9:
            # Points are collinear
            radius = 0
            cx, cy = 0, 0
        else:
            ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
            uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d

            cx, cy = ux, uy
            radius = math.sqrt((cx - x1)**2 + (cy - y1)**2)

        return MethodResult(
            value=radius,
            description=f"Circle through ({x1},{y1}), ({x2},{y2}), ({x3},{y3}): center=({cx:.2f},{cy:.2f}), r={radius:.4f}",
            metadata={"center": (cx, cy), "radius": radius, "points": [(x1, y1), (x2, y2), (x3, y3)]}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CircumferenceToRadius(MethodBlock):
    """
    Convert circumference to radius: r = C / (2π).
    """

    def __init__(self):
        super().__init__()
        self.name = "circumference_to_radius"
        self.input_type = "numeric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "circle", "circumference", "radius"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate random circumference."""
        circumference = random.uniform(10, 100)
        return {"circumference": circumference}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        C = params.get("circumference", 2 * math.pi)

        radius = C / (2 * math.pi)

        return MethodResult(
            value=radius,
            description=f"Radius from circumference {C:.4f}: r = {C:.4f}/(2π) = {radius:.4f}",
            metadata={"circumference": C}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Convert radius back to circumference."""
        r = output_value
        C = 2 * math.pi * r

        return MethodResult(
            value=C,
            description=f"Circumference from radius {r:.4f}: C = 2π·{r:.4f} = {C:.4f}",
            metadata={"radius": r}
        )


@register_technique
class CircleRadius(MethodBlock):
    """
    Extract radius from a circle representation.
    Circle is represented as (center_x, center_y, radius).
    Returns the radius.
    """

    def __init__(self):
        super().__init__()
        self.name = "circle_radius"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "circle", "radius"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate circle parameters."""
        cx = random.randint(-10, 10)
        cy = random.randint(-10, 10)
        r = random.uniform(1, 20)

        return {"center": (cx, cy), "radius": r}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", (0, 0)), "center", 2)
        r = params.get("radius", 5)

        return MethodResult(
            value=r,
            description=f"Circle with center ({cx}, {cy}) and radius {r:.4f}",
            metadata={"center": (cx, cy), "radius": r}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# ADVANCED GEOMETRY METHODS (22)
# ============================================================================


@register_technique
class CircumradiusLawOfSines(MethodBlock):
    """
    Calculate circumradius using the law of sines.

    Formula: R = a / (2·sin(A))
    where R is circumradius, a is a side, and A is the opposite angle.
    """

    def __init__(self):
        super().__init__()
        self.name = "circumradius_law_of_sines"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "circumradius", "law_of_sines"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate side and opposite angle."""
        a = random.randint(5, 30)
        # Angle A in degrees (avoid very small or very large to keep R reasonable)
        A_deg = random.randint(30, 150)
        A_rad = math.radians(A_deg)
        return {"a": a, "A": A_rad, "A_deg": A_deg}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        A = params.get("A", math.pi / 3)
        A_deg = params.get("A_deg", 60)

        # R = a / (2·sin(A))
        circumradius = a / (2 * math.sin(A))

        return MethodResult(
            value=circumradius,
            description=f"Circumradius R = {a}/(2·sin({A_deg}°)) = {circumradius:.4f}",
            metadata={"side": a, "angle_rad": A, "angle_deg": A_deg}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find side length given circumradius and angle."""
        R = output_value
        A = params.get("A", math.pi / 3)
        A_deg = params.get("A_deg", 60)

        a = 2 * R * math.sin(A)

        return MethodResult(
            value=a,
            description=f"Side a = 2·{R:.4f}·sin({A_deg}°) = {a:.4f}",
            metadata={"circumradius": R, "angle_rad": A, "angle_deg": A_deg}
        )


@register_technique
class CircumradiusEqualityCondition(MethodBlock):
    """
    Check if two circumradii are equal.

    This is useful in problems involving cyclic quadrilaterals or
    multiple triangles sharing the same circumcircle.

    Returns 1 if R₁ = R₂ (within tolerance), 0 otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "circumradius_equality_condition"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "circumradius", "equality", "condition"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two circumradii."""
        R1 = random.uniform(5, 20)
        # 50% chance they're equal
        if random.random() < 0.5:
            R2 = R1
        else:
            R2 = random.uniform(5, 20)
        return {"R1": R1, "R2": R2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        R1 = params.get("R1", 10)
        R2 = params.get("R2", 10)

        # Check equality within tolerance
        tolerance = 1e-6
        are_equal = abs(R1 - R2) < tolerance
        result = 1 if are_equal else 0

        return MethodResult(
            value=result,
            description=f"Circumradius equality: R₁={R1:.4f}, R₂={R2:.4f}, equal={are_equal}",
            metadata={"R1": R1, "R2": R2, "difference": abs(R1 - R2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TangentLengthConcentricCircles(MethodBlock):
    """
    Calculate tangent length between two concentric circles.

    For two concentric circles with radii r₁ and r₂ (r₁ < r₂),
    the length of the tangent from a point on the outer circle
    to the inner circle is √(r₂² - r₁²).
    """

    def __init__(self):
        super().__init__()
        self.name = "tangent_length_concentric_circles"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "circle", "tangent", "concentric"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two radii."""
        r1 = random.randint(3, 15)
        r2 = random.randint(r1 + 5, r1 + 20)
        return {"r1": r1, "r2": r2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        r1 = params.get("r1", 5)
        r2 = params.get("r2", 13)

        # Tangent length = √(r₂² - r₁²)
        tangent_length = math.sqrt(r2**2 - r1**2)

        return MethodResult(
            value=tangent_length,
            description=f"Tangent length between concentric circles r₁={r1}, r₂={r2}: √({r2}²-{r1}²) = {tangent_length:.4f}",
            metadata={"r1": r1, "r2": r2}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find outer radius given inner radius and tangent length."""
        r1 = params.get("r1", 5)
        t = output_value  # tangent length

        # t² = r₂² - r₁²  =>  r₂ = √(t² + r₁²)
        r2 = math.sqrt(t**2 + r1**2)

        return MethodResult(
            value=r2,
            description=f"Outer radius r₂ = √({t:.4f}² + {r1}²) = {r2:.4f}",
            metadata={"r1": r1, "tangent_length": t}
        )


