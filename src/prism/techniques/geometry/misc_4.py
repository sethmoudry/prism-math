"""
Geometry Techniques - Misc (Part 4)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class HyperbolaSemiAxes(MethodBlock):
    """
    Extract the semi-axes a and b from a hyperbola.

    Standard form: (x-h)²/a² - (y-k)²/b² = 1
    Returns the product a·b.
    """

    def __init__(self):
        super().__init__()
        self.name = "hyperbola_semi_axes"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "hyperbola", "conic_section", "semi_axes"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate hyperbola parameters."""
        a = random.randint(3, 15)
        b = random.randint(3, 15)
        h = random.randint(-10, 10)
        k = random.randint(-10, 10)
        return {"a": a, "b": b, "center": (h, k)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 5)
        b = params.get("b", 5)
        h, k = ensure_tuple(params.get("center", (0, 0)), "center", 2)

        # Product of semi-axes
        product = a * b

        return MethodResult(
            value=product,
            description=f"Hyperbola (x-{h})²/{a}² - (y-{k})²/{b}² = 1, a·b = {product}",
            metadata={"a": a, "b": b, "center": (h, k)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class HyperbolaFociDistance(MethodBlock):
    """
    Calculate the distance between foci of a hyperbola.

    For hyperbola (x-h)²/a² - (y-k)²/b² = 1:
    Distance between foci = 2c where c² = a² + b²
    """

    def __init__(self):
        super().__init__()
        self.name = "hyperbola_foci_distance"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "hyperbola", "foci", "distance"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate hyperbola parameters."""
        a = random.randint(3, 15)
        b = random.randint(3, 15)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 5)
        b = params.get("b", 5)

        # c² = a² + b², distance = 2c
        c_squared = a**2 + b**2
        c = math.sqrt(c_squared)
        distance = 2 * c

        return MethodResult(
            value=distance,
            description=f"Hyperbola foci distance 2c = 2√({a}²+{b}²) = {distance:.4f}",
            metadata={"a": a, "b": b, "c": c}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VolumeFromBounds(MethodBlock):
    """
    Calculate volume of a rectangular region from given bounds.
    Volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    """

    def __init__(self):
        super().__init__()
        self.name = "volume_from_bounds"
        self.input_type = "integer"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "volume", "3d", "bounds"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate bounds for a 3D rectangular region."""
        # Generate bounds ensuring positive dimensions
        x_min = random.randint(-20, 10)
        x_max = random.randint(x_min + 1, x_min + 30)

        y_min = random.randint(-20, 10)
        y_max = random.randint(y_min + 1, y_min + 30)

        z_min = random.randint(-20, 10)
        z_max = random.randint(z_min + 1, z_min + 30)

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x_min = params.get("x_min", 0)
        x_max = params.get("x_max", 10)
        y_min = params.get("y_min", 0)
        y_max = params.get("y_max", 10)
        z_min = params.get("z_min", 0)
        z_max = params.get("z_max", 10)

        # Calculate volume of rectangular region
        length = x_max - x_min
        width = y_max - y_min
        height = z_max - z_min

        volume = length * width * height

        return MethodResult(
            value=volume,
            description=f"Volume of region [{x_min},{x_max}]×[{y_min},{y_max}]×[{z_min},{z_max}] = {length}×{width}×{height} = {volume}",
            metadata={
                "bounds": {
                    "x": (x_min, x_max),
                    "y": (y_min, y_max),
                    "z": (z_min, z_max)
                },
                "dimensions": {
                    "length": length,
                    "width": width,
                    "height": height
                }
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PolesAndPolars(MethodBlock):
    """Compute the polar line of a point with respect to a circle.

    In projective geometry, given a circle with center O and radius r, and a
    point P (the pole), the polar of P is a line perpendicular to OP.

    If d = distance(O, P), the polar is at distance r^2/d from O along OP.

    Properties:
    - If P is outside the circle (d > r): polar intersects circle at the
      tangent points from P
    - If P is on the circle (d = r): polar is the tangent line at P
    - If P is inside the circle (d < r): polar is outside the circle

    The technique computes the distance from the center to the polar line,
    which is r^2/d (an integer when computed with appropriate scaling).

    Examples:
    - Circle r=6, point at d=9: polar at distance 36/9 = 4 from center
    - Circle r=10, point at d=5: polar at distance 100/5 = 20 from center

    Returns: The scaled distance from center to polar (integer).
    """

    def __init__(self):
        super().__init__()
        self.name = "poles_and_polars"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "projective", "circle", "polar", "inversion"]

    def validate_params(self, params, prev_value=None):
        """Validate circle radius and point distance are positive."""
        r = params.get("radius")
        d = params.get("distance")
        if r is None or d is None:
            return False
        return r > 0 and d > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        """Generate circle radius and point distance that give integer polar distance."""
        # Choose r and d such that r^2 is divisible by d for integer result
        # Strategy: pick r^2 first, then choose d as a divisor

        if input_value is not None and input_value > 0:
            # Use input as radius
            r = input_value
        else:
            r = random.randint(4, 20)

        r_squared = r * r

        # Find divisors of r^2 for clean integer results
        divisors = [i for i in range(1, r_squared + 1) if r_squared % i == 0]
        # Prefer divisors that give reasonable polar distances
        valid_divisors = [d for d in divisors if r_squared // d <= 1000]
        d = random.choice(valid_divisors) if valid_divisors else r

        return {"radius": r, "distance": d}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute the distance from circle center to the polar line.

        For a circle with radius r and a point at distance d from center,
        the polar line is at distance r^2/d from the center.
        """
        r = params.get("radius", 10)
        d = params.get("distance", 5)

        r_squared = r * r

        # Polar distance = r^2 / d
        # If not exactly divisible, we scale to get an integer
        if r_squared % d == 0:
            polar_distance = r_squared // d
            description = f"Polar of point at distance {d} from center of circle r={r}: "
            description += f"polar at distance r^2/d = {r}^2/{d} = {polar_distance}"
        else:
            # Return scaled result (multiply by d to get integer)
            polar_distance = r_squared  # scaled by d
            description = f"Polar distance (scaled by {d}): {r}^2 = {polar_distance}"

        # Determine relationship
        if d > r:
            position = "outside"
        elif d == r:
            position = "on"
        else:
            position = "inside"

        return MethodResult(
            value=polar_distance,
            description=description,
            params=params,
            metadata={
                "radius": r,
                "point_distance": d,
                "polar_distance": polar_distance,
                "point_position": position,
                "formula": "r^2/d"
            }
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> Optional[Any]:
        """Given polar distance, find point distance."""
        r = params.get("radius", 10)
        polar_dist = output_value
        if polar_dist <= 0:
            return None
        # d = r^2 / polar_dist
        r_squared = r * r
        if r_squared % polar_dist == 0:
            return r_squared // polar_dist
        return None


# ============================================================================
# BRIANCHON'S THEOREM (Projective Geometry)
# ============================================================================


@register_technique
class BrianchonTheorem(MethodBlock):
    """Verify Brianchon's theorem for hexagons circumscribed about a conic.

    Brianchon's theorem states that for a hexagon circumscribed about a conic
    (i.e., all six sides are tangent to the conic), the three main diagonals
    (connecting opposite vertices) are concurrent.

    This is the dual of Pascal's theorem (which applies to inscribed hexagons).

    For a hexagon with vertices A, B, C, D, E, F circumscribed about a circle:
    - Main diagonals are AD, BE, CF
    - These three lines meet at a single point (the Brianchon point)

    The technique verifies concurrency by checking if the determinant of the
    system of lines equals zero (within numerical tolerance).

    For computational purposes, we work with a circle and generate tangent
    hexagons, then verify the concurrency condition.

    Returns: 1 if diagonals are concurrent (theorem verified), 0 otherwise.
    """

    def __init__(self):
        super().__init__()
        self.name = "brianchon_theorem"
        self.input_type = "geometric"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["geometry", "projective", "conic", "brianchon", "concurrency"]

    def validate_params(self, params, prev_value=None):
        """Validate that we have valid tangent points."""
        tangent_points = params.get("tangent_points")
        if tangent_points is None:
            return False
        if not isinstance(tangent_points, (list, tuple)) or len(tangent_points) != 6:
            return False
        return True

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        """Generate 6 tangent points on a unit circle for a circumscribed hexagon."""
        # Generate 6 angles in increasing order for tangent points
        # The tangent lines at these points will form the hexagon sides

        if input_value is not None:
            # Use input to seed the angle distribution
            random.seed(input_value)

        # Generate 6 distinct angles in [0, 2pi)
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(6)])

        # Tangent points on unit circle
        tangent_points = [(math.cos(a), math.sin(a)) for a in angles]

        return {"tangent_points": tangent_points, "radius": 1, "angles": angles}

    def _tangent_line_at_point(self, point: Tuple[float, float], center: Tuple[float, float] = (0, 0)) -> Tuple[float, float, float]:
        """Get line equation ax + by + c = 0 for tangent at point on circle centered at origin."""
        px, py = point
        cx, cy = center
        # For unit circle at origin, tangent at (px, py) is px*x + py*y = 1
        # In general form: px*x + py*y - 1 = 0
        a = px - cx
        b = py - cy
        r_squared = a*a + b*b
        c = -r_squared  # normalized: ax + by + c = 0 where point is on circle
        # Actually for tangent: px*x + py*y = px^2 + py^2 = 1 (for unit circle)
        return (a, b, -1)  # px*x + py*y = 1

    def _line_through_points(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
        """Get line equation ax + by + c = 0 through two points."""
        x1, y1 = p1
        x2, y2 = p2
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        return (a, b, c)

    def _lines_concurrent(self, line1: Tuple, line2: Tuple, line3: Tuple, tol: float = 1e-6) -> bool:
        """Check if three lines are concurrent using determinant."""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        a3, b3, c3 = line3

        # Three lines are concurrent iff determinant is zero:
        # | a1 b1 c1 |
        # | a2 b2 c2 | = 0
        # | a3 b3 c3 |
        det = (a1 * (b2 * c3 - b3 * c2) -
               b1 * (a2 * c3 - a3 * c2) +
               c1 * (a2 * b3 - a3 * b2))

        return abs(det) < tol

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Verify Brianchon's theorem for the given circumscribed hexagon.

        Given 6 tangent points T0, T1, T2, T3, T4, T5 on a circle:
        1. Compute tangent lines at each point (these form hexagon sides)
        2. Find hexagon vertices as intersections of consecutive tangent lines
        3. Compute the three main diagonals
        4. Check if they are concurrent
        """
        tangent_points = params.get("tangent_points")
        radius = params.get("radius", 1)

        # Get tangent lines at each point
        tangent_lines = [self._tangent_line_at_point(p) for p in tangent_points]

        # Find hexagon vertices (intersection of consecutive tangent lines)
        vertices = []
        for i in range(6):
            l1 = tangent_lines[i]
            l2 = tangent_lines[(i + 1) % 6]
            # Solve intersection
            intersection = line_intersection(
                ((0, -l1[2]/l1[1]) if abs(l1[1]) > 1e-10 else ((-l1[2]/l1[0], 0)),
                 (1, (-l1[0] - l1[2])/l1[1]) if abs(l1[1]) > 1e-10 else ((-l1[2]/l1[0], 1))),
                ((0, -l2[2]/l2[1]) if abs(l2[1]) > 1e-10 else ((-l2[2]/l2[0], 0)),
                 (1, (-l2[0] - l2[2])/l2[1]) if abs(l2[1]) > 1e-10 else ((-l2[2]/l2[0], 1)))
            )
            if intersection:
                vertices.append(intersection)
            else:
                # Parallel lines - shouldn't happen for valid configuration
                vertices.append((0, 0))

        if len(vertices) != 6:
            return MethodResult(
                value=0,
                description="Failed to compute hexagon vertices",
                params=params,
                metadata={"error": "vertex_computation_failed"}
            )

        # Main diagonals: connect opposite vertices
        # V0-V3, V1-V4, V2-V5
        diagonal_AD = self._line_through_points(vertices[0], vertices[3])
        diagonal_BE = self._line_through_points(vertices[1], vertices[4])
        diagonal_CF = self._line_through_points(vertices[2], vertices[5])

        # Check concurrency
        concurrent = self._lines_concurrent(diagonal_AD, diagonal_BE, diagonal_CF)
        result = 1 if concurrent else 0

        return MethodResult(
            value=result,
            description=f"Brianchon's theorem: diagonals {'are' if concurrent else 'are NOT'} concurrent",
            params=params,
            metadata={
                "tangent_points": tangent_points,
                "vertices": vertices,
                "concurrent": concurrent,
                "theorem": "Brianchon"
            }
        )

    def can_invert(self) -> bool:
        return False


