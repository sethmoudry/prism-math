"""
Geometry Techniques - Polygons

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class ShoelaceArea(MethodBlock):
    """
    Compute polygon area using shoelace formula:
    A = (1/2)|Σ(x_i·y_{i+1} - x_{i+1}·y_i)|.
    """

    def __init__(self):
        super().__init__()
        self.name = "shoelace_area"
        self.input_type = "polygon"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "coordinate", "area", "polygon"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Generate random convex polygon
        n = random.randint(3, 6)
        vertices = []
        for _ in range(n):
            x = random.randint(-20, 20)
            y = random.randint(-20, 20)
            vertices.append((x, y))
        return {"vertices": vertices}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        vertices = params.get("vertices")

        # Handle case where input_value is an integer from chaining
        # Use it to generate vertices deterministically
        if vertices is None or not isinstance(vertices, (list, tuple)):
            if isinstance(input_value, (int, float)):
                # Use input_value as seed for deterministic polygon generation
                seed_val = int(abs(input_value)) if input_value else 42
                local_random = random.Random(seed_val)
                # Number of vertices: 3-6 based on seed
                n_vertices = 3 + (seed_val % 4)
                vertices = []
                for _ in range(n_vertices):
                    x = local_random.randint(-20, 20)
                    y = local_random.randint(-20, 20)
                    vertices.append((x, y))
            else:
                # Default fallback: generate a simple triangle
                vertices = [(0, 0), (10, 0), (5, 10)]

        # Validate each vertex is a 2-tuple
        vertices = [ensure_tuple(v, f"vertices[{i}]", 2) for i, v in enumerate(vertices)]
        n = len(vertices)

        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2
        area_rounded = int(round(area))

        return MethodResult(
            value=area_rounded,
            description=f"Polygon area (shoelace) = {area:.4f} ≈ {area_rounded}",
            metadata={"vertices": vertices, "exact_area": area}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ShoelaceInverseVertex(MethodBlock):
    """Find vertex coordinates given area and other vertices."""

    def __init__(self):
        super().__init__()
        self.name = "shoelace_inverse_vertex"
        self.input_type = "geometric_value"
        self.output_type = "point"
        self.difficulty = 4
        self.tags = ["geometry", "coordinate", "area", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        vertices = [(0, 0), (random.randint(5, 15), 0), (random.randint(0, 10), random.randint(5, 15))]
        target_area = random.uniform(20, 100)
        return {"known_vertices": vertices, "target_area": target_area}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        known = params.get("known_vertices", [(0, 0), (10, 0), (5, 10)])
        target_area = params.get("target_area", 50)

        x1, y1 = known[0]
        x2, y2 = known[1]
        orig_x3 = known[2][0] if len(known) > 2 else 5
        orig_y3 = known[2][1] if len(known) > 2 else 10

        two_A = 2 * target_area
        C = x1 * y2 - x2 * y1
        coeff_x3 = y1 - y2

        if abs(coeff_x3) > 1e-10:
            B = orig_y3 * (x2 - x1)
            sol1 = (two_A - C - B) / coeff_x3
            sol2 = (-two_A - C - B) / coeff_x3
            if abs(sol1 - orig_x3) <= abs(sol2 - orig_x3):
                x3 = int(round(sol1))
            else:
                x3 = int(round(sol2))
            y3 = orig_y3
        else:
            coeff_y3 = x2 - x1
            if abs(coeff_y3) > 1e-10:
                sol1 = (two_A - C) / coeff_y3
                sol2 = (-two_A - C) / coeff_y3
                if abs(sol1 - orig_y3) <= abs(sol2 - orig_y3):
                    y3 = int(round(sol1))
                else:
                    y3 = int(round(sol2))
                x3 = orig_x3
            else:
                x3 = orig_x3
                y3 = orig_y3

        return MethodResult(
            value=(x3, y3),
            description=f"Found vertex ({x3}, {y3}) to achieve target area {target_area:.2f}",
            metadata={"known_vertices": known, "target_area": target_area}
        )


# ============================================================================
# LATTICE POINTS (4)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleConstraintsTilings(MethodBlock):
    """
    Angle constraints in geometric tilings.

    In regular tilings and tessellations, angles between adjacent polygons
    follow strict constraints. For example, in a tiling of regular n-gons,
    the internal angle is (n-2)*180/n degrees.

    Key properties:
    - Regular n-gons tile the plane if angle = 360/k for integer k
    - Common tilings: hexagon (120°), square (90°), triangle (60°)
    - Angle constraint: sum of angles around a vertex = 360°
    """
    def __init__(self):
        super().__init__()
        self.name = "angle_constraints_tilings"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "tiling"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(3, 12)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 6)
        # Internal angle of regular n-gon: (n-2)*180/n degrees
        internal_angle = (n - 2) * 180 // n
        # For tiling constraint: how many n-gons fit around a vertex
        # We need k*internal_angle = 360, so k = 360/internal_angle
        if internal_angle > 0:
            k = 360 // internal_angle if 360 % internal_angle == 0 else 360 / internal_angle
            result = int(k) if isinstance(k, float) else k
        else:
            result = 0
        return MethodResult(
            value=result,
            description=f"Regular {n}-gon: interior angle={(n-2)*180}/{n}°, {result} fit around vertex",
            metadata={"n": n, "interior_angle": internal_angle, "fit_count": result}
        )

    def can_invert(self):
        return False


@register_technique
class ConvexHullAnalysis(MethodBlock):
    """
    Convex hull analysis for a set of points.

    Computes properties of the convex hull:
    - Number of vertices on the convex hull
    - Perimeter of convex hull
    - Area of convex hull (using shoelace formula)

    Uses gift wrapping or similar approach for hull construction.
    """
    def __init__(self):
        super().__init__()
        self.name = "convex_hull_analysis"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "convex_hull", "computational"]

    def validate_params(self, params, prev_value=None):
        """Validate at least 3 points for meaningful convex hull."""
        points = params.get("points")
        return points is not None and len(points) >= 3

    def generate_parameters(self, input_value=None):
        # Generate random points for convex hull
        n_points = random.randint(5, 12)
        points = [(random.randint(-10, 10), random.randint(-10, 10)) for _ in range(n_points)]
        operation = random.choice(["vertices", "area", "perimeter"])
        return {"points": points, "operation": operation}

    def _cross(self, o, a, b):
        """Cross product of vectors OA and OB."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _convex_hull(self, points):
        """Compute convex hull using Andrew's monotone chain algorithm."""
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def compute(self, input_value, params):
        points = params.get("points")
        operation = params.get("operation", "vertices")

        # Handle case where input_value is an integer from chaining
        # Use it to generate points deterministically
        if points is None or not isinstance(points, (list, tuple)):
            if isinstance(input_value, (int, float)):
                # Use input_value as seed for deterministic point generation
                seed_val = int(abs(input_value)) if input_value else 42
                local_random = random.Random(seed_val)
                # Number of points: 5-12 based on seed
                n_points = 5 + (seed_val % 8)
                points = []
                for _ in range(n_points):
                    x = local_random.randint(-10, 10)
                    y = local_random.randint(-10, 10)
                    points.append((x, y))
            else:
                # Fallback to default points
                points = [(0, 0), (4, 0), (2, 3)]

        # Ensure points are tuples
        points = [tuple(p) if isinstance(p, list) else p for p in points]

        hull = self._convex_hull(points)
        n_hull = len(hull)

        if operation == "vertices":
            result = n_hull
            description = f"Convex hull has {n_hull} vertices from {len(points)} points"
        elif operation == "area":
            # Shoelace formula for area
            if n_hull < 3:
                area = 0
            else:
                area = 0
                for i in range(n_hull):
                    j = (i + 1) % n_hull
                    area += hull[i][0] * hull[j][1]
                    area -= hull[j][0] * hull[i][1]
                area = abs(area) / 2
            result = int(round(area))
            description = f"Convex hull area = {result}"
        else:  # perimeter
            perimeter = 0
            for i in range(n_hull):
                j = (i + 1) % n_hull
                dx = hull[j][0] - hull[i][0]
                dy = hull[j][1] - hull[i][1]
                perimeter += math.sqrt(dx * dx + dy * dy)
            result = int(round(perimeter))
            description = f"Convex hull perimeter = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"n_points": len(points), "hull_vertices": n_hull, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class CyclicQuadrilaterals(MethodBlock):
    """
    Properties of cyclic quadrilaterals.

    A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle.

    Key properties:
    - Ptolemy's Theorem: For cyclic quadrilateral ABCD: AC*BD = AB*CD + AD*BC
    - Opposite angles sum to 180°: angle A + angle C = 180°
    - Area by Brahmagupta: A = sqrt((s-a)(s-b)(s-c)(s-d)) where s is semi-perimeter
    - Inscribed angle relationships and power of a point theorems apply
    """
    def __init__(self):
        super().__init__()
        self.name = "cyclic_quadrilaterals"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "quadrilateral"]

    def generate_parameters(self, input_value=None):
        angle = input_value if input_value is not None else random.randint(30, 150)
        return {"angle": angle}

    def compute(self, input_value, params):
        angle = params.get("angle", 90)
        # Normalize angle to [1, 179] range to handle nonsensical chain inputs
        angle = ((int(angle) - 1) % 178) + 1
        # Ptolemy's theorem application: For cyclic quadrilateral with sides a, b, c, d
        # and diagonals p, q: p*q = a*c + b*d
        # For a regular approach: opposite angles sum to 180°
        opposite_angle = 180 - angle
        # Using Ptolemy for a specific case with angle as input
        # For an inscribed quadrilateral with one angle given
        result = opposite_angle
        return MethodResult(
            value=result,
            description=f"Cyclic quad opposite angle: 180 - {angle} = {result}",
            metadata={"angle": angle, "theorem": "Ptolemy and opposite angles"}
        )

    def can_invert(self):
        return False


@register_technique
class RegularPolygonProperties(MethodBlock):
    """Properties of regular polygons."""
    def __init__(self):
        super().__init__()
        self.name = "regular_polygon_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "polygon"]

    def generate_parameters(self, input_value=None):
        n = random.randint(3, 8)
        s = random.randint(2, 10)
        return {"n": n, "s": s}

    def compute(self, input_value, params):
        n = params.get("n", 6)
        s = params.get("s", 4)
        result = n * s
        return MethodResult(
            value=result,
            description=f"Regular {n}-gon perimeter with side {s}: {result}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class CyclicQuadrilateral(MethodBlock):
    """Properties of cyclic quadrilaterals (inscribed in circles)."""
    def __init__(self):
        super().__init__()
        self.name = "cyclic_quadrilateral"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "quadrilateral", "circle"]

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 15)
        b = random.randint(3, 15)
        c = random.randint(3, 15)
        d = random.randint(3, 15)
        return {"a": a, "b": b, "c": c, "d": d}

    def validate_params(self, params, prev_value=None):
        """Validate cyclic quadrilateral parameters: all sides must be positive."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        d = params.get("d")
        if any(v is None for v in [a, b, c, d]):
            return False
        try:
            return all(float(v) > 0 for v in [a, b, c, d])
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        a = params.get("a", 5)
        b = params.get("b", 6)
        c = params.get("c", 7)
        d = params.get("d", 8)
        result = a * c + b * d
        return MethodResult(
            value=result,
            description=f"Cyclic quad with sides {a},{b},{c},{d}: Ptolemy product = {result}",
            metadata={"sides": [a, b, c, d]}
        )

    def can_invert(self):
        return False


