"""
Ported geometry starter and primitive methods from the old composable_techniques system.

Methods ported:
- random_point (from geometry_starters.py)
- random_point_pair (from geometry_starters.py)
- random_circle_pair (from geometry_starters.py)
- random_triangle (from geometry_starters.py)
- random_hexagon (from geometry_starters.py)
- distance_squared (from utility_primitives.py)
- volume_rectangular_prism (from utility_methods.py)
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    random, math, Dict, Any, Optional,
)


# Pythagorean triples for clean integer calculations
PYTHAGOREAN_TRIPLES = [
    (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
    (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61),
    (6, 8, 10), (9, 12, 15), (12, 16, 20), (15, 20, 25),
]


# ============================================================================
# GEOMETRY STARTERS (3 techniques)
# ============================================================================

@register_technique
class RandomPoint(MethodBlock):
    """Generate a random point with integer coordinates."""

    def __init__(self):
        super().__init__()
        self.name = "random_point"
        self.input_type = "none"
        self.output_type = "point"
        self.difficulty = 1
        self.tags = ["geometry", "starter"]

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate integer coordinates for a clean point."""
        x = random.randint(1, 20)
        y = random.randint(1, 20)
        return {"x": x, "y": y}

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Validate point parameters."""
        return "x" in params and "y" in params

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Generate the point."""
        x = params["x"]
        y = params["y"]
        return MethodResult(
            value=(x, y),
            description=f"Point at ({x}, {y})",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class RandomPointPair(MethodBlock):
    """Generate a pair of random points with integer coordinates."""

    def __init__(self):
        super().__init__()
        self.name = "random_point_pair"
        self.input_type = "none"
        self.output_type = "point_pair"
        self.difficulty = 1
        self.tags = ["geometry", "starter"]

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate two distinct points with integer coordinates."""
        x1 = random.randint(0, 15)
        y1 = random.randint(0, 15)
        x2 = random.randint(0, 15)
        y2 = random.randint(0, 15)
        while (x2, y2) == (x1, y1):
            x2 = random.randint(0, 15)
            y2 = random.randint(0, 15)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Validate that we have two distinct points."""
        return (
            all(k in params for k in ["x1", "y1", "x2", "y2"]) and
            (params["x1"], params["y1"]) != (params["x2"], params["y2"])
        )

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Generate the point pair."""
        p1 = (params["x1"], params["y1"])
        p2 = (params["x2"], params["y2"])
        return MethodResult(
            value=(p1, p2),
            description=f"Points at {p1} and {p2}",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class RandomCirclePair(MethodBlock):
    """Generate a pair of circles for radical axis problems."""

    def __init__(self):
        super().__init__()
        self.name = "random_circle_pair"
        self.input_type = "none"
        self.output_type = "circle_pair"
        self.difficulty = 2
        self.tags = ["geometry", "starter", "circles"]

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate two non-identical intersecting circles."""
        c1x = random.randint(0, 10)
        c1y = random.randint(0, 10)
        r1 = random.randint(3, 8)

        angle = random.uniform(0, 2 * math.pi)
        r2 = random.randint(3, 8)
        min_dist = abs(r1 - r2) + 1
        max_dist = r1 + r2 - 1
        dist = random.uniform(min_dist, max_dist)

        c2x = int(c1x + dist * math.cos(angle))
        c2y = int(c1y + dist * math.sin(angle))

        return {
            "c1x": c1x, "c1y": c1y, "r1": r1,
            "c2x": c2x, "c2y": c2y, "r2": r2
        }

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Validate that circles are distinct."""
        keys = ["c1x", "c1y", "r1", "c2x", "c2y", "r2"]
        if not all(k in params for k in keys):
            return False
        return not (
            params["c1x"] == params["c2x"] and
            params["c1y"] == params["c2y"] and
            params["r1"] == params["r2"]
        )

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Generate the circle pair."""
        c1 = ((params["c1x"], params["c1y"]), params["r1"])
        c2 = ((params["c2x"], params["c2y"]), params["r2"])
        return MethodResult(
            value=(c1, c2),
            description=f"Circles: center {c1[0]} radius {c1[1]}, center {c2[0]} radius {c2[1]}",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class RandomTriangle(MethodBlock):
    """Generate a random triangle with clean integer coordinates using Pythagorean triples."""

    def __init__(self):
        super().__init__()
        self.name = "random_triangle"
        self.input_type = "none"
        self.output_type = "triangle"
        self.difficulty = 1
        self.tags = ["geometry", "starter"]

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate a right triangle using Pythagorean triples for clean math."""
        # Pick a Pythagorean triple
        a, b, c = random.choice(PYTHAGOREAN_TRIPLES)

        # Place at a random offset
        ox = random.randint(0, 10)
        oy = random.randint(0, 10)

        # Right angle at origin (offset)
        return {
            "ax": ox, "ay": oy,           # Right angle vertex
            "bx": ox + a, "by": oy,       # Leg along x-axis
            "cx": ox, "cy": oy + b        # Leg along y-axis
        }

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Validate that we have 3 non-collinear points."""
        keys = ["ax", "ay", "bx", "by", "cx", "cy"]
        if not all(k in params for k in keys):
            return False

        # Check not collinear using cross product
        ax, ay = params["ax"], params["ay"]
        bx, by = params["bx"], params["by"]
        cx, cy = params["cx"], params["cy"]

        cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        return cross != 0

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Generate the triangle."""
        A = (params["ax"], params["ay"])
        B = (params["bx"], params["by"])
        C = (params["cx"], params["cy"])
        return MethodResult(
            value=(A, B, C),
            description=f"Triangle with vertices {A}, {B}, {C}",
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class RandomHexagon(MethodBlock):
    """Generate a hexagon for Pascal's theorem problems."""

    def __init__(self):
        super().__init__()
        self.name = "random_hexagon"
        self.input_type = "none"
        self.output_type = "hexagon"
        self.difficulty = 3
        self.tags = ["geometry", "starter", "projective", "pascal"]

    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate 6 points on a conic (circle for simplicity)."""
        cx, cy = random.randint(10, 20), random.randint(10, 20)
        r = random.randint(5, 12)

        # 6 angles, sorted
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(6)])

        params = {"cx": cx, "cy": cy, "r": r}
        for i, angle in enumerate(angles):
            params[f"x{i+1}"] = int(cx + r * math.cos(angle))
            params[f"y{i+1}"] = int(cy + r * math.sin(angle))

        return params

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Validate hexagon parameters."""
        keys = [f"x{i}" for i in range(1, 7)] + [f"y{i}" for i in range(1, 7)]
        if not all(k in params for k in keys):
            return False

        # Check all 6 points are distinct
        points = [(params[f"x{i}"], params[f"y{i}"]) for i in range(1, 7)]
        return len(set(points)) == 6

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Generate the hexagon."""
        points = tuple((params[f"x{i}"], params[f"y{i}"]) for i in range(1, 7))
        return MethodResult(
            value=points,
            description=(
                f"Hexagon with vertices on circle centered at "
                f"({params.get('cx', 0)}, {params.get('cy', 0)})"
            ),
            params=params
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# BASIC GEOMETRY PRIMITIVES (2 techniques)
# ============================================================================

@register_technique
class DistanceSquared(MethodBlock):
    """
    Calculate squared distance between two points: (x2-x1)^2 + (y2-y1)^2

    Avoids square root for exact integer arithmetic.
    """

    def __init__(self):
        super().__init__()
        self.name = "distance_squared"
        self.input_type = "points"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "coordinate", "basic"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        x1 = random.randint(-10, 10)
        y1 = random.randint(-10, 10)
        x2 = random.randint(-10, 10)
        y2 = random.randint(-10, 10)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1 = params.get("x1", 10)
        y1 = params.get("y1", 10)
        x2 = params.get("x2", 10)
        y2 = params.get("y2", 10)

        dx = x2 - x1
        dy = y2 - y1
        result = dx * dx + dy * dy

        return MethodResult(
            value=result,
            description=f"distance^2 from ({x1},{y1}) to ({x2},{y2}) = {dx}^2 + {dy}^2 = {result}",
            params=params,
            metadata={"dx": dx, "dy": dy}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class VolumeRectangularPrism(MethodBlock):
    """Compute volume of rectangular prism: l * w * h."""

    def __init__(self):
        super().__init__()
        self.name = "volume_rectangular_prism"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["geometry", "volume"]

    def generate_parameters(self, input_value=None):
        """Generate random dimensions."""
        return {
            "l": random.randint(1, 20),
            "w": random.randint(1, 20),
            "h": random.randint(1, 20)
        }

    def compute(self, input_value, params):
        """Compute volume of rectangular prism.

        Examples:
            volume_rectangular_prism(2, 3, 4) -> 24
            volume_rectangular_prism(5, 5, 5) -> 125
        """
        l = params.get("l", 2)
        w = params.get("w", 3)
        h = params.get("h", 4)

        result = l * w * h

        return MethodResult(
            value=result,
            description=f"Volume of rectangular prism ({l} x {w} x {h}) = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False
