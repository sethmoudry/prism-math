"""
Geometry Techniques - Transformations

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class SpiralSimilarity(MethodBlock):
    """Apply spiral similarity: rotate by θ and scale by k about center O."""

    def __init__(self):
        super().__init__()
        self.name = "spiral_similarity"
        self.input_type = "point"
        self.output_type = "point"
        self.difficulty = 3
        self.tags = ["geometry", "transformation", "spiral", "similarity"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        center = (random.randint(-10, 10), random.randint(-10, 10))
        angle = random.uniform(0, 360)
        scale = random.uniform(0.5, 2.0)
        point = (random.randint(-20, 20), random.randint(-20, 20))
        return {"center": center, "angle": angle, "scale": scale, "point": point}

    def validate_params(self, params, prev_value=None):
        """Validate spiral similarity parameters: scale must be non-zero."""
        scale = params.get("scale")
        if scale is None:
            return False
        try:
            return float(scale) != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", 10), "center", 2)
        theta = math.radians(params.get("angle", 10))
        k = params.get("scale", 10)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)

        # Translate to origin
        dx, dy = px - cx, py - cy

        # Rotate and scale
        new_x = k * (dx * math.cos(theta) - dy * math.sin(theta))
        new_y = k * (dx * math.sin(theta) + dy * math.cos(theta))

        # Translate back
        result = (cx + new_x, cy + new_y)

        return MethodResult(
            value=result,
            description=f"Spiral similarity: ({px}, {py}) → ({result[0]:.2f}, {result[1]:.2f})",
            metadata={"center": (cx, cy), "angle": params.get("angle", 10), "scale": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SpiralSimilarityInverse(MethodBlock):
    """Find center of spiral similarity given image and pre-image."""

    def __init__(self):
        super().__init__()
        self.name = "spiral_similarity_inverse"
        self.input_type = "point_pair"
        self.output_type = "point"
        self.difficulty = 4
        self.tags = ["geometry", "transformation", "spiral", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        p1 = (random.randint(-20, 20), random.randint(-20, 20))
        p2 = (random.randint(-20, 20), random.randint(-20, 20))
        angle = random.uniform(0, 360)
        scale = random.uniform(0.5, 2.0)
        return {"p1": p1, "p2": p2, "angle": angle, "scale": scale}

    def validate_params(self, params, prev_value=None):
        """Validate spiral similarity inverse parameters: scale must be non-zero."""
        scale = params.get("scale")
        if scale is None:
            return False
        try:
            return float(scale) != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", 10), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", 10), "p2", 2)
        theta = math.radians(params.get("angle", 10))
        k = params.get("scale", 10)

        P1 = complex(x1, y1)
        P2 = complex(x2, y2)
        w = k * complex(math.cos(theta), math.sin(theta))

        denom = 1 - w
        if abs(denom) < 1e-10:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            O = (P2 - w * P1) / denom
            center = (O.real, O.imag)

        return MethodResult(
            value=center,
            description=f"Center of spiral similarity: ({center[0]:.2f}, {center[1]:.2f})",
            metadata={"p1": (x1, y1), "p2": (x2, y2), "angle": params.get("angle", 10), "scale": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Homothety(MethodBlock):
    """Apply homothety: scale by k about center O."""

    def __init__(self):
        super().__init__()
        self.name = "homothety"
        self.input_type = "point"
        self.output_type = "point"
        self.difficulty = 2
        self.tags = ["geometry", "transformation", "homothety", "scaling"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        center = (random.randint(-10, 10), random.randint(-10, 10))
        # Use integer scale to avoid floating point errors in chain verification
        scale = random.randint(1, 4)
        point = (random.randint(-20, 20), random.randint(-20, 20))
        return {"center": center, "scale": scale, "point": point}

    def validate_params(self, params, prev_value=None):
        """Validate homothety parameters: scale must be non-zero."""
        scale = params.get("scale")
        if scale is None:
            return False
        try:
            return float(scale) != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", 10), "center", 2)
        k = params.get("scale", 10)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)

        # P' = O + k(P - O)
        new_x = cx + k * (px - cx)
        new_y = cy + k * (py - cy)

        return MethodResult(
            value=(new_x, new_y),
            description=f"Homothety: ({px}, {py}) → ({new_x:.2f}, {new_y:.2f}) with scale {k}",
            metadata={"center": (cx, cy), "scale": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class HomothetyInverse(MethodBlock):
    """Find center/ratio of homothety given two points."""

    def __init__(self):
        super().__init__()
        self.name = "homothety_inverse"
        self.input_type = "point_pair"
        self.output_type = "point"
        self.difficulty = 3
        self.tags = ["geometry", "transformation", "homothety", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        p1 = (random.randint(-20, 20), random.randint(-20, 20))
        p2 = (random.randint(-20, 20), random.randint(-20, 20))
        # Use integer scale to avoid floating point errors in chain verification
        scale = random.randint(2, 4)
        return {"p1": p1, "p2": p2, "scale": scale}

    def validate_params(self, params, prev_value=None):
        """Validate homothety inverse parameters: scale must be non-zero."""
        scale = params.get("scale")
        if scale is None:
            return False
        try:
            return float(scale) != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", 10), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", 10), "p2", 2)
        k = params.get("scale", 10)

        # O = (kP1 - P2)/(k-1)
        if abs(k - 1) < 0.001:
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
        else:
            cx = (k * x1 - x2) / (k - 1)
            cy = (k * y1 - y2) / (k - 1)
            center = (cx, cy)

        return MethodResult(
            value=center,
            description=f"Homothety center: ({center[0]:.2f}, {center[1]:.2f})",
            metadata=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class InversiveGeometry(MethodBlock):
    """Inversion in circle: P' = O + r²(P-O)/|P-O|²."""

    def __init__(self):
        super().__init__()
        self.name = "inversive_geometry"
        self.input_type = "point"
        self.output_type = "point"
        self.difficulty = 4
        self.tags = ["geometry", "transformation", "inversion", "circle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        center = (random.randint(-10, 10), random.randint(-10, 10))
        radius = random.randint(5, 20)
        # Generate point ensuring it's not at center (for valid inversion)
        # Use offsets that guarantee we're away from center
        offset_x = random.choice(list(range(-20, -2)) + list(range(3, 25)))
        offset_y = random.choice(list(range(-20, -2)) + list(range(3, 25)))
        point = (center[0] + offset_x, center[1] + offset_y)
        return {"center": center, "radius": radius, "point": point}

    def validate_params(self, params, prev_value=None):
        """Validate inversive geometry parameters: radius must be positive and point not at center."""
        radius = params.get("radius")
        if radius is None:
            return False
        try:
            r = float(radius)
            if r <= 0:
                return False
        except (ValueError, TypeError):
            return False

        # Ensure point is not at center
        center = params.get("center", (0, 0))
        point = params.get("point", (0, 0))
        if isinstance(center, (list, tuple)) and isinstance(point, (list, tuple)):
            cx, cy = center[0], center[1]
            px, py = point[0], point[1]
            dist_sq = (px - cx)**2 + (py - cy)**2
            if dist_sq < 0.001:
                return False
        return True

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", 10), "center", 2)
        r = params.get("radius", 10)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)

        dx, dy = px - cx, py - cy
        dist_sq = dx**2 + dy**2

        if dist_sq < 0.001:
            return MethodResult(
                value=(px, py),
                description="Point at center is undefined under inversion",
                metadata=params
            )

        scale = r**2 / dist_sq
        new_x = cx + scale * dx
        new_y = cy + scale * dy

        return MethodResult(
            value=(new_x, new_y),
            description=f"Inversion of ({px}, {py}) in circle: ({new_x:.2f}, {new_y:.2f})",
            metadata={"center": (cx, cy), "radius": r}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Reflection(MethodBlock):
    """Reflect point across line."""

    def __init__(self):
        super().__init__()
        self.name = "reflection"
        self.input_type = "point"
        self.output_type = "point"
        self.difficulty = 2
        self.tags = ["geometry", "transformation", "reflection"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Line through origin with slope m
        m = random.uniform(-2, 2)
        point = (random.randint(-20, 20), random.randint(-20, 20))
        return {"slope": m, "point": point}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        m = params.get("slope", 10)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)

        # Reflection across y = mx
        # Formula: P' = ((1-m²)x + 2my, 2mx + (m²-1)y) / (1+m²)
        denom = 1 + m**2
        new_x = ((1 - m**2) * px + 2 * m * py) / denom
        new_y = (2 * m * px + (m**2 - 1) * py) / denom

        return MethodResult(
            value=(new_x, new_y),
            description=f"Reflection of ({px}, {py}) across y={m}x: ({new_x:.2f}, {new_y:.2f})",
            metadata={"slope": m, "original": (px, py)}
        )


# ============================================================================
# PROJECTIVE (4)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class ReflectionGeometry(MethodBlock):
    """
    Compute properties of geometric reflections. Given a point and a mirror line
    (axis of reflection), computes distances and properties of the reflected point.
    The reflected point is at the same distance from the mirror line as the original,
    but on the opposite side. Uses the formula for reflection across an axis.
    """
    def __init__(self):
        super().__init__()
        self.name = "reflection_geometry"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "reflection", "transformation"]

    def generate_parameters(self, input_value=None):
        """Generate point and reflection parameters."""
        px = input_value if input_value is not None else random.randint(-20, 20)
        py = random.randint(-20, 20)
        # Reflection axis: either x-axis, y-axis, or line y=x
        axis_type = random.randint(0, 2)  # 0=x-axis, 1=y-axis, 2=y=x
        return {"px": px, "py": py, "axis_type": axis_type}

    def compute(self, input_value, params):
        px = params.get("px", 5)
        py = params.get("py", 3)
        axis_type = params.get("axis_type", 0)

        if axis_type == 0:
            # Reflection across x-axis: (px, py) -> (px, -py)
            rx, ry = px, -py
            axis_name = "x-axis"
        elif axis_type == 1:
            # Reflection across y-axis: (px, py) -> (-px, py)
            rx, ry = -px, py
            axis_name = "y-axis"
        else:
            # Reflection across y=x line: (px, py) -> (py, px)
            rx, ry = py, px
            axis_name = "line y=x"

        # Distance from original point to reflected point
        distance = math.sqrt((px - rx)**2 + (py - ry)**2)

        # Return distance as integer
        result = round(distance)

        return MethodResult(
            value=result,
            description=f"Reflection of ({px}, {py}) across {axis_name}: ({rx}, {ry}), distance={distance:.2f}",
            metadata={"original": (px, py), "reflected": (rx, ry), "axis": axis_name, "distance": distance}
        )

    def can_invert(self):
        return False


@register_technique
class SymmetryRotation(MethodBlock):
    """
    Compute rotational symmetry properties of regular polygons and patterns.
    For an n-sided regular polygon, the smallest rotation angle that maps it to itself
    is 360/n degrees. This method computes rotation angles and counts rotational
    symmetries, which is fundamental in analyzing symmetric patterns.
    """
    def __init__(self):
        super().__init__()
        self.name = "symmetry_rotation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "symmetry", "rotation", "polygon"]

    def validate_params(self, params, prev_value=None):
        """Validate that n is a positive integer >= 3."""
        n = params.get("n")
        return isinstance(n, int) and n >= 3

    def generate_parameters(self, input_value=None):
        """Generate number of sides for regular polygon."""
        n = input_value if input_value is not None else random.randint(3, 12)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 6)

        # Basic rotation angle for n-sided regular polygon
        rotation_angle = 360 / n

        # Number of rotational symmetries (including identity) equals n
        num_symmetries = n

        # If we compute the total symmetry value (counting all symmetry operations)
        # For a regular n-gon with all rotational symmetries
        total_symmetry_value = round(rotation_angle * num_symmetries / 10)

        return MethodResult(
            value=total_symmetry_value,
            description=f"Regular {n}-gon: rotation angle={rotation_angle:.1f}°, {num_symmetries} rotational symmetries, total value={total_symmetry_value}",
            metadata={"n": n, "rotation_angle": rotation_angle, "num_symmetries": num_symmetries}
        )

    def can_invert(self):
        return False


@register_technique
class ReflectionPrinciple(MethodBlock):
    """Reflection principle for path counting."""
    def __init__(self):
        super().__init__()
        self.name = "reflection_principle"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "combinatorics"]

    def generate_parameters(self, input_value=None):
        m = random.randint(2, 6)
        n = random.randint(2, 6)
        return {"m": m, "n": n}

    def compute(self, input_value, params):
        m = params.get("m", 3)
        n = params.get("n", 4)
        total = math.comb(m + n, m)
        return MethodResult(
            value=total,
            description=f"Lattice paths: C({m+n},{m}) = {total}",
            params=params
        )

    def can_invert(self):
        return False


