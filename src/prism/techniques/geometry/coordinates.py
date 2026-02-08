"""
Geometry Techniques - Coordinates

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class Distance(MethodBlock):
    """Euclidean distance between two points: d = √((x2-x1)² + (y2-y1)²)."""

    def __init__(self):
        super().__init__()
        self.name = "distance"
        self.input_type = "point_pair"
        self.output_type = "geometric_value"
        self.difficulty = 1
        self.tags = ["geometry", "coordinate", "distance"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        x1, y1 = random.randint(-50, 50), random.randint(-50, 50)
        x2, y2 = random.randint(-50, 50), random.randint(-50, 50)
        # Support both distance(p1, p2) with tuples and distance(x1, y1, x2, y2) with 4 coords
        # x1, y1, x2, y2 first for positional arg mapping
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "p1": (x1, y1), "p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Check if 4 separate coords are passed: distance(x1, y1, x2, y2)
        if "x1" in params and "y1" in params and "x2" in params and "y2" in params:
            x1 = params.get("x1", 10)
            y1 = params.get("y1", 10)
            x2 = params.get("x2", 10)
            y2 = params.get("y2", 10)
            d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            return MethodResult(
                value=d,
                description=f"Distance from ({x1}, {y1}) to ({x2}, {y2}) = {d:.4f}",
                metadata={"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )

        p1 = params.get("p1")
        p2 = params.get("p2")

        # Handle None or invalid inputs
        if p1 is None or p2 is None:
            return MethodResult(
                value=0,
                description="Invalid points provided (None)",
                metadata={"error": "null_point"}
            )

        # Handle case where p1/p2 are scalars (shouldn't happen with updated generate_parameters)
        if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
            # Treat as 1D distance
            d = abs(p2 - p1)
            return MethodResult(
                value=d,
                description=f"Distance from {p1} to {p2} = {d}",
                metadata={"p1": p1, "p2": p2}
            )

        # Handle different formats - could be 2D or 3D points
        if len(p1) == 2:
            x1, y1 = p1
            x2, y2 = p2
            d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            point_str_1 = f"({x1}, {y1})"
            point_str_2 = f"({x2}, {y2})"
        elif len(p1) == 3:
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            d = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            point_str_1 = f"({x1}, {y1}, {z1})"
            point_str_2 = f"({x2}, {y2}, {z2})"
        else:
            # General n-dimensional case
            d = math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2)))
            point_str_1 = f"{p1}"
            point_str_2 = f"{p2}"

        d_rounded = int(round(d))

        return MethodResult(
            value=d_rounded,
            description=f"Distance from {point_str_1} to {point_str_2} = {d:.4f} ≈ {d_rounded}",
            metadata={"p1": p1, "p2": p2, "exact_distance": d}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class DistanceInversePoint(MethodBlock):
    """Find point at given distance from another point."""

    def __init__(self):
        super().__init__()
        self.name = "distance_inverse_point"
        self.input_type = "geometric_value"
        self.output_type = "point"
        self.difficulty = 2
        self.tags = ["geometry", "coordinate", "distance", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        x1, y1 = random.randint(-50, 50), random.randint(-50, 50)
        distance = random.uniform(5, 50)
        angle = random.uniform(0, 2 * math.pi)
        return {"p1": (x1, y1), "distance": distance, "angle": angle}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", 10), "p1", 2)
        distance = params.get("distance", 10)
        angle = params.get("angle", 10)

        x2 = x1 + distance * math.cos(angle)
        y2 = y1 + distance * math.sin(angle)

        return MethodResult(
            value=(x2, y2),
            description=f"Point at distance {distance:.2f} from ({x1}, {y1}) is ({x2:.2f}, {y2:.2f})",
            metadata={"origin": (x1, y1), "distance": distance, "angle": angle}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CoordinateGeometry(MethodBlock):
    """
    Basic coordinate geometry calculations.

    Supports multiple operations:
    - 'distance': Euclidean distance between two points
    - 'midpoint': Midpoint coordinates (returns x+y as integer)
    - 'slope': Slope of line through two points (returns integer approximation)
    - 'line_length': Length of line segment
    - 'triangle_area': Area of triangle from 3 vertices (shoelace formula)

    Parameters:
        operation: 'distance', 'midpoint', 'slope', 'line_length', 'triangle_area'
        p1: first point as (x, y) or [x, y]
        p2: second point as (x, y) or [x, y]
        p3: third point (for triangle_area)
        x1, y1, x2, y2: alternative point specification
    """
    def __init__(self):
        super().__init__()
        self.name = "coordinate_geometry"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "coordinate", "distance"]

    def generate_parameters(self, input_value=None):
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        return {
            "operation": "distance",
            "p1": [x1, y1],
            "p2": [x2, y2]
        }

    def _get_point(self, params, point_name, x_name, y_name, default_x=0, default_y=0):
        """Extract point from params, supporting both (x,y) tuple and separate x,y params."""
        if point_name in params:
            p = params[point_name]
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return float(p[0]), float(p[1])
            elif isinstance(p, (int, float)):
                return float(p), 0.0
        # Fallback to separate x, y params
        x = params.get(x_name, default_x)
        y = params.get(y_name, default_y)
        return float(x), float(y)

    def compute(self, input_value, params):
        operation = params.get("operation", "distance")

        # Extract points
        x1, y1 = self._get_point(params, "p1", "x1", "y1", 0, 0)
        x2, y2 = self._get_point(params, "p2", "x2", "y2", 1, 1)

        if operation == "distance" or operation == "line_length":
            # Euclidean distance
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            result = int(round(dist))
            description = f"Distance from ({x1},{y1}) to ({x2},{y2}) = {dist:.2f} -> {result}"
            metadata = {"p1": (x1, y1), "p2": (x2, y2), "exact_distance": dist}

        elif operation == "midpoint":
            # Midpoint: return sum of coordinates as integer
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            result = int(round(mx + my))
            description = f"Midpoint of ({x1},{y1})-({x2},{y2}) = ({mx},{my}), sum={result}"
            metadata = {"p1": (x1, y1), "p2": (x2, y2), "midpoint": (mx, my)}

        elif operation == "slope":
            # Slope of line
            if abs(x2 - x1) < 1e-10:
                # Vertical line - undefined slope
                result = float('inf')
                description = f"Slope undefined (vertical line)"
            else:
                slope = (y2 - y1) / (x2 - x1)
                # Return slope * 10 to preserve some precision
                result = int(round(slope * 10))
                description = f"Slope from ({x1},{y1}) to ({x2},{y2}) = {slope:.2f}, encoded as {result}"
            metadata = {"p1": (x1, y1), "p2": (x2, y2)}

        elif operation == "triangle_area":
            # Get third point
            x3, y3 = self._get_point(params, "p3", "x3", "y3", 2, 0)
            # Shoelace formula
            area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)
            result = int(round(area))
            description = f"Triangle area with vertices ({x1},{y1}), ({x2},{y2}), ({x3},{y3}) = {area:.2f}"
            metadata = {"vertices": [(x1, y1), (x2, y2), (x3, y3)], "exact_area": area}

        else:
            # Default to distance
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            result = int(round(dist))
            description = f"Coordinate geometry ({operation}): distance = {result}"
            metadata = {"operation": operation}

        return MethodResult(
            value=result,
            description=description,
            metadata=metadata
        )

    def can_invert(self):
        return False


@register_technique
class CoordinateGeometryArea(MethodBlock):
    """Compute polygon area using coordinates (Shoelace formula)."""
    def __init__(self):
        super().__init__()
        self.name = "coordinate_geometry_area"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "coordinate", "area"]

    def generate_parameters(self, input_value=None):
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}

    def compute(self, input_value, params):
        x1, y1 = params.get("x1", 10), params.get("y1", 10)
        x2, y2 = params.get("x2", 10), params.get("y2", 10)
        x3, y3 = params.get("x3", 10), params.get("y3", 10)
        area = abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2)
        result = int(round(area * 2))
        return MethodResult(
            value=result,
            description=f"Triangle area = {area}",
            metadata={"vertices": [(x1,y1), (x2,y2), (x3,y3)]}
        )

    def can_invert(self):
        return False


@register_technique
class CoordinateGeometrySynthetic(MethodBlock):
    """
    Synthetic coordinate geometry combining multiple operations.

    Combines coordinate geometry with classical geometry theorems:
    - Find circumcenter of triangle using coordinates
    - Find orthocenter using coordinates
    - Find centroid and compute distances
    - Verify collinearity of derived points
    """
    def __init__(self):
        super().__init__()
        self.name = "coordinate_geometry_synthetic"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "coordinate", "synthetic"]

    def generate_parameters(self, input_value=None):
        # Generate triangle vertices
        x1, y1 = random.randint(0, 10), random.randint(0, 10)
        x2, y2 = random.randint(0, 10), random.randint(0, 10)
        x3, y3 = random.randint(0, 10), random.randint(0, 10)
        operation = random.choice(["circumcenter", "centroid", "orthocenter", "euler_line"])
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3, "operation": operation}

    def compute(self, input_value, params):
        x1, y1 = params.get("x1", 0), params.get("y1", 0)
        x2, y2 = params.get("x2", 6), params.get("y2", 0)
        x3, y3 = params.get("x3", 3), params.get("y3", 4)
        operation = params.get("operation", "centroid")

        if operation == "centroid":
            # Centroid G = ((x1+x2+x3)/3, (y1+y2+y3)/3)
            gx = (x1 + x2 + x3) / 3
            gy = (y1 + y2 + y3) / 3
            # Return sum of coordinates scaled
            result = int(round(gx * 3 + gy * 3))
            description = f"Centroid at ({gx:.2f}, {gy:.2f}), encoded as {result}"
        elif operation == "circumcenter":
            # Circumcenter: intersection of perpendicular bisectors
            # Using the determinant formula
            d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(d) < 1e-10:
                result = 0
                description = "Degenerate triangle (collinear points)"
            else:
                ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
                uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
                result = int(round(abs(ux) + abs(uy)))
                description = f"Circumcenter at ({ux:.2f}, {uy:.2f}), sum = {result}"
        elif operation == "orthocenter":
            # Orthocenter H = A + B + C - 2*circumcenter (for triangle with circumcenter at origin)
            # Simpler: H = (x1 + x2 + x3 - 2*ux, y1 + y2 + y3 - 2*uy)
            d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(d) < 1e-10:
                result = 0
                description = "Degenerate triangle"
            else:
                ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
                uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
                hx = x1 + x2 + x3 - 2 * ux
                hy = y1 + y2 + y3 - 2 * uy
                result = int(round(abs(hx) + abs(hy)))
                description = f"Orthocenter at ({hx:.2f}, {hy:.2f}), sum = {result}"
        else:  # euler_line
            # Distance between centroid and circumcenter
            d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(d) < 1e-10:
                result = 0
                description = "Degenerate triangle"
            else:
                gx, gy = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
                ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
                uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
                dist = math.sqrt((gx - ux)**2 + (gy - uy)**2)
                result = int(round(dist * 10))  # Scale for precision
                description = f"Euler line G-O distance = {dist:.2f}, encoded as {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "vertices": [(x1, y1), (x2, y2), (x3, y3)]}
        )

    def can_invert(self):
        return False


@register_technique
class DistanceFormula(MethodBlock):
    """
    Compute Euclidean distance between two points.

    Supports 2D and 3D distances:
    - 2D: sqrt((x2-x1)^2 + (y2-y1)^2)
    - 3D: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)

    Parameters:
        p1: first point as (x, y) or (x, y, z)
        p2: second point as (x, y) or (x, y, z)
        x1, y1, z1, x2, y2, z2: alternative separate coordinate specification
        squared: if True, return distance squared (integer)
    """
    def __init__(self):
        super().__init__()
        self.name = "distance_formula"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "distance", "coordinate"]

    def generate_parameters(self, input_value=None):
        # Generate points that give a nice integer distance (Pythagorean triples)
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        dx, dy, dist = random.choice(triples)
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = x1 + dx, y1 + dy
        return {
            "p1": [x1, y1],
            "p2": [x2, y2],
            "squared": False
        }

    def compute(self, input_value, params):
        squared = params.get("squared", False)

        # Extract points - support both p1/p2 format and x1,y1,x2,y2 format
        p1 = params.get("p1")
        p2 = params.get("p2")

        if p1 is not None and p2 is not None:
            # Convert to lists if needed
            if isinstance(p1, (list, tuple)):
                coords1 = list(p1)
            else:
                coords1 = [p1, 0]

            if isinstance(p2, (list, tuple)):
                coords2 = list(p2)
            else:
                coords2 = [p2, 0]
        else:
            # Fallback to x1, y1, z1, x2, y2, z2 format
            x1 = params.get("x1", 0)
            y1 = params.get("y1", 0)
            z1 = params.get("z1", None)
            x2 = params.get("x2", 1)
            y2 = params.get("y2", 1)
            z2 = params.get("z2", None)

            if z1 is not None and z2 is not None:
                coords1 = [x1, y1, z1]
                coords2 = [x2, y2, z2]
            else:
                coords1 = [x1, y1]
                coords2 = [x2, y2]

        # Ensure same dimensionality
        max_dim = max(len(coords1), len(coords2))
        while len(coords1) < max_dim:
            coords1.append(0)
        while len(coords2) < max_dim:
            coords2.append(0)

        # Compute squared distance
        dist_sq = sum((float(c2) - float(c1))**2 for c1, c2 in zip(coords1, coords2))

        if squared:
            result = int(round(dist_sq))
            description = f"Distance squared from {coords1} to {coords2} = {result}"
        else:
            dist = math.sqrt(dist_sq)
            result = int(round(dist))
            description = f"Distance from {coords1} to {coords2} = {dist:.4f} -> {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={
                "p1": coords1,
                "p2": coords2,
                "distance_squared": dist_sq,
                "exact_distance": math.sqrt(dist_sq)
            }
        )

    def can_invert(self):
        return False


@register_technique
class MinimizeDistancePlane(MethodBlock):
    """Minimize distance from point to plane."""
    def __init__(self):
        super().__init__()
        self.name = "minimize_distance_plane"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "3d", "optimization"]

    def generate_parameters(self, input_value=None):
        a, b, c = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
        d = random.randint(5, 20)
        x0, y0, z0 = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
        return {"a": a, "b": b, "c": c, "d": d, "x0": x0, "y0": y0, "z0": z0}

    def compute(self, input_value, params):
        a = params.get("a", 1)
        b = params.get("b", 1)
        c = params.get("c", 1)
        d = params.get("d", 10)
        x0 = params.get("x0", 5)
        y0 = params.get("y0", 5)
        z0 = params.get("z0", 5)
        denom = max(int(math.sqrt(a**2 + b**2 + c**2)), 1)
        dist = abs(a*x0 + b*y0 + c*z0 - d) // denom
        return MethodResult(
            value=int(dist),
            description=f"Distance to plane: {int(dist)}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class PlaneGeometry(MethodBlock):
    """Compute properties of planar geometric figures."""
    def __init__(self):
        super().__init__()
        self.name = "plane_geometry"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "plane"]

    def generate_parameters(self, input_value=None):
        n = random.randint(3, 8)
        side = random.randint(2, 10)
        return {"n": n, "side": side}

    def compute(self, input_value, params):
        n = params.get("n", 10)
        angle_sum = (n - 2) * 180
        return MethodResult(
            value=angle_sum,
            description=f"Interior angle sum of {n}-gon: {angle_sum}°",
            metadata={"n": n, "angle_sum": angle_sum}
        )

    def can_invert(self):
        return False


@register_technique
class PowerPointCoordinate(MethodBlock):
    """
    Power of a point with respect to a circle using coordinates.

    Power of point P(x, y) with respect to circle centered at (h, k) with radius r:
    pow(P) = (x - h)^2 + (y - k)^2 - r^2

    Properties:
    - pow(P) > 0 if P is outside the circle
    - pow(P) = 0 if P is on the circle
    - pow(P) < 0 if P is inside the circle
    - For external point: pow(P) = (tangent length)^2
    """
    def __init__(self):
        super().__init__()
        self.name = "power_point_coordinate"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "circle", "power", "coordinate"]

    def generate_parameters(self, input_value=None):
        # Circle center and radius
        h, k = random.randint(-5, 5), random.randint(-5, 5)
        r = random.randint(3, 10)
        # Point coordinates
        px, py = random.randint(-10, 10), random.randint(-10, 10)
        return {"h": h, "k": k, "r": r, "px": px, "py": py}

    def compute(self, input_value, params):
        h = params.get("h", 0)
        k = params.get("k", 0)
        r = params.get("r", 5)
        px = params.get("px", 8)
        py = params.get("py", 0)

        # Power of point = distance^2 - radius^2
        dist_sq = (px - h)**2 + (py - k)**2
        power = dist_sq - r**2

        # Absolute value for integer result
        result = abs(int(power))

        if power > 0:
            tangent_length = math.sqrt(power)
            description = f"Power of ({px},{py}) wrt circle at ({h},{k}) radius {r}: {power} (tangent length = {tangent_length:.2f})"
            position = "outside"
        elif power < 0:
            description = f"Power of ({px},{py}) wrt circle at ({h},{k}) radius {r}: {power} (point inside)"
            position = "inside"
        else:
            description = f"Point ({px},{py}) is on the circle centered at ({h},{k}) with radius {r}"
            position = "on_circle"

        return MethodResult(
            value=result,
            description=description,
            metadata={"power": power, "position": position, "circle": (h, k, r), "point": (px, py)}
        )

    def can_invert(self):
        return False


@register_technique
class SlopeCollinearity(MethodBlock):
    """Check collinearity using slopes."""
    def __init__(self):
        super().__init__()
        self.name = "slope_collinearity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "coordinate", "collinearity"]

    def generate_parameters(self, input_value=None):
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}

    def compute(self, input_value, params):
        x1, y1 = params.get("x1", 10), params.get("y1", 10)
        x2, y2 = params.get("x2", 10), params.get("y2", 10)
        x3, y3 = params.get("x3", 10), params.get("y3", 10)
        det = (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
        result = 0 if det == 0 else 1
        return MethodResult(
            value=result,
            description=f"Points are {'collinear' if result == 0 else 'not collinear'}",
            metadata={"collinear": result == 0}
        )

    def can_invert(self):
        return False


