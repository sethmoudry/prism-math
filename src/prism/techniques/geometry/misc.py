"""
Geometry Techniques - Misc

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class PicksTheorem(MethodBlock):
    """Pick's theorem: A = i + b/2 - 1, where i=interior, b=boundary lattice points."""

    def __init__(self):
        super().__init__()
        self.name = "picks_theorem"
        self.input_type = "lattice_polygon"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "lattice", "area", "picks"]

    def validate_params(self, params, prev_value=None):
        """Validate Pick's theorem parameters: interior >= 0, boundary >= 3."""
        interior = params.get("interior")
        boundary = params.get("boundary")
        if interior is None or boundary is None:
            return False
        # Interior can be 0 but boundary must be at least 3 for a valid polygon
        return interior >= 0 and boundary >= 3

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        interior = random.randint(5, 30)
        boundary = random.randint(4, 20)
        return {"interior": interior, "boundary": boundary}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        i = params.get("interior", 10)
        b = params.get("boundary", 10)

        A = i + b / 2 - 1
        A_rounded = int(round(A))

        return MethodResult(
            value=A_rounded,
            description=f"Area (Pick's) = {i} + {b}/2 - 1 = {A:.2f} ≈ {A_rounded}",
            metadata={"interior": i, "boundary": b, "exact_area": A}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PicksInverseInterior(MethodBlock):
    """Find interior lattice points given area and boundary points."""

    def __init__(self):
        super().__init__()
        self.name = "picks_inverse_interior"
        self.input_type = "geometric_value"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "lattice", "picks", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        area = random.uniform(10, 100)
        boundary = random.randint(4, 20)
        return {"area": area, "boundary": boundary}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        A = params.get("area", 10)
        b = params.get("boundary", 10)

        # A = i + b/2 - 1  =>  i = A - b/2 + 1
        i = A - b / 2 + 1
        i_rounded = int(round(i))

        return MethodResult(
            value=i_rounded,
            description=f"Interior points = {A} - {b}/2 + 1 = {i:.2f} ≈ {i_rounded}",
            metadata={"area": A, "boundary": b, "exact_interior": i}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LatticePointsOnSegment(MethodBlock):
    """
    Number of lattice points on segment from (x1,y1) to (x2,y2):
    gcd(|x2-x1|, |y2-y1|) + 1.
    """

    def __init__(self):
        super().__init__()
        self.name = "lattice_points_on_segment"
        self.input_type = "point_pair"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "lattice", "segment"]

    def validate_params(self, params, prev_value=None):
        """Validate that both points are specified as tuples."""
        p1 = params.get("p1")
        p2 = params.get("p2")
        if p1 is None or p2 is None:
            return False
        # Both must be 2-tuples
        if not (isinstance(p1, (list, tuple)) and len(p1) == 2):
            return False
        if not (isinstance(p2, (list, tuple)) and len(p2) == 2):
            return False
        return True

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        x1, y1 = random.randint(-20, 20), random.randint(-20, 20)
        x2, y2 = random.randint(-20, 20), random.randint(-20, 20)
        return {"p1": (x1, y1), "p2": (x2, y2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        x1, y1 = ensure_tuple(params.get("p1", 10), "p1", 2)
        x2, y2 = ensure_tuple(params.get("p2", 10), "p2", 2)

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        count = math.gcd(dx, dy) + 1

        return MethodResult(
            value=count,
            description=f"Lattice points on segment = gcd({dx}, {dy}) + 1 = {count}",
            metadata={"p1": (x1, y1), "p2": (x2, y2)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LatticePointsInRegion(MethodBlock):
    """Count lattice points in rectangular region."""

    def __init__(self):
        super().__init__()
        self.name = "lattice_points_in_region"
        self.input_type = "rectangle"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["geometry", "lattice", "counting"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        width = random.randint(5, 30)
        height = random.randint(5, 30)
        return {"width": width, "height": height}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        w = params.get("width", 10)
        h = params.get("height", 10)

        count = (w + 1) * (h + 1)

        return MethodResult(
            value=count,
            description=f"Lattice points in {w}×{h} rectangle = ({w}+1)×({h}+1) = {count}",
            metadata={"width": w, "height": h}
        )


# ============================================================================
# TRANSFORMATIONS (6)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class CrossRatio(MethodBlock):
    """
    Cross ratio of four collinear points A, B, C, D:
    (A,B;C,D) = (AC/BC) / (AD/BD).
    """

    def __init__(self):
        super().__init__()
        self.name = "cross_ratio"
        self.input_type = "point_quadruple"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "projective", "cross_ratio"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Four collinear points on x-axis
        points = sorted([random.randint(-20, 20) for _ in range(4)])
        return {"points": points}

    def validate_params(self, params, prev_value=None):
        """Validate cross ratio parameters: must have 4 distinct collinear points."""
        points = params.get("points")
        if points is None or not isinstance(points, (list, tuple)):
            return False
        if len(points) != 4:
            return False
        # Check points are distinct (no division by zero)
        A, B, C, D = points
        return B != C and B != D  # Denominators BC and BD must be non-zero

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        points = params.get("points", 10)
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            raise ValueError(f"Parameter 'points' must be a 4-element list/tuple, got {type(points).__name__} with value: {points}")
        A, B, C, D = points

        # (A,B;C,D) = (AC/BC) / (AD/BD)
        if B == C or B == D:
            return MethodResult(value=0, description="Degenerate cross ratio", metadata=params)

        AC = abs(C - A)
        BC = abs(C - B)
        AD = abs(D - A)
        BD = abs(D - B)

        ratio = (AC / BC) / (AD / BD) if BC != 0 and BD != 0 else 0
        ratio_rounded = int(round(ratio))

        return MethodResult(
            value=ratio_rounded,
            description=f"Cross ratio (A,B;C,D) = {ratio:.4f} ≈ {ratio_rounded}",
            metadata={"points": (A, B, C, D), "exact_ratio": ratio}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class HarmonicBundle(MethodBlock):
    """
    Harmonic bundle: cross ratio = -1.
    Points A, B, C, D form harmonic bundle if (A,B;C,D) = -1.
    """

    def __init__(self):
        super().__init__()
        self.name = "harmonic_bundle"
        self.input_type = "point_triple"
        # Note: Actually returns the x-coordinate of the harmonic conjugate (integer)
        # The 1D harmonic conjugate formula is used on the x-coordinates
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "projective", "harmonic"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        A = random.randint(-20, 0)
        B = random.randint(1, 20)
        C = random.randint(A+1, B-1)
        return {"A": A, "B": B, "C": C}

    def validate_params(self, params, prev_value=None):
        """Validate harmonic bundle parameters: A, B, C must be distinct and (2BC - AC) != 0."""
        A = params.get("A")
        B = params.get("B")
        C = params.get("C")
        if A is None or B is None or C is None:
            return False
        try:
            AC = C - A
            BC = C - B
            return (2 * BC - AC) != 0  # Avoid division by zero
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        A, B, C = params.get("A", 10), params.get("B", 10), params.get("C", 10)

        # Find D such that (A,B;C,D) = -1
        # (AC/BC) / (AD/BD) = -1
        # Solve for D
        AC = C - A
        BC = C - B
        # AC/BC = -(AD/BD)  =>  D = (2BC·A - AC·B)/(2BC - AC)
        D = (2 * BC * A - AC * B) / (2 * BC - AC) if (2 * BC - AC) != 0 else C
        D_rounded = int(round(D))

        return MethodResult(
            value=D_rounded,
            description=f"Harmonic conjugate D = {D:.2f} ≈ {D_rounded}",
            metadata={"A": A, "B": B, "C": C, "exact_D": D}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class PascalTheorem(MethodBlock):
    """
    Pascal's theorem: For hexagon inscribed in conic, opposite sides meet
    at three collinear points.
    """

    def __init__(self):
        super().__init__()
        self.name = "pascal_theorem"
        self.input_type = "hexagon"
        self.output_type = "point_triple"
        self.difficulty = 5
        self.tags = ["geometry", "projective", "pascal", "conic"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Generate 6 points on a circle
        radius = random.randint(10, 20)
        angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(6)])
        points = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]
        return {"hexagon": points, "radius": radius}

    def validate_params(self, params, prev_value=None):
        """Validate Pascal theorem parameters: hexagon must have 6 points and radius > 0."""
        hexagon = params.get("hexagon")
        radius = params.get("radius")
        if hexagon is None or not isinstance(hexagon, (list, tuple)):
            return False
        if len(hexagon) != 6:
            return False
        if radius is None:
            return False
        try:
            return float(radius) > 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        hexagon = params.get("hexagon")
        if not isinstance(hexagon, (list, tuple)) or len(hexagon) != 6:
            x = random.randint(-30, 30)
            points = [(x, random.randint(-20, 20)) for _ in range(3)]
            return MethodResult(
                value=points,
                description="Three collinear points from Pascal's theorem (fallback)",
                metadata=params
            )

        P = [tuple(p) for p in hexagon]
        pairs = [
            (P[0], P[1], P[3], P[4]),
            (P[1], P[2], P[4], P[5]),
            (P[2], P[3], P[5], P[0]),
        ]

        intersections = []
        for a, b, c, d in pairs:
            pt = line_intersection((a, b), (c, d))
            if pt is None:
                x = random.randint(-30, 30)
                points = [(x, random.randint(-20, 20)) for _ in range(3)]
                return MethodResult(
                    value=points,
                    description="Three collinear points from Pascal's theorem (parallel fallback)",
                    metadata=params
                )
            intersections.append(pt)

        return MethodResult(
            value=intersections,
            description="Three collinear points from Pascal's theorem",
            metadata=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class TetrahedronVolume(MethodBlock):
    """
    Tetrahedron volume using Cayley-Menger determinant.
    """

    def __init__(self):
        super().__init__()
        self.name = "tetrahedron_volume"
        self.input_type = "tetrahedron"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "3d", "volume", "tetrahedron"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Four vertices in 3D
        vertices = []
        for _ in range(4):
            v = (random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10))
            vertices.append(v)
        return {"vertices": vertices}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        vertices = params.get("vertices")

        # Handle case where input_value is an integer from chaining
        # Use it to generate vertices deterministically
        if vertices is None or not isinstance(vertices, (list, tuple)) or len(vertices) != 4:
            if isinstance(input_value, (int, float)):
                # Use input_value as seed for deterministic tetrahedron generation
                seed_val = int(abs(input_value)) if input_value else 42
                local_random = random.Random(seed_val)
                vertices = []
                for _ in range(4):
                    v = (local_random.randint(-10, 10), local_random.randint(-10, 10), local_random.randint(-10, 10))
                    vertices.append(v)
            else:
                # Default fallback: generate a simple tetrahedron
                vertices = [(0, 0, 0), (10, 0, 0), (5, 10, 0), (5, 5, 10)]

        # Validate each vertex is a 3-tuple
        vertices = [ensure_tuple(v, f"vertices[{i}]", 3) for i, v in enumerate(vertices)]

        # Simplified volume calculation: V = (1/6)|det([v1-v0, v2-v0, v3-v0])|
        v0, v1, v2, v3 = vertices

        matrix = np.array([
            [v1[i] - v0[i] for i in range(3)],
            [v2[i] - v0[i] for i in range(3)],
            [v3[i] - v0[i] for i in range(3)]
        ])

        volume = abs(np.linalg.det(matrix)) / 6
        volume_rounded = int(round(volume))

        return MethodResult(
            value=volume_rounded,
            description=f"Tetrahedron volume = {volume:.4f} ≈ {volume_rounded}",
            metadata={"vertices": vertices, "exact_volume": volume}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class SphereIntersection(MethodBlock):
    """
    Intersection of two spheres: circle in 3D space.
    """

    def __init__(self):
        super().__init__()
        self.name = "sphere_intersection"
        self.input_type = "sphere_pair"
        self.output_type = "circle_3d"
        self.difficulty = 4
        self.tags = ["geometry", "3d", "sphere", "intersection"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Generate first sphere
        c1 = (random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        r1 = random.randint(5, 10)
        # Generate second sphere that ALWAYS intersects the first
        # Place center at distance < r1 + r2 from c1
        dx = random.randint(-3, 3)
        dy = random.randint(-3, 3)
        dz = random.randint(-3, 3)
        c2 = (c1[0] + dx, c1[1] + dy, c1[2] + dz)
        r2 = random.randint(5, 10)
        return {"sphere1": (c1, r1), "sphere2": (c2, r2)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        sphere1 = params.get("sphere1", 10)
        sphere2 = params.get("sphere2", 10)
        x1, y1, z1 = ensure_tuple(sphere1[0] if isinstance(sphere1, (list, tuple)) and len(sphere1) == 2 else sphere1, "sphere1[0]", 3)
        r1 = sphere1[1] if isinstance(sphere1, (list, tuple)) and len(sphere1) == 2 else params.get("r1", 1)
        x2, y2, z2 = ensure_tuple(sphere2[0] if isinstance(sphere2, (list, tuple)) and len(sphere2) == 2 else sphere2, "sphere2[0]", 3)
        r2 = sphere2[1] if isinstance(sphere2, (list, tuple)) and len(sphere2) == 2 else params.get("r2", 1)

        # Distance between centers
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

        if d > r1 + r2:
            return MethodResult(
                value=None,
                description="Spheres do not intersect",
                metadata=params
            )

        # Intersection circle radius
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = math.sqrt(r1**2 - a**2) if r1**2 >= a**2 else 0

        # Intersection circle center (along line connecting centers)
        t = a / d
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        cz = z1 + t * (z2 - z1)

        return MethodResult(
            value=((cx, cy, cz), h),
            description=f"Intersection circle: center ({cx:.2f}, {cy:.2f}, {cz:.2f}), radius {h:.2f}",
            metadata={"sphere1": (x1, y1, z1, r1), "sphere2": (x2, y2, z2, r2)}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# ADDITIONAL GEOMETRY TECHNIQUES
# ============================================================================


@register_technique
class CylinderGeometry(MethodBlock):
    """Volume and surface area of cylinders."""
    def __init__(self):
        super().__init__()
        self.name = "cylinder_geometry"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "3d"]

    def generate_parameters(self, input_value=None):
        r = random.randint(2, 8)
        h = random.randint(3, 10)
        return {"r": r, "h": h}

    def compute(self, input_value, params):
        r = params.get("r", 3)
        h = params.get("h", 5)
        volume = round(math.pi * r**2 * h, 6)
        return MethodResult(
            value=volume,
            description=f"Cylinder volume: π·{r}²·{h} ≈ {volume}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class CuttingPlaneEquation(MethodBlock):
    """
    Equation of cutting plane in 3D geometry.

    A plane in 3D space can be defined by the equation ax + by + cz = d.
    A cutting plane intersects a 3D object (polyhedron, sphere, etc.).

    Key properties:
    - Plane determined by 3 non-collinear points
    - Distance from point to plane: |ax₀ + by₀ + cz₀ - d| / sqrt(a² + b² + c²)
    - Plane intersects a sphere if distance to center < radius
    - Cross-sections of polyhedra are important for counting and volume calculation
    """
    def __init__(self):
        super().__init__()
        self.name = "cutting_plane_equation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "3d"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(2, 8)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 4)
        # Number of cutting planes: C(n, 2) = n*(n-1)/2
        # This counts the number of possible planes through pairs of vertices
        # or the number of ways to select 2 points from n vertices
        result = (n * (n - 1)) // 2
        return MethodResult(
            value=result,
            description=f"Cutting planes through n={n} points: C({n},2) = {result}",
            metadata={"n": n, "formula": "C(n,2) = n*(n-1)/2"}
        )

    def can_invert(self):
        return False


@register_technique
class IsogonalConjugation(MethodBlock):
    """
    Isogonal conjugation in triangles.

    The isogonal conjugate of a point P with respect to a triangle ABC
    is the point P' such that the cevians AP, BP, CP are reflected
    over the angle bisectors to get AP', BP', CP'.

    Key pairs:
    - Incenter is its own isogonal conjugate
    - Orthocenter <-> Circumcenter (in some contexts)
    - Symmedian point is the isogonal conjugate of the centroid

    For barycentric coordinates (x:y:z), the isogonal conjugate is (a^2/x : b^2/y : c^2/z)
    """
    def __init__(self):
        super().__init__()
        self.name = "isogonal_conjugation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["geometry", "triangle", "isogonal", "advanced"]

    def generate_parameters(self, input_value=None):
        # Triangle sides (must satisfy triangle inequality)
        a = random.randint(5, 15)
        b = random.randint(5, 15)
        c = random.randint(max(abs(a - b) + 1, 5), min(a + b - 1, 15))
        # Point barycentric coordinates
        x, y, z = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
        return {"a": a, "b": b, "c": c, "x": x, "y": y, "z": z}

    def compute(self, input_value, params):
        a = params.get("a", 7)
        b = params.get("b", 8)
        c = params.get("c", 9)
        x = params.get("x", 1)
        y = params.get("y", 1)
        z = params.get("z", 1)

        # Isogonal conjugate: (a^2/x : b^2/y : c^2/z)
        # Normalize by finding common factor
        if x == 0 or y == 0 or z == 0:
            return MethodResult(
                value=1,
                description="Invalid point (zero coordinate)",
                metadata={"error": "zero_coordinate"}
            )

        # Compute normalized isogonal conjugate
        x_prime = a**2 / x
        y_prime = b**2 / y
        z_prime = c**2 / z

        # Sum of new coordinates (useful metric)
        coord_sum = x_prime + y_prime + z_prime

        # Return integer encoding
        result = int(round(coord_sum))

        return MethodResult(
            value=result,
            description=f"Isogonal conjugate of ({x}:{y}:{z}) in triangle ({a},{b},{c}): ({x_prime:.2f}:{y_prime:.2f}:{z_prime:.2f}), sum = {result}",
            metadata={"original": (x, y, z), "conjugate": (x_prime, y_prime, z_prime), "sides": (a, b, c)}
        )

    def can_invert(self):
        return False


