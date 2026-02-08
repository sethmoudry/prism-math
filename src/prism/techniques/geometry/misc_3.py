"""
Geometry Techniques - Misc (Part 3)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class VolumeCalculation(MethodBlock):
    """
    Calculate volumes of various 3D geometric shapes including rectangular prisms
    (boxes), cylinders, spheres, cones, and pyramids. Uses standard formulas:
    - Box: V = l * w * h
    - Cylinder: V = π * r² * h
    - Sphere: V = (4/3) * π * r³
    - Cone: V = (1/3) * π * r² * h
    - Pyramid: V = (1/3) * base_area * h
    """
    def __init__(self):
        super().__init__()
        self.name = "volume_calculation"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "3d", "volume", "shape"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for various 3D shapes."""
        shape_type = random.randint(0, 4)  # 0=box, 1=cylinder, 2=sphere, 3=cone, 4=pyramid

        if shape_type == 0:  # Box
            l = input_value if input_value is not None else random.randint(3, 20)
            w = random.randint(3, 20)
            h = random.randint(3, 20)
            return {"shape": "box", "l": l, "w": w, "h": h}
        elif shape_type == 1:  # Cylinder
            r = random.randint(2, 10)
            h = random.randint(3, 20)
            return {"shape": "cylinder", "r": r, "h": h}
        elif shape_type == 2:  # Sphere
            r = input_value if input_value is not None else random.randint(2, 10)
            return {"shape": "sphere", "r": r}
        elif shape_type == 3:  # Cone
            r = random.randint(2, 10)
            h = random.randint(3, 20)
            return {"shape": "cone", "r": r, "h": h}
        else:  # Pyramid
            base = random.randint(3, 15)
            h = random.randint(3, 20)
            return {"shape": "pyramid", "base": base, "h": h}

    def compute(self, input_value, params):
        shape = params.get("shape", "box")

        if shape == "box":
            l = params.get("l", 4)
            w = params.get("w", 5)
            h = params.get("h", 6)
            volume = l * w * h
            description = f"Box volume: {l} × {w} × {h} = {volume}"

        elif shape == "cylinder":
            r = params.get("r", 5)
            h = params.get("h", 10)
            volume = math.pi * r * r * h
            description = f"Cylinder volume: π × {r}² × {h} = {volume:.2f}"

        elif shape == "sphere":
            r = params.get("r", 5)
            volume = (4.0 / 3.0) * math.pi * r * r * r
            description = f"Sphere volume: (4/3)π × {r}³ = {volume:.2f}"

        elif shape == "cone":
            r = params.get("r", 5)
            h = params.get("h", 10)
            volume = (1.0 / 3.0) * math.pi * r * r * h
            description = f"Cone volume: (1/3)π × {r}² × {h} = {volume:.2f}"

        else:  # pyramid
            base = params.get("base", 6)
            h = params.get("h", 10)
            # Square pyramid base area
            base_area = base * base
            volume = (1.0 / 3.0) * base_area * h
            description = f"Pyramid volume: (1/3) × {base}² × {h} = {volume:.2f}"

        # Return rounded volume as integer
        result = round(volume)

        return MethodResult(
            value=result,
            description=description,
            metadata={"shape": shape, "volume": volume}
        )

    def can_invert(self):
        return False


@register_technique
class GeometricSeriesSum(MethodBlock):
    """
    Geometric series sums with applications to geometry.

    For a geometric series: a + ar + ar^2 + ... + ar^(n-1)
    - Finite sum: a(1 - r^n)/(1 - r) for r != 1
    - Infinite sum: a/(1 - r) for |r| < 1

    Applications in geometry:
    - Nested squares with ratio r
    - Spiral lengths
    - Fractal areas/perimeters
    """
    def __init__(self):
        super().__init__()
        self.name = "geometric_series_sum"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "series", "sequence"]

    def generate_parameters(self, input_value=None):
        operation = random.choice(["finite_sum", "infinite_sum", "nested_squares"])
        a = random.randint(1, 10)
        r_num = random.randint(1, 3)
        r_den = random.randint(2, 4)
        n = random.randint(3, 8)
        return {"operation": operation, "a": a, "r_num": r_num, "r_den": r_den, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "finite_sum")
        a = params.get("a", 1)
        r_num = params.get("r_num", 1)
        r_den = params.get("r_den", 2)
        n = params.get("n", 5)

        r = r_num / r_den

        if operation == "finite_sum":
            # Sum of first n terms: a(1 - r^n)/(1 - r)
            if r == 1:
                total = a * n
            else:
                total = a * (1 - r ** n) / (1 - r)
            result = int(round(total))
            description = f"Geometric series sum (n={n}, a={a}, r={r_num}/{r_den}): {result}"
        elif operation == "infinite_sum":
            # Infinite sum for |r| < 1: a/(1-r)
            if abs(r) >= 1:
                result = float('inf')
                description = "Series diverges (|r| >= 1)"
            else:
                total = a / (1 - r)
                result = int(round(total))
                description = f"Infinite geometric series (a={a}, r={r_num}/{r_den}): {result}"
        else:  # nested_squares
            # Total area of nested squares where each is r^2 times the previous
            # First square has side a, next has side a*r, etc.
            # Areas: a^2, (ar)^2, (ar^2)^2, ... = a^2(1 + r^2 + r^4 + ...)
            r_sq = r ** 2
            if r_sq >= 1:
                result = float('inf')
                description = "Nested squares area diverges"
            else:
                total_area = (a ** 2) / (1 - r_sq)
                result = int(round(total_area))
                description = f"Nested squares total area (side={a}, r={r_num}/{r_den}): {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "a": a, "r": r, "n": n}
        )

    def can_invert(self):
        return False


@register_technique
class TangentAddition(MethodBlock):
    """
    Tangent addition formula and related calculations.

    tan(A + B) = (tan(A) + tan(B)) / (1 - tan(A) * tan(B))
    tan(A - B) = (tan(A) - tan(B)) / (1 + tan(A) * tan(B))

    Applications:
    - Finding angle sums in geometry problems
    - Computing tangent of compound angles
    - Relating angles in triangles
    """
    def __init__(self):
        super().__init__()
        self.name = "tangent_addition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "trigonometry", "angles"]

    def generate_parameters(self, input_value=None):
        # Use nice angles with known tangent values
        angles = [15, 30, 45, 60, 75]  # degrees
        angle_a = random.choice(angles)
        angle_b = random.choice(angles)
        operation = random.choice(["sum", "difference", "double"])
        return {"angle_a": angle_a, "angle_b": angle_b, "operation": operation}

    def compute(self, input_value, params):
        angle_a = params.get("angle_a", 30)
        angle_b = params.get("angle_b", 45)
        operation = params.get("operation", "sum")

        tan_a = math.tan(math.radians(angle_a))
        tan_b = math.tan(math.radians(angle_b))

        if operation == "sum":
            # tan(A + B) = (tan A + tan B) / (1 - tan A * tan B)
            denominator = 1 - tan_a * tan_b
            if abs(denominator) < 1e-10:
                result = float('inf')
                description = f"tan({angle_a}+{angle_b}) is undefined"
            else:
                tan_sum = (tan_a + tan_b) / denominator
                result = int(round(tan_sum * 100))  # Scale for precision
                description = f"tan({angle_a}+{angle_b}) = {tan_sum:.4f}, encoded as {result}"
        elif operation == "difference":
            # tan(A - B) = (tan A - tan B) / (1 + tan A * tan B)
            denominator = 1 + tan_a * tan_b
            if abs(denominator) < 1e-10:
                result = float('inf')
                description = f"tan({angle_a}-{angle_b}) is undefined"
            else:
                tan_diff = (tan_a - tan_b) / denominator
                result = int(round(tan_diff * 100))
                description = f"tan({angle_a}-{angle_b}) = {tan_diff:.4f}, encoded as {result}"
        else:  # double angle
            # tan(2A) = 2tan(A) / (1 - tan^2(A))
            denominator = 1 - tan_a ** 2
            if abs(denominator) < 1e-10:
                result = float('inf')
                description = f"tan(2*{angle_a}) is undefined"
            else:
                tan_double = (2 * tan_a) / denominator
                result = int(round(tan_double * 100))
                description = f"tan(2*{angle_a}) = {tan_double:.4f}, encoded as {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"angle_a": angle_a, "angle_b": angle_b, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class ThreeDCombinatorics(MethodBlock):
    """
    Combinatorics in 3D geometry.

    Counting problems:
    - Points in n x n x n cube: n^3
    - Space diagonals in cube: 4
    - Face diagonals in cube: 12
    - Ways to select k points from n x n x n grid
    - Planes determined by points
    """
    def __init__(self):
        super().__init__()
        self.name = "3d_combinatorics"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "3d", "combinatorics"]

    def generate_parameters(self, input_value=None):
        operation = random.choice(["grid_points", "space_diagonals", "face_diagonals", "planes_from_points"])
        n = input_value if input_value is not None else random.randint(2, 8)
        return {"operation": operation, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "grid_points")
        n = params.get("n", 4)

        if operation == "grid_points":
            # Total lattice points in n x n x n cube
            result = (n + 1) ** 3
            description = f"Lattice points in {n}x{n}x{n} cube: {result}"
        elif operation == "space_diagonals":
            # Space diagonals of a rectangular box: 4
            # For n x n x n with interior lattice points
            result = 4
            description = f"Space diagonals of cube: {result}"
        elif operation == "face_diagonals":
            # Face diagonals of cube: 2 per face, 6 faces = 12
            result = 12
            description = f"Face diagonals of cube: {result}"
        elif operation == "planes_from_points":
            # Number of planes determined by n non-coplanar points in 3D
            # C(n, 3) for points in general position
            if n < 3:
                result = 0
            else:
                result = math.comb(n, 3)
            description = f"Planes from {n} general points: C({n},3) = {result}"
        else:
            # Default: n^3
            result = n ** 3
            description = f"3D grid count: {n}^3 = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "n": n}
        )

    def can_invert(self):
        return False


@register_technique
class ThreeDDistancePythagorean(MethodBlock):
    """3D distance using Pythagorean theorem."""
    def __init__(self):
        super().__init__()
        self.name = "3d_distance_pythagorean"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "3d"]

    def generate_parameters(self, input_value=None):
        x = random.randint(1, 8)
        y = random.randint(1, 8)
        z = random.randint(1, 8)
        return {"x": x, "y": y, "z": z}

    def compute(self, input_value, params):
        x = params.get("x", 3)
        y = params.get("y", 4)
        z = params.get("z", 5)
        dist_sq = x**2 + y**2 + z**2
        dist = int(math.sqrt(dist_sq))
        return MethodResult(
            value=dist,
            description=f"3D distance: √({x}² + {y}² + {z}²) ≈ {dist}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# PROBLEM 1: TRIANGLE WITH ANGLE BISECTOR CEVIAN TECHNIQUES
# ============================================================================


@register_technique
class CoprimeDiffSquaresParam(MethodBlock):
    """
    Given constraint (a/(b+c))² = (b-c)/b, find parametric family.
    Returns formulas:
    a = k * y * (2*x² - y²)
    b = k * x³
    c = k * x * (x² - y²)
    with constraints: x > y, gcd(x,y) = 1, k ≥ 1
    """

    def __init__(self):
        super().__init__()
        self.name = "coprime_diff_squares_param"
        self.input_type = "constraint"
        self.output_type = "family"
        self.difficulty = 4
        self.tags = ["geometry", "parametric", "number_theory"]

    def generate_parameters(self, input_value=None):
        # No additional parameters needed - the formulas are fixed
        return {}

    def compute(self, input_value, params):
        # Compute example with k=1, x=2, y=1 (satisfies constraints)
        # a = k * y * (2*x² - y²) = 1 * 1 * (2*4 - 1) = 7
        # b = k * x³ = 1 * 8 = 8
        # c = k * x * (x² - y²) = 1 * 2 * (4 - 1) = 6
        k, x, y = 1, 2, 1
        a = k * y * (2*x**2 - y**2)  # 7
        b = k * x**3  # 8
        c = k * x * (x**2 - y**2)  # 6
        product = a * b * c  # 336

        return MethodResult(
            value=product,  # Return product of example values, not dict
            description=f"Parametric family: a = k*y*(2x²-y²), b = k*x³, c = k*x*(x²-y²). Example (k={k},x={x},y={y}): a={a}, b={b}, c={c}, product={product}",
            params=params,
            metadata={
                "example": {"k": k, "x": x, "y": y, "a": a, "b": b, "c": c},
                "constraints": ["x > y", "gcd(x,y) == 1", "k >= 1"]
            }
        )

    def can_invert(self):
        return False


@register_technique
class ProductMod(MethodBlock):
    """
    Compute (a * b * c) % modulus.
    """

    def __init__(self):
        super().__init__()
        self.name = "product_mod"
        self.input_type = "any"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "modular"]

    def generate_parameters(self, input_value=None):
        a = random.randint(2, 500)
        b = random.randint(2, 500)
        c = random.randint(2, 500)
        # Pick a reasonable modulus: random prime or power of 10
        modulus_choices = [97, 101, 997, 1009, 10007, 100003, 1000, 10000, 100000]
        modulus = random.choice(modulus_choices)
        return {"a": a, "b": b, "c": c, "modulus": modulus}

    def validate_params(self, params, prev_value=None):
        """Validate modulus is non-zero."""
        modulus = params.get("modulus", 1)
        return modulus is not None and modulus != 0

    def compute(self, input_value, params):
        # Accept input_value as tuple (a, b, c) or individual params
        if isinstance(input_value, (tuple, list)) and len(input_value) >= 3:
            a, b, c = input_value[0], input_value[1], input_value[2]
        else:
            a = params.get("a", 1)
            b = params.get("b", 1)
            c = params.get("c", 1)
        modulus = params.get("modulus", 997)

        result = (int(a) * int(b) * int(c)) % modulus

        return MethodResult(
            value=result,
            description=f"({a} * {b} * {c}) mod {modulus} = {result}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class VolumeBall(MethodBlock):
    """
    Compute the volume of an n-dimensional ball with given radius.

    Formula: V_n(r) = π^(n/2) * r^n / Γ(n/2 + 1)

    Special cases:
    - n=2 (disk): V = π*r²
    - n=3 (sphere): V = (4/3)*π*r³

    Parameters:
    - radius: r (the radius of the ball)
    - dimension: n (the dimension, default 3 for sphere)

    Returns the volume as a float.
    """

    def __init__(self):
        super().__init__()
        self.name = "volume_ball"
        self.input_type = "integer"
        self.output_type = "float"
        self.difficulty = 3
        self.tags = ["geometry", "volume", "ball", "sphere", "multidimensional"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate random parameters for volume calculation."""
        # Radius between 1 and 20
        radius = random.uniform(1, 20)

        # Dimension between 2 and 10
        dimension = random.randint(2, 10)

        return {"radius": radius, "dimension": dimension}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute volume of n-dimensional ball."""
        radius = params.get("radius", 3)
        dimension = params.get("dimension", 3)

        # Ensure valid parameters
        if radius <= 0:
            radius = abs(radius) + 1
        if dimension < 1:
            dimension = 3

        # Formula: V_n(r) = π^(n/2) * r^n / Γ(n/2 + 1)
        pi_power = math.pi ** (dimension / 2)
        radius_power = radius ** dimension
        gamma_value = math.gamma(dimension / 2 + 1)

        volume = (pi_power * radius_power) / gamma_value

        # For readability, scale results to reasonable ranges
        display_value = volume

        # Create a descriptive string based on dimension
        if dimension == 2:
            formula_used = f"π*r² = π*{radius}² ≈ {volume:.4f}"
        elif dimension == 3:
            formula_used = f"(4/3)*π*r³ = (4/3)*π*{radius}³ ≈ {volume:.4f}"
        else:
            formula_used = f"π^({dimension}/2)*{radius}^{dimension}/Γ({dimension}/2+1) ≈ {volume:.4f}"

        return MethodResult(
            value=volume,
            description=f"Volume of {dimension}D ball with radius {radius}: {formula_used}",
            metadata={
                "radius": radius,
                "dimension": dimension,
                "pi_power": pi_power,
                "radius_power": radius_power,
                "gamma_value": gamma_value,
                "formula": f"π^(n/2) * r^n / Γ(n/2 + 1)"
            }
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# GEOMETRY CORE METHODS (20)
# ============================================================================


@register_technique
class EllipseStandardForm(MethodBlock):
    """
    Convert ellipse parameters to standard form equation.

    Standard form: (x-h)²/a² + (y-k)²/b² = 1
    where (h,k) is center, a is semi-major axis, b is semi-minor axis.

    Returns the equation coefficients.
    """

    def __init__(self):
        super().__init__()
        self.name = "ellipse_standard_form"
        self.input_type = "geometric"
        self.output_type = "geometric"
        self.difficulty = 2
        self.tags = ["geometry", "ellipse", "conic_section", "standard_form"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate ellipse parameters."""
        a = random.randint(5, 20)  # semi-major axis
        b = random.randint(3, min(a, 15))  # semi-minor axis (b ≤ a)
        h = random.randint(-10, 10)  # center x
        k = random.randint(-10, 10)  # center y
        return {"a": a, "b": b, "center": (h, k)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        b = params.get("b", 5)
        h, k = ensure_tuple(params.get("center", (0, 0)), "center", 2)

        # Compute area (pi * a * b) - return as integer approximation
        import math
        area = int(round(math.pi * a * b))

        return MethodResult(
            value=area,  # Return area as simple int, not dict
            description=f"Ellipse: (x-{h})²/{a}² + (y-{k})²/{b}² = 1, area ≈ {area}",
            metadata={
                "center": (h, k),
                "a": a,
                "b": b,
                "a_squared": a**2,
                "b_squared": b**2,
                "area": area
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class EllipseVertices(MethodBlock):
    """
    Find the vertices of an ellipse.

    For ellipse centered at (h,k) with semi-axes a,b:
    - Major axis vertices (if a > b): (h±a, k)
    - Minor axis vertices: (h, k±b)

    Returns sum of x-coordinates of all 4 vertices.
    """

    def __init__(self):
        super().__init__()
        self.name = "ellipse_vertices"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "ellipse", "vertices"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate ellipse parameters."""
        a = random.randint(5, 20)
        b = random.randint(3, min(a, 15))
        h = random.randint(-10, 10)
        k = random.randint(-10, 10)
        return {"a": a, "b": b, "center": (h, k)}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        b = params.get("b", 5)
        h, k = ensure_tuple(params.get("center", (0, 0)), "center", 2)

        # Vertices: (h±a, k) and (h, k±b)
        vertices = [
            (h + a, k),  # right
            (h - a, k),  # left
            (h, k + b),  # top
            (h, k - b)   # bottom
        ]

        # Sum of x-coordinates
        x_sum = sum(v[0] for v in vertices)

        return MethodResult(
            value=x_sum,
            description=f"Ellipse vertices: {vertices}, sum of x-coords = {x_sum}",
            metadata={"vertices": vertices, "center": (h, k), "a": a, "b": b}
        )

    def can_invert(self) -> bool:
        return False


