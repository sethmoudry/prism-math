"""
Geometry Techniques - Circles

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class PowerOfPoint(MethodBlock):
    """
    Compute power of a point P with respect to a circle.
    Power = |OP|² - r² where O is center, r is radius.
    If P is outside, power = (tangent length)².
    """

    def __init__(self):
        super().__init__()
        self.name = "power_of_point"
        self.input_type = "geometric"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "circle", "power"]

    def validate_params(self, params, prev_value=None):
        """Validate that radius is positive for power of point."""
        r = params.get("radius")
        if r is None or r <= 0:
            return False
        return True

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate circle center, radius, and point."""
        # Circle center at origin for simplicity
        cx, cy = 0, 0
        r = random.randint(5, 20)

        # Point P outside the circle
        distance = random.randint(r + 5, r + 30)
        angle = random.uniform(0, 2 * math.pi)
        px = cx + distance * math.cos(angle)
        py = cy + distance * math.sin(angle)

        return {
            "center": (cx, cy),
            "radius": r,
            "point": (px, py),
            "distance": distance
        }

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cx, cy = ensure_tuple(params.get("center", 10), "center", 2)
        r = params.get("radius", 10)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)

        # Power = |OP|² - r²
        dist_squared = (px - cx)**2 + (py - cy)**2
        power = dist_squared - r**2

        return MethodResult(
            value=power,
            description=f"Power of point ({px:.2f}, {py:.2f}) w.r.t circle center ({cx}, {cy}), r={r}: {power:.2f}",
            metadata={"dist_squared": dist_squared, "radius_squared": r**2}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find radius given power and point location."""
        cx, cy = ensure_tuple(params.get("center", 10), "center", 2)
        px, py = ensure_tuple(params.get("point", 10), "point", 2)
        power = output_value

        dist_squared = (px - cx)**2 + (py - cy)**2
        r_squared = dist_squared - power
        r = math.sqrt(abs(r_squared))

        return MethodResult(
            value=r,
            description=f"Found radius={r:.2f} from power={power:.2f}",
            metadata={"power": power, "dist_squared": dist_squared}
        )


@register_technique
class PowerOfPointInverseRadius(MethodBlock):
    """Find radius given power of a point."""

    def __init__(self):
        super().__init__()
        self.name = "power_of_point_inverse_radius"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "circle", "power", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        distance = random.randint(10, 50)
        power = random.randint(50, 500)
        return {"distance": distance, "power": power}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        distance = params.get("distance", 10)
        power = params.get("power", 2)

        # power = distance² - r²  =>  r = √(distance² - power)
        r_squared = distance**2 - power
        if r_squared < 0:
            r_squared = abs(r_squared)
        r = math.sqrt(r_squared)

        return MethodResult(
            value=r,
            description=f"Radius from power={power}, distance={distance}: r={r:.2f}",
            params=params,
            metadata={"distance": distance, "power": power}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Circumradius(MethodBlock):
    """
    Compute circumradius R of a triangle using R = abc/(4K).
    Where a, b, c are sides and K is area.
    """

    def __init__(self):
        super().__init__()
        self.name = "circumradius"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "circle", "circumradius"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for circumradius calculation."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate valid triangle sides."""
        while True:
            a = random.randint(5, 30)
            b = random.randint(5, 30)
            c = random.randint(abs(a-b)+1, a+b-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)

        # Heron's formula for area
        s = (a + b + c) / 2
        K_squared = s * (s - a) * (s - b) * (s - c)
        K = math.sqrt(max(K_squared, 1e-10))

        # R = abc/(4K)
        R = (a * b * c) / (4 * K)

        # Deterministic scaling to get integer result in reasonable range
        # Use hash of inputs for reproducible amplification
        seed = (a * 1000003 + b * 1009 + c * 17) % 401 + 100  # multiplier in [100, 500]
        offset = (a * 7 + b * 31 + c * 127) % 4001 + 1000  # offset in [1000, 5000]
        R_int = int(R * seed + offset)

        return MethodResult(
            value=R_int,
            description=f"Circumradius R = {a}*{b}*{c}/(4*{K:.2f}) = {R:.2f}, scaled = {R_int}",
            metadata={"area": K, "sides": (a, b, c), "original_circumradius": R}
        )

    def can_invert(self) -> bool:
        return True


@register_technique
class CircumradiusInverseSide(MethodBlock):
    """Find third side of triangle given circumradius R and two sides."""

    def __init__(self):
        super().__init__()
        self.name = "circumradius_inverse_side"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "circumradius", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(a - b) + 1, a + b - 1)
            if a + b > c and b + c > a and a + c > b:
                s = (a + b + c) / 2
                K = math.sqrt(s * (s - a) * (s - b) * (s - c))
                if K > 0.001:
                    R = (a * b * c) / (4 * K)
                    return {"a": a, "b": b, "R": R}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        b = params.get("b", 10)
        R = params.get("R", 10)

        sin_A = min(a / (2 * R), 1.0)
        sin_B = min(b / (2 * R), 1.0)
        A_angle = math.asin(sin_A)
        B_angle = math.asin(sin_B)
        C_angle = math.pi - A_angle - B_angle
        c = 2 * R * math.sin(C_angle)
        c_rounded = int(round(c))

        return MethodResult(
            value=c_rounded,
            description=f"Found side c={c_rounded} given R={R:.2f}, a={a}, b={b}",
            metadata={"R": R, "a": a, "b": b, "exact_c": c}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Inradius(MethodBlock):
    """
    Compute inradius r of a triangle using r = K/s.
    Where K is area and s is semiperimeter.
    """

    def __init__(self):
        super().__init__()
        self.name = "inradius"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "circle", "inradius"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality: a + b > c, b + c > a, a + c > b."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        return a > 0 and b > 0 and c > 0 and a + b > c and b + c > a and a + c > b

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 30)
            b = random.randint(5, 30)
            c = random.randint(abs(a-b)+1, a+b-1)
            if a + b > c and b + c > a and a + c > b:
                # Make scaling deterministic by seeding from triangle sides
                multiplier = random.randint(100, 500)
                offset = random.randint(1000, 5000)
                return {"a": a, "b": b, "c": c, "multiplier": multiplier, "offset": offset}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)

        # Get scaling parameters (use defaults if not provided for backward compatibility)
        multiplier = params.get("multiplier", 1)
        offset = params.get("offset", 0)

        s = (a + b + c) / 2
        K = math.sqrt(s * (s - a) * (s - b) * (s - c))
        r = K / s

        # Apply scaling if parameters are provided
        if multiplier != 1 or offset != 0:
            r_amplified = int(r * multiplier + offset)
            description = f"Inradius r = {K:.2f}/{s:.2f} = {r:.2f}, scaled by {multiplier} + {offset} = {r_amplified}"
            return MethodResult(
                value=r_amplified,
                description=description,
                metadata={"area": K, "semiperimeter": s, "original_inradius": r, "multiplier": multiplier, "offset": offset}
            )
        else:
            return MethodResult(
                value=r,
                description=f"Inradius r = {K:.2f}/{s:.2f} = {r:.2f}",
                metadata={"area": K, "semiperimeter": s}
            )

    def can_invert(self) -> bool:
        return False


@register_technique
class InradiusInverseSide(MethodBlock):
    """Find side given inradius and other sides."""

    def __init__(self):
        super().__init__()
        self.name = "inradius_inverse_side"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "inradius", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(a - b) + 1, a + b - 1)
            if a + b > c and b + c > a and a + c > b:
                s = (a + b + c) / 2
                K = math.sqrt(s * (s - a) * (s - b) * (s - c))
                if K > 0.001:
                    r = K / s
                    return {"a": a, "b": b, "r": r}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        b = params.get("b", 10)
        r = params.get("r", 5)

        best_c = abs(a - b) + 1
        best_err = float('inf')

        c_min = abs(a - b) + 0.01
        c_max = a + b - 0.01

        for i in range(1001):
            c_try = c_min + (c_max - c_min) * i / 1000
            s = (a + b + c_try) / 2
            val = s * (s - a) * (s - b) * (s - c_try)
            if val > 0:
                K = math.sqrt(val)
                r_try = K / s
                err = abs(r_try - r)
                if err < best_err:
                    best_err = err
                    best_c = c_try

        c_rounded = int(round(best_c))

        return MethodResult(
            value=c_rounded,
            description=f"Found side c={c_rounded} given r={r:.2f}, a={a}, b={b}",
            metadata={"r": r, "a": a, "b": b, "exact_c": best_c}
        )


# ============================================================================
# TRIANGLE TECHNIQUES (8)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class RadicalAxis(MethodBlock):
    """
    Radical axis of two circles: locus of points with equal power to both circles.
    """

    def __init__(self):
        super().__init__()
        self.name = "radical_axis"
        self.input_type = "circle_pair"
        self.output_type = "line"
        self.difficulty = 3
        self.tags = ["geometry", "circle", "radical_axis", "power"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        c1 = (random.randint(-20, 0), random.randint(-20, 20))
        r1 = random.randint(5, 15)
        c2 = (random.randint(1, 20), random.randint(-20, 20))
        r2 = random.randint(5, 15)
        return {"circle1": (c1, r1), "circle2": (c2, r2)}

    def validate_params(self, params, prev_value=None):
        """Validate radical axis parameters: two circles with distinct centers."""
        circle1 = params.get("circle1")
        circle2 = params.get("circle2")
        if circle1 is None or circle2 is None:
            return False
        if not isinstance(circle1, (list, tuple)) or len(circle1) != 2:
            return False
        if not isinstance(circle2, (list, tuple)) or len(circle2) != 2:
            return False
        # Centers must be distinct
        try:
            c1, r1 = circle1
            c2, r2 = circle2
            if c1 == c2:
                return False  # Same center - no radical axis
            return float(r1) > 0 and float(r2) > 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        circle1 = params.get("circle1", 10)
        circle2 = params.get("circle2", 10)
        x1, y1 = ensure_tuple(circle1[0] if isinstance(circle1, (list, tuple)) and len(circle1) == 2 else circle1, "circle1[0]", 2)
        r1 = circle1[1] if isinstance(circle1, (list, tuple)) and len(circle1) == 2 else params.get("r1", 1)
        x2, y2 = ensure_tuple(circle2[0] if isinstance(circle2, (list, tuple)) and len(circle2) == 2 else circle2, "circle2[0]", 2)
        r2 = circle2[1] if isinstance(circle2, (list, tuple)) and len(circle2) == 2 else params.get("r2", 1)

        # Radical axis: 2(x2-x1)x + 2(y2-y1)y + (r1²-r2²+x1²-x2²+y1²-y2²) = 0
        # Coefficient form: ax + by + c = 0
        a = 2 * (x2 - x1)
        b = 2 * (y2 - y1)
        c = r1**2 - r2**2 + x1**2 - x2**2 + y1**2 - y2**2

        return MethodResult(
            value=(a, b, c),
            description=f"Radical axis: {a}x + {b}y + {c} = 0",
            metadata={"circle1": (x1, y1, r1), "circle2": (x2, y2, r2)}
        )


# ============================================================================
# ADVANCED (4)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class NinePointCircle(MethodBlock):
    """
    Nine-point circle: passes through midpoints of sides, feet of altitudes,
    and midpoints from vertices to orthocenter.
    """

    def __init__(self):
        super().__init__()
        self.name = "nine_point_circle"
        self.input_type = "triangle"
        self.output_type = "circle"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "nine_point", "circle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Triangle vertices
        A = (0, 0)
        B = (random.randint(10, 30), 0)
        C = (random.randint(5, 20), random.randint(10, 30))
        return {"A": A, "B": B, "C": C}

    def validate_params(self, params, prev_value=None):
        """Validate nine-point circle parameters: must form a valid non-degenerate triangle."""
        A = params.get("A")
        B = params.get("B")
        C = params.get("C")
        if A is None or B is None or C is None:
            return False
        try:
            # Check for non-collinearity (area != 0)
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            area_2 = abs(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
            return area_2 > 0  # Non-degenerate triangle
        except (ValueError, TypeError, IndexError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        A, B, C = params.get("A", 10), params.get("B", 10), params.get("C", 10)
        ax, ay = A
        bx, by = B
        cx, cy = C

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D

        hx = ax + bx + cx - 2 * ux
        hy = ay + by + cy - 2 * uy

        center = ((ux + hx) / 2, (uy + hy) / 2)
        R = math.sqrt((ux - ax)**2 + (uy - ay)**2)
        radius = R / 2

        return MethodResult(
            value=(center, radius),
            description=f"Nine-point circle center: ({center[0]:.2f}, {center[1]:.2f}), r={radius:.2f}",
            metadata={"triangle": (A, B, C), "circumcenter": (ux, uy), "orthocenter": (hx, hy)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExcircleProperties(MethodBlock):
    """
    Excircle radius: r_a = K/(s-a) where K is area, s is semiperimeter.
    """

    def __init__(self):
        super().__init__()
        self.name = "excircle_properties"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "excircle"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(b-a)+1, b+a-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate excircle parameters: sides must satisfy triangle inequality."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        try:
            a_val = float(a)
            b_val = float(b)
            c_val = float(c)
            # Triangle inequality
            return (a_val + b_val > c_val and
                    b_val + c_val > a_val and
                    a_val + c_val > b_val and
                    a_val > 0 and b_val > 0 and c_val > 0)
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)

        s = (a + b + c) / 2
        K = math.sqrt(s * (s - a) * (s - b) * (s - c))
        r_a = K / (s - a)
        r_a_rounded = int(round(r_a))

        return MethodResult(
            value=r_a_rounded,
            description=f"Excircle radius r_a = {K:.2f}/{s-a:.2f} = {r_a:.2f} ≈ {r_a_rounded}",
            metadata={"area": K, "semiperimeter": s, "exact_excircle_radius": r_a}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CircumradiusInradiusFormulas(MethodBlock):
    """Circumradius and inradius formulas for triangles."""
    def __init__(self):
        super().__init__()
        self.name = "circumradius_inradius_formulas"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality: a + b > c, b + c > a, a + c > b."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        return a > 0 and b > 0 and c > 0 and a + b > c and b + c > a and a + c > b

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 10)
        b = random.randint(3, 10)
        c = random.randint(3, 10)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)

        # Scale sides proportionally when input creates degenerate triangles
        if input_value is not None and abs(input_value) > 10:
            scale = abs(input_value) / 10.0
            a = max(1, int(a * scale)) if a * scale < 1000 else a
            b = max(1, int(b * scale)) if b * scale < 1000 else b
            c = max(1, int(c * scale)) if c * scale < 1000 else c

        # Validate triangle inequality - fix if needed
        sides = sorted([a, b, c])
        if sides[0] + sides[1] <= sides[2]:
            # Make valid triangle by adjusting largest side
            sides[2] = sides[0] + sides[1] - 1
            a, b, c = sides

        s = (a + b + c) / 2
        area = math.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        circumradius = (a * b * c) / (4 * area) if area > 0 else 1
        result = int(circumradius * 10)
        return MethodResult(
            value=result,
            description=f"Circumradius R ≈ {result/10}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class InscribedAngleTheorem(MethodBlock):
    """Inscribed angle theorem."""
    def __init__(self):
        super().__init__()
        self.name = "inscribed_angle_theorem"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "circle"]

    def generate_parameters(self, input_value=None):
        arc = random.randint(30, 150)
        return {"arc": arc}

    def validate_params(self, params, prev_value=None):
        """Validate inscribed angle theorem parameters: arc must be positive and <= 360."""
        arc = params.get("arc")
        if arc is None:
            return False
        try:
            arc_val = float(arc)
            return 0 < arc_val <= 360
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        arc = params.get("arc", 90)
        inscribed = arc // 2
        return MethodResult(
            value=inscribed,
            description=f"Inscribed angle = arc/2 = {arc}/2 = {inscribed}°",
            params=params
        )

    def can_invert(self):
        return False


