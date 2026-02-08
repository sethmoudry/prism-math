"""
Geometry Techniques - Triangles

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class AreaHeron(MethodBlock):
    """Compute triangle area using Heron's formula: K = √(s(s-a)(s-b)(s-c))."""

    def __init__(self):
        super().__init__()
        self.name = "area_heron"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "area", "triangle", "heron"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for Heron's formula."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(3, 20)
            b = random.randint(3, 20)
            c = random.randint(abs(a-b)+1, a+b-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)
        s = (a + b + c) / 2
        K_squared = s * (s - a) * (s - b) * (s - c)
        K = math.sqrt(K_squared)
        K_rounded = int(round(K))

        return MethodResult(
            value=K_rounded,
            description=f"Area = √({s}·{s-a}·{s-b}·{s-c}) = {K:.4f} ≈ {K_rounded}",
            metadata={"semiperimeter": s, "sides": (a, b, c), "exact_area": K}
        )

    def can_invert(self) -> bool:
        return True


@register_technique
class AreaHeronInverseSide(MethodBlock):
    """Find third side given area and two sides using Heron's formula."""

    def __init__(self):
        super().__init__()
        self.name = "area_heron_inverse_side"
        self.input_type = "geometric_value"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "area", "triangle", "heron", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(a - b) + 1, a + b - 1)
            if a + b > c and b + c > a and a + c > b:
                s = (a + b + c) / 2
                K = math.sqrt(s * (s - a) * (s - b) * (s - c))
                if K > 0.001:
                    return {"a": a, "b": b, "K": K}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", 10)
        b = params.get("b", 10)
        K = params.get("K", 10)

        # Solve for c using Heron's formula: 16K² = 2a²b² + 2b²c² + 2c²a² - a⁴ - b⁴ - c⁴
        # This is quadratic in u = c²:
        # u = (a² + b²) ± 2·sqrt(a²b² - 4K²)
        disc_inner = a**2 * b**2 - 4 * K**2
        if disc_inner < 0:
            # Area too large for given sides — clamp area
            K = a * b / 2.001
            disc_inner = a**2 * b**2 - 4 * K**2

        sqrt_disc = math.sqrt(max(disc_inner, 0))

        u1 = (a**2 + b**2) + 2 * sqrt_disc
        u2 = (a**2 + b**2) - 2 * sqrt_disc

        # Prefer u2 (smaller root) since generate_parameters picks c
        # from a range that typically matches the smaller solution
        candidates = []
        for u in [u2, u1]:
            if u > 0:
                c = math.sqrt(u)
                if abs(a - b) < c < a + b:
                    candidates.append(c)

        if candidates:
            c = candidates[0]
        else:
            c = math.sqrt(max(u1, u2, 0.01))

        c_rounded = int(round(c))

        return MethodResult(
            value=c_rounded,
            description=f"Found side c={c:.4f} ≈ {c_rounded} given K={K:.2f}, a={a}, b={b}",
            metadata={"K": K, "a": a, "b": b, "exact_c": c}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class Stewart(MethodBlock):
    """
    Stewart's Theorem: For cevian AD in triangle ABC,
    b²·m + c²·n = a(d² + mn) where m=BD, n=DC, d=AD.
    """

    def __init__(self):
        super().__init__()
        self.name = "stewart_theorem"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "stewart", "cevian"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality and valid cevian segments."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        m = params.get("m")
        n = params.get("n")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        # Triangle inequality
        if not (a + b > c and b + c > a and a + c > b):
            return False
        # Cevian segments must be positive and sum to a
        if m is None or n is None:
            return True  # Optional for some uses
        return m > 0 and n > 0 and abs(m + n - a) < 0.001

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        a = random.randint(10, 30)
        b = random.randint(10, 30)
        c = random.randint(abs(b-a)+1, b+a-1)
        m = random.randint(2, a-2)
        n = a - m
        return {"a": a, "b": b, "c": c, "m": m, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)
        m, n = params.get("m", 10), params.get("n", 10)

        # b²m + c²n = a(d² + mn)  =>  d² = (b²m + c²n - amn)/a
        d_squared = (b**2 * m + c**2 * n - a * m * n) / a
        d = math.sqrt(abs(d_squared))
        d_rounded = int(round(d))

        return MethodResult(
            value=d_rounded,
            description=f"Cevian length d={d:.2f} ≈ {d_rounded} via Stewart's theorem",
            metadata={"sides": (a, b, c), "segments": (m, n), "exact_cevian": d}
        )

    def can_invert(self) -> bool:
        return False



# DISABLED: Delegates to Stewart() but produces inconsistent geometry data
# @register_technique


class StewartInverseCevian(MethodBlock):
    """Find cevian segment length using Stewart's theorem."""

    def __init__(self):
        super().__init__()
        self.name = "stewart_inverse_cevian"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "stewart", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        return Stewart().generate_parameters(target_output)

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        return Stewart().compute(input_value, params)

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleBisectorLength(MethodBlock):
    """
    Length of angle bisector from A to side BC:
    t_a = (2bc·cos(A/2))/(b+c) or t_a = √(bc((b+c)²-a²))/(b+c).
    """

    def __init__(self):
        super().__init__()
        self.name = "angle_bisector_length"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "angle_bisector"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for angle bisector length."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(b-a)+1, b+a-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)

        # t_a = √(bc((b+c)²-a²))/(b+c)
        numerator = math.sqrt(b * c * ((b + c)**2 - a**2))
        t_a = numerator / (b + c)
        t_a_rounded = int(round(t_a))

        return MethodResult(
            value=t_a_rounded,
            description=f"Angle bisector length t_a = {t_a:.4f} ≈ {t_a_rounded}",
            metadata={"sides": (a, b, c), "exact_length": t_a}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MedianLength(MethodBlock):
    """
    Length of median from A to midpoint of BC:
    m_a = (1/2)√(2b² + 2c² - a²).
    """

    def __init__(self):
        super().__init__()
        self.name = "median_length"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "median"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for median length."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(b-a)+1, b+a-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)

        m_a = 0.5 * math.sqrt(2 * b**2 + 2 * c**2 - a**2)
        m_a_rounded = int(round(m_a))

        return MethodResult(
            value=m_a_rounded,
            description=f"Median m_a = {m_a:.4f} ≈ {m_a_rounded}",
            metadata={"sides": (a, b, c), "exact_median": m_a}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleBisectorTheorem(MethodBlock):
    """
    Angle bisector theorem: If AD bisects angle A, then BD/DC = AB/AC.
    """

    def __init__(self):
        super().__init__()
        self.name = "angle_bisector_theorem"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "angle_bisector", "ratio"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        AB = random.randint(5, 20)
        AC = random.randint(5, 20)
        BC = random.randint(abs(AB-AC)+1, AB+AC-1)
        return {"AB": AB, "AC": AC, "BC": BC}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        AB = params.get("AB", 10)
        AC = params.get("AC", 10)
        BC = params.get("BC", 10)

        # BD/DC = AB/AC
        # BD + DC = BC
        # BD = BC·AB/(AB+AC), DC = BC·AC/(AB+AC)
        BD = BC * AB / (AB + AC)
        DC = BC * AC / (AB + AC)
        ratio = BD / DC
        ratio_rounded = int(round(ratio))

        return MethodResult(
            value=ratio_rounded,
            description=f"Angle bisector ratio BD/DC = {ratio:.4f} ≈ {ratio_rounded}",
            metadata={"AB": AB, "AC": AC, "BD": BD, "DC": DC, "exact_ratio": ratio}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CevaTheorem(MethodBlock):
    """
    Ceva's theorem: Cevians AA', BB', CC' are concurrent iff
    (BA'/A'C)·(CB'/B'A)·(AC'/C'B) = 1.
    """

    def __init__(self):
        super().__init__()
        self.name = "ceva_theorem"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "ceva", "concurrent"]

    def validate_params(self, params, prev_value=None):
        """Validate that ratios are non-zero for Ceva's theorem."""
        r1 = params.get("r1")
        r2 = params.get("r2")
        r3 = params.get("r3")
        if r1 is None or r2 is None or r3 is None:
            return False
        # Ratios must be positive for valid cevians
        return r1 > 0 and r2 > 0 and r3 > 0

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Generate ratios that multiply to 1
        r1 = random.uniform(0.5, 2.0)
        r2 = random.uniform(0.5, 2.0)
        r3 = 1.0 / (r1 * r2)
        return {"r1": r1, "r2": r2, "r3": r3}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        r1, r2, r3 = params.get("r1", 10), params.get("r2", 10), params.get("r3", 10)
        product = r1 * r2 * r3

        return MethodResult(
            value=product,
            description=f"Ceva product = {product:.6f} (should be 1)",
            metadata={"ratios": (r1, r2, r3)}
        )


# ============================================================================
# THEOREMS (4)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class MenelausTheorem(MethodBlock):
    """
    Menelaus theorem: Points A', B', C' on sides BC, CA, AB are collinear iff
    (BA'/A'C)·(CB'/B'A)·(AC'/C'B) = -1 (signed ratios).
    """

    def __init__(self):
        super().__init__()
        self.name = "menelaus_theorem"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "menelaus", "collinear"]

    def validate_params(self, params, prev_value=None):
        """Validate that ratios are non-zero for Menelaus theorem."""
        r1 = params.get("r1")
        r2 = params.get("r2")
        r3 = params.get("r3")
        if r1 is None or r2 is None or r3 is None:
            return False
        # Ratios must be non-zero (can be negative for Menelaus)
        return r1 != 0 and r2 != 0 and r3 != 0

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        # Generate signed ratios that multiply to -1
        r1 = random.uniform(-2.0, 2.0)
        r2 = random.uniform(-2.0, 2.0)
        r3 = -1.0 / (r1 * r2) if r1 * r2 != 0 else 1.0
        return {"r1": r1, "r2": r2, "r3": r3}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        r1, r2, r3 = params.get("r1", 10), params.get("r2", 10), params.get("r3", 10)
        product = r1 * r2 * r3

        return MethodResult(
            value=product,
            description=f"Menelaus product = {product:.6f} (should be -1)",
            metadata={"signed_ratios": (r1, r2, r3)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LawOfCosines(MethodBlock):
    """Law of cosines: c² = a² + b² - 2ab·cos(C)."""

    def __init__(self):
        super().__init__()
        self.name = "law_of_cosines"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "law_of_cosines"]

    def validate_params(self, params, prev_value=None):
        """Validate law of cosines parameters: sides positive, angle in (0, 180)."""
        a = params.get("a")
        b = params.get("b")
        if a is None or b is None:
            return False
        if a <= 0 or b <= 0:
            return False
        angle_C = params.get("angle_C", 90.0)
        # Angle must be in (0, 180) degrees for a valid triangle
        return 0 < angle_C < 180

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        a = random.randint(5, 20)
        b = random.randint(5, 20)
        angle_C = random.uniform(30, 150)  # degrees
        return {"a": a, "b": b, "angle_C": angle_C}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Get required parameters with validation
        a = params.get("a")
        b = params.get("b")

        if a is None or b is None:
            raise ValueError(
                f"law_of_cosines requires parameters 'a' and 'b' (sides of triangle). "
                f"Got: {params}. Usage: law_of_cosines(a, b, angle_C) or law_of_cosines(a, b) for right triangle."
            )

        # angle_C is optional - defaults to 90 degrees (right triangle)
        angle_C = params.get("angle_C", 90.0)

        C_rad = math.radians(angle_C)
        c_squared = a**2 + b**2 - 2 * a * b * math.cos(C_rad)

        # Ensure non-negative value under square root
        if c_squared < 0:
            raise ValueError(
                f"Invalid triangle: law of cosines gives negative value under square root. "
                f"a={a}, b={b}, angle_C={angle_C}°"
            )

        c = math.sqrt(c_squared)
        c_rounded = int(round(c))

        return MethodResult(
            value=c_rounded,
            description=f"Side c = √({a}² + {b}² - 2·{a}·{b}·cos({angle_C}°)) = {c:.4f} ≈ {c_rounded}",
            metadata={"a": a, "b": b, "angle_C": angle_C, "exact_side": c}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LawOfCosinesInverseAngle(MethodBlock):
    """Find angle given three sides using law of cosines: cos(C) = (a²+b²-c²)/(2ab)."""

    def __init__(self):
        super().__init__()
        self.name = "law_of_cosines_inverse_angle"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "law_of_cosines", "inverse"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(b-a)+1, b+a-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Get required parameters with validation
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")

        if a is None or b is None or c is None:
            raise ValueError(
                f"law_of_cosines_inverse_angle requires all three sides 'a', 'b', 'c'. "
                f"Got: {params}. Usage: law_of_cosines_inverse_angle(a, b, c)"
            )

        # Validate triangle inequality
        if not (a + b > c and b + c > a and a + c > b):
            raise ValueError(
                f"Invalid triangle: sides do not satisfy triangle inequality. "
                f"a={a}, b={b}, c={c}"
            )

        # Calculate and validate cos_C
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

        # Ensure cos_C is in valid range [-1, 1] for acos
        if cos_C < -1 or cos_C > 1:
            # This shouldn't happen with valid triangle, but floating point errors...
            cos_C = max(-1, min(1, cos_C))

        angle_C = math.degrees(math.acos(cos_C))
        angle_C_rounded = int(round(angle_C))

        return MethodResult(
            value=angle_C_rounded,
            description=f"Angle C = arccos(({a}²+{b}²-{c}²)/(2·{a}·{b})) = {angle_C:.2f}° ≈ {angle_C_rounded}°",
            metadata={"sides": (a, b, c), "exact_angle": angle_C}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LawOfSinesRatio(MethodBlock):
    """Law of sines: a/sin(A) = b/sin(B) = c/sin(C) = 2R."""

    def __init__(self):
        super().__init__()
        self.name = "law_of_sines_ratio"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "law_of_sines", "circumradius"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        while True:
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(abs(b-a)+1, b+a-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        # Get required parameters with validation
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")

        if a is None or b is None or c is None:
            raise ValueError(
                f"law_of_sines_ratio requires all three sides 'a', 'b', 'c'. "
                f"Got: {params}. Usage: law_of_sines_ratio(a, b, c)"
            )

        # Validate triangle inequality
        if not (a + b > c and b + c > a and a + c > b):
            raise ValueError(
                f"Invalid triangle: sides do not satisfy triangle inequality. "
                f"a={a}, b={b}, c={c}"
            )

        # Compute area and circumradius
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)

        # Ensure non-negative area (should always be true for valid triangle)
        if area_squared < 0:
            raise ValueError(
                f"Invalid triangle: Heron's formula gives negative area. "
                f"a={a}, b={b}, c={c}"
            )

        K = math.sqrt(area_squared)

        # Avoid division by zero
        if K == 0:
            raise ValueError(
                f"Degenerate triangle: area is zero. "
                f"a={a}, b={b}, c={c}"
            )

        R = (a * b * c) / (4 * K)
        result = 2 * R
        result_rounded = int(round(result))

        return MethodResult(
            value=result_rounded,
            description=f"Law of sines ratio 2R = {result:.4f} ≈ {result_rounded}",
            metadata={"circumradius": R, "area": K, "exact_ratio": result}
        )


# ============================================================================
# COORDINATES (4)
# ============================================================================

    def can_invert(self) -> bool:
        return False


@register_technique
class AngleBisectorTheoremExtended(MethodBlock):
    """Extended angle bisector theorem for segment ratios in triangles."""
    def __init__(self):
        super().__init__()
        self.name = "angle_bisector_theorem_extended"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "bisector"]

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 15)
        b = random.randint(3, 15)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        from math import gcd
        a = params.get("a", 10)
        b = params.get("b", 10)
        g = gcd(a, b)
        ratio_num = b // g
        ratio_den = a // g
        return MethodResult(
            value=ratio_num + ratio_den,
            description=f"Angle bisector divides opposite side in ratio {ratio_num}:{ratio_den}",
            metadata={"a": a, "b": b, "ratio": f"{ratio_num}:{ratio_den}"}
        )

    def can_invert(self):
        return False


