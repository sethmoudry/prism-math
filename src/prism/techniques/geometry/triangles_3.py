"""
Geometry Techniques - Triangles (Part 3)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class OrthocenterAltitudeProperties(MethodBlock):
    """Properties of orthocenter and altitudes in triangles."""
    def __init__(self):
        super().__init__()
        self.name = "orthocenter_altitude_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "orthocenter"]

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 15)
        b = random.randint(3, 15)
        c = random.randint(3, 15)
        return {"a": a, "b": b, "c": c}

    def validate_params(self, params, prev_value=None):
        """Validate orthocenter parameters: sides must satisfy triangle inequality."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        try:
            a_val = float(a)
            b_val = float(b)
            c_val = float(c)
            return (a_val + b_val > c_val and
                    b_val + c_val > a_val and
                    a_val + c_val > b_val and
                    a_val > 0 and b_val > 0 and c_val > 0)
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)
        result = a + b + c
        return MethodResult(
            value=result,
            description=f"Perimeter of triangle with sides {a}, {b}, {c}: {result}",
            metadata={"a": a, "b": b, "c": c}
        )

    def can_invert(self):
        return False


@register_technique
class RightTriangleProperties(MethodBlock):
    """
    Properties of right triangles.

    Key formulas for right triangle with legs a, b and hypotenuse c:
    - Pythagorean theorem: a^2 + b^2 = c^2
    - Area = (1/2) * a * b
    - Circumradius R = c/2 (hypotenuse is diameter)
    - Inradius r = (a + b - c) / 2
    - Altitude to hypotenuse h = ab/c
    - Median to hypotenuse = c/2

    Supports multiple operations based on Pythagorean theorem applications.
    """
    def __init__(self):
        super().__init__()
        self.name = "right_triangle_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "right_triangle", "pythagorean"]

    def generate_parameters(self, input_value=None):
        # Use Pythagorean triples for integer results
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29)]
        a, b, c = random.choice(triples)
        scale = random.randint(1, 4)
        operation = random.choice(["area", "inradius", "circumradius", "altitude_h", "perimeter"])
        return {"a": a * scale, "b": b * scale, "c": c * scale, "operation": operation}

    def compute(self, input_value, params):
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)
        operation = params.get("operation", "area")

        if operation == "area":
            result = (a * b) // 2
            description = f"Right triangle area: (1/2)*{a}*{b} = {result}"
        elif operation == "inradius":
            # r = (a + b - c) / 2
            r = (a + b - c) // 2
            result = r
            description = f"Right triangle inradius: ({a}+{b}-{c})/2 = {result}"
        elif operation == "circumradius":
            # R = c/2 (hypotenuse is diameter)
            result = c // 2
            description = f"Right triangle circumradius: {c}/2 = {result}"
        elif operation == "altitude_h":
            # Altitude to hypotenuse: h = ab/c
            h = (a * b) // c
            result = h
            description = f"Altitude to hypotenuse: {a}*{b}/{c} = {result}"
        elif operation == "perimeter":
            result = a + b + c
            description = f"Right triangle perimeter: {a}+{b}+{c} = {result}"
        else:
            # Default: hypotenuse from legs
            result = c
            description = f"Hypotenuse: sqrt({a}^2 + {b}^2) = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"a": a, "b": b, "c": c, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class AltitudeFeetCyclic(MethodBlock):
    """
    Properties of altitude feet in triangles forming cyclic quadrilateral.

    Given a triangle ABC with altitudes from each vertex, the feet of these
    altitudes form a cyclic quadrilateral (the orthic triangle and its properties).

    Key properties:
    - The four points (three altitude feet + one vertex) form a cyclic quadrilateral
    - If angle A = α, then the angle subtended by the altitude feet is 180° - 2α
    - The orthic triangle has specific angle relationships with the original triangle
    """
    def __init__(self):
        super().__init__()
        self.name = "altitude_feet_cyclic"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "cyclic"]

    def generate_parameters(self, input_value=None):
        angle = input_value if input_value is not None else random.randint(30, 90)
        return {"angle": angle}

    def validate_params(self, params, prev_value=None):
        """Validate angle is within valid triangle angle range (0, 180)."""
        angle = params.get("angle", prev_value) if prev_value is not None else params.get("angle")
        return angle is not None and 0 < angle < 180

    def compute(self, input_value, params):
        angle = params.get("angle", 60)
        # In the orthic triangle, the angle at the foot of the altitude from vertex A
        # is supplementary to 2*angle_A. For altitude feet forming a cyclic quadrilateral:
        # The angle relationship is 180 - 2*angle for the inscribed angle.
        result = abs(180 - 2 * angle)
        return MethodResult(
            value=result,
            description=f"Altitude feet cyclic quad angle: |180 - 2*{angle}| = {result}",
            metadata={"angle": angle, "formula": "180 - 2*angle"}
        )

    def can_invert(self):
        return False


@register_technique
class CircumcenterProperties(MethodBlock):
    """Properties of triangle circumcenter."""
    def __init__(self):
        super().__init__()
        self.name = "circumcenter_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle"]

    def generate_parameters(self, input_value=None):
        r = random.randint(3, 10)
        return {"r": r}

    def validate_params(self, params, prev_value=None):
        """Validate circumcenter parameters: radius must be positive."""
        r = params.get("r")
        if r is None:
            return False
        try:
            return float(r) > 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        r = params.get("r", 5)
        result = r * 2
        return MethodResult(
            value=result,
            description=f"Circumcircle diameter: 2R = 2·{r} = {result}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class CentroidProperties(MethodBlock):
    """Properties of centroid (center of mass) in triangles."""
    def __init__(self):
        super().__init__()
        self.name = "centroid_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "centroid"]

    def generate_parameters(self, input_value=None):
        x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
        x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        x3, y3 = random.randint(-10, 10), random.randint(-10, 10)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}

    def validate_params(self, params, prev_value=None):
        """Validate centroid parameters: must be a non-degenerate triangle."""
        x1, y1 = params.get("x1"), params.get("y1")
        x2, y2 = params.get("x2"), params.get("y2")
        x3, y3 = params.get("x3"), params.get("y3")
        if any(v is None for v in [x1, y1, x2, y2, x3, y3]):
            return False
        try:
            # Check non-collinearity
            area_2 = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            return area_2 > 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        x1, y1 = params.get("x1", 10), params.get("y1", 10)
        x2, y2 = params.get("x2", 10), params.get("y2", 10)
        x3, y3 = params.get("x3", 10), params.get("y3", 10)
        cx = (x1 + x2 + x3) // 3
        cy = (y1 + y2 + y3) // 3
        result = abs(cx) + abs(cy)
        return MethodResult(
            value=result,
            description=f"Centroid distance from origin: {result}",
            metadata={"centroid": (cx, cy)}
        )

    def can_invert(self):
        return False


@register_technique
class TrianglesSharedAngle(MethodBlock):
    """
    Properties of triangles sharing a common angle.

    When two triangles share an angle:
    - Area ratio = product of sides ratio enclosing the angle
    - If triangles ABC and ADE share angle A:
      Area(ADE)/Area(ABC) = (AD*AE)/(AB*AC)

    Used for comparing areas, finding ratios, and proving similarity.
    """
    def __init__(self):
        super().__init__()
        self.name = "triangles_shared_angle"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "area", "ratio"]

    def generate_parameters(self, input_value=None):
        # First triangle sides enclosing shared angle
        ab = random.randint(4, 12)
        ac = random.randint(4, 12)
        # Second triangle sides enclosing same angle
        ad = random.randint(2, 8)
        ae = random.randint(2, 8)
        # Shared angle (in degrees)
        angle = random.randint(30, 120)
        operation = random.choice(["area_ratio", "area_diff", "find_side"])
        return {"ab": ab, "ac": ac, "ad": ad, "ae": ae, "angle": angle, "operation": operation}

    def compute(self, input_value, params):
        ab = params.get("ab", 6)
        ac = params.get("ac", 8)
        ad = params.get("ad", 3)
        ae = params.get("ae", 4)
        angle = params.get("angle", 60)
        operation = params.get("operation", "area_ratio")

        # Calculate areas using (1/2)*a*b*sin(C)
        sin_angle = math.sin(math.radians(angle))
        area_abc = 0.5 * ab * ac * sin_angle
        area_ade = 0.5 * ad * ae * sin_angle

        if operation == "area_ratio":
            # Area ratio = (AD*AE)/(AB*AC)
            ratio = (ad * ae) / (ab * ac)
            result = int(round(ratio * 100))  # Encode as percentage
            description = f"Area ratio ADE/ABC = {ad}*{ae}/({ab}*{ac}) = {ratio:.4f}, encoded as {result}"
        elif operation == "area_diff":
            # Difference in areas
            diff = abs(area_abc - area_ade)
            result = int(round(diff))
            description = f"Area difference = |{area_abc:.2f} - {area_ade:.2f}| = {result}"
        else:  # find_side
            # Given areas equal, find what ad should be if ae, ab, ac are known
            # area_abc = area_ade => ab*ac = ad*ae => ad = ab*ac/ae
            required_ad = (ab * ac) / ae
            result = int(round(required_ad))
            description = f"For equal areas, AD = {ab}*{ac}/{ae} = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"ab": ab, "ac": ac, "ad": ad, "ae": ae, "angle": angle, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class PythagoreanTheorem(MethodBlock):
    """Generate problems using the Pythagorean theorem: a^2 + b^2 = c^2."""

    def __init__(self):
        super().__init__()
        self.name = "pythagorean_theorem"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["geometry", "triangles", "pythagorean"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for a Pythagorean theorem problem."""
        problem_type = random.choice(["hypotenuse", "leg", "area"])

        # Generate a Pythagorean triple (scaled)
        # Common primitive triples: (3,4,5), (5,12,13), (8,15,17), (7,24,25)
        base_triples = [
            (3, 4, 5),
            (5, 12, 13),
            (8, 15, 17),
            (7, 24, 25),
            (20, 21, 29),
            (9, 40, 41),
            (12, 35, 37),
            (11, 60, 61),
        ]

        # Choose a base triple and scale it
        base_a, base_b, base_c = random.choice(base_triples)
        scale = random.randint(1, 5)

        a = base_a * scale
        b = base_b * scale
        c = base_c * scale

        return {
            "a": a,
            "b": b,
            "c": c,
            "problem_type": problem_type
        }

    def compute(self, input_value, params):
        """Compute the answer based on the problem type."""
        # Get required parameters with validation
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        problem_type = params.get("problem_type", "hypotenuse")

        if a is None or b is None:
            raise ValueError(
                f"pythagorean_theorem requires at least 'a' and 'b' (legs of right triangle). "
                f"Got: {params}. Usage: pythagorean_theorem(a, b) or pythagorean_theorem(a, b, c, problem_type)"
            )

        # If c is not provided, compute it from a and b
        if c is None:
            c = int(math.sqrt(a**2 + b**2))

        if problem_type == "hypotenuse":
            # Given two legs, find hypotenuse
            answer = c
            description = f"Right triangle with legs {a} and {b}. Find hypotenuse c where {a}^2 + {b}^2 = c^2. Answer: {c}"
        elif problem_type == "leg":
            # Given hypotenuse and one leg, find other leg
            # Use 'which_leg' param if specified (for backward generation), otherwise random
            which_leg = params.get("which_leg", "random")
            if which_leg == "a" or (which_leg == "random" and random.random() < 0.5):
                answer = a
                description = f"Right triangle with hypotenuse {c} and one leg {b}. Find other leg a where a^2 + {b}^2 = {c}^2. Answer: {a}"
            else:
                answer = b
                description = f"Right triangle with hypotenuse {c} and one leg {a}. Find other leg b where {a}^2 + b^2 = {c}^2. Answer: {b}"
        else:  # area
            # Area of right triangle = (1/2) * a * b
            answer = (a * b) // 2
            description = f"Right triangle with legs {a} and {b}. Find area = (1/2) * {a} * {b} = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"triple": (a, b, c), "problem_type": problem_type}
        )

    def can_invert(self):
        return False

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        """Find Pythagorean triple parameters that produce the target answer."""
        # Generate a comprehensive list of Pythagorean triples
        base_triples = [
            (3, 4, 5),
            (5, 12, 13),
            (8, 15, 17),
            (7, 24, 25),
            (20, 21, 29),
            (9, 40, 41),
            (12, 35, 37),
            (11, 60, 61),
            (13, 84, 85),
            (36, 77, 85),
        ]

        # Try to find a triple that works for the target
        for base_a, base_b, base_c in base_triples:
            for scale in range(1, 20):
                a = base_a * scale
                b = base_b * scale
                c = base_c * scale

                # Stop if values get too large
                if c > 10000:
                    break

                # Try different problem types
                # Type 1: hypotenuse
                if c == target:
                    return {"a": a, "b": b, "c": c, "problem_type": "hypotenuse"}

                # Type 2: leg
                if a == target:
                    return {"a": a, "b": b, "c": c, "problem_type": "leg", "which_leg": "a"}
                if b == target:
                    return {"a": a, "b": b, "c": c, "problem_type": "leg", "which_leg": "b"}

                # Type 3: area
                area = (a * b) // 2
                if area == target:
                    return {"a": a, "b": b, "c": c, "problem_type": "area"}

        # If no exact match found, try to generate from scratch for area
        # For area: (1/2) * a * b = target => a * b = 2 * target
        # Find factor pairs of 2 * target that form Pythagorean triples
        double_target = 2 * target
        if double_target <= 0 or double_target > 50000:
            return None

        # Try factor pairs
        for a in range(1, min(int(double_target ** 0.5) + 1, 200)):
            if double_target % a == 0:
                b = double_target // a
                # Check if a^2 + b^2 is a perfect square
                c_squared = a * a + b * b
                c = int(c_squared ** 0.5)
                if c * c == c_squared:
                    return {"a": a, "b": b, "c": c, "problem_type": "area"}

        return None


@register_technique
class TriangleSimilarity(MethodBlock):
    """
    Triangle similarity calculations.

    Two triangles are similar if:
    - AAA: All corresponding angles are equal
    - SAS: Two sides proportional and included angle equal
    - SSS: All three sides proportional

    For similar triangles with ratio k:
    - Corresponding sides have ratio k
    - Corresponding areas have ratio k^2
    - Corresponding perimeters have ratio k
    """
    def __init__(self):
        super().__init__()
        self.name = "triangle_similarity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "similarity"]

    def generate_parameters(self, input_value=None):
        # First triangle sides
        a1 = random.randint(3, 10)
        b1 = random.randint(3, 10)
        c1 = random.randint(max(abs(a1 - b1) + 1, 3), min(a1 + b1 - 1, 10))
        # Scale factor
        k_num = random.randint(2, 5)
        k_den = random.randint(1, 3)
        operation = random.choice(["find_side", "area_ratio", "perimeter_ratio"])
        return {"a1": a1, "b1": b1, "c1": c1, "k_num": k_num, "k_den": k_den, "operation": operation}

    def compute(self, input_value, params):
        a1 = params.get("a1", 3)
        b1 = params.get("b1", 4)
        c1 = params.get("c1", 5)
        k_num = params.get("k_num", 2)
        k_den = params.get("k_den", 1)
        operation = params.get("operation", "find_side")

        k = k_num / k_den

        if operation == "find_side":
            # Find corresponding side in similar triangle
            a2 = a1 * k
            result = int(round(a2))
            description = f"Similar triangle side: {a1} * {k_num}/{k_den} = {result}"
        elif operation == "area_ratio":
            # Area ratio = k^2
            area_ratio = k ** 2
            result = int(round(area_ratio * 100))  # Encode as percentage * 100
            description = f"Area ratio = k^2 = ({k_num}/{k_den})^2 = {area_ratio:.4f}, encoded as {result}"
        else:  # perimeter_ratio
            # Perimeter ratio = k
            perimeter1 = a1 + b1 + c1
            perimeter2 = perimeter1 * k
            result = int(round(perimeter2))
            description = f"Similar triangle perimeter: {perimeter1} * {k_num}/{k_den} = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"triangle1": (a1, b1, c1), "k": k, "operation": operation}
        )

    def can_invert(self):
        return False


@register_technique
class AreaRatios(MethodBlock):
    """Ratios of areas in geometric figures."""
    def __init__(self):
        super().__init__()
        self.name = "area_ratios"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry"]

    def generate_parameters(self, input_value=None):
        a1 = random.randint(4, 16)
        a2 = random.randint(4, 16)
        return {"a1": a1, "a2": a2}

    def validate_params(self, params, prev_value=None):
        """Validate area ratios parameters: both areas must be positive."""
        a1 = params.get("a1")
        a2 = params.get("a2")
        if a1 is None or a2 is None:
            return False
        try:
            return float(a1) > 0 and float(a2) > 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        a1 = params.get("a1", 12)
        a2 = params.get("a2", 8)
        result = a1 + a2
        return MethodResult(
            value=result,
            description=f"Area sum: {a1} + {a2} = {result}",
            params=params
        )

    def can_invert(self):
        return False


