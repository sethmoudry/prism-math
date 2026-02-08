"""
Geometry Techniques - Triangles (Part 2)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class AngleChasing(MethodBlock):
    """
    Compute unknown angle from known angles using angle sum properties.

    Supports:
    - Triangle angle sum (180 degrees)
    - Quadrilateral angle sum (360 degrees)
    - Polygon angle sum ((n-2)*180 degrees)
    - Supplementary angles (180 degrees)
    - Complementary angles (90 degrees)

    Parameters:
        known_angles: list of known angles in degrees
        polygon_sides: number of sides (default 3 for triangle)
        angle_type: 'interior' (polygon), 'supplementary', or 'complementary'
    """
    def __init__(self):
        super().__init__()
        self.name = "angle_chasing"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "angles", "triangle"]

    def generate_parameters(self, input_value=None):
        # Generate a valid triangle with one unknown angle
        angle1 = random.randint(30, 80)
        angle2 = random.randint(30, 80)
        # The third angle will be computed
        return {
            "known_angles": [angle1, angle2],
            "polygon_sides": 3,
            "angle_type": "interior"
        }

    def compute(self, input_value, params):
        known_angles = params.get("known_angles", [])
        polygon_sides = params.get("polygon_sides", 3)
        angle_type = params.get("angle_type", "interior")

        # Convert to list if needed
        if isinstance(known_angles, (int, float)):
            known_angles = [known_angles]
        elif not isinstance(known_angles, list):
            known_angles = list(known_angles) if known_angles else []

        # Determine total angle sum based on type
        if angle_type == "supplementary":
            total_sum = 180
        elif angle_type == "complementary":
            total_sum = 90
        else:  # interior polygon
            total_sum = (polygon_sides - 2) * 180

        # Calculate unknown angle
        known_sum = sum(known_angles)
        unknown_angle = total_sum - known_sum

        # Round to integer
        result = int(round(unknown_angle))

        description = f"Angle chasing: total={total_sum}°, known={known_angles}, unknown={result}°"

        return MethodResult(
            value=result,
            description=description,
            metadata={
                "known_angles": known_angles,
                "polygon_sides": polygon_sides,
                "angle_type": angle_type,
                "total_sum": total_sum
            }
        )

    def can_invert(self):
        return False


@register_technique
class ApolloniusTheorem(MethodBlock):
    """Apollonius's theorem on medians."""
    def __init__(self):
        super().__init__()
        self.name = "apollonius_theorem"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle"]

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 10)
        b = random.randint(3, 10)
        c = random.randint(3, 10)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        a = params.get("a", 5)
        b = params.get("b", 6)
        c = params.get("c", 7)
        result = 2 * (a**2 + b**2) - c**2
        return MethodResult(
            value=result,
            description=f"Apollonius: 2({a}² + {b}²) - {c}² = {result}",
            params=params
        )

    def can_invert(self):
        return False


@register_technique
class AreaDecomposition(MethodBlock):
    """
    Decompose area into parts or compute unknown part from total and known parts.

    Supports multiple operations:
    - 'find_unknown': Given total_area and known_parts, find the remaining area
    - 'sum_parts': Sum a list of partial areas to get total
    - 'ratio_split': Split total area by ratio into parts
    - 'subtract': Subtract inner area from outer area

    Parameters:
        operation: 'find_unknown', 'sum_parts', 'ratio_split', or 'subtract'
        total_area: total area (for find_unknown, ratio_split)
        known_parts: list of known partial areas (for find_unknown, sum_parts)
        ratio: list of ratio parts (for ratio_split, e.g., [2, 3] means 2:3 split)
        outer_area: outer area (for subtract)
        inner_area: inner area (for subtract)
    """
    def __init__(self):
        super().__init__()
        self.name = "area_decomposition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "area", "decomposition"]

    def generate_parameters(self, input_value=None):
        # Default: find unknown area in a triangle split into regions
        total = random.randint(50, 200)
        part1 = random.randint(10, total // 3)
        part2 = random.randint(10, total // 3)
        return {
            "operation": "find_unknown",
            "total_area": total,
            "known_parts": [part1, part2]
        }

    def compute(self, input_value, params):
        operation = params.get("operation", "find_unknown")

        if operation == "find_unknown":
            total_area = params.get("total_area", 100)
            known_parts = params.get("known_parts", [])

            # Convert to list if single value
            if isinstance(known_parts, (int, float)):
                known_parts = [known_parts]
            elif not isinstance(known_parts, list):
                known_parts = list(known_parts) if known_parts else []

            known_sum = sum(known_parts)
            result = int(round(total_area - known_sum))
            if result < 0:
                result = abs(result)

            description = f"Area decomposition: total={total_area}, known parts={known_parts}, unknown={result}"
            metadata = {"total_area": total_area, "known_parts": known_parts}

        elif operation == "sum_parts":
            known_parts = params.get("known_parts", [])
            if isinstance(known_parts, (int, float)):
                known_parts = [known_parts]
            elif not isinstance(known_parts, list):
                known_parts = list(known_parts) if known_parts else []

            result = int(round(sum(known_parts)))
            description = f"Sum of areas: {known_parts} = {result}"
            metadata = {"parts": known_parts}

        elif operation == "ratio_split":
            total_area = params.get("total_area", 100)
            ratio = params.get("ratio", [1, 1])
            if isinstance(ratio, (int, float)):
                ratio = [ratio, 1]

            ratio_sum = sum(ratio)
            # Return the first ratio's portion
            result = int(round(total_area * ratio[0] / ratio_sum))
            description = f"Ratio split: {total_area} in ratio {ratio[0]}:{ratio[1] if len(ratio) > 1 else 1} = {result}"
            metadata = {"total_area": total_area, "ratio": ratio}

        elif operation == "subtract":
            outer_area = params.get("outer_area", 100)
            inner_area = params.get("inner_area", 25)
            result = int(round(outer_area - inner_area))
            if result < 0:
                result = abs(result)
            description = f"Subtract areas: {outer_area} - {inner_area} = {result}"
            metadata = {"outer_area": outer_area, "inner_area": inner_area}

        else:
            # Default fallback
            result = input_value if input_value is not None else 0
            description = f"Area decomposition with unknown operation: {operation}"
            metadata = {"operation": operation}

        return MethodResult(
            value=result,
            description=description,
            metadata=metadata
        )

    def can_invert(self):
        return False


@register_technique
class AreaSineFormula(MethodBlock):
    """Compute triangle area using sine formula: A = (1/2)ab*sin(C)."""
    def __init__(self):
        super().__init__()
        self.name = "area_sine_formula"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "area"]

    def validate_params(self, params, prev_value=None):
        """Validate a > 0, b > 0, and 0 < angle < 180 for area sine formula."""
        a = params.get("a")
        b = params.get("b")
        angle = params.get("angle", 90)
        if a is None or b is None:
            return False
        return a > 0 and b > 0 and 0 < angle < 180

    def generate_parameters(self, input_value=None):
        a = random.randint(3, 20)
        b = random.randint(3, 20)
        angle = random.choice([30, 45, 60, 90])
        return {"a": a, "b": b, "angle": angle}

    def compute(self, input_value, params):
        # Get required parameters with validation
        a = params.get("a")
        b = params.get("b")
        angle = params.get("angle", 90)  # Default to 90 degrees for right triangle

        if a is None or b is None:
            raise ValueError(
                f"area_sine_formula requires 'a' and 'b' (two sides of triangle). "
                f"Got: {params}. Usage: area_sine_formula(a, b, angle) or area_sine_formula(a, b) for right triangle."
            )

        angle_rad = math.radians(angle)
        area = 0.5 * a * b * math.sin(angle_rad)
        result = int(round(area))
        return MethodResult(
            value=result,
            description=f"Triangle area with sides {a}, {b} and angle {angle}°: A = {result}",
            metadata={"a": a, "b": b, "angle": angle}
        )

    def can_invert(self):
        return False


@register_technique
class AreaMaximizationTriangles(MethodBlock):
    """
    Area maximization for triangles with fixed constraints.

    Given fixed perimeter or fixed base and height constraints,
    find the maximum area configuration.

    For fixed perimeter P, max area is achieved by equilateral triangle:
    Area_max = (P^2 * sqrt(3)) / 36

    For fixed base b and perimeter P, the max area triangle is isosceles.
    """
    def __init__(self):
        super().__init__()
        self.name = "area_maximization_triangles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "optimization", "triangle"]

    def generate_parameters(self, input_value=None):
        # Generate a perimeter that gives nice integer-ish areas
        perimeter = random.choice([12, 18, 24, 30, 36, 42, 48, 54, 60])
        problem_type = random.choice(["equilateral_max", "isosceles_fixed_base"])
        base = perimeter // 4 if problem_type == "isosceles_fixed_base" else None
        return {"perimeter": perimeter, "problem_type": problem_type, "base": base}

    def compute(self, input_value, params):
        perimeter = params.get("perimeter", 12)
        problem_type = params.get("problem_type", "equilateral_max")
        base = params.get("base")

        if problem_type == "equilateral_max":
            # For fixed perimeter, equilateral triangle has max area
            # Side = P/3, Area = (side^2 * sqrt(3))/4 = (P^2 * sqrt(3))/36
            side = perimeter / 3
            area = (side ** 2 * math.sqrt(3)) / 4
            result = int(round(area))
            description = f"Max area for perimeter {perimeter}: equilateral with side {side:.2f}, area = {result}"
        else:
            # Isosceles triangle with fixed base b and perimeter P
            # Legs = (P - b) / 2
            # Height = sqrt(leg^2 - (b/2)^2)
            # Area = (1/2) * b * h
            if base is None:
                base = perimeter // 4
            leg = (perimeter - base) / 2
            half_base = base / 2
            if leg > half_base:
                height = math.sqrt(leg ** 2 - half_base ** 2)
                area = 0.5 * base * height
            else:
                area = 0
            result = int(round(area))
            description = f"Isosceles with base {base}, legs {leg:.2f}: area = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"perimeter": perimeter, "problem_type": problem_type}
        )

    def can_invert(self):
        return False


@register_technique
class CevaMenelausCombined(MethodBlock):
    """
    Combined application of Ceva and Menelaus theorems.

    Ceva's Theorem: Three cevians (lines from vertices through opposite sides) are
    concurrent if and only if (AF/FB) * (BD/DC) * (CE/EA) = 1.

    Menelaus' Theorem: A line intersects the sides of a triangle at three points
    if and only if (AF/FB) * (BD/DC) * (CE/EA) = -1 (with signed ratios).

    Combined: These theorems provide powerful tools for proving collinearity and concurrency.
    """
    def __init__(self):
        super().__init__()
        self.name = "ceva_menelaus_combined"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "theorem"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value is not None else random.randint(2, 10)
        return {"a": a}

    def compute(self, input_value, params):
        a = params.get("a", 5)
        # Ceva's theorem: product of three ratios = 1
        # For a configuration with a, a+1, a+2 as segment lengths
        # The product (a/(a+1)) * ((a+1)/(a+2)) * ((a+2)/a) = 1
        # But we can use the binomial coefficient C(a+2, 3) as a related quantity
        result = (a * (a + 1) * (a + 2)) // 6
        return MethodResult(
            value=result,
            description=f"Ceva-Menelaus: C({a+2},3) = {a}*{a+1}*{a+2}/6 = {result}",
            metadata={"a": a, "formula": "C(a+2,3)"}
        )

    def can_invert(self):
        return False


@register_technique
class EulerOiDistance(MethodBlock):
    """Compute Euler distance OI between circumcenter and incenter."""
    def __init__(self):
        super().__init__()
        self.name = "euler_oi_distance"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "centers"]

    def generate_parameters(self, input_value=None):
        # Generate valid triangle sides using triangle inequality
        a = random.randint(5, 30)
        b = random.randint(5, 30)
        c_min = abs(a - b) + 1
        c_max = a + b - 1
        if c_min > c_max:
            a, b = 7, 10
            c_min, c_max = 4, 16
        c = random.randint(c_min, min(c_max, 30))
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        import math
        a, b, c = params.get("a", 10), params.get("b", 10), params.get("c", 10)
        # Euler's formula: OI^2 = R^2 - 2*R*r
        s = (a + b + c) / 2.0
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        R = (a * b * c) / (4.0 * area)
        r = area / s
        oi_sq = R * R - 2.0 * R * r
        oi = math.sqrt(max(oi_sq, 0))
        result = round(oi)
        return MethodResult(
            value=result,
            description=f"Euler OI distance for triangle ({a},{b},{c}): OI = sqrt(R²-2Rr) = sqrt({R:.2f}²-2·{R:.2f}·{r:.2f}) ≈ {oi:.4f} → {result}",
            metadata={"a": a, "b": b, "c": c, "R": R, "r": r, "OI": oi}
        )

    def can_invert(self):
        return False


@register_technique
class EulerOgDistance(MethodBlock):
    """
    Compute Euler distance OG between circumcenter O and centroid G.

    For a triangle with circumradius R and sides a, b, c:
    OG^2 = R^2 - (a^2 + b^2 + c^2) / 9

    The centroid G divides the Euler line in ratio OG:GH = 1:2
    """
    def __init__(self):
        super().__init__()
        self.name = "euler_og_distance"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "triangle", "euler_line", "centers"]

    def generate_parameters(self, input_value=None):
        # Use Pythagorean triples for nice calculations
        triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
        a, b, c = random.choice(triples)
        scale = random.randint(1, 3)
        return {"a": a * scale, "b": b * scale, "c": c * scale}

    def compute(self, input_value, params):
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)

        # Calculate circumradius R = abc / (4 * Area)
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            return MethodResult(
                value=1,
                description="Invalid triangle",
                metadata={"error": "degenerate"}
            )

        area = math.sqrt(area_sq)
        R = (a * b * c) / (4 * area)

        # OG^2 = R^2 - (a^2 + b^2 + c^2) / 9
        og_squared = R**2 - (a**2 + b**2 + c**2) / 9

        if og_squared < 0:
            og_squared = abs(og_squared)  # Handle numerical issues

        og = math.sqrt(og_squared)
        result = int(round(og * 10))  # Scale for precision

        return MethodResult(
            value=result,
            description=f"Euler distance OG for triangle ({a},{b},{c}): OG = {og:.3f}, encoded as {result}",
            metadata={"a": a, "b": b, "c": c, "circumradius": R, "og_distance": og}
        )

    def can_invert(self):
        return False


@register_technique
class IsoscelesProperties(MethodBlock):
    """Compute properties of isosceles triangles."""
    def __init__(self):
        super().__init__()
        self.name = "isosceles_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "isosceles"]

    def generate_parameters(self, input_value=None):
        leg = random.randint(5, 20)
        base = random.randint(3, 20)
        return {"leg": leg, "base": base}

    def compute(self, input_value, params):
        leg = params.get("leg", 10)
        base = params.get("base", 2)
        altitude_sq = leg**2 - (base/2)**2
        if altitude_sq < 0:
            altitude = 0
        else:
            altitude = math.sqrt(altitude_sq)
        result = int(round(altitude))
        return MethodResult(
            value=result,
            description=f"Isosceles triangle with legs {leg} and base {base}: altitude = {result}",
            metadata={"leg": leg, "base": base}
        )

    def can_invert(self):
        return False


@register_technique
class MedialTriangleProperties(MethodBlock):
    """
    Compute properties of the medial triangle formed by connecting the midpoints
    of a triangle's sides. The medial triangle has sides equal to half the original
    triangle's sides and area equal to 1/4 of the original triangle's area.
    Uses Heron's formula to compute areas given three side lengths.
    """
    def __init__(self):
        super().__init__()
        self.name = "medial_triangle_properties"
        self.input_type = "triangle"
        self.output_type = "geometric_value"
        self.difficulty = 3
        self.tags = ["geometry", "triangle", "medial", "properties"]

    def validate_params(self, params, prev_value=None):
        """Validate triangle inequality for medial triangle calculation."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        if a <= 0 or b <= 0 or c <= 0:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, input_value=None):
        """Generate valid triangle sides for medial triangle."""
        while True:
            a = random.randint(4, 20)
            b = random.randint(4, 20)
            c = random.randint(abs(a-b)+1, a+b-1)
            if a + b > c and b + c > a and a + c > b:
                return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        a = params.get("a", 8)
        b = params.get("b", 10)
        c = params.get("c", 12)

        # Medial triangle sides are half the original triangle sides
        med_a = a / 2.0
        med_b = b / 2.0
        med_c = c / 2.0

        # Compute area of original triangle using Heron's formula
        s = (a + b + c) / 2.0
        area_squared = s * (s - a) * (s - b) * (s - c)
        original_area = math.sqrt(abs(area_squared))

        # Medial triangle has 1/4 the area of original
        medial_area = original_area / 4.0

        # Return rounded medial area as integer
        result = round(medial_area)

        return MethodResult(
            value=result,
            description=f"Medial triangle area: Original area={original_area:.2f}, Medial area={medial_area:.2f}",
            metadata={"a": a, "b": b, "c": c, "medial_area": medial_area, "original_area": original_area}
        )

    def can_invert(self):
        return False


@register_technique
class CircumcenterCentroidProperties(MethodBlock):
    """Properties relating circumcenter and centroid."""
    def __init__(self):
        super().__init__()
        self.name = "circumcenter_centroid_properties"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "triangle"]

    def generate_parameters(self, input_value=None):
        x1, y1 = random.randint(0, 10), random.randint(0, 10)
        x2, y2 = random.randint(0, 10), random.randint(0, 10)
        x3, y3 = random.randint(0, 10), random.randint(0, 10)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}

    def compute(self, input_value, params):
        x1 = params.get("x1", 0)
        y1 = params.get("y1", 0)
        x2 = params.get("x2", 4)
        y2 = params.get("y2", 0)
        x3 = params.get("x3", 2)
        y3 = params.get("y3", 3)
        gx = (x1 + x2 + x3) // 3
        gy = (y1 + y2 + y3) // 3
        result = gx + gy
        return MethodResult(
            value=result,
            description=f"Centroid: ({gx}, {gy}), sum = {result}",
            params=params
        )

    def can_invert(self):
        return False


