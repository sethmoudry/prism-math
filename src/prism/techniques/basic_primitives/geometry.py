"""
Basic geometry primitives.
"""

import math
import random
from typing import List, Tuple, Union, Optional
from sympy import (
    symbols, simplify, expand, sin, cos, tan, exp, log, sqrt, sympify
)

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class Midpoint(MethodBlock):
    """Compute midpoint of line segment between two points."""

    def __init__(self):
        super().__init__()
        self.name = "midpoint"
        self.input_type = "none"
        self.output_type = "point"
        self.difficulty = 1
        self.tags = ["geometry", "coordinates"]

    def generate_parameters(self, input_value=None):
        """Generate random points."""
        return {
            "p1": (random.randint(-10, 10), random.randint(-10, 10)),
            "p2": (random.randint(-10, 10), random.randint(-10, 10))
        }

    def compute(self, input_value, params):
        """Compute midpoint of line segment between two points."""
        p1 = params.get("p1", (0, 0))
        p2 = params.get("p2", (4, 6))

        # Validate that p1 and p2 are tuples/lists, not scalars
        if not isinstance(p1, (tuple, list)) or len(p1) != 2:
            p1 = (0, 0)
        if not isinstance(p2, (tuple, list)) or len(p2) != 2:
            p2 = (4, 6)

        result = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        return MethodResult(
            value=result,
            description=f"Midpoint of {p1} and {p2} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TriangleAreaHeron(MethodBlock):
    """Compute triangle area using Heron's formula."""

    def __init__(self):
        super().__init__()
        self.name = "triangle_area_heron"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "area"]

    def validate_params(self, params, prev_value=None):
        """Triangle requires sides satisfying triangle inequality."""
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        if a is None or b is None or c is None:
            return False
        return a + b > c and b + c > a and a + c > b

    def generate_parameters(self, input_value=None):
        """Generate random triangle sides satisfying triangle inequality."""
        a = random.randint(3, 10)
        b = random.randint(3, 10)
        # Ensure triangle inequality
        c = random.randint(abs(a - b) + 1, a + b - 1)
        return {"a": a, "b": b, "c": c}

    def compute(self, input_value, params):
        """Compute triangle area using Heron's formula."""
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)

        # Check triangle inequality
        if a + b <= c or b + c <= a or c + a <= b:
            raise ValueError("Invalid triangle: sides don't satisfy triangle inequality")

        s = (a + b + c) / 2  # semi-perimeter
        area_squared = s * (s - a) * (s - b) * (s - c)

        if area_squared < 0:
            raise ValueError("Invalid triangle configuration")

        result = math.sqrt(area_squared)
        return MethodResult(
            value=result,
            description=f"Area of triangle with sides {a}, {b}, {c} = {result:.4f} (Heron's formula)",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class PerpendicularBisector(MethodBlock):
    """Find perpendicular bisector of line segment."""

    def __init__(self):
        super().__init__()
        self.name = "perpendicular_bisector"
        self.input_type = "none"
        self.output_type = "line"
        self.difficulty = 2
        self.tags = ["geometry", "lines"]

    def generate_parameters(self, input_value=None):
        """Generate random segment."""
        return {
            "p1": (random.randint(-10, 10), random.randint(-10, 10)),
            "p2": (random.randint(-10, 10), random.randint(-10, 10))
        }

    def compute(self, input_value, params):
        """Find perpendicular bisector of line segment."""
        p1 = params.get("p1", (0, 0))
        p2 = params.get("p2", (4, 6))

        # Midpoint
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        midpoint = (mx, my)

        # Direction vector of segment
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Perpendicular direction (rotate 90 degrees)
        perp_direction = (-dy, dx)

        # Return segment length as the value (useful for chaining)
        import math
        segment_length = math.sqrt(dx**2 + dy**2)

        return MethodResult(
            value=int(round(segment_length)),  # Return segment length, not dict
            description=f"Perpendicular bisector of segment {p1} to {p2}, segment length = {segment_length:.2f}",
            params=params,
            metadata={"midpoint": midpoint, "direction": perp_direction, "segment_length": segment_length}
        )

    def can_invert(self):
        return False


@register_technique
class LineIntersection(MethodBlock):
    """Find intersection point of two lines.

    Lines can be specified as:
    - {"midpoint": (x, y), "direction": (dx, dy)} - point + direction format
    - ((x1, y1), (x2, y2)) - two points on the line
    """

    def __init__(self):
        super().__init__()
        self.name = "line_intersection"
        self.input_type = "none"
        self.output_type = "point"
        self.difficulty = 2
        self.tags = ["geometry", "lines", "intersection"]

    def generate_parameters(self, input_value=None):
        """Generate two intersecting lines."""
        return {
            "line1": {"midpoint": (0, 0), "direction": (1, 1)},
            "line2": {"midpoint": (1, 0), "direction": (1, -1)}
        }

    def compute(self, input_value, params):
        """Find intersection of two lines."""
        line1 = params.get("line1")
        line2 = params.get("line2")

        # Extract point and direction from line1
        if isinstance(line1, dict):
            p1 = line1.get("midpoint", (0, 0))
            d1 = line1.get("direction", (1, 0))
        elif isinstance(line1, (list, tuple)) and len(line1) == 2:
            p1 = line1[0]
            d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
        else:
            p1, d1 = (0, 0), (1, 0)

        # Extract point and direction from line2
        if isinstance(line2, dict):
            p2 = line2.get("midpoint", (0, 0))
            d2 = line2.get("direction", (0, 1))
        elif isinstance(line2, (list, tuple)) and len(line2) == 2:
            p2 = line2[0]
            d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
        else:
            p2, d2 = (0, 0), (0, 1)

        # Solve for intersection: p1 + t*d1 = p2 + s*d2
        # d1x*t - d2x*s = p2x - p1x
        # d1y*t - d2y*s = p2y - p1y
        det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
        if abs(det) < 1e-10:
            # Parallel lines
            return MethodResult(
                value=None,
                description="Lines are parallel, no intersection",
                params=params
            )

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        t = (dx * (-d2[1]) - dy * (-d2[0])) / det

        # Intersection point
        ix = p1[0] + t * d1[0]
        iy = p1[1] + t * d1[1]
        intersection = (ix, iy)

        return MethodResult(
            value=intersection,
            description=f"Intersection of two lines at {intersection}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TriangleArea(MethodBlock):
    """Compute triangle area from vertex coordinates."""

    def __init__(self):
        super().__init__()
        self.name = "triangle_area"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["geometry", "triangle", "area"]

    def generate_parameters(self, input_value=None):
        """Generate random triangle vertices."""
        return {
            "p1": (random.randint(-10, 10), random.randint(-10, 10)),
            "p2": (random.randint(-10, 10), random.randint(-10, 10)),
            "p3": (random.randint(-10, 10), random.randint(-10, 10))
        }

    def compute(self, input_value, params):
        """Compute triangle area from vertex coordinates using cross product."""
        p1 = params.get("p1", (0, 0))
        p2 = params.get("p2", (4, 0))
        p3 = params.get("p3", (0, 3))

        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area = abs(
            p1[0] * (p2[1] - p3[1]) +
            p2[0] * (p3[1] - p1[1]) +
            p3[0] * (p1[1] - p2[1])
        ) / 2

        return MethodResult(
            value=area,
            description=f"Area of triangle with vertices {p1}, {p2}, {p3} = {area}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TriangleCentroid(MethodBlock):
    """Compute centroid (center of mass) of triangle."""

    def __init__(self):
        super().__init__()
        self.name = "triangle_centroid"
        self.input_type = "none"
        self.output_type = "point"
        self.difficulty = 1
        self.tags = ["geometry", "triangle", "centroids"]

    def generate_parameters(self, input_value=None):
        """Generate random triangle vertices."""
        return {
            "p1": (random.randint(-10, 10), random.randint(-10, 10)),
            "p2": (random.randint(-10, 10), random.randint(-10, 10)),
            "p3": (random.randint(-10, 10), random.randint(-10, 10))
        }

    def compute(self, input_value, params):
        """Compute centroid (average of vertices)."""
        p1 = params.get("p1", (0, 0))
        p2 = params.get("p2", (3, 0))
        p3 = params.get("p3", (0, 3))

        cx = (p1[0] + p2[0] + p3[0]) / 3
        cy = (p1[1] + p2[1] + p3[1]) / 3
        result = (cx, cy)

        return MethodResult(
            value=result,
            description=f"Centroid of triangle with vertices {p1}, {p2}, {p3} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TriangleInequality(MethodBlock):
    """Check if three sides can form a valid triangle."""

    def __init__(self):
        super().__init__()
        self.name = "triangle_inequality"
        self.input_type = "none"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["geometry", "triangle", "inequalities"]

    def generate_parameters(self, input_value=None):
        """Generate random side lengths."""
        return {
            "a": random.randint(1, 10),
            "b": random.randint(1, 10),
            "c": random.randint(1, 10)
        }

    def compute(self, input_value, params):
        """Check if sides satisfy triangle inequality."""
        a = params.get("a", 3)
        b = params.get("b", 4)
        c = params.get("c", 5)

        result = (a + b > c) and (b + c > a) and (c + a > b)

        return MethodResult(
            value=1 if result else 0,  # Convert to int for consistency
            description=f"Triangle inequality for sides {a}, {b}, {c}: {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class IsAcuteAngle(MethodBlock):
    """Check if an angle is acute (< 90 degrees)."""

    def __init__(self):
        super().__init__()
        self.name = "is_acute_angle"
        self.input_type = "number"
        self.output_type = "boolean"
        self.difficulty = 1
        self.tags = ["geometry", "angles"]

    def generate_parameters(self, input_value=None):
        """Generate random angle in degrees."""
        return {"angle": random.uniform(0, 180)}

    def compute(self, input_value, params):
        """Check if angle < 90 degrees."""
        angle = input_value or params.get("angle", 45)

        if angle is None:
            raise ValueError("angle must be provided")

        # Check if acute (0 < angle < 90)
        result = 0 < angle < 90

        return MethodResult(
            value=1 if result else 0,
            description=f"Is {angle}° acute? {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class DivideArea(MethodBlock):
    """Divide area by n: area / n."""

    def __init__(self):
        super().__init__()
        self.name = "divide_area"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["geometry", "area", "division"]

    def generate_parameters(self, input_value=None):
        """Generate random area and divisor."""
        return {
            "area": random.uniform(1, 1000),
            "n": random.randint(2, 10)
        }

    def compute(self, input_value, params):
        """Divide area by n."""
        area = input_value or params.get("area", 100)
        n = params.get("n", 4)

        if area is None or n is None:
            raise ValueError("Both area and n must be provided")

        if n == 0:
            raise ValueError("Cannot divide by zero")

        result = area / n

        return MethodResult(
            value=result,
            description=f"{area} / {n} = {result}",
            params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False

