"""
Ported geometry and optimization methods from the old composable_techniques system.

Methods ported:
- line_covering_threshold (from ramsey_combinatorics.py)
- max_points_not_forcing_line_cover (from ramsey_combinatorics.py)
- max_triangle_area (from optimization_techniques.py)
- max_rectangle_area (from optimization_techniques.py)
- optimal_box_volume (from optimization_techniques.py)
- constrained_polygon_perimeter (from optimization_techniques.py)
- max_distinct_perimeter_rectangles (from utility_primitives.py)
- sum_min_areas_rectangles (from utility_primitives.py)
- counting_regions_space (from combinatorics.py)
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    random, math, Dict, Any, Optional,
)


def heron_area(a: int, b: int, c: int) -> float:
    """Compute triangle area using Heron's formula."""
    if a + b <= c or a + c <= b or b + c <= a:
        return 0.0
    s = (a + b + c) / 2
    area_squared = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(max(0, area_squared))


# ============================================================================
# LINE COVERING / SYLVESTER-GALLAI TYPE (2 techniques)
# ============================================================================

@register_technique
class LineCoveringThreshold(MethodBlock):
    """Compute the threshold n for line covering problems.

    For a set A in R^2, we ask: what is the minimum n such that
    "every n points can be covered by k lines" implies "all points are on k lines"?

    Formula for k lines: threshold = 2k + 1
    - k=1 line: n = 3 (any 3 collinear points)
    - k=2 lines: n = 5
    - k=3 lines: n = 7
    """

    def __init__(self):
        super().__init__()
        self.name = "line_covering_threshold"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 6
        self.tags = ["combinatorics", "extremal", "geometry", "sylvester_gallai", "incidence"]

    # Known thresholds for k-line covering
    LINE_COVERING_THRESHOLDS = {
        1: 3,    # Any 3 collinear points (trivial)
        2: 5,    # Classic result: n=5 for 2-line covering
        3: 7,    # n=7 for 3-line covering
        4: 9,    # n=9 for 4-line covering
        5: 11,   # Pattern: 2k+1 for k >= 1
    }

    def generate_parameters(self, input_value=None):
        """Generate parameters for line covering threshold."""
        k = input_value if input_value else random.randint(1, 5)
        return {"k": k}

    def compute(self, input_value, params):
        k = params.get("k", input_value) if input_value else params.get("k", 2)

        if k in self.LINE_COVERING_THRESHOLDS:
            result = self.LINE_COVERING_THRESHOLDS[k]
        else:
            # General formula: threshold = 2k + 1 for k >= 1
            result = 2 * k + 1

        description = (
            f"Minimum n such that 'every n points coverable by {k} lines' "
            f"implies 'all points on {k} lines': n = {result}"
        )

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"k": k, "formula": "2k+1"}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Invert to find k from threshold n = 2k+1."""
        if output_value < 3 or output_value % 2 == 0:
            return None
        k = (output_value - 1) // 2
        return k


@register_technique
class MaxPointsNotForcingLineCover(MethodBlock):
    """Compute max points that can avoid k-line covering property.

    This is the complementary bound: what's the maximum number of points
    that can have "every m points coverable by k lines" without "all points on k lines"?

    For k=2 lines, the answer is 4 (since 5 forces the property).
    In general: max = 2k for k lines.
    """

    def __init__(self):
        super().__init__()
        self.name = "max_points_not_forcing_line_cover"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["combinatorics", "extremal", "geometry", "incidence"]

    def generate_parameters(self, input_value=None):
        """Generate parameters."""
        k = input_value if input_value else random.randint(1, 5)
        return {"k": k}

    def compute(self, input_value, params):
        k = params.get("k", input_value) if input_value else params.get("k", 2)

        # For k lines, max non-forcing m is 2k
        result = 2 * k

        description = (
            f"Max m such that 'every m points coverable by {k} lines' "
            f"doesn't force 'all points on {k} lines': m = {result}"
        )

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"k": k}
        )

    def can_invert(self):
        return True

    def invert(self, output_value, params):
        """Invert to find k from max = 2k."""
        if output_value < 2 or output_value % 2 != 0:
            return None
        return output_value // 2


# ============================================================================
# CONSTRAINED GEOMETRIC OPTIMIZATION (4 techniques)
# ============================================================================

@register_technique
class MaxTriangleArea(MethodBlock):
    """Find triangle with max area given perimeter constraint."""

    def __init__(self):
        super().__init__()
        self.name = "max_triangle_area"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["optimization", "geometry", "triangle"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        perimeter = input_value if input_value is not None else random.randint(12, 60)
        return {"perimeter": perimeter}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        p = params.get("perimeter", input_value)

        # Bound perimeter to prevent O(p^2) explosion
        p = min(abs(p) if p else 30, 200)

        # For integer sides, search for best triangle
        max_area = 0
        best_triangle = (0, 0, 0)

        for a in range(1, p - 1):
            for b in range(a, p - a):
                c = p - a - b
                if c >= b and a + b > c:  # Valid triangle
                    area = heron_area(a, b, c)
                    if area > max_area:
                        max_area = area
                        best_triangle = (a, b, c)

        # Scale area to integer (multiply by 100 for precision)
        area_scaled = int(max_area * 100)

        return MethodResult(
            value=area_scaled,
            description=f"Max triangle area with perimeter {p}: {max_area:.2f} from sides {best_triangle}",
            params=params,
            metadata={"perimeter": p, "best_triangle": best_triangle, "area": max_area}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MaxRectangleArea(MethodBlock):
    """Find rectangle with max area given perimeter constraint."""

    def __init__(self):
        super().__init__()
        self.name = "max_rectangle_area"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["optimization", "geometry", "rectangle"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        perimeter = input_value if input_value is not None else random.randint(20, 100)
        return {"perimeter": perimeter}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        p = params.get("perimeter", input_value)

        # For integer sides: perimeter = 2(a + b), so a + b = p/2
        if p % 2 != 0:
            return MethodResult(
                value=0,
                description=f"No integer rectangle with odd perimeter {p}",
                params=params,
                metadata={"perimeter": p}
            )

        half_p = p // 2
        max_area = 0
        best_rect = (0, 0)

        for a in range(1, half_p):
            b = half_p - a
            area = a * b
            if area > max_area:
                max_area = area
                best_rect = (a, b)

        return MethodResult(
            value=max_area,
            description=f"Max rectangle area with perimeter {p}: {max_area} from sides {best_rect}",
            params=params,
            metadata={"perimeter": p, "best_rectangle": best_rect}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class OptimalBoxVolume(MethodBlock):
    """Find box dimensions maximizing volume given surface area constraint."""

    def __init__(self):
        super().__init__()
        self.name = "optimal_box_volume"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["optimization", "geometry", "3d"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        surface_area = input_value if input_value is not None else random.randint(50, 200)
        return {"surface_area": surface_area}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        sa = params.get("surface_area", input_value)

        # Surface area = 2(ab + bc + ca)
        # Search for integer solutions
        max_volume = 0
        best_box = (0, 0, 0)

        max_dim = int(math.sqrt(sa)) + 10

        for a in range(1, max_dim):
            for b in range(1, max_dim):
                # Solve for c: 2(ab + c(b + a)) = sa
                # c(b + a) = sa/2 - ab
                if (sa - 2 * a * b) % (2 * (a + b)) == 0:
                    c = (sa - 2 * a * b) // (2 * (a + b))
                    if c > 0:
                        volume = a * b * c
                        if volume > max_volume:
                            max_volume = volume
                            best_box = (a, b, c)

        return MethodResult(
            value=max_volume,
            description=f"Max box volume with surface area {sa}: {max_volume} from dimensions {best_box}",
            params=params,
            metadata={"surface_area": sa, "best_box": best_box}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ConstrainedPolygonPerimeter(MethodBlock):
    """Find regular polygon with n sides fitting in circle of radius r."""

    def __init__(self):
        super().__init__()
        self.name = "constrained_polygon_perimeter"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["optimization", "geometry", "polygon"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 12)
        r = random.randint(10, 50)
        return {"n": n, "r": r}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        r = params.get("r", 5)

        # Regular n-gon inscribed in circle of radius r
        # Side length = 2r * sin(pi/n)
        side_length = 2 * r * math.sin(math.pi / n)
        perimeter = n * side_length

        # Scale to integer
        perimeter_int = int(perimeter * 100)

        return MethodResult(
            value=perimeter_int,
            description=f"Perimeter of regular {n}-gon in circle r={r}: {perimeter:.2f}",
            params=params,
            metadata={"n": n, "r": r, "perimeter": perimeter}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# RECTANGLE TILING OPTIMIZATION (2 techniques)
# ============================================================================

@register_technique
class SumMinAreasRectangles(MethodBlock):
    """
    Sum of minimum areas for k rectangles with distinct perimeters in N x N grid.

    For a rectangle with semi-perimeter s in an N x N square:
    - If s <= N: min_area(s) = s-1 (e.g., 1 x (s-1))
    - If s > N: min_area(s) = N*(s-N) (constrained by square side)

    This computes sum of min_area(s) for s = 2, 3, ..., k+1.
    """

    def __init__(self):
        super().__init__()
        self.name = "sum_min_areas_rectangles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "geometry", "optimization", "rectangles"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        N = input_value if input_value is not None else random.randint(20, 200)
        k = random.randint(5, min(N, 100))
        return {"k": k, "N": N}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        k = params.get("k", 5)
        N = params.get("N", input_value)

        # Sum of minimum areas for rectangles with semi-perimeters 2, 3, ..., k+1
        total = 0
        for s in range(2, k + 2):
            if s <= N:
                # min_area = s - 1
                total += s - 1
            else:
                # min_area = N * (s - N)
                total += N * (s - N)

        description = (
            f"Sum of min areas for {k} rectangles "
            f"(semi-perimeters 2..{k+1}) in {N}x{N} = {total}"
        )
        return MethodResult(
            value=total,
            description=description,
            params=params,
            metadata={
                "k": k,
                "N": N,
                "formula": "sum(min_area(s) for s=2..k+1)"
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MaxDistinctPerimeterRectangles(MethodBlock):
    """
    Find the maximum number of rectangles with distinct perimeters
    that can tile an N x N square.

    Algorithm:
    - Binary search for max k where sum_min_areas(k, N) <= N^2
    - sum_min_areas(k, N) = sum of minimum areas for k rectangles
      with semi-perimeters 2..k+1

    For N=500: Answer is 520.
    """

    def __init__(self):
        super().__init__()
        self.name = "max_distinct_perimeter_rectangles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["combinatorics", "geometry", "optimization", "binary_search"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        N = input_value if input_value is not None else random.randint(100, 1000)
        return {"N": N}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        N = params.get("N", input_value)

        def sum_min_areas(k, N):
            """Sum of min areas for k rectangles with semi-perimeters 2..k+1."""
            total = 0
            for s in range(2, k + 2):
                if s <= N:
                    total += s - 1
                else:
                    total += N * (s - N)
            return total

        # Binary search for max k where sum_min_areas(k, N) <= N^2
        target = N * N
        lo, hi = 1, 2 * N
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sum_min_areas(mid, N) <= target:
                lo = mid
            else:
                hi = mid - 1

        result = lo
        description = (
            f"Max k where sum_min_areas(k, {N}) <= {N}^2 = {result}"
        )
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"N": N, "target_area": target, "actual_sum": sum_min_areas(result, N)}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# COUNTING / SPACE DIVISION (1 technique)
# ============================================================================

@register_technique
class CountingRegionsSpace(MethodBlock):
    """Count regions created by n hyperplanes in d-dimensional space.

    The number of regions formed by n hyperplanes in general position in R^d
    is given by: R(n,d) = sum_{i=0}^{d} C(n,i)

    This represents how many distinct regions n hyperplanes can divide
    d-dimensional space into when positioned in general position.
    """

    def __init__(self):
        super().__init__()
        self.name = "counting_regions_space"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "hyperplanes", "space-division"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(3, 15)
        # Generate both number of hyperplanes and dimension
        d = random.randint(2, min(5, n))
        return {"n": n, "d": d}

    def compute(self, input_value, params):
        n = params.get("n", 5)
        d = params.get("d", 3)

        # Compute sum of binomial coefficients C(n,i) for i=0 to d
        result = 0
        for i in range(min(d + 1, n + 1)):
            result += math.comb(n, i)

        description = (
            f"Regions from {n} hyperplanes in {d}D: "
            f"sum(C({n},{i}) for i=0..{d}) = {result}"
        )
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "d": d}
        )

    def can_invert(self) -> bool:
        return False
