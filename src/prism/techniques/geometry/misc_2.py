"""
Geometry Techniques - Misc (Part 2)

Auto-generated from composable_techniques/geometry.py
"""

from ._common import (
    MethodBlock, MethodResult, register_technique,
    ensure_tuple, line_intersection,
    random, math, np, Dict, Any, Optional, Tuple, List
)


@register_technique
class LatticePointCounting(MethodBlock):
    """
    Count lattice points in geometric regions using Pick's theorem.

    Pick's theorem: A = i + b/2 - 1
    where A = area, i = interior points, b = boundary points

    Therefore: i = A - b/2 + 1

    Supports:
    - Rectangle counting: interior and boundary points
    - Triangle counting using Pick's theorem
    - Circle counting (approximate using bounds)
    """
    def __init__(self):
        super().__init__()
        self.name = "lattice_point_counting"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "lattice", "picks_theorem", "counting"]

    def generate_parameters(self, input_value=None):
        shape = random.choice(["rectangle", "triangle", "segment"])
        if shape == "rectangle":
            width = random.randint(2, 15)
            height = random.randint(2, 15)
            return {"shape": shape, "width": width, "height": height}
        elif shape == "triangle":
            # Generate lattice triangle vertices
            x1, y1 = 0, 0
            x2, y2 = random.randint(2, 10), 0
            x3, y3 = random.randint(0, 10), random.randint(2, 10)
            return {"shape": shape, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}
        else:  # segment
            x1, y1 = 0, 0
            x2, y2 = random.randint(2, 20), random.randint(2, 20)
            return {"shape": shape, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _gcd(self, a, b):
        """Compute GCD of two integers."""
        a, b = abs(int(a)), abs(int(b))
        while b:
            a, b = b, a % b
        return a

    def _boundary_points_segment(self, x1, y1, x2, y2):
        """Count lattice points on segment from (x1,y1) to (x2,y2) excluding endpoints."""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        return self._gcd(dx, dy) - 1

    def compute(self, input_value, params):
        shape = params.get("shape", "rectangle")

        if shape == "rectangle":
            width = params.get("width", 5)
            height = params.get("height", 4)
            # Interior points: (width-1) * (height-1)
            interior = (width - 1) * (height - 1)
            # Boundary points: 2*width + 2*height - 4
            boundary = 2 * width + 2 * height - 4
            # Total = interior + boundary
            total = (width + 1) * (height + 1)
            result = total
            description = f"Rectangle {width}x{height}: {total} lattice points (interior={interior}, boundary={boundary})"

        elif shape == "triangle":
            x1, y1 = params.get("x1", 0), params.get("y1", 0)
            x2, y2 = params.get("x2", 4), params.get("y2", 0)
            x3, y3 = params.get("x3", 2), params.get("y3", 3)

            # Area using shoelace (times 2 for integer arithmetic)
            double_area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

            # Boundary points on each edge (gcd gives interior points on edge + 1)
            b1 = self._gcd(abs(x2 - x1), abs(y2 - y1))  # points including endpoints
            b2 = self._gcd(abs(x3 - x2), abs(y3 - y2))
            b3 = self._gcd(abs(x1 - x3), abs(y1 - y3))
            boundary = b1 + b2 + b3

            # Pick's theorem: A = i + b/2 - 1, so i = A - b/2 + 1
            # 2*A = 2i + b - 2, so 2i = 2A - b + 2
            interior = (double_area - boundary + 2) // 2

            total = interior + boundary
            result = total
            description = f"Triangle with vertices ({x1},{y1}), ({x2},{y2}), ({x3},{y3}): {total} lattice points"

        else:  # segment
            x1, y1 = params.get("x1", 0), params.get("y1", 0)
            x2, y2 = params.get("x2", 6), params.get("y2", 8)
            # Points on segment: gcd(dx, dy) + 1 (including endpoints)
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            result = self._gcd(dx, dy) + 1
            description = f"Segment from ({x1},{y1}) to ({x2},{y2}): {result} lattice points"

        return MethodResult(
            value=result,
            description=description,
            metadata={"shape": shape}
        )

    def can_invert(self):
        return False


@register_technique
class LatticeConvexPolygon(MethodBlock):
    """
    Maximum vertices in convex polygon on n×n lattice grid.
    Formula: sum(totient(k) for k in range(1, n//2 + 1)) + 1

    Based on the fact that convex hull vertices correspond to coprime direction vectors.
    Uses Euler's totient function to count valid direction vectors.
    """
    def __init__(self):
        super().__init__()
        self.name = "lattice_convex_polygon"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 8
        self.tags = ["geometry", "lattice", "totient", "extremal"]

    def generate_parameters(self, input_value=None):
        # For backward generation, we need to find n given target answer
        # This is complex, so we'll use a search approach
        if input_value is not None:
            n = self._find_n_for_answer(input_value)
            if n is None:
                # Fallback: use a random grid size
                n = random.randint(100, 5000)
                n = (n // 2) * 2  # Make it even
        else:
            n = random.randint(100, 5000)
            n = (n // 2) * 2  # Make it even for cleaner formulas
        return {"n": n}

    def compute(self, input_value, params):
        from sympy import totient

        n = params.get("n", 10)

        # Formula: sum(totient(k) for k in range(1, n//2 + 1)) + 1
        total = 1  # Start with 1
        for k in range(1, n // 2 + 1):
            total += int(totient(k))

        return MethodResult(
            value=total,
            description=f"Max convex {total}-gon on {n}×{n} lattice (totient sum + 1)",
            metadata={"n": n, "grid_size": f"{n}×{n}", "max_vertices": total}
        )

    def can_invert(self):
        return True

    def generate(self, target_answer=None):
        """
        Generate a lattice convex polygon problem, optionally targeting a specific answer.
        """
        try:
            if target_answer is not None:
                # Find n such that the formula gives target_answer
                n = self._find_n_for_answer(target_answer)
                if n is None:
                    return {
                        "success": False,
                        "error": f"Cannot find grid size n with max vertices = {target_answer}",
                        "unsupported_target": target_answer,
                        "technique": self.name
                    }
                params = {"n": n}
            else:
                # Forward generation
                params = self.generate_parameters()

            # Compute result
            result = self.compute(None, params)
            n = params.get("n", 10)
            answer = result.value

            # Generate problem statement
            relationship = (
                f"Draw a ${n} \\times {n}$ array of points. What is the largest integer $k$ "
                f"for which it is possible to draw a convex $k$-gon whose vertices are "
                f"chosen from the points in the array?"
            )

            return {
                "success": True,
                "answer": answer,
                "relationship": relationship,
                "technique": self.name,
                "params": params,
                "description": result.description,
                "metadata": result.metadata
            }
        except Exception as e:
            # Return None on exception (not an error dict)
            return None

    def _find_n_for_answer(self, target: int):
        """
        Find grid size n such that sum(totient(k) for k=1..n//2) + 1 = target.

        Uses binary search since the totient sum is monotonically increasing.
        """
        from sympy import totient

        if target < 2:
            return None

        # Binary search for the right n
        # Lower bound: n=2 gives sum(totient(1)) + 1 = 2
        # Upper bound: estimate based on totient sum approximation
        # sum(totient(k) for k=1..m) ≈ 3*m²/π² ≈ 0.304*m²
        # So for target T, m ≈ sqrt(T/0.304), and n ≈ 2*m

        import math
        estimated_m = int(math.sqrt(target / 0.304))
        low = 2
        high = max(estimated_m * 4, 10000)  # Give extra room for search

        best_n = None
        best_diff = float('inf')

        # Binary search
        for _ in range(100):  # Limit iterations
            mid = (low + high) // 2
            mid = (mid // 2) * 2  # Make even

            # Compute sum(totient(k) for k=1..mid//2) + 1
            total = 1
            for k in range(1, mid // 2 + 1):
                total += int(totient(k))

            diff = abs(total - target)
            if diff < best_diff:
                best_diff = diff
                best_n = mid

            if total == target:
                return mid
            elif total < target:
                low = mid + 2
            else:
                high = mid - 2

            if low > high:
                break

        # Return best match if within 10% of target (relaxed for backward generation)
        # Note: Some answers may not be exactly achievable due to formula constraints
        if best_n is not None and best_diff <= max(target * 0.10, 50):
            return best_n

        # If no close match, return None
        return None


@register_technique
class ProjectiveGeometry(MethodBlock):
    """
    Projective geometry calculations.

    Supports:
    - Projective plane PG(2, q): q^2 + q + 1 points and lines
    - Homogeneous coordinates
    - Cross-ratio calculations
    - Duality principles

    For finite projective plane of order n:
    - Number of points = n^2 + n + 1
    - Number of lines = n^2 + n + 1
    - Points per line = n + 1
    - Lines through a point = n + 1
    """
    def __init__(self):
        super().__init__()
        self.name = "projective_geometry"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["geometry", "projective", "finite_geometry"]

    def generate_parameters(self, input_value=None):
        operation = random.choice(["points", "lines", "points_per_line", "cross_ratio"])
        if operation == "cross_ratio":
            # Four collinear points for cross-ratio
            a, b, c, d = 1, 2, 4, 7
            return {"operation": operation, "a": a, "b": b, "c": c, "d": d}
        else:
            # Order of finite projective plane
            n = input_value if input_value is not None else random.randint(2, 10)
            return {"operation": operation, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "points")
        n = params.get("n", 5)

        if operation == "points" or operation == "lines":
            # Finite projective plane PG(2, n): n^2 + n + 1 points and lines
            result = n**2 + n + 1
            description = f"Projective plane PG(2,{n}): {result} points (and lines)"
        elif operation == "points_per_line":
            # Each line has n + 1 points
            result = n + 1
            description = f"Points per line in PG(2,{n}): {result}"
        elif operation == "cross_ratio":
            # Cross-ratio (a,b;c,d) = (a-c)(b-d) / (a-d)(b-c)
            a = params.get("a", 1)
            b = params.get("b", 2)
            c = params.get("c", 4)
            d = params.get("d", 7)

            numerator = (a - c) * (b - d)
            denominator = (a - d) * (b - c)

            if denominator == 0:
                result = float('inf')
                description = "Cross-ratio undefined (points not in general position)"
            else:
                cross_ratio = numerator / denominator
                # Encode as scaled integer
                result = int(round(abs(cross_ratio) * 100))
                description = f"Cross-ratio ({a},{b};{c},{d}) = {cross_ratio:.4f}, encoded as {result}"
        else:
            # Default: number of lines through n general points
            result = n * (n - 1) // 2
            description = f"Lines through {n} general points: {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "n": n}
        )

    def can_invert(self):
        return False


@register_technique
class SurfaceAreaCalculation(MethodBlock):
    """
    Surface area calculation for 3D solids.

    Formulas:
    - Cube: 6 * a^2
    - Rectangular prism: 2(lw + wh + lh)
    - Sphere: 4 * pi * r^2
    - Cylinder: 2*pi*r^2 + 2*pi*r*h (two bases + lateral)
    - Cone: pi*r^2 + pi*r*l (base + lateral, l = slant height)
    """
    def __init__(self):
        super().__init__()
        self.name = "surface_area_calculation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["geometry", "3d", "surface_area"]

    def generate_parameters(self, input_value=None):
        shape = random.choice(["cube", "rectangular_prism", "sphere", "cylinder", "cone"])
        if shape == "cube":
            a = random.randint(2, 10)
            return {"shape": shape, "a": a}
        elif shape == "rectangular_prism":
            l, w, h = random.randint(2, 10), random.randint(2, 10), random.randint(2, 10)
            return {"shape": shape, "l": l, "w": w, "h": h}
        elif shape == "sphere":
            r = random.randint(2, 10)
            return {"shape": shape, "r": r}
        elif shape == "cylinder":
            r, h = random.randint(2, 8), random.randint(3, 12)
            return {"shape": shape, "r": r, "h": h}
        else:  # cone
            r, h = random.randint(2, 8), random.randint(3, 12)
            return {"shape": shape, "r": r, "h": h}

    def compute(self, input_value, params):
        shape = params.get("shape", "cube")

        if shape == "cube":
            a = params.get("a", 5)
            surface_area = 6 * a ** 2
            result = surface_area
            description = f"Cube surface area: 6 * {a}^2 = {result}"
        elif shape == "rectangular_prism":
            l = params.get("l", 3)
            w = params.get("w", 4)
            h = params.get("h", 5)
            surface_area = 2 * (l * w + w * h + l * h)
            result = surface_area
            description = f"Rectangular prism surface area: 2({l}*{w} + {w}*{h} + {l}*{h}) = {result}"
        elif shape == "sphere":
            r = params.get("r", 5)
            surface_area = 4 * math.pi * r ** 2
            result = int(round(surface_area))
            description = f"Sphere surface area: 4*pi*{r}^2 = {result}"
        elif shape == "cylinder":
            r = params.get("r", 3)
            h = params.get("h", 5)
            surface_area = 2 * math.pi * r ** 2 + 2 * math.pi * r * h
            result = int(round(surface_area))
            description = f"Cylinder surface area: 2*pi*{r}^2 + 2*pi*{r}*{h} = {result}"
        else:  # cone
            r = params.get("r", 3)
            h = params.get("h", 4)
            slant = math.sqrt(r ** 2 + h ** 2)
            surface_area = math.pi * r ** 2 + math.pi * r * slant
            result = int(round(surface_area))
            description = f"Cone surface area with slant {slant:.2f}: {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"shape": shape}
        )

    def can_invert(self):
        return False


@register_technique
class TilingTheory(MethodBlock):
    """
    Tiling theory for regular and semi-regular tilings.

    Regular tilings (only 3 exist):
    - Triangles (3.3.3.3.3.3): 6 triangles meet at each vertex
    - Squares (4.4.4.4): 4 squares meet at each vertex
    - Hexagons (6.6.6): 3 hexagons meet at each vertex

    Counting tiles to cover an m x n rectangle:
    - Unit squares: m * n
    - 1x2 dominoes: (m * n) / 2 (if even)
    - 2x2 squares: floor(m/2) * floor(n/2)
    """
    def __init__(self):
        super().__init__()
        self.name = "tiling_theory"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "tiling", "combinatorics"]

    def generate_parameters(self, input_value=None):
        operation = random.choice(["unit_squares", "dominoes", "checkerboard", "vertex_type"])
        m = random.randint(2, 12)
        n = random.randint(2, 12)
        return {"operation": operation, "m": m, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "unit_squares")
        m = params.get("m", 4)
        n = params.get("n", 6)

        if operation == "unit_squares":
            # Count unit squares to tile m x n rectangle
            result = m * n
            description = f"Unit squares to tile {m}x{n} rectangle: {result}"
        elif operation == "dominoes":
            # Count 1x2 dominoes to tile m x n rectangle (if area is even)
            area = m * n
            if area % 2 == 0:
                result = area // 2
                description = f"Dominoes to tile {m}x{n} rectangle: {result}"
            else:
                result = 0
                description = f"Cannot tile {m}x{n} with dominoes (odd area)"
        elif operation == "checkerboard":
            # Count squares of each color on m x n checkerboard
            # Black squares: ceil(m*n/2), White squares: floor(m*n/2)
            black = (m * n + 1) // 2
            white = m * n // 2
            result = black  # Return count of black squares
            description = f"Black squares on {m}x{n} checkerboard: {result}"
        else:  # vertex_type
            # At each vertex of regular n-gon tiling
            # For triangles: 6 meet (interior angle 60)
            # For squares: 4 meet (interior angle 90)
            # For hexagons: 3 meet (interior angle 120)
            n_sides = random.choice([3, 4, 6])
            interior_angle = (n_sides - 2) * 180 / n_sides
            tiles_at_vertex = 360 / interior_angle
            result = int(tiles_at_vertex)
            description = f"Regular {n_sides}-gons at vertex: {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"operation": operation, "m": m, "n": n}
        )

    def can_invert(self):
        return False


@register_technique
class TrigIdentitiesSynthetic(MethodBlock):
    """Apply trigonometric identities synthetically."""
    def __init__(self):
        super().__init__()
        self.name = "trig_identities_synthetic"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "trigonometry"]

    def generate_parameters(self, input_value=None):
        angle = random.choice([30, 45, 60])
        return {"angle": angle}

    def compute(self, input_value, params):
        angle = params.get("angle", 10)
        angle_rad = math.radians(angle)
        result = int(100 * (math.sin(angle_rad)**2 + math.cos(angle_rad)**2))
        return MethodResult(
            value=result,
            description=f"sin²({angle}°) + cos²({angle}°) = 1 (scaled: {result/100})",
            metadata={"angle": angle}
        )

    def can_invert(self):
        return False


@register_technique
class UnfoldingNetsPathfinding(MethodBlock):
    """
    Shortest paths on surfaces by unfolding to nets.

    Classic problem: Ant on a box
    - For a box with dimensions l x w x h
    - Find shortest path from one corner to opposite corner
    - Unfold the box and use straight-line distance

    For rectangular box, unfold along different edges:
    - Path 1: sqrt((l + w)^2 + h^2)
    - Path 2: sqrt((l + h)^2 + w^2)
    - Path 3: sqrt((w + h)^2 + l^2)
    - Shortest path is minimum of these
    """
    def __init__(self):
        super().__init__()
        self.name = "unfolding_nets_pathfinding"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["geometry", "3d", "pathfinding", "unfolding"]

    def generate_parameters(self, input_value=None):
        l = random.randint(3, 12)
        w = random.randint(3, 12)
        h = random.randint(3, 12)
        operation = random.choice(["shortest_path", "all_paths", "cube_diagonal"])
        return {"l": l, "w": w, "h": h, "operation": operation}

    def compute(self, input_value, params):
        l = params.get("l", 3)
        w = params.get("w", 4)
        h = params.get("h", 5)
        operation = params.get("operation", "shortest_path")

        if operation == "shortest_path":
            # Compute all three unfolding paths and return minimum
            path1 = math.sqrt((l + w) ** 2 + h ** 2)
            path2 = math.sqrt((l + h) ** 2 + w ** 2)
            path3 = math.sqrt((w + h) ** 2 + l ** 2)
            shortest = min(path1, path2, path3)
            result = int(round(shortest * 10))  # Scale for precision
            description = f"Shortest surface path on {l}x{w}x{h} box: {shortest:.3f}, encoded as {result}"
        elif operation == "all_paths":
            # Return the sum of all three paths (encoded)
            path1 = math.sqrt((l + w) ** 2 + h ** 2)
            path2 = math.sqrt((l + h) ** 2 + w ** 2)
            path3 = math.sqrt((w + h) ** 2 + l ** 2)
            total = path1 + path2 + path3
            result = int(round(total))
            description = f"Sum of all unfolding paths: {total:.2f}, rounded to {result}"
        else:  # cube_diagonal
            # For a cube with side s, diagonal across surface
            # Best path = sqrt((2s)^2 + s^2) = s*sqrt(5)
            s = l  # Use l as side length
            diagonal = s * math.sqrt(5)
            result = int(round(diagonal * 10))
            description = f"Cube side {s} surface diagonal: {s}*sqrt(5) = {diagonal:.3f}, encoded as {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"l": l, "w": w, "h": h, "operation": operation}
        )

    def can_invert(self):
        return False


