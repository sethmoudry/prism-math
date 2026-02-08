"""
Miscellaneous combinatorics methods.

This module contains:
- Counting constraints (CountingConstraints, CountingIntersections, etc.)
- Permutation methods (PermutationDecomposition, PermutationWithRepetition, etc.)
- Multiset permutations
- Rectangle counting
- Legendre valuation
- Utility techniques
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .special_numbers_derangements_partitions import derangement


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def multinomial_coefficient(n: int, *ks) -> int:
    """Compute multinomial coefficient n! / (k1! * k2! * ... * km!)."""
    if n != sum(ks):
        return 0
    result = math.factorial(n)
    for k in ks:
        result //= math.factorial(k)
    return result


def count_non_overlapping_placements(word_len: int, pattern_len: int, num_patterns: int) -> int:
    """Count ways to place num_patterns non-overlapping patterns in a word."""
    if num_patterns == 0:
        return 1
    if num_patterns * pattern_len > word_len:
        return 0

    def count_placements(first_available: int, remaining: int) -> int:
        if remaining == 0:
            return 1
        if first_available + pattern_len * remaining > word_len:
            return 0
        total = 0
        for start in range(first_available, word_len - pattern_len * remaining + 1):
            total += count_placements(start + pattern_len, remaining - 1)
        return total

    return count_placements(0, num_patterns)


# ============================================================================
# COUNTING CONSTRAINTS
# ============================================================================

@register_technique
class CountingConstraints(MethodBlock):
    """Count objects satisfying given constraints."""
    def __init__(self):
        super().__init__()
        self.name = "counting_constraints"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "constraints", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        constraint_types = ["no_consecutive", "derangement", "ascending_digits", "sum_to_n"]
        constraint_type = random.choice(constraint_types)
        n = input_value if input_value is not None else random.randint(4, 12)
        params = {"n": n, "constraint_type": constraint_type}
        if constraint_type == "sum_to_n":
            params["k"] = random.randint(2, min(5, n))
        return params

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value if input_value is not None else 6)
        constraint_type = params.get("constraint_type", "no_consecutive")

        if constraint_type == "no_consecutive":
            if n <= 0:
                result = 1
            else:
                a, b = 1, 2
                for _ in range(n - 1):
                    a, b = b, a + b
                result = b if n >= 1 else a
            description = f"Binary strings of length {n} with no consecutive 1s = F({n}+2) = {result}"

        elif constraint_type == "derangement":
            result = derangement(n)
            description = f"Derangements of {n} elements = !{n} = {result}"

        elif constraint_type == "ascending_digits":
            result = 0 if n > 9 else math.comb(9, n)
            description = f"n-digit numbers with strictly ascending digits = C(9,{n}) = {result}"

        elif constraint_type == "sum_to_n":
            k = params.get("k", 2)
            result = 0 if n < k else math.comb(n - 1, k - 1)
            description = f"Ways to write {n} as sum of {k} positive integers = C({n-1},{k-1}) = {result}"

        else:
            result = n
            description = f"Counting with constraints for n={n}: {result}"

        return MethodResult(value=result, description=description, params=params,
                           metadata={"n": n, "constraint_type": constraint_type})

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingIntersections(MethodBlock):
    """Count intersections of lines in general position."""
    def __init__(self):
        super().__init__()
        self.name = "counting_intersections"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "counting"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(2, 15)}

    def compute(self, input_value, params):
        n = params.get("n", 5)
        result = n * (n - 1) // 2
        return MethodResult(value=result,
                           description=f"{n} lines in general position have {result} intersection points",
                           metadata={"n": n})

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingPaths(MethodBlock):
    """Count lattice paths from (0,0) to (m,n)."""
    def __init__(self):
        super().__init__()
        self.name = "counting_paths"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "paths", "lattice", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        m = input_value if input_value is not None else random.randint(2, 15)
        n = random.randint(2, 15)
        path_type = random.choice(["basic", "ballot", "catalan"])
        return {"m": m, "n": n, "path_type": path_type}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        m = params.get("m", input_value if input_value is not None else 5)
        n = params.get("n", 5)
        path_type = params.get("path_type", "basic")

        if path_type == "basic":
            result = math.comb(m + n, n)
            description = f"Lattice paths from (0,0) to ({m},{n}) = C({m+n},{n}) = {result}"
        elif path_type == "ballot":
            if m <= n:
                result = 0
                description = f"Ballot paths require m > n, but m={m}, n={n}"
            else:
                total = math.comb(m + n, m)
                result = (m - n) * total // (m + n)
                description = f"Ballot paths: ({m}-{n})/({m}+{n}) * C({m+n},{m}) = {result}"
        elif path_type == "catalan":
            result = math.comb(2 * n, n) // (n + 1)
            description = f"Dyck paths from (0,0) to ({n},{n}) = C_{n} = {result}"
        else:
            result = math.comb(m + n, n)
            description = f"Lattice paths = C({m+n},{n}) = {result}"

        return MethodResult(value=result, description=description, params=params,
                           metadata={"m": m, "n": n, "path_type": path_type})

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingRegions(MethodBlock):
    """Count regions formed by lines in plane."""
    def __init__(self):
        super().__init__()
        self.name = "counting_regions"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "geometry"]

    def generate_parameters(self, input_value=None):
        return {"n": input_value or random.randint(2, 10)}

    def compute(self, input_value, params):
        n = params.get("n", 5)
        result = 1 + n + n * (n - 1) // 2
        return MethodResult(value=result, description=f"Regions from {n} lines: {result}", params=params)

    def can_invert(self) -> bool:
        return False


# ============================================================================
# PERMUTATION METHODS
# ============================================================================

@register_technique
class PermutationDecomposition(MethodBlock):
    """Decompose permutations into disjoint cycles (Stirling first kind)."""
    def __init__(self):
        super().__init__()
        self.name = "permutation_decomposition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "cycle-decomposition", "stirling-numbers"]

    def generate_parameters(self, input_value=None):
        from .special_numbers_stirling_bell import stirling_first
        n = input_value if input_value is not None else random.randint(4, 12)
        k = random.randint(1, n)
        return {"n": n, "k": k}

    def compute(self, input_value, params):
        from .special_numbers_stirling_bell import stirling_first
        n = params.get("n", 6)
        k = params.get("k", 3)
        result = stirling_first(n, k)
        description = f"Unsigned Stirling s({n},{k}): permutations with {k} cycles = {result}"
        return MethodResult(value=result, description=description,
                           metadata={"n": n, "k": k, "type": "stirling_first_kind"})

    def can_invert(self) -> bool:
        return False


@register_technique
class PermutationWithRepetition(MethodBlock):
    """Count permutations with repetition: n^k."""
    def __init__(self):
        super().__init__()
        self.name = "permutation_with_repetition"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(2, 6), "k": random.randint(1, 4)}

    def compute(self, input_value, params):
        n, k = params.get("n", 3), params.get("k", 2)
        result = n ** k
        return MethodResult(value=result, description=f"Permutations with repetition: {n}^{k} = {result}",
                           params=params)

    def can_invert(self) -> bool:
        return False


@register_technique
class PermutationOrder(MethodBlock):
    """Compute the order of a permutation (LCM of cycle lengths)."""
    def __init__(self):
        super().__init__()
        self.name = "permutation_order"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "permutations", "group_theory"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        num_cycles = random.randint(2, 5)
        cycle_lengths = [random.randint(2, 10) for _ in range(num_cycles)]
        return {"cycle_lengths": cycle_lengths}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        cycle_lengths = params.get("cycle_lengths", [2, 3])
        if isinstance(cycle_lengths, str):
            try:
                import ast
                cycle_lengths = ast.literal_eval(cycle_lengths)
            except Exception:
                cycle_lengths = [2, 3]

        try:
            cycle_lengths = [int(c) for c in cycle_lengths]
        except Exception:
            cycle_lengths = [2, 3]

        if not cycle_lengths:
            return MethodResult(value=1, description="Empty permutation has order 1", params=params)

        from math import gcd
        def lcm(a, b):
            return abs(a * b) // gcd(a, b) if a and b else 0

        result = cycle_lengths[0]
        for length in cycle_lengths[1:]:
            result = lcm(result, length)

        cycles_str = ", ".join(str(c) for c in cycle_lengths)
        description = f"Permutation with cycles [{cycles_str}]: order = LCM = {result}"
        return MethodResult(value=result, description=description, params=params,
                           metadata={"cycle_lengths": cycle_lengths, "order": result})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# MULTISET PERMUTATION METHODS
# ============================================================================

@register_technique
class MultisetPermutationTotal(MethodBlock):
    """Compute total permutations of a multiset n!/(k1! * k2! * ... * km!)."""

    def __init__(self):
        super().__init__()
        self.name = "multiset_permutation_total"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "multinomial", "permutation", "multiset"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        num_letters = random.randint(2, 4)
        frequencies = [random.randint(1, 4) for _ in range(num_letters)]
        return {"frequencies": frequencies}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        frequencies = params.get("frequencies", [3, 3, 3])
        n = sum(frequencies)
        result = multinomial_coefficient(n, *frequencies)
        fact_str = " * ".join(f"{f}!" for f in frequencies)
        return MethodResult(value=result, description=f"Multiset permutations: {n}!/({fact_str}) = {result}",
                           params=params, metadata={"total_elements": n, "frequencies": frequencies})

    def can_invert(self) -> bool:
        return False


@register_technique
class MultisetPermutationAvoidingPattern(MethodBlock):
    """Count multiset permutations avoiding a specific substring pattern."""

    def __init__(self):
        super().__init__()
        self.name = "multiset_permutation_avoiding_pattern"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 5
        self.tags = ["combinatorics", "inclusion_exclusion", "forbidden_pattern", "multiset"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        pattern_reps = random.randint(2, 4)
        pattern_length = random.randint(2, 4)
        frequencies = [pattern_reps] * pattern_length
        pattern_frequencies = [1] * pattern_length
        return {"frequencies": frequencies, "pattern_frequencies": pattern_frequencies,
                "pattern_length": pattern_length}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        frequencies = params.get("frequencies", [3, 3, 3])
        pattern_freq = params.get("pattern_frequencies", [1, 1, 1])
        pattern_length = params.get("pattern_length", 3)

        n = sum(frequencies)
        total = multinomial_coefficient(n, *frequencies)

        max_by_length = n // pattern_length
        max_by_letters = min(freq // pf for freq, pf in zip(frequencies, pattern_freq) if pf > 0)
        max_patterns = min(max_by_length, max_by_letters)

        at_least_one = 0
        for m in range(1, max_patterns + 1):
            remaining_freq = [f - m * pf for f, pf in zip(frequencies, pattern_freq)]
            if any(rf < 0 for rf in remaining_freq):
                break
            remaining_n = sum(remaining_freq)
            remaining_perms = multinomial_coefficient(remaining_n, *remaining_freq)
            num_placements = count_non_overlapping_placements(n, pattern_length, m)
            sign = (-1) ** (m + 1)
            at_least_one += sign * num_placements * remaining_perms

        result = total - at_least_one
        return MethodResult(value=result,
                           description=f"Multiset permutations avoiding pattern: {result} (from {total} total)",
                           params=params, metadata={"total": total, "avoiding_pattern": result})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# RECTANGLE COUNTING
# ============================================================================

@register_technique
class MinAreaFunction(MethodBlock):
    """Compute minimum area for a rectangle with semi-perimeter s in an NxN square."""
    def __init__(self):
        super().__init__()
        self.name = "min_area_function"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "rectangle", "optimization"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        N = random.randint(10, 100)
        s = random.randint(2, N + 50)
        return {"s": s, "N": N}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        s, N = params.get("s", 10), params.get("N", 50)
        result = s - 1 if s <= N else N * (s - N)
        description = f"f({s}, {N}) = {result} (min area for semi-perimeter {s} in {N}x{N})"
        return MethodResult(value=result, description=description, params=params,
                           metadata={"s": s, "N": N})

    def can_invert(self) -> bool:
        return False


@register_technique
class SumMinAreas(MethodBlock):
    """Sum minimum areas for k rectangles with distinct semi-perimeters."""
    def __init__(self):
        super().__init__()
        self.name = "sum_min_areas"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "rectangle", "optimization"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        N = random.randint(10, 100)
        k = random.randint(5, N)
        return {"k": k, "N": N}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        k, N = params.get("k", 10), params.get("N", 50)
        total = sum((s - 1 if s <= N else N * (s - N)) for s in range(2, k + 2))
        return MethodResult(value=total, description=f"Sum of min areas for {k} rectangles in {N}x{N}: {total}",
                           params=params, metadata={"k": k, "N": N})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# LEGENDRE VALUATION
# ============================================================================

@register_technique
class LegendreValuationFactorial(MethodBlock):
    """Compute Legendre's formula for p-adic valuation of n!."""
    def __init__(self):
        super().__init__()
        self.name = "legendre_valuation_factorial"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "number_theory", "valuation", "legendre"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(100, 10000)
        p = random.choice([2, 3, 5, 7])
        return {"n": n, "p": p}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n, p = params.get("n", input_value), params.get("p", 2)
        valuation, p_power = 0, p
        while p_power <= n:
            valuation += n // p_power
            p_power *= p
        return MethodResult(value=valuation, description=f"v_{p}({n}!) = {valuation}", params=params,
                           metadata={"n": n, "p": p})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# UTILITY TECHNIQUES
# ============================================================================

@register_technique
class PlaceValueSum(MethodBlock):
    """Compute place value sum for a list of digits."""
    def __init__(self):
        super().__init__()
        self.name = "place_value_sum"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["combinatorics", "arithmetic", "base_conversion"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and isinstance(input_value, list):
            return {"digits": input_value}
        return {"digits": [random.randint(0, 9) for _ in range(random.randint(2, 6))]}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        digits = params.get("digits", input_value)
        if not isinstance(digits, list):
            digits = [digits]
        result = sum(d * (10 ** i) for i, d in enumerate(digits))
        return MethodResult(value=result, description=f"Place value sum of {digits}: {result}",
                           params=params, metadata={"digits": digits, "base": 10})

    def can_invert(self) -> bool:
        return True


@register_technique
class FactorialBaseToDecimal(MethodBlock):
    """Convert a number from factorial base to decimal."""
    def __init__(self):
        super().__init__()
        self.name = "factorial_base_to_decimal"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["combinatorics", "base_conversion", "factorial"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and isinstance(input_value, list):
            return {"factorial_base": input_value}
        length = random.randint(3, 6)
        fact_base = [random.randint(0, i + 1) for i in range(length)]
        return {"factorial_base": fact_base}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        fact_base = params.get("factorial_base", input_value)
        result = sum(d * math.factorial(i + 1) for i, d in enumerate(fact_base))
        return MethodResult(value=result, description=f"Factorial base {fact_base} to decimal: {result}",
                           params=params, metadata={"factorial_base": fact_base, "decimal": result})

    def can_invert(self) -> bool:
        return True


@register_technique
class CountOdd(MethodBlock):
    """Count odd numbers in a list."""
    def __init__(self):
        super().__init__()
        self.name = "count_odd"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["combinatorics", "filtering", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and isinstance(input_value, list):
            return {"numbers": input_value}
        return {"numbers": [random.randint(1, 50) for _ in range(random.randint(5, 15))]}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        numbers = params.get("numbers", input_value)
        if not isinstance(numbers, list):
            numbers = [numbers]
        result = sum(1 for n in numbers if n % 2 == 1)
        return MethodResult(value=result, description=f"Count of odd numbers: {result}",
                           params=params, metadata={"numbers": numbers, "count": result})

    def can_invert(self) -> bool:
        return False


@register_technique
class CountEven(MethodBlock):
    """Count even numbers in a list."""
    def __init__(self):
        super().__init__()
        self.name = "count_even"
        self.input_type = "list"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["combinatorics", "filtering", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and isinstance(input_value, list):
            return {"numbers": input_value}
        return {"numbers": [random.randint(1, 50) for _ in range(random.randint(5, 15))]}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        numbers = params.get("numbers", input_value)
        if not isinstance(numbers, list):
            numbers = [numbers]
        result = sum(1 for n in numbers if n % 2 == 0)
        return MethodResult(value=result, description=f"Count of even numbers: {result}",
                           params=params, metadata={"numbers": numbers, "count": result})

    def can_invert(self) -> bool:
        return False


@register_technique
class CircularArrangementsWithAdjacentConstraint(MethodBlock):
    """Count circular arrangements of n items with adjacent pair constraints."""

    def __init__(self):
        super().__init__()
        self.name = "circular_arrangements_with_adjacent_constraint"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "circular_permutations", "constraints"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(4, 8)
        num_pairs = random.randint(1, min(2, n // 2))
        used, pairs = set(), []
        for _ in range(num_pairs):
            for _ in range(20):
                a, b = random.randint(1, n), random.randint(1, n)
                if a != b and a not in used and b not in used:
                    pairs.append((min(a, b), max(a, b)))
                    used.update([a, b])
                    break
        return {"n": n, "pairs": pairs}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n, pairs = params.get("n", input_value), params.get("pairs", [])
        num_units = n - len(pairs)
        result = math.factorial(num_units - 1) if num_units > 0 else 1
        result *= (2 ** len(pairs))
        return MethodResult(value=result,
                           description=f"Circular arrangements of {n} items with {len(pairs)} adjacent constraints: {result}",
                           params=params, metadata={"n": n, "pairs": pairs, "num_units": num_units})

    def can_invert(self) -> bool:
        return False
