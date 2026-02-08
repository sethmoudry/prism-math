"""
Ported utility/primitive methods - batch 2.

Methods ported (sorted by impact):
- greater_than_or_equal (16 failing problems)
- deep_stirling (2 failing problems) - alias for stirling_second
- modular_reduction (2 failing problems) - alias for modulo
- extract_mn_sum (2 failing problems) - simple m+n utility

Passthrough stubs for LLM-hallucinated variable-as-function patterns:
- board_size (7 failing problems)
- grid_size (4 failing problems)
- total_sum (3 failing problems)
- total_people (3 failing problems)
- side_length (3 failing problems)
- total_vertices (3 failing problems)
- max_val (3 failing problems)
"""

import random
from typing import Any

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# GREATER THAN OR EQUAL (16 failing problems)
# ============================================================================

@register_technique
class GreaterThanOrEqual(MethodBlock):
    """Check if a >= b. Returns 1 if true, 0 if false."""

    def __init__(self):
        super().__init__()
        self.name = "greater_than_or_equal"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["comparison", "primitive"]

    def generate_parameters(self, input_value=None):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        return {"a": a, "b": b}

    def compute(self, input_value, params):
        a = params.get("a", 5)
        b = params.get("b", 10)
        result = 1 if a >= b else 0
        return MethodResult(
            value=result,
            description=f"Compare {a} >= {b}: {bool(result)}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# DEEP STIRLING (2 failing problems) - Stirling numbers of the second kind
# ============================================================================

@register_technique
class DeepStirling(MethodBlock):
    """Stirling numbers of the second kind S(n,k).

    S(n,k) counts the number of ways to partition a set of n elements
    into exactly k non-empty subsets.

    Recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    Base cases: S(0,0)=1, S(n,0)=0 for n>0, S(0,k)=0 for k>0
    """

    def __init__(self):
        super().__init__()
        self.name = "deep_stirling"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "stirling", "partition"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(3, 12)
        k = random.randint(1, max(1, n))
        return {"n": n, "k": k}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value is not None else 5)
        k = params.get("k", 2)

        # Clamp to avoid excessive computation
        n = min(abs(int(n)), 100)
        k = min(abs(int(k)), n)

        # Base cases
        if n == 0 and k == 0:
            result = 1
        elif n == 0 or k == 0:
            result = 0
        elif k == 1 or k == n:
            result = 1
        else:
            # DP computation
            dp = [[0] * (k + 1) for _ in range(n + 1)]
            dp[0][0] = 1
            for i in range(1, n + 1):
                for j in range(1, min(i, k) + 1):
                    dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
            result = dp[n][k]

        return MethodResult(
            value=result,
            description=f"S({n},{k}) = {result}",
            params=params,
            metadata={"n": n, "k": k}
        )

    def can_invert(self):
        return False


# ============================================================================
# MODULAR REDUCTION (2 failing problems) - same as mod(a, m)
# ============================================================================

@register_technique
class ModularReduction(MethodBlock):
    """Reduce a modulo m. Equivalent to mod(a, m)."""

    def __init__(self):
        super().__init__()
        self.name = "modular_reduction"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["number_theory", "modular", "primitive"]

    def generate_parameters(self, input_value=None):
        a = input_value if input_value is not None else random.randint(10, 10000)
        m = random.randint(2, 1000)
        return {"a": a, "m": m}

    def compute(self, input_value, params):
        a = params.get("a", input_value if input_value is not None else 100)
        m = params.get("m", 7)

        if m == 0:
            result = a
            desc = f"{a} mod 0 is undefined, returning {a}"
        else:
            result = a % m
            desc = f"{a} mod {m} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# EXTRACT MN SUM (2 failing problems) - returns m + n
# ============================================================================

@register_technique
class ExtractMnSum(MethodBlock):
    """Extract and sum m + n. Simple utility for final answer extraction."""

    def __init__(self):
        super().__init__()
        self.name = "extract_mn_sum"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["utility", "extraction"]

    def generate_parameters(self, input_value=None):
        m = random.randint(1, 100)
        n = random.randint(1, 100)
        return {"m": m, "n": n}

    def compute(self, input_value, params):
        m = params.get("m", 0)
        n = params.get("n", 0)
        result = m + n
        return MethodResult(
            value=result,
            description=f"m + n = {m} + {n} = {result}",
            params=params
        )

    def can_invert(self):
        return False


# ============================================================================
# PASSTHROUGH STUBS for LLM-hallucinated variable-as-function patterns
#
# These are NOT real mathematical methods. They exist because the LLM
# sometimes formats variable assignments as function calls, e.g.:
#   board_size(7)  instead of  board_size = 7
#   grid_size(4)   instead of  grid_size = 4
#
# Each returns its first argument unchanged (identity function).
# ============================================================================

def _make_passthrough_class(method_name: str, count: int):
    """Factory to create a passthrough MethodBlock class."""

    @register_technique
    class PassthroughStub(MethodBlock):
        __doc__ = (
            f"Passthrough stub for '{method_name}'. "
            f"Returns first argument unchanged. "
            f"({count} LLM-hallucinated uses as variable-as-function.)"
        )

        def __init__(self):
            super().__init__()
            self.name = method_name
            self.input_type = "any"
            self.output_type = "any"
            self.difficulty = 0
            self.tags = ["passthrough", "stub"]

        def generate_parameters(self, input_value=None):
            val = input_value if input_value is not None else random.randint(1, 20)
            return {"value": val}

        def compute(self, input_value, params):
            val = params.get("value", input_value if input_value is not None else 0)
            return MethodResult(
                value=val,
                description=f"{method_name}({val}) = {val} (passthrough)",
                params=params
            )

        def can_invert(self):
            return False

    # Give each class a unique name for debugging
    PassthroughStub.__name__ = f"Passthrough_{method_name}"
    PassthroughStub.__qualname__ = f"Passthrough_{method_name}"
    return PassthroughStub


# Hallucinated variable names with their occurrence counts
_PASSTHROUGH_STUBS = {
    "board_size": 7,
    "grid_size": 4,
    "total_sum": 3,
    "total_people": 3,
    "side_length": 3,
    "total_vertices": 3,
    "max_val": 3,
}

# Register all passthrough stubs
BoardSize = _make_passthrough_class("board_size", 7)
GridSize = _make_passthrough_class("grid_size", 4)
TotalSum = _make_passthrough_class("total_sum", 3)
TotalPeople = _make_passthrough_class("total_people", 3)
SideLength = _make_passthrough_class("side_length", 3)
TotalVertices = _make_passthrough_class("total_vertices", 3)
MaxVal = _make_passthrough_class("max_val", 3)
