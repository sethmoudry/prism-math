"""
Stirling numbers and Bell numbers.

This module contains:
- Helper functions for Stirling numbers
- Stirling numbers (4 techniques)
- Bell numbers (1 technique)
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def stirling_first(n: int, k: int) -> int:
    """Compute unsigned Stirling number of first kind s(n,k)."""
    n = min(abs(n), 100)
    k = min(abs(k), n)
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    dp = [[0] * (k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, min(i, k)+1):
            dp[i][j] = dp[i-1][j-1] + (i-1) * dp[i-1][j]
    return dp[n][k]


def stirling_second(n: int, k: int) -> int:
    """Compute Stirling number of second kind S(n,k)."""
    n = min(abs(n), 100)
    k = min(abs(k), n)
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    dp = [[0] * (k+1) for _ in range(n+1)]
    dp[0][0] = 1
    for i in range(1, n+1):
        for j in range(1, min(i, k)+1):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    return dp[n][k]


# ============================================================================
# STIRLING NUMBERS (4 techniques)
# ============================================================================

@register_technique
class StirlingFirst(MethodBlock):
    """Compute unsigned Stirling number of first kind s(n,k)."""

    def __init__(self):
        super().__init__()
        self.name = "stirling_first"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "stirling", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return n >= k >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = input_value
            k = random.randint(1, min(n, 10))
        else:
            n = random.randint(5, 15)
            k = random.randint(2, min(n, 10))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 2)
        result = stirling_first(n, k)
        return MethodResult(value=result, description=f"s({n},{k}) = {result}", params=params, metadata={"n": n, "k": k, "type": "first"})

    def can_invert(self) -> bool:
        return False


@register_technique
class StirlingSecond(MethodBlock):
    """Compute Stirling number of second kind S(n,k)."""

    def __init__(self):
        super().__init__()
        self.name = "stirling_second"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "stirling", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return n >= k >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = input_value
            k = random.randint(1, min(n, 10))
        else:
            n = random.randint(5, 15)
            k = random.randint(2, min(n, 10))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 2)
        result = stirling_second(n, k)
        return MethodResult(value=result, description=f"S({n},{k}) = {result}", params=params, metadata={"n": n, "k": k, "type": "second"})

    def can_invert(self) -> bool:
        return False


@register_technique
class StirlingSecondInverseN(MethodBlock):
    """Find n such that S(n,k) = target."""

    def __init__(self):
        super().__init__()
        self.name = "stirling_second_inverse_n"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "stirling", "inverse"]

    def validate_params(self, params, prev_value=None):
        k = params.get("k")
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if k is None or target is None:
            return False
        return k >= 0 and target > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        k = random.randint(2, 8)
        if input_value is not None:
            target = input_value
        else:
            n = random.randint(k+1, 15)
            target = stirling_second(n, k)
        return {"target": target, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        k = params.get("k", 2)
        for n in range(k, 50):
            s = stirling_second(n, k)
            if s == target:
                return MethodResult(value=n, description=f"Find n where S(n,{k}) = {target}: n = {n}", params=params, metadata={"target": target, "k": k, "found": True})
            if s > target:
                break
        return MethodResult(value=-1, description=f"No n found where S(n,{k}) = {target}", params=params, metadata={"target": target, "k": k, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class Bell(MethodBlock):
    """Compute Bell number B_n = sum of S(n,k) for k=0..n."""

    def __init__(self):
        super().__init__()
        self.name = "bell"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "bell", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 12)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 10, 20)

        if n == 0:
            result = 1
        else:
            bell = [0] * (n + 1)
            bell[0] = 1
            for i in range(1, n + 1):
                bell[i] = sum(math.comb(i - 1, k) * bell[k] for k in range(i))
            result = bell[n]

        return MethodResult(value=result, description=f"B_{n} = {result}", params=params, metadata={"n": n, "formula": "Bell triangle"})

    def can_invert(self) -> bool:
        return False
