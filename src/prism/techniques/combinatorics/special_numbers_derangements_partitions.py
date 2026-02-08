"""
Derangements, partitions, and explosive growth techniques.

This module contains:
- Helper functions for derangements and partitions
- Derangements (3 techniques)
- Partitions (4 techniques)
- Large factorial/binomial/catalan/stirling (explosive growth)
- Binary tree counting
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from ..decomposition import Decomposition
from .special_numbers_stirling_bell import stirling_first, stirling_second


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def derangement(n: int) -> int:
    """Compute number of derangements D_n = !n."""
    n = min(abs(n), 1000)
    if n == 0:
        return 1
    if n == 1:
        return 0
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0
    for i in range(2, n+1):
        dp[i] = (i - 1) * (dp[i-1] + dp[i-2])
    return dp[n]


def partition_count(n: int, max_val: int = None) -> int:
    """Compute number of integer partitions of n."""
    n = min(abs(n), 500)
    if max_val is None:
        max_val = n
    max_val = min(max_val, n)
    dp = [[0] * (max_val + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for j in range(max_val + 1):
        dp[0][j] = 1
    for i in range(1, n + 1):
        for j in range(1, max_val + 1):
            dp[i][j] = dp[i][j-1]
            if i >= j:
                dp[i][j] += dp[i-j][j]
    return dp[n][max_val]


def partition_k_parts(n: int, k: int) -> int:
    """Compute number of partitions of n into exactly k parts."""
    n = min(abs(n), 500)
    k = min(abs(k), n)
    if k > n or k <= 0:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i-1][j-1]
            if i >= j:
                dp[i][j] += dp[i-j][j]
    return dp[n][k]


# ============================================================================
# DERANGEMENTS (3 techniques)
# ============================================================================

@register_technique
class Derangement(MethodBlock):
    """Compute derangement count D_n (subfactorial !n)."""

    def __init__(self):
        super().__init__()
        self.name = "derangement"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "derangement", "counting"]
        self.decomposition = Decomposition(
            expression="floor(add(divide(factorial(n), 2.71828), 0.5))",
            param_map={"n": "n"},
            notes="D_n = round(n!/e) - approximation formula"
        )

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = max(8, min(abs(input_value) % 20 + 8, 20))
        else:
            n = random.randint(8, 15)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 10, 100)
        result = derangement(n)
        description = f"D_{n} = !{n} = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"n": n, "formula": "!(n)"})

    def can_invert(self) -> bool:
        return False


@register_technique
class DerangementInverse(MethodBlock):
    """Find n such that D_n = target."""

    def __init__(self):
        super().__init__()
        self.name = "derangement_inverse"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "derangement", "inverse"]

    def validate_params(self, params, prev_value=None):
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if target is None:
            return False
        return target >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            target = input_value
        else:
            n = random.randint(3, 12)
            target = derangement(n)
        return {"target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        for n in range(0, 50):
            d = derangement(n)
            if d == target:
                return MethodResult(value=n, description=f"Find n where D_n = {target}: n = {n}", params=params, metadata={"target": target, "found": True})
            if d > target:
                break
        return MethodResult(value=-1, description=f"No n found where D_n = {target}", params=params, metadata={"target": target, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class Subfactorial(MethodBlock):
    """Compute subfactorial !n (alias for derangement)."""

    def __init__(self):
        super().__init__()
        self.name = "subfactorial"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "subfactorial", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 100:
            n = input_value + random.randint(8, 15)
        elif input_value is not None:
            n = min(input_value, 20)
        else:
            n = random.randint(8, 15)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        result = derangement(n)
        description = f"!{n} = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"n": n, "formula": "subfactorial"})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# PARTITIONS (4 techniques)
# ============================================================================

@register_technique
class Partition(MethodBlock):
    """Compute number of integer partitions p(n)."""

    def __init__(self):
        super().__init__()
        self.name = "partition"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "partition", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = max(25, min(abs(input_value) % 60 + 25, 80))
        else:
            n = random.randint(25, 50)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 30, 100)
        result = partition_count(n)
        description = f"p({n}) = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"n": n, "formula": "partition count"})

    def can_invert(self) -> bool:
        return False


@register_technique
class PartitionInverse(MethodBlock):
    """Find n such that p(n) = target."""

    def __init__(self):
        super().__init__()
        self.name = "partition_inverse"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "partition", "inverse"]

    def validate_params(self, params, prev_value=None):
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if target is None:
            return False
        return target >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            target = input_value
        else:
            n = random.randint(5, 15)
            target = partition_count(n)
        return {"target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        for n in range(0, 100):
            p = partition_count(n)
            if p == target:
                return MethodResult(value=n, description=f"Find n where p(n) = {target}: n = {n}", params=params, metadata={"target": target, "found": True})
            if p > target:
                break
        return MethodResult(value=-1, description=f"No n found where p(n) = {target}", params=params, metadata={"target": target, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class PartitionKParts(MethodBlock):
    """Compute partitions of n into exactly k parts."""

    def __init__(self):
        super().__init__()
        self.name = "partition_k_parts"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "partition", "counting"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return n >= k >= 1

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 200:
            n = input_value + random.randint(20, 40)
            k = random.randint(max(2, n // 8), min(n, n // 3))
        elif input_value is not None:
            n = min(input_value, 80)
            k = random.randint(2, min(n, 15))
        else:
            n = random.randint(25, 50)
            k = random.randint(max(2, n // 8), min(n, 15))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 2)
        result = partition_k_parts(n, k)
        description = f"Partitions of {n} into {k} parts = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"n": n, "k": k})

    def can_invert(self) -> bool:
        return False


@register_technique
class IntegerPartitionsIntoK(MethodBlock):
    """Count integer partitions of n into exactly k parts."""

    def __init__(self):
        super().__init__()
        self.name = "integer_partitions_into_k"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "partitions", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = input_value
            k = random.randint(2, min(n, 10))
        else:
            n = random.randint(6, 20)
            k = random.randint(2, min(n, 10))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 2)

        result = partition_k_parts(n, k)

        return MethodResult(
            value=result,
            description=f"Number of partitions of {n} into {k} parts: {result}",
            params=params,
            metadata={"n": n, "k": k, "formula": "p(n,k)"}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# EXPLOSIVE GROWTH TECHNIQUES
# ============================================================================

@register_technique
class LargeFactorial(MethodBlock):
    """Compute n! for n in range 12-20, producing 10^8 to 10^18 values."""

    def __init__(self):
        super().__init__()
        self.name = "large_factorial"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["combinatorics", "factorial", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return 12 <= n <= 20

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = random.randint(12, 18)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", 15)
        n = max(12, min(n, 20))
        result = math.factorial(n)

        return MethodResult(
            value=result,
            description=f"{n}! = {result}",
            params=params,
            metadata={
                "n": n,
                "magnitude": len(str(result)),
                "formula": "n!",
                "growth_type": "explosive_factorial"
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LargeBinomial(MethodBlock):
    """Compute C(n, k) with large n (50-100) and k near n/2."""

    def __init__(self):
        super().__init__()
        self.name = "large_binomial"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["combinatorics", "binomial", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return n >= 50 and 0 <= k <= n

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = random.randint(50, 80)
        k = n // 2 + random.randint(-5, 5)
        k = max(0, min(k, n))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", 60)
        k = params.get("k", n // 2)
        n = max(0, int(n))
        k = max(0, min(k, n))
        result = math.comb(n, k)

        return MethodResult(
            value=result,
            description=f"C({n},{k}) = {result}",
            params=params,
            metadata={
                "n": n,
                "k": k,
                "magnitude": len(str(result)),
                "formula": "n!/(k!(n-k)!)",
                "growth_type": "explosive_binomial"
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LargeCatalan(MethodBlock):
    """Compute Catalan number C_n for n in range 12-18."""

    def __init__(self):
        super().__init__()
        self.name = "large_catalan"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "catalan", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return 12 <= n <= 18

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = random.randint(12, 18)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", 15)
        n = max(12, min(n, 18))
        result = math.comb(2 * n, n) // (n + 1)

        return MethodResult(
            value=result,
            description=f"C_{n} = C(2*{n},{n})/({n}+1) = {result}",
            params=params,
            metadata={
                "n": n,
                "magnitude": len(str(result)),
                "formula": "C(2n,n)/(n+1)",
                "growth_type": "explosive_catalan"
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class LargeStirling(MethodBlock):
    """Compute Stirling numbers with large parameters."""

    def __init__(self):
        super().__init__()
        self.name = "large_stirling"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "stirling", "large_values", "explosive_growth"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return n >= 15 and 2 <= k <= n

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        kind = random.choice(["first", "second"])
        if kind == "first":
            n = random.randint(15, 20)
            k = random.randint(2, 5)
        else:
            n = random.randint(18, 25)
            k = random.randint(4, 8)
        return {"n": n, "k": k, "kind": kind}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", 20)
        k = params.get("k", 5)
        kind = params.get("kind", "second")
        n = max(15, min(n, 25))
        k = max(2, min(k, n))

        if kind == "first":
            result = stirling_first(n, k)
            desc = f"s({n},{k}) = {result}"
            formula = "unsigned Stirling first kind"
        else:
            result = stirling_second(n, k)
            desc = f"S({n},{k}) = {result}"
            formula = "Stirling second kind"

        return MethodResult(
            value=result,
            description=desc,
            params=params,
            metadata={
                "n": n,
                "k": k,
                "kind": kind,
                "magnitude": len(str(result)),
                "formula": formula,
                "growth_type": "explosive_stirling"
            }
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# BINARY TREE COUNTING
# ============================================================================

@register_technique
class BinaryTreeCountRecurrence(MethodBlock):
    """Count binary trees using Catalan numbers recurrence."""

    def __init__(self):
        super().__init__()
        self.name = "binary_tree_count_recurrence"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "catalan", "binary_trees", "recurrence"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 15)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)

        if n < 0:
            return MethodResult(
                value=0,
                description=f"Binary trees with negative nodes: 0",
                params=params,
                metadata={"n": n, "formula": "invalid"}
            )

        catalan = [0] * (n + 1)
        catalan[0] = 1

        for i in range(1, n + 1):
            for j in range(i):
                catalan[i] += catalan[j] * catalan[i - 1 - j]

        result = catalan[n]

        if n <= 4:
            terms = []
            for i in range(n):
                terms.append(f"C_{i}*C_{n-1-i}")
            recurrence_str = " + ".join(terms)
            description = f"C_{n} = {recurrence_str} = {result} (binary trees with {n} nodes)"
        else:
            description = f"C_{n} = {result} (binary trees with {n} nodes via recurrence)"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "n": n,
                "catalan_number": result,
                "formula": "C_n = sum C_i * C_{n-1-i} for i=0 to n-1"
            }
        )

    def can_invert(self) -> bool:
        return False
