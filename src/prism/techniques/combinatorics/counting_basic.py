"""
Basic counting and combinatorics methods.

This module contains:
- Catalan numbers (2 techniques)
- Binomial coefficients (4 techniques)
- Paths/Lattice counting (4 techniques)
- Basic counting techniques (2 techniques)
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from ..decomposition import Decomposition


# ============================================================================
# CATALAN NUMBERS (2 techniques)
# ============================================================================

@register_technique
class Catalan(MethodBlock):
    """Compute nth Catalan number C_n = C(2n,n)/(n+1)."""

    def __init__(self):
        super().__init__()
        self.name = "catalan"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "catalan", "counting"]
        self.decomposition = Decomposition(
            expression="divide(binomial(multiply(2, n), n), add(n, constant_one()))",
            param_map={"n": "n"},
            notes="C_n = C(2n,n)/(n+1)"
        )

    def validate_params(self, params, prev_value=None):
        """Validate n >= 0 for Catalan numbers."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 15)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 10, 20)
        c_n = math.comb(2*n, n) // (n + 1)
        return MethodResult(value=c_n, description=f"C_{n} = C(2*{n},{n})/({n}+1) = {c_n}", params=params, metadata={"n": n, "formula": "C(2n,n)/(n+1)"})

    def can_invert(self) -> bool:
        return False


@register_technique
class CatalanInverse(MethodBlock):
    """Find n such that C_n equals a given value."""

    def __init__(self):
        super().__init__()
        self.name = "catalan_inverse"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "catalan", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            target = input_value
        else:
            n = random.randint(3, 12)
            target = math.comb(2*n, n) // (n + 1)
        return {"target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        for n in range(1, 100):
            c_n = math.comb(2*n, n) // (n + 1)
            if c_n == target:
                return MethodResult(value=n, description=f"Find n where C_n = {target}: n = {n}", params=params, metadata={"target": target, "found": True})
            if c_n > target:
                break
        return MethodResult(value=-1, description=f"No n found where C_n = {target}", params=params, metadata={"target": target, "found": False})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# BINOMIAL COEFFICIENTS (4 techniques)
# ============================================================================

@register_technique
class Binomial(MethodBlock):
    """Compute binomial coefficient C(n,k)."""

    def __init__(self):
        super().__init__()
        self.name = "binomial"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 1
        self.tags = ["combinatorics", "binomial", "counting"]
        self.is_primitive = True

    def validate_params(self, params, prev_value=None):
        """Binomial requires 0 <= k <= n."""
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        return 0 <= k <= n

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = input_value
            k = random.randint(1, min(n, 20))
        else:
            n = random.randint(5, 50)
            k = random.randint(1, min(n, 20))
        return {"n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 2)
        result = math.comb(n, k)
        return MethodResult(value=result, description=f"C({n},{k}) = {result}", params=params, metadata={"n": n, "k": k})

    def can_invert(self) -> bool:
        return False


@register_technique
class BinomialInverseN(MethodBlock):
    """Find n such that C(n,k) = target."""

    def __init__(self):
        super().__init__()
        self.name = "binomial_inverse_n"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "binomial", "inverse"]

    def validate_params(self, params, prev_value=None):
        k = params.get("k")
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if k is None or target is None:
            return False
        return k >= 0 and target > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        k = random.randint(2, 10)
        if input_value is not None:
            target = input_value
        else:
            n = random.randint(k+1, 30)
            target = math.comb(n, k)
        return {"target": target, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        k = params.get("k", 2)
        for n in range(k, 200):
            c = math.comb(n, k)
            if c == target:
                return MethodResult(value=n, description=f"Find n where C(n,{k}) = {target}: n = {n}", params=params, metadata={"target": target, "k": k, "found": True})
            if c > target:
                break
        return MethodResult(value=-1, description=f"No n found where C(n,{k}) = {target}", params=params, metadata={"target": target, "k": k, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class BinomialInverseK(MethodBlock):
    """Find k such that C(n,k) = target."""

    def __init__(self):
        super().__init__()
        self.name = "binomial_inverse_k"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "binomial", "inverse"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if n is None or target is None:
            return False
        return n >= 0 and target > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = random.randint(10, 40)
        k = random.randint(2, n//2)
        target = math.comb(n, k)
        return {"target": target, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        n = params.get("n", 10)
        for k in range(0, n+1):
            c = math.comb(n, k)
            if c == target:
                return MethodResult(value=k, description=f"Find k where C({n},k) = {target}: k = {k}", params=params, metadata={"target": target, "n": n, "found": True})
        return MethodResult(value=-1, description=f"No k found where C({n},k) = {target}", params=params, metadata={"target": target, "n": n, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class BinomialSum(MethodBlock):
    """Compute sum of binomial coefficients: sum C(n,k) for k=0..n = 2^n."""

    def __init__(self):
        super().__init__()
        self.name = "binomial_sum"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "binomial", "sum"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        if n is None:
            return False
        return n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 20)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 10, 60)
        result = 2 ** n
        return MethodResult(value=result, description=f"Sum of C({n},k) for k=0..{n} = 2^{n} = {result}", params=params, metadata={"n": n, "formula": "2^n"})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# PATHS/LATTICE (4 techniques)
# ============================================================================

@register_technique
class LatticePaths(MethodBlock):
    """Count lattice paths from (0,0) to (m,n) = C(m+n, n)."""

    def __init__(self):
        super().__init__()
        self.name = "lattice_paths"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "paths", "counting"]

    def validate_params(self, params, prev_value=None):
        m = params.get("m", prev_value) if prev_value is not None else params.get("m")
        n = params.get("n")
        if m is None or n is None:
            return False
        return m >= 0 and n >= 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            m = input_value
            n = random.randint(1, 20)
        else:
            m = random.randint(1, 20)
            n = random.randint(1, 20)
        return {"m": m, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        m = params.get("m", input_value)
        n = params.get("n", 1)
        result = math.comb(m + n, n)
        return MethodResult(value=result, description=f"Paths from (0,0) to ({m},{n}) = C({m+n},{n}) = {result}", params=params, metadata={"m": m, "n": n})

    def can_invert(self) -> bool:
        return False


@register_technique
class LatticePathsInverseM(MethodBlock):
    """Find m such that lattice paths from (0,0) to (m,n) equals target."""

    def __init__(self):
        super().__init__()
        self.name = "lattice_paths_inverse_m"
        self.input_type = "count"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "paths", "inverse"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        target = params.get("target", prev_value) if prev_value is not None else params.get("target")
        if n is None or target is None:
            return False
        return n >= 0 and target > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = random.randint(2, 10)
        if input_value is not None:
            target = input_value
        else:
            m = random.randint(2, 15)
            target = math.comb(m + n, n)
        return {"target": target, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        target = params.get("target", input_value)
        n = params.get("n", 2)
        for m in range(0, 100):
            paths = math.comb(m + n, n)
            if paths == target:
                return MethodResult(value=m, description=f"Find m where paths to ({m},{n}) = {target}: m = {m}", params=params, metadata={"target": target, "n": n, "found": True})
            if paths > target:
                break
        return MethodResult(value=-1, description=f"No m found where paths to (m,{n}) = {target}", params=params, metadata={"target": target, "n": n, "found": False})

    def can_invert(self) -> bool:
        return False


@register_technique
class Ballot(MethodBlock):
    """Ballot problem: paths where a > b throughout. Count = (a-b)/(a+b) * C(a+b, a)."""

    def __init__(self):
        super().__init__()
        self.name = "ballot"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "ballot", "paths"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            a = input_value
            b = random.randint(1, a-1)
        else:
            a = random.randint(5, 20)
            b = random.randint(1, a-1)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        a = params.get("a", input_value)
        b = params.get("b", 1)
        total = math.comb(a + b, a)
        result = (a - b) * total // (a + b)
        return MethodResult(value=result, description=f"Ballot({a},{b}) = ({a}-{b})/({a}+{b}) * C({a+b},{a}) = {result}", params=params, metadata={"a": a, "b": b})

    def can_invert(self) -> bool:
        return False


@register_technique
class DyckPaths(MethodBlock):
    """Count Dyck paths of length 2n (equals Catalan number C_n)."""

    def __init__(self):
        super().__init__()
        self.name = "dyck_paths"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "dyck", "catalan", "paths"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 15)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n), 500) if n else 10
        result = math.comb(2*n, n) // (n + 1)
        return MethodResult(value=result, description=f"Dyck paths of length 2*{n} = C_{n} = {result}", params=params, metadata={"n": n, "formula": "Catalan"})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# BASIC COUNTING TECHNIQUES
# ============================================================================

@register_technique
class Counting(MethodBlock):
    """Basic counting problems for arrangements and selections."""

    def __init__(self):
        super().__init__()
        self.name = "counting"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "arrangements", "selections"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(5, 20)
        k = random.randint(1, min(n, 8))
        counting_type = random.choice(["permutation", "combination"])
        return {"n": n, "k": k, "counting_type": counting_type}

    def compute(self, input_value, params):
        n = params.get("n", 10)
        k = params.get("k", 5)
        counting_type = params.get("counting_type", "permutation")

        if counting_type == "permutation":
            result = math.perm(n, k)
            description = f"Permutations P({n},{k}) = {n}!/({n}-{k})! = {result}"
        else:
            result = math.comb(n, k)
            description = f"Combinations C({n},{k}) = {n}!/({k}!{n-k}!) = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={"n": n, "k": k, "counting_type": counting_type}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingPrinciples(MethodBlock):
    """Apply addition and multiplication principles to count outcomes."""

    def __init__(self):
        super().__init__()
        self.name = "counting_principles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "multiplication-principle", "addition-principle"]

    def generate_parameters(self, input_value=None):
        num_stages = random.randint(2, 4)
        outcomes_per_stage = [random.randint(2, 8) for _ in range(num_stages)]
        use_addition = random.choice([True, False])
        return {
            "outcomes_per_stage": outcomes_per_stage,
            "use_addition": use_addition
        }

    def compute(self, input_value, params):
        outcomes_per_stage = params.get("outcomes_per_stage", [5, 4, 3])
        use_addition = params.get("use_addition", False)

        if use_addition:
            result = sum(outcomes_per_stage)
            stages_str = " + ".join(str(x) for x in outcomes_per_stage)
            description = f"Addition principle (mutually exclusive): {stages_str} = {result}"
        else:
            result = 1
            for outcome_count in outcomes_per_stage:
                result *= outcome_count
            stages_str = " * ".join(str(x) for x in outcomes_per_stage)
            description = f"Multiplication principle (sequential): {stages_str} = {result}"

        return MethodResult(
            value=result,
            description=description,
            metadata={
                "outcomes_per_stage": outcomes_per_stage,
                "use_addition": use_addition,
                "principle": "addition" if use_addition else "multiplication"
            }
        )

    def can_invert(self) -> bool:
        return False
