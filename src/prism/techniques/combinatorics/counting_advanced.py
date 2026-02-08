"""
Advanced counting methods: Pigeonhole and Bijections.

This module contains:
- Pigeonhole principle (3 techniques)
- Counting with bijections (2 techniques)
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# PIGEONHOLE PRINCIPLE
# ============================================================================

@register_technique
class Pigeonhole(MethodBlock):
    """Apply pigeonhole principle: n+1 pigeons in n holes => collision."""

    def __init__(self):
        super().__init__()
        self.name = "pigeonhole"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "pigeonhole", "counting"]

    def validate_params(self, params, prev_value=None):
        n_holes = params.get("n_holes")
        n_pigeons = params.get("n_pigeons")
        if n_holes is None or n_pigeons is None:
            return False
        return n_holes > 0 and n_pigeons > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 500:
            n_holes = input_value * random.randint(5, 20)
            n_pigeons = n_holes + input_value * random.randint(10, 50)
        elif input_value is not None:
            n_holes = input_value
            n_pigeons = n_holes + random.randint(100, 500)
        else:
            n_holes = random.randint(100, 1000)
            n_pigeons = n_holes + random.randint(200, 1000)
        return {"n_holes": n_holes, "n_pigeons": n_pigeons}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n_holes = params.get("n_holes", input_value)
        n_pigeons = params.get("n_pigeons", n_holes + 1)
        min_in_hole = (n_pigeons + n_holes - 1) // n_holes
        description = f"{n_pigeons} pigeons in {n_holes} holes => >={min_in_hole} in some hole"
        return MethodResult(value=min_in_hole, description=description, params=params, metadata={"n_holes": n_holes, "n_pigeons": n_pigeons})

    def can_invert(self) -> bool:
        return False


@register_technique
class PigeonholeGeneralized(MethodBlock):
    """Generalized pigeonhole: n pigeons in k holes => ceil(n/k) in some hole."""

    def __init__(self):
        super().__init__()
        self.name = "pigeonhole_generalized"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "pigeonhole", "counting"]

    def validate_params(self, params, prev_value=None):
        k_holes = params.get("k_holes")
        n_pigeons = params.get("n_pigeons")
        if k_holes is None or n_pigeons is None:
            return False
        return k_holes > 0 and n_pigeons > 0

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 500:
            n_pigeons = input_value * random.randint(10, 50)
            k_holes = random.randint(max(3, n_pigeons // 100), max(15, n_pigeons // 20))
        elif input_value is not None:
            n_pigeons = input_value
            k_holes = random.randint(3, max(15, n_pigeons // 50))
        else:
            n_pigeons = random.randint(500, 5000)
            k_holes = random.randint(10, 50)
        return {"k_holes": k_holes, "n_pigeons": n_pigeons}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n_pigeons = params.get("n_pigeons", input_value)
        k_holes = params.get("k_holes", 10)
        result = (n_pigeons + k_holes - 1) // k_holes
        description = f"{n_pigeons} pigeons in {k_holes} holes => ceil({n_pigeons}/{k_holes}) = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"k_holes": k_holes, "n_pigeons": n_pigeons})

    def can_invert(self) -> bool:
        return False


@register_technique
class DoubleCount(MethodBlock):
    """Double counting argument: count same set two ways."""

    def __init__(self):
        super().__init__()
        self.name = "double_count"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "double_counting", "proof"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 500:
            n = input_value + random.randint(40, 150)
        elif input_value is not None:
            n = input_value
        else:
            n = random.randint(50, 200)
        return {"n": n, "object": "edges_K_n"}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        obj = params.get("object", "edges_K_n")
        if obj == "edges_K_n":
            result = n * (n - 1) // 2
            description = f"Edges in K_{n}: C({n},2) = {n}*{n-1}/2 = {result}"
        else:
            result = 0
            description = f"Double count {obj} with n={n}"
        return MethodResult(value=result, description=description, params=params, metadata={"n": n, "object": obj})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# COUNTING WITH BIJECTIONS
# ============================================================================

@register_technique
class Bijection(MethodBlock):
    """Establish bijection between two sets and return the count."""

    def __init__(self):
        super().__init__()
        self.name = "bijection"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "bijection", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        bijection_types = ["binary_strings", "compositions", "lattice_paths", "dyck_paths"]
        bijection_type = random.choice(bijection_types)
        if input_value is not None:
            n = input_value
        else:
            n = random.randint(3, 15)
        params = {"n": n, "bijection_type": bijection_type}
        if bijection_type == "lattice_paths":
            params["m"] = random.randint(2, 10)
        return params

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value if input_value is not None else 5)
        bijection_type = params.get("bijection_type", "binary_strings")

        if bijection_type == "binary_strings":
            result = 2 ** n
            description = f"Bijection: {n}-bit binary strings <-> subsets of {{1,...,{n}}} = 2^{n} = {result}"
        elif bijection_type == "compositions":
            result = 2 ** (n - 1)
            description = f"Bijection: compositions of {n} <-> binary strings of length {n-1} = 2^{n-1} = {result}"
        elif bijection_type == "lattice_paths":
            m = params.get("m", n)
            result = math.comb(m + n, n)
            description = f"Bijection: paths (0,0)->({m},{n}) <-> sequences of {m} R's and {n} U's = C({m+n},{n}) = {result}"
        elif bijection_type == "dyck_paths":
            result = math.comb(2 * n, n) // (n + 1)
            description = f"Bijection: Dyck paths of length 2*{n} <-> valid parentheses = C_{n} = {result}"
        else:
            result = 2 ** n
            description = f"Bijection count for n={n}: {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "bijection_type": bijection_type}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingBijection(MethodBlock):
    """Count using bijection between sets."""

    def __init__(self):
        super().__init__()
        self.name = "counting_bijection"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics"]

    def generate_parameters(self, input_value=None):
        n = input_value or random.randint(5, 15)
        k = random.randint(2, min(n-1, 10))
        return {"n": n, "k": k}

    def compute(self, input_value, params):
        n = params.get("n", 10)
        k = params.get("k", 5)
        result = math.comb(n, k)
        return MethodResult(
            value=result,
            description=f"Bijection count C({n},{k}) = {result}",
            params=params
        )

    def can_invert(self) -> bool:
        return False
