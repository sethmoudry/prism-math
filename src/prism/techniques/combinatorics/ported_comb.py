"""
Ported combinatorics, probability, and game theory methods.

Methods sorted by impact:
- linearity_of_expectation (16)
- probabilistic_method (11)
- extremal_construction (8)
- inclusion_exclusion (8)
- derangement_via_inclusion_exclusion (8)
- ramsey_bound (8)
- construct_isomorphism (7)
- compute_nim_value (7)
- graph_arboricity_bound (7)
- optional_stopping_theorem (6)
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# Helper: derangement count (reimplemented to avoid circular imports)
def _derangement(n: int) -> int:
    """Compute D_n = n! * sum_{k=0}^{n} (-1)^k / k!"""
    if n == 0:
        return 1
    if n == 1:
        return 0
    result = 0
    factorial_n = math.factorial(n)
    for k in range(n + 1):
        term = factorial_n // math.factorial(k)
        if k % 2 == 0:
            result += term
        else:
            result -= term
    return result


# ============================================================================
# PROBABILITY & EXPECTATION (3 techniques)
# ============================================================================

@register_technique
class LinearityOfExpectation(MethodBlock):
    """Apply linearity of expectation: E[X+Y] = E[X] + E[Y]."""

    def __init__(self):
        super().__init__()
        self.name = "linearity_of_expectation"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 2
        self.tags = ["combinatorics", "probability", "expectation"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(5, 20)
        p_numerator = random.randint(1, 10)
        p_denominator = random.randint(p_numerator + 1, 20)
        return {"n": n, "p_num": p_numerator, "p_den": p_denominator}

    def validate_params(self, params, prev_value=None):
        """Validate linearity of expectation parameters: p_den must be non-zero."""
        p_den = params.get("p_den")
        if p_den is None:
            return False
        try:
            return float(p_den) != 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        p_num = params.get("p_num", 1)
        p_den = params.get("p_den", 2)
        result = (n * p_num * 100) // p_den
        return MethodResult(
            value=result,
            description=f"E[X] = {n} * {p_num}/{p_den} (scaled x100) = {result}",
            params=params,
            metadata={"n": n, "p": f"{p_num}/{p_den}"}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ProbabilisticMethod(MethodBlock):
    """Use probabilistic method to prove existence via expectation."""

    def __init__(self):
        super().__init__()
        self.name = "probabilistic_method"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "probability", "existence"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None and input_value < 500:
            n = input_value + random.randint(20, 50)
            k = random.randint(n // 4, n // 2)
        elif input_value is not None:
            n = input_value
            k = random.randint(n // 4, n // 2)
        else:
            n = random.randint(30, 100)
            k = random.randint(n // 4, n // 2)
        return {"n": n, "k": k}

    def validate_params(self, params, prev_value=None):
        """Validate probabilistic method parameters: n >= k >= 0."""
        n = params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        try:
            n_val = int(n)
            k_val = int(k)
            return n_val >= k_val and k_val >= 0
        except (ValueError, TypeError):
            return False

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        k = params.get("k", 5)
        prob_success = 0.6

        try:
            total_subsets = math.comb(n, k)
            if total_subsets > 10**15:
                result = min(99999, max(int(n * prob_success * 100), 1))
            else:
                expected_count = int(prob_success * total_subsets)
                result = max(expected_count, 1)
        except (OverflowError, ValueError):
            result = min(99999, max(int(n * prob_success * 100), 1))

        return MethodResult(
            value=result,
            description=f"Probabilistic method: expected {result} k-subsets (n={n}, k={k}) with property",
            params=params,
            metadata={"n": n, "k": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class OptionalStoppingTheorem(MethodBlock):
    """Apply the optional stopping theorem: E[M_t] = E[M_0] for bounded stopping times."""

    def __init__(self):
        super().__init__()
        self.name = "optional_stopping_theorem"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["probability", "stochastic", "martingale", "stopping_time"]

    def generate_parameters(self, input_value=None):
        return {
            "initial_value": random.uniform(0, 100),
            "is_bounded_stopping_time": random.choice([True, False])
        }

    def compute(self, input_value, params):
        initial_value = params.get("initial_value", 0.0)
        is_bounded = params.get("is_bounded_stopping_time", True)

        if is_bounded:
            result = initial_value
            description = f"E[M_t] = E[M_0] = {initial_value:.4f} (bounded stopping time)"
        else:
            result = -1
            description = f"Stopping time not bounded, theorem doesn't apply -> {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "techniques_used": [self.name],
                "is_bounded_stopping_time": is_bounded,
                "initial_value": initial_value
            }
        )

    def can_invert(self):
        return False


# ============================================================================
# COUNTING & INCLUSION-EXCLUSION (3 techniques)
# ============================================================================

@register_technique
class InclusionExclusion(MethodBlock):
    """Apply inclusion-exclusion principle to count union of sets."""

    def __init__(self):
        super().__init__()
        self.name = "inclusion_exclusion"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "inclusion-exclusion", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n_sets = random.randint(2, 4)
        if input_value is not None and input_value < 500:
            universe_size = input_value * random.randint(10, 30)
        elif input_value is not None:
            universe_size = input_value
        else:
            universe_size = random.randint(500, 5000)
        set_sizes = [random.randint(universe_size // 4, universe_size // 2) for _ in range(n_sets)]
        return {"universe_size": universe_size, "set_sizes": set_sizes}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        universe_size = params.get("universe_size", input_value)
        set_sizes = params.get("set_sizes", [100, 150])
        n = len(set_sizes)
        if n == 2:
            a, b = set_sizes
            overlap = (a * b) // universe_size
            result = a + b - overlap
        else:
            result = sum(set_sizes) - sum(
                set_sizes[i] * set_sizes[j] // universe_size
                for i in range(n) for j in range(i + 1, n)
            )
        result = min(result, universe_size)
        description = f"PIE estimate: |A u B u ...| ~ {result} from sets {set_sizes}"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"universe_size": universe_size, "set_sizes": set_sizes}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class DerangementViaInclusionExclusion(MethodBlock):
    """Derive derangement count using inclusion-exclusion principle."""

    def __init__(self):
        super().__init__()
        self.name = "derangement_via_inclusion_exclusion"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 3
        self.tags = ["combinatorics", "derangement", "inclusion-exclusion"]

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
        result = _derangement(n)
        description = f"D_{n} via PIE = {result}"
        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "formula": "n! * sum(-1)^k/k!"}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ExtremalConstruction(MethodBlock):
    """Extremal argument: construct optimal example to prove bound."""

    def __init__(self):
        super().__init__()
        self.name = "extremal_construction"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["reasoning", "proof", "extremal", "construction"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value else random.randint(5, 50)
        objective = random.choice(["max_independent_set", "min_cover", "max_matching"])
        return {"n": n, "objective": objective}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value if input_value else 10)
        objective = params.get("objective", "max_independent_set")

        if objective == "max_independent_set":
            result = 1
            desc = f"Extremal: max independent set in K_{n} = 1"
        elif objective == "min_cover":
            result = max(1, n - 1)
            desc = f"Extremal: min vertex cover of K_{n} = {result}"
        elif objective == "max_matching":
            result = n // 2
            desc = f"Extremal: max matching in K_{n} = {result}"
        else:
            result = n
            desc = f"Extremal construction for n={n}"

        return MethodResult(value=result, description=desc, params=params, metadata={"n": n})

    def validate_params(self, params, prev_value=None) -> bool:
        n = params.get("n", prev_value)
        return n is not None and n >= 1

    def can_invert(self) -> bool:
        return False


# ============================================================================
# GRAPH THEORY (2 techniques)
# ============================================================================

@register_technique
class GraphArboricityBound(MethodBlock):
    """Compute chromatic number upper bound from arboricity: chi(G) <= 2 * arboricity(G)."""

    def __init__(self):
        super().__init__()
        self.name = "graph_arboricity_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["graph_theory", "arboricity", "chromatic_number", "nash_williams"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(1, 20)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", 1)
        chi_upper_bound = 2 * n

        return MethodResult(
            value=chi_upper_bound,
            description=f"For a graph with arboricity {n}, chi(G) <= 2 * {n} = {chi_upper_bound}",
            params=params,
            metadata={
                "arboricity": n,
                "chromatic_bound": chi_upper_bound,
                "theorem": "Nash-Williams arboricity bound"
            }
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value: Any, params: Dict[str, Any]) -> Optional[Any]:
        if output_value % 2 == 0:
            return output_value // 2
        return None

    def _find_params_for_answer(self, target_answer: int) -> Optional[Dict[str, Any]]:
        if target_answer % 2 == 0 and target_answer >= 2:
            return {"n": target_answer // 2}
        return None


@register_technique
class ChromaticNumberBound(MethodBlock):
    """Return chromatic number bound given a bound type and parameter."""

    def __init__(self):
        super().__init__()
        self.name = "chromatic_number_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["graph_theory", "chromatic_number", "bounds"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        bound_type = random.choice(["arboricity", "degeneracy", "maximum_degree"])
        param_value = input_value if input_value is not None else random.randint(2, 15)
        return {"bound_type": bound_type, "param": param_value}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        bound_type = params.get("bound_type", "arboricity")
        param = params.get("param", input_value if input_value is not None else 2)

        if bound_type == "arboricity":
            bound = 2 * param
            description = f"chi(G) <= 2 * arboricity = 2 * {param} = {bound}"
        elif bound_type == "degeneracy":
            bound = param + 1
            description = f"chi(G) <= degeneracy + 1 = {param} + 1 = {bound}"
        elif bound_type == "maximum_degree":
            bound = param + 1
            description = f"chi(G) <= Delta + 1 = {param} + 1 = {bound}"
        else:
            bound = 2 * param
            description = f"chi(G) <= 2 * {param} = {bound}"

        return MethodResult(
            value=bound,
            description=description,
            params=params,
            metadata={"bound_type": bound_type, "parameter": param, "chromatic_bound": bound}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# RAMSEY THEORY (1 technique)
# ============================================================================

@register_technique
class RamseyBound(MethodBlock):
    """Compute bounds related to Ramsey numbers."""

    RAMSEY_VALUES = {
        (3, 3): 6,
        (3, 4): 9,
        (3, 5): 14,
        (3, 6): 18,
        (3, 7): 23,
        (3, 8): 28,
        (3, 9): 36,
        (4, 4): 18,
        (4, 5): 25,
    }

    def __init__(self):
        super().__init__()
        self.name = "ramsey_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "ramsey_theory", "extremal"]

    def generate_parameters(self, input_value=None):
        m, n = random.choice(list(self.RAMSEY_VALUES.keys()))
        return {"m": m, "n": n}

    def compute(self, input_value, params):
        m = params.get("m", 3)
        n = params.get("n", 3)

        if m > n:
            m, n = n, m

        key = (m, n)
        if key in self.RAMSEY_VALUES:
            result = self.RAMSEY_VALUES[key]
            description = f"R({m}, {n}) = {result}"
        else:
            result = math.comb(m + n - 2, m - 1)
            description = f"Upper bound for R({m}, {n}) <= C({m+n-2}, {m-1}) = {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"m": m, "n": n}
        )

    def can_invert(self):
        return False
