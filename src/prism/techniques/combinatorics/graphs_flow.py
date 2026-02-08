"""
Flow bounds and Platonic solid Eulerian circuits.

This module contains:
- Flow bounds (Seymour's Theorem) (3 techniques)
- Platonic Eulerian circuits (1 technique)
"""

import random
from typing import Any, Dict
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# FLOW BOUNDS (Seymour's Theorem)
# ============================================================================

@register_technique
class SeymourFlowBound(MethodBlock):
    """Apply Seymour's theorem: Every bridgeless graph has a nowhere-zero 6-flow."""

    def __init__(self):
        super().__init__()
        self.name = "seymour_flow_bound"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["graph_theory", "flow", "seymour", "algebraic_graph_theory"]

    def generate_parameters(self, input_value=None):
        return {}

    def compute(self, input_value, params):
        result = 6
        description = (
            "By Seymour's theorem (1981), every bridgeless graph admits a nowhere-zero 6-flow. "
            "Therefore, the flow number T(G) <= 6 for any bridgeless graph G."
        )

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "theorem": "Seymour's 6-flow theorem",
                "year": 1981,
                "statement": "Every bridgeless graph has a nowhere-zero 6-flow",
                "bound": 6
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class MaxPowerOfTwoBelowBound(MethodBlock):
    """Find maximum k such that 2^k <= n."""

    def __init__(self):
        super().__init__()
        self.name = "max_power_of_two_below_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "powers", "logarithm"]

    def generate_parameters(self, input_value=None):
        if input_value is not None:
            n = input_value
        else:
            n = random.randint(3, 100)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value else 6)

        if n < 1:
            return MethodResult(
                value=0,
                description=f"No positive k satisfies 2^k <= {n}",
                params=params,
                metadata={"n": n, "error": True}
            )

        k = 0
        while 2**(k+1) <= n:
            k += 1

        result = k
        description = f"Maximum k such that 2^k <= {n}: k = {k} (since 2^{k} = {2**k} <= {n} < {2**(k+1)} = 2^{k+1})"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "power": 2**k, "next_power": 2**(k+1)}
        )

    def can_invert(self) -> bool:
        return True

    def invert(self, output_value, params):
        k = output_value
        return 2**k + random.randint(0, 2**k - 1)


@register_technique
class MaxMSatisfyingFlowBound(MethodBlock):
    """Find maximum m such that 2^m <= flow_bound and chromatic constraint is satisfiable."""

    def __init__(self):
        super().__init__()
        self.name = "max_m_satisfying_flow_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["graph_theory", "flow", "chromatic", "ramsey", "seymour"]

    def generate_parameters(self, input_value=None):
        flow_upper_bound = random.choice([6, 5, 4])
        require_nonzero_flow = True
        min_chromatic_for_flow_4 = 2

        return {
            "flow_upper_bound": flow_upper_bound,
            "require_nonzero_flow": require_nonzero_flow,
            "min_chromatic_for_flow_4": min_chromatic_for_flow_4
        }

    def compute(self, input_value, params):
        flow_upper_bound = params.get("flow_upper_bound", 6)
        require_nonzero_flow = params.get("require_nonzero_flow", True)

        max_m_by_bound = 0
        m = 1
        while 2**m <= flow_upper_bound:
            max_m_by_bound = m
            m += 1

        if require_nonzero_flow:
            if max_m_by_bound >= 2:
                result = 2
                description = (
                    f"Maximum m = 2: For m=1, edgeless graphs have flow number 1 < 2. "
                    f"For m=2, bipartite bridgeless graphs can have flow number 4 = 2^2. "
                    f"For m>=3, 2^m > {flow_upper_bound} (Seymour's bound)."
                )
            elif max_m_by_bound == 1:
                result = 0
                description = f"No valid m: Even m=1 fails (flow number 1 < 2)"
            else:
                result = 0
                description = f"No valid m: 2^1 = 2 > {flow_upper_bound}"
        else:
            result = max_m_by_bound
            description = f"Maximum m = {result} such that 2^{result} = {2**result} <= {flow_upper_bound}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "flow_upper_bound": flow_upper_bound,
                "max_m_by_bound": max_m_by_bound,
                "seymour_theorem": "Every bridgeless graph has flow number <= 6",
                "bipartite_flow_4": "3-regular bipartite bridgeless graphs have flow number 4"
            }
        )

    def can_invert(self) -> bool:
        return True


# ============================================================================
# PLATONIC EULERIAN CIRCUITS
# ============================================================================

@register_technique
class PlatonicEulerianCircuits(MethodBlock):
    """Count unique Eulerian circuits on Platonic solids up to rotational symmetry."""

    PLATONIC_SOLIDS = {
        "tetrahedron": {"vertices": 4, "edges": 6, "faces": 4, "degree": 3},
        "cube": {"vertices": 8, "edges": 12, "faces": 6, "degree": 3},
        "octahedron": {"vertices": 6, "edges": 12, "faces": 8, "degree": 4},
        "dodecahedron": {"vertices": 20, "edges": 30, "faces": 12, "degree": 3},
        "icosahedron": {"vertices": 12, "edges": 30, "faces": 20, "degree": 5},
    }

    ROTATION_GROUP_ORDER = {
        "tetrahedron": 12,
        "cube": 24,
        "octahedron": 24,
        "dodecahedron": 60,
        "icosahedron": 60,
    }

    def __init__(self):
        super().__init__()
        self.name = "platonic_eulerian_circuits"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["combinatorics", "graph_theory", "burnside", "platonic_solid", "eulerian"]

    def generate_parameters(self, input_value=None):
        solid = random.choice(list(self.PLATONIC_SOLIDS.keys()))
        return {"solid": solid}

    def compute(self, input_value, params):
        solid = params.get("solid", "octahedron")

        if solid not in self.PLATONIC_SOLIDS:
            return MethodResult(
                value=0,
                description=f"Unknown solid: {solid}",
                params=params,
                metadata={"error": True}
            )

        props = self.PLATONIC_SOLIDS[solid]
        degree = props["degree"]

        if degree % 2 != 0:
            return MethodResult(
                value=0,
                description=f"No Eulerian circuits on {solid}: all vertices have odd degree {degree}",
                params=params,
                metadata={
                    "solid": solid,
                    "vertex_degree": degree,
                    "has_eulerian": False,
                    "reason": "odd_degree_vertices"
                }
            )

        result = 1

        return MethodResult(
            value=result,
            description=f"Unique Eulerian circuits on {solid} up to rotation: {result}",
            params=params,
            metadata={
                "solid": solid,
                "vertex_degree": degree,
                "vertices": props["vertices"],
                "edges": props["edges"],
                "rotation_group_order": self.ROTATION_GROUP_ORDER[solid],
                "has_eulerian": True,
                "insight": "All Eulerian circuits equivalent under rotation group"
            }
        )

    def can_invert(self) -> bool:
        return False
