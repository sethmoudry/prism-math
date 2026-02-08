"""
Topological and algebraic topology combinatorics methods.

This module contains:
- TopologicalComplexityGraph (Farber's theorem)
- BouquetOfCircles
- KleitmanRothschildHeightBound
- EssentialVertexCount
- FigureEightTopology
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class TopologicalComplexityGraph(MethodBlock):
    """Compute topological complexity (TC) for graphs using Farber's theorem.

    Farber's theorem: For a connected graph X, TC(X) = 2 * (number of essential vertices) + 1,
    where an essential vertex is one with degree >= 3.
    """

    def __init__(self):
        super().__init__()
        self.name = "topological_complexity_graph"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["topology", "graph_theory", "farber_theorem", "algebraic_topology"]

    def generate_parameters(self, input_value=None):
        graph_type = random.choice(['bouquet', 'tree', 'general'])

        if graph_type == 'bouquet':
            n_circles = input_value if input_value is not None else random.randint(1, 10)
            n_essential_vertices = 1 if n_circles >= 1 else 0
            return {"graph_type": graph_type, "n_circles": n_circles, "n_essential_vertices": n_essential_vertices}
        elif graph_type == 'tree':
            n_essential_vertices = input_value if input_value is not None else random.randint(0, 5)
            return {"graph_type": graph_type, "n_essential_vertices": n_essential_vertices}
        else:
            n_essential_vertices = input_value if input_value is not None else random.randint(1, 5)
            return {"graph_type": graph_type, "n_essential_vertices": n_essential_vertices}

    def compute(self, input_value, params):
        graph_type = params.get("graph_type", "bouquet")
        n_essential_vertices = params.get("n_essential_vertices", 1)
        n_circles = params.get("n_circles", 2)

        if graph_type == 'interval':
            tc = 1
            description = f"TC([0,1]) = 1 (interval is contractible up to homotopy)"
        elif n_essential_vertices == 0:
            tc = 1
            description = f"TC(X) = 1 (no essential vertices, graph is a tree homeomorphic to interval)"
        else:
            tc = 2 * n_essential_vertices + 1
            if graph_type == 'bouquet':
                description = (
                    f"TC(bouquet of {n_circles} circles) = 2*{n_essential_vertices} + 1 = {tc} "
                    f"(by Farber's theorem)"
                )
            else:
                description = (
                    f"TC(X) = 2*{n_essential_vertices} + 1 = {tc} "
                    f"(by Farber's theorem with {n_essential_vertices} essential vertices)"
                )

        return MethodResult(
            value=tc,
            description=description,
            params=params,
            metadata={
                "graph_type": graph_type,
                "n_essential_vertices": n_essential_vertices,
                "topological_complexity": tc,
                "formula": "TC(X) = 2 * (number of essential vertices) + 1"
            }
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target_answer: int):
        """Support backward generation for TC values."""
        if target_answer == 1:
            return {"graph_type": "interval", "n_essential_vertices": 0}
        elif target_answer >= 3 and target_answer % 2 == 1:
            v = (target_answer - 1) // 2
            if v == 1:
                n = random.randint(2, 10)
                return {"graph_type": "bouquet", "n_circles": n, "n_essential_vertices": 1}
            else:
                return {"graph_type": "general", "n_essential_vertices": v}
        return None


@register_technique
class BouquetOfCircles(MethodBlock):
    """Compute properties of a bouquet of n circles (wedge sum of n copies of S^1).

    A bouquet of n circles has:
    - Fundamental group: Free group on n generators F_n
    - One vertex (the basepoint) with degree 2n
    - n edges (loops)
    - Topological complexity TC = 3 (by Farber's theorem, since 1 essential vertex)
    """

    def __init__(self):
        super().__init__()
        self.name = "bouquet_of_circles"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["topology", "algebraic_topology", "graph_theory"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(1, 10)
        property_type = random.choice(['tc', 'degree', 'fundamental_group_rank'])
        return {"n_circles": n, "property": property_type}

    def compute(self, input_value, params):
        n = params.get("n_circles", input_value if input_value else 2)
        property_type = params.get("property", "tc")

        if property_type == 'tc':
            result = 3
            description = f"TC(bouquet of {n} circles) = 3 (by Farber's theorem)"
        elif property_type == 'degree':
            result = 2 * n
            description = f"Degree of vertex in bouquet of {n} circles = 2*{n} = {result}"
        elif property_type == 'fundamental_group_rank':
            result = n
            description = f"Rank of fundamental group pi_1(bouquet of {n} circles) = {n} (free group F_{n})"
        else:
            result = n
            description = f"Bouquet has {n} circles"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n_circles": n, "property": property_type, "result": result}
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target_answer: int):
        """Support backward generation."""
        if target_answer == 3:
            return {"n_circles": random.randint(2, 10), "property": "tc"}
        elif target_answer >= 2:
            if target_answer % 2 == 0:
                return {"n_circles": target_answer // 2, "property": "degree"}
            return {"n_circles": target_answer, "property": "fundamental_group_rank"}
        return None


@register_technique
class KleitmanRothschildHeightBound(MethodBlock):
    """Compute upper bound on expected height of random poset.

    The Kleitman-Rothschild theorem (1975) shows that asymptotically almost all
    finite posets have a three-layer structure (A, B, C where A < B < C).
    Therefore, almost all posets have height exactly 3.
    """

    def __init__(self):
        super().__init__()
        self.name = "kleitman_rothschild_height_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "poset", "random", "theorem", "bounds"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value is not None else random.randint(10, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value if input_value else 100)
        height_bound = 3

        description = (
            f"By the Kleitman-Rothschild theorem, almost all posets on {n} elements "
            f"have a three-layer structure with height exactly 3. "
            f"Upper bound on expected height: {height_bound}"
        )

        return MethodResult(
            value=height_bound,
            description=description,
            params=params,
            metadata={"n": n, "theorem": "Kleitman-Rothschild (1975)", "height_bound": height_bound}
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target_answer: int):
        """Support backward generation - only valid for target=3."""
        if target_answer == 3:
            return {"n": random.randint(10, 1000)}
        return None


@register_technique
class EssentialVertexCount(MethodBlock):
    """Count essential vertices in a graph.

    An essential vertex is one with degree >= 3.
    Used in Farber's theorem for topological complexity: TC = 2v + 1
    """

    def __init__(self):
        super().__init__()
        self.name = "essential_vertex_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["topology", "graph_theory", "farber_theorem"]

    def generate_parameters(self, input_value=None):
        graph_type = random.choice(['bouquet', 'tree', 'general'])

        if graph_type == 'bouquet':
            n_circles = input_value if input_value else random.randint(2, 10)
            return {"graph_type": "bouquet", "n_circles": n_circles}
        else:
            n_essential = input_value if input_value else random.randint(1, 5)
            return {"graph_type": graph_type, "n_essential_vertices": n_essential}

    def compute(self, input_value, params):
        graph_type = params.get("graph_type", "bouquet")

        if graph_type == 'bouquet':
            n_circles = params.get("n_circles", 2)
            result = 1
            description = f"Bouquet of {n_circles} circles has 1 essential vertex (degree {2*n_circles} >= 3)"
        else:
            result = params.get("n_essential_vertices", 1)
            description = f"Graph has {result} essential vertices (vertices with degree >= 3)"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"graph_type": graph_type, "essential_vertex_count": result}
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target_answer: int):
        """Support backward generation."""
        if target_answer == 1:
            return {"graph_type": "bouquet", "n_circles": random.randint(2, 10)}
        elif target_answer >= 1:
            return {"graph_type": "general", "n_essential_vertices": target_answer}
        return None


@register_technique
class FigureEightTopology(MethodBlock):
    """Compute topological properties of a figure-eight (wedge of two circles).

    The figure-eight is the simplest non-trivial bouquet of circles.
    It has:
    - Fundamental group: Free group on 2 generators F_2
    - One essential vertex with degree 4
    - Topological complexity TC = 3
    """

    def __init__(self):
        super().__init__()
        self.name = "figure_eight_topology"
        self.input_type = "string"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["topology", "algebraic_topology", "graph_theory"]

    def generate_parameters(self, input_value=None):
        property_type = random.choice(['tc', 'essential_vertices', 'vertex_degree', 'fundamental_group_rank'])
        return {"property": property_type}

    def compute(self, input_value, params):
        property_type = params.get("property", "tc")

        if property_type == 'tc':
            result = 3
            description = "TC(figure-eight) = 3 (by Farber's theorem: 1 essential vertex, TC = 2*1 + 1)"
        elif property_type == 'essential_vertices':
            result = 1
            description = "Figure-eight has 1 essential vertex (the wedge point with degree 4)"
        elif property_type == 'vertex_degree':
            result = 4
            description = "The wedge point of figure-eight has degree 4 (each circle contributes 2)"
        elif property_type == 'fundamental_group_rank':
            result = 2
            description = "Fundamental group of figure-eight is F_2 (free group on 2 generators)"
        else:
            result = 3
            description = "TC(figure-eight) = 3"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"space": "figure-eight", "property": property_type, "result": result}
        )

    def can_invert(self) -> bool:
        return False

    def _find_params_for_answer(self, target_answer: int):
        """Support backward generation."""
        if target_answer == 3:
            return {"property": "tc"}
        elif target_answer == 1:
            return {"property": "essential_vertices"}
        elif target_answer == 4:
            return {"property": "vertex_degree"}
        elif target_answer == 2:
            return {"property": "fundamental_group_rank"}
        return None
