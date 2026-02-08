"""
Basic graph theory combinatorics methods.

This module contains:
- Chromatic polynomials (2 techniques)
- Spanning trees (1 technique)
- Eulerian paths (1 technique)
- Hall marriage (1 technique)
- Graph theory basics (1 technique)
- Hamiltonian paths (1 technique)
- Turan bound (1 technique)
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# CHROMATIC POLYNOMIALS
# ============================================================================

@register_technique
class ChromaticPoly(MethodBlock):
    """Compute chromatic polynomial P(G,k) for simple graphs."""

    def __init__(self):
        super().__init__()
        self.name = "chromatic_poly"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "graph", "chromatic"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        graph_type = random.choice(["tree", "cycle"])
        if input_value is not None and input_value < 100:
            n = input_value + random.randint(5, 15)
            k = n + random.randint(2, 10)
        elif input_value is not None:
            n = min(input_value, 25)
            k = n + random.randint(2, 8)
        else:
            n = random.randint(8, 20)
            k = n + random.randint(2, 8)
        return {"graph_type": graph_type, "n": n, "k": k}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        graph_type = params.get("graph_type", "tree")
        n = params.get("n", input_value)
        k = params.get("k", 3)
        if graph_type == "tree":
            result = k * ((k - 1) ** (n - 1))
        elif graph_type == "complete":
            result = 1
            for i in range(n):
                result *= (k - i)
            result = max(0, result)
        elif graph_type == "cycle":
            result = (k - 1) ** n + ((-1) ** n) * (k - 1)
        else:
            result = 0
        description = f"P({graph_type}_{n}, {k}) = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"graph_type": graph_type, "n": n, "k": k})

    def can_invert(self) -> bool:
        return False


@register_technique
class GraphColoring(MethodBlock):
    """Compute chromatic polynomial or number of colorings."""

    def __init__(self):
        super().__init__()
        self.name = "graph_coloring"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "graph_theory"]

    def generate_parameters(self, input_value=None):
        n = random.randint(2, 8)
        k = random.randint(2, 5)
        return {"n": n, "k": k}

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        k = params.get("k")
        if n is None or k is None:
            return False
        try:
            n_val = int(n)
            k_val = int(k)
            return n_val >= 1 and k_val >= 1
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        try:
            n = int(params.get("n", 3))
            k = int(params.get("k", 2))
        except (ValueError, TypeError) as e:
            raise TypeError(f"graph_coloring requires integer parameters: n={params.get('n')}, k={params.get('k')}") from e

        n = min(abs(n) if n else 3, 20)
        k = min(abs(k) if k else 2, 10)

        result = k * ((k - 1) ** (n - 1))
        return MethodResult(
            value=result,
            description=f"Complete graph K_{n} with {k} colors: {result} proper colorings",
            metadata={"n": n, "k": k, "graph_type": "complete"}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# SPANNING TREES
# ============================================================================

@register_technique
class SpanningTrees(MethodBlock):
    """Count spanning trees using Kirchhoff's theorem."""

    def __init__(self):
        super().__init__()
        self.name = "spanning_trees"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "graph", "spanning_trees"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 2

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        graph_type = random.choice(["complete", "cycle"])
        if input_value is not None and input_value < 100:
            if graph_type == "complete":
                n = input_value + random.randint(3, 8)
            else:
                n = input_value * random.randint(10, 50)
        elif input_value is not None:
            n = min(input_value, 20) if graph_type == "complete" else input_value
        else:
            n = random.randint(6, 15) if graph_type == "complete" else random.randint(50, 500)
        return {"graph_type": graph_type, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        graph_type = params.get("graph_type", "complete")
        n = params.get("n", input_value)

        if graph_type == "complete":
            n = min(abs(n) if n else 6, 15)
            result = n ** (n - 2) if n >= 2 else 0
        elif graph_type == "cycle":
            n = min(abs(n) if n else 50, 1000)
            result = n
        else:
            result = 0

        description = f"Spanning trees of {graph_type}_{n} = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"graph_type": graph_type, "n": n})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# EULERIAN PATHS
# ============================================================================

@register_technique
class EulerianPaths(MethodBlock):
    """Count Eulerian paths in a graph (simplified cases)."""

    def __init__(self):
        super().__init__()
        self.name = "eulerian_paths"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "graph", "eulerian"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 3

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        graph_type = "cycle"
        if input_value is not None and input_value < 1000:
            n = input_value * random.randint(10, 50)
        elif input_value is not None:
            n = input_value
        else:
            n = random.randint(500, 5000)
        return {"graph_type": graph_type, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        graph_type = params.get("graph_type", "cycle")
        n = params.get("n", input_value)
        if graph_type == "cycle":
            result = n
        else:
            result = 0
        description = f"Eulerian paths in {graph_type}_{n} = {result}"
        return MethodResult(value=result, description=description, params=params, metadata={"graph_type": graph_type, "n": n})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# HALL MARRIAGE
# ============================================================================

@register_technique
class HallMarriage(MethodBlock):
    """Check Hall's marriage condition for bipartite matching."""

    def __init__(self):
        super().__init__()
        self.name = "hall_marriage"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "graph", "matching"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n", prev_value) if prev_value is not None else params.get("n")
        return n is not None and n >= 1

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        n = input_value if input_value is not None else random.randint(3, 10)
        return {"n": n, "graph_type": "complete_bipartite"}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        n = min(abs(n) if n else 5, 20)
        graph_type = params.get("graph_type", "complete_bipartite")
        if graph_type == "complete_bipartite":
            result = math.factorial(n)
        else:
            result = 0
        return MethodResult(value=result, description=f"Perfect matchings in K_{{{n},{n}}} = {result}", params=params, metadata={"n": n, "graph_type": graph_type})

    def can_invert(self) -> bool:
        return False


# ============================================================================
# GRAPH THEORY BASICS
# ============================================================================

@register_technique
class GraphTheory(MethodBlock):
    """Basic graph theory operations: vertices, edges, degree calculations."""

    def __init__(self):
        super().__init__()
        self.name = "graph_theory"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "graphs", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        operations = ["complete_graph_edges", "complete_bipartite_edges", "sum_of_degrees",
                      "spanning_trees", "cycle_edges", "path_edges"]
        operation = random.choice(operations)
        if input_value is not None:
            n = input_value
        else:
            n = random.randint(3, 15)
        params = {"n": n, "operation": operation}
        if operation == "complete_bipartite_edges":
            params["m"] = random.randint(2, 10)
        if operation == "sum_of_degrees":
            max_edges = n * (n - 1) // 2
            params["edges"] = random.randint(n - 1, min(max_edges, 3 * n))
        return params

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value if input_value is not None else 5)
        operation = params.get("operation", "complete_graph_edges")

        if operation == "complete_graph_edges":
            result = n * (n - 1) // 2
            description = f"Complete graph K_{n} has {n}*({n}-1)/2 = {result} edges"
        elif operation == "complete_bipartite_edges":
            m = params.get("m", n)
            result = m * n
            description = f"Complete bipartite graph K_{{{m},{n}}} has {m}*{n} = {result} edges"
        elif operation == "sum_of_degrees":
            edges = params.get("edges", n)
            result = 2 * edges
            description = f"Sum of degrees in graph with {edges} edges = 2*{edges} = {result} (handshaking lemma)"
        elif operation == "spanning_trees":
            if n < 2:
                result = 1
            else:
                result = n ** (n - 2)
            description = f"Number of labeled spanning trees of K_{n} = {n}^({n}-2) = {result} (Cayley's formula)"
        elif operation == "cycle_edges":
            result = n
            description = f"Cycle graph C_{n} has {n} edges"
        elif operation == "path_edges":
            result = n - 1 if n > 0 else 0
            description = f"Path graph P_{n} has {n}-1 = {result} edges"
        else:
            result = n * (n - 1) // 2
            description = f"Graph theory computation for n={n}: {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"n": n, "operation": operation}
        )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# HAMILTONIAN PATHS
# ============================================================================

@register_technique
class HamiltonianPath(MethodBlock):
    """Count or check Hamiltonian paths and cycles in graphs."""

    def __init__(self):
        super().__init__()
        self.name = "hamiltonian_path"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "graph_theory", "hamiltonian"]

    def generate_parameters(self, input_value=None):
        graph_type = random.choice(["complete", "complete_cycle", "path_exists"])
        n = input_value if input_value is not None else random.randint(3, 8)
        return {"graph_type": graph_type, "n": n}

    def compute(self, input_value, params):
        graph_type = params.get("graph_type", "complete")
        n = params.get("n", input_value if input_value is not None else 5)

        if n < 1:
            return MethodResult(value=0, description="Invalid graph size",
                                metadata={"graph_type": graph_type, "n": n})

        if graph_type == "complete":
            if n == 1:
                result = 1
            else:
                result = math.factorial(n) // 2
            return MethodResult(
                value=result,
                description=f"Complete graph K_{n} has {n}!/2 = {result} Hamiltonian paths",
                metadata={"graph_type": graph_type, "n": n}
            )
        elif graph_type == "complete_cycle":
            if n <= 2:
                result = 1 if n == 2 else 0
            else:
                result = math.factorial(n - 1) // 2
            return MethodResult(
                value=result,
                description=f"Complete graph K_{n} has ({n}-1)!/2 = {result} Hamiltonian cycles",
                metadata={"graph_type": graph_type, "n": n}
            )
        else:
            result = 1 if n >= 1 else 0
            return MethodResult(
                value=result,
                description=f"K_{n} {'has' if result else 'does not have'} a Hamiltonian path",
                metadata={"graph_type": graph_type, "n": n, "exists": bool(result)}
            )

    def can_invert(self) -> bool:
        return False


# ============================================================================
# TURAN BOUND
# ============================================================================

@register_technique
class TuranBound(MethodBlock):
    """Compute Turan number ex(n, K_r) - max edges in n-vertex graph without K_r."""

    def __init__(self):
        super().__init__()
        self.name = "turan_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "graph_theory", "extremal", "turan"]

    def validate_params(self, params, prev_value=None):
        n = params.get("n")
        r = params.get("r")
        if n is None or r is None:
            return False
        return n >= 1 and r >= 2

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        if input_value is not None:
            n = input_value
        else:
            n = random.randint(5, 50)
        r = random.randint(3, min(6, n))
        return {"n": n, "r": r}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        n = params.get("n", input_value)
        r = params.get("r", 3)

        if n is None:
            n = 10
        if n < 1:
            return MethodResult(
                value=0,
                description=f"Turan bound undefined for n={n} < 1, returning 0",
                params=params,
                metadata={"n": n, "r": r, "error": "n < 1"}
            )
        if r < 2:
            r = 2

        num_parts = r - 1
        base_size = n // num_parts
        remainder = n % num_parts

        sum_of_squares = remainder * (base_size + 1) ** 2 + (num_parts - remainder) * base_size ** 2
        turan_edges = (n * n - sum_of_squares) // 2

        return MethodResult(
            value=turan_edges,
            description=f"ex({n}, K_{r}) = {turan_edges} (Turan bound)",
            params=params,
            metadata={
                "n": n,
                "r": r,
                "num_parts": num_parts,
                "formula": f"(1 - 1/{r-1}) * {n}^2 / 2"
            }
        )

    def can_invert(self) -> bool:
        return False
