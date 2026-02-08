"""
Inequality and optimization algebra techniques.

Contains:
- AM-GM (2)
- Cauchy-Schwarz (1)
- Holder's inequality (1)
- Jensen's inequality (1)
- Lagrange optimization (1)
- Floor/ceiling operations (5)
"""

import random
import math
from typing import Any, Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# OPTIMIZATION/INEQUALITIES (6 techniques)
# ============================================================================

@register_technique
class ArithmeticGeometricMeanMin(MethodBlock):
    """Find minimum using AM-GM inequality."""

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_geometric_mean_min"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "inequalities", "amgm", "optimization"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate constraint for AM-GM."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            P = random.randint(100, 500)
        else:
            P = random.randint(10, 50)
        return {"product": P, "n": 2}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply AM-GM: (x+y)/2 >= sqrt(xy), so x+y >= 2*sqrt(P)."""
        P = params.get("product", 20)
        n = params.get("n", 10)

        minimum = n * (P ** (1/n))
        minimum = int(round(minimum))

        description = f"Minimum of sum with product={P} is {minimum} by AM-GM"

        return MethodResult(
            value=minimum,
            description=description,
            params=params, metadata={"product": P, "n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ArithmeticGeometricMeanInverseVars(MethodBlock):
    """Find variable values at AM-GM equality."""

    def __init__(self):
        super().__init__()
        self.name = "arithmetic_geometric_mean_inverse_vars"
        self.input_type = "integer"
        self.output_type = "sequence"
        self.difficulty = 3
        self.tags = ["algebra", "inequalities", "amgm", "optimization"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate product constraint."""
        if input_value is not None and input_value > 1000:
            P = random.randint(100, 500)
        else:
            P = random.randint(10, 50)
        n = 2
        return {"product": P, "n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """At equality, all variables are equal: x = y = sqrt(P)."""
        P = params.get("product", 20)
        n = params.get("n", 10)

        value = P ** (1/n)
        values = [value] * n
        try:
            description = f"At AM-GM equality with product={P}: all vars = {value}"
        except ValueError:
            description = f"At AM-GM equality with product={P}: all vars = <large value>"

        return MethodResult(
            value=values,
            description=description,
            params=params, metadata={"product": P, "n": n}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class CauchySchwarzBound(MethodBlock):
    """Compute bound using Cauchy-Schwarz inequality."""

    def __init__(self):
        super().__init__()
        self.name = "cauchy_schwarz_bound"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "inequalities", "cauchy_schwarz"]

    def validate_params(self, params, prev_value=None):
        """Validate that sequences a and b have the same length and are non-empty."""
        a = params.get("a")
        b = params.get("b")
        if a is None or b is None:
            return False
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        if len(a) == 0 or len(b) == 0:
            return False
        return len(a) == len(b)

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate two sequences."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            n = random.randint(4, 8)
            a = [random.randint(10, 50) for _ in range(n)]
            b = [random.randint(10, 50) for _ in range(n)]
        else:
            n = random.randint(2, 4)
            a = [random.randint(1, 5) for _ in range(n)]
            b = [random.randint(1, 5) for _ in range(n)]
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply (Sigma a_i*b_i)^2 <= (Sigma a_i^2)(Sigma b_i^2)."""
        a = params.get("a", 10)
        b = params.get("b", 10)

        dot_product = sum(a[i] * b[i] for i in range(len(a)))
        sum_a_sq = sum(x**2 for x in a)
        sum_b_sq = sum(x**2 for x in b)

        bound = int(math.sqrt(sum_a_sq * sum_b_sq))

        description = f"Cauchy-Schwarz bound: {dot_product}^2 <= {sum_a_sq}*{sum_b_sq}, so |Sigma a_i*b_i| <= {bound}"

        return MethodResult(
            value=bound,
            description=description,
            params=params, metadata={"a": a, "b": b, "dot_product": dot_product}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class HolderInequality(MethodBlock):
    """Apply Holder's inequality."""

    def __init__(self):
        super().__init__()
        self.name = "holder_inequality"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "inequalities", "holder"]

    def validate_params(self, params, prev_value=None):
        """Validate Holder inequality parameters."""
        a = params.get("a")
        b = params.get("b")
        p = params.get("p")
        q = params.get("q")
        if a is None or b is None:
            return False
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            return False
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            return False
        if p is not None and q is not None:
            return p > 1 and q > 1
        return True

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate sequences and exponents."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            n = random.randint(4, 8)
            a = [random.randint(10, 50) for _ in range(n)]
            b = [random.randint(10, 50) for _ in range(n)]
        else:
            n = random.randint(2, 3)
            a = [random.randint(1, 5) for _ in range(n)]
            b = [random.randint(1, 5) for _ in range(n)]
        p = 2
        q = 2
        return {"a": a, "b": b, "p": p, "q": q}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply Holder: Sigma|a_i*b_i| <= (Sigma|a_i|^p)^(1/p) * (Sigma|b_i|^q)^(1/q)."""
        a = params.get("a", 10)
        b = params.get("b", 10)
        p = params.get("p", 5)
        q = params.get("q", 10)

        sum_product = sum(abs(a[i] * b[i]) for i in range(len(a)))
        norm_a = sum(abs(x)**p for x in a) ** (1/p)
        norm_b = sum(abs(x)**q for x in b) ** (1/q)

        bound = int(norm_a * norm_b)

        description = f"Holder bound: {sum_product} <= {bound}"

        return MethodResult(
            value=bound,
            description=description,
            params=params, metadata={"a": a, "b": b, "p": p, "q": q}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class JensenInequality(MethodBlock):
    """Apply Jensen's inequality for convex functions."""

    def __init__(self):
        super().__init__()
        self.name = "jensen_inequality"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "inequalities", "jensen", "convex"]

    def validate_params(self, params, prev_value=None):
        """Validate Jensen inequality parameters."""
        values = params.get("values")
        func = params.get("func")
        if values is None:
            return False
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return False
        if func is not None and func not in ["square", "exp", "log"]:
            return False
        return True

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate values for Jensen."""
        if isinstance(input_value, (int, float)) and input_value > 1000:
            n = random.randint(4, 8)
            values = [random.randint(20, 100) for _ in range(n)]
        else:
            n = random.randint(2, 4)
            values = [random.randint(1, 10) for _ in range(n)]
        func = "square"
        return {"values": values, "func": func}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply Jensen: f(E[X]) <= E[f(X)] for convex f."""
        values = params.get("values", 10)
        func = params.get("func", 10)

        n = len(values)
        mean = sum(values) / n

        if func == "square":
            f_mean = mean ** 2
            mean_f = sum(x**2 for x in values) / n

            result = int(f_mean)

            description = f"Jensen: f(mean) = {f_mean:.2f} <= {mean_f:.2f} = mean(f(x))"

            return MethodResult(
                value=result,
                description=description,
                params=params, metadata={"values": values, "func": func, "mean": mean}
            )

        return MethodResult(value=0, description="Unknown function", params=params, metadata={})

    def can_invert(self) -> bool:
        return False


@register_technique
class LagrangeOptimize(MethodBlock):
    """Optimize using Lagrange multipliers."""

    def __init__(self):
        super().__init__()
        self.name = "lagrange_optimize"
        self.input_type = "sequence"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "optimization", "lagrange", "calculus"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate optimization problem."""
        S = random.randint(10, 30)
        return {"sum": S, "objective": "product"}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Solve using Lagrange multipliers."""
        S = params.get("sum", 10)

        optimal_value = (S**2) // 4

        return MethodResult(
            value=optimal_value,
            description=f"Maximum product with sum={S} is {optimal_value} at x=y={S/2}",
            params=params, metadata={"sum": S}
        )

    def _find_params_for_answer(self, target_answer: int) -> Optional[Dict[str, Any]]:
        """Find parameters that produce target_answer."""
        if target_answer < 0:
            return None

        approximate_S = int(2 * math.sqrt(target_answer))

        for S in range(max(1, approximate_S - 5), approximate_S + 10):
            if (S**2) // 4 == target_answer:
                return {"sum": S, "objective": "product"}

        return None

    def can_invert(self) -> bool:
        return False


# ============================================================================
# FLOOR/CEILING (5 techniques)
# ============================================================================

@register_technique
class FloorSum(MethodBlock):
    """Compute Sigma floor((ai+b)/m) for i=0..n-1."""

    def __init__(self):
        super().__init__()
        self.name = "floor_sum"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "floor", "sum"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate floor sum parameters."""
        if input_value is not None and input_value > 1000:
            n = random.randint(20, 50)
            a = random.randint(10, 30)
            b = random.randint(0, 20)
            m = random.randint(3, 10)
        else:
            n = random.randint(5, 15)
            a = random.randint(1, 5)
            b = random.randint(0, 10)
            m = random.randint(3, 10)
        return {"n": n, "a": a, "b": b, "m": m}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute floor sum."""
        n, a, b, m = params.get("n", 10), params.get("a", 10), params.get("b", 10), params.get("m", 10)
        n = min(abs(n) if n else 10, 10000)

        result = sum((a * i + b) // m for i in range(n))

        description = f"Sigma floor(({a}i+{b})/{m}) for i=0..{n-1} = {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FloorSumInverseN(MethodBlock):
    """Find n such that floor sum equals target."""

    def __init__(self):
        super().__init__()
        self.name = "floor_sum_inverse_n"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebra", "floor", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters."""
        if input_value is not None and input_value > 1000:
            a = random.randint(1, 3)
            b = random.randint(0, 5)
            m = random.randint(3, 8)
            target = random.randint(500, 2000)
        else:
            a = random.randint(1, 3)
            b = random.randint(0, 5)
            m = random.randint(3, 8)
            target = random.randint(20, 50)
        return {"a": a, "b": b, "m": m, "target": target}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find smallest n where floor sum >= target."""
        a, b, m = params.get("a", 10), params.get("b", 10), params.get("m", 10)
        target = params.get("target", 10)

        current_sum = 0
        n = 0
        while current_sum < target and n < 5000:
            current_sum += (a * n + b) // m
            n += 1

        description = f"Smallest n where floor_sum >= {target} is n={n}"

        return MethodResult(
            value=n,
            description=description,
            params=params
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FloorDiv(MethodBlock):
    """Compute floor(a/b)."""

    def __init__(self):
        super().__init__()
        self.name = "floor_div"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 1
        self.tags = ["algebra", "floor", "division"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate division parameters."""
        if input_value is not None and input_value > 1000:
            a = random.randint(1000, 5000)
            b = random.randint(3, 15)
        else:
            a = random.randint(10, 100)
            b = random.randint(3, 15)
        return {"a": a, "b": b}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute floor division."""
        a, b = params.get("a", 10), params.get("b", 10)
        result = a // b

        description = f"floor({a}/{b}) = {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params
        )

    def validate_params(self, params, prev_value=None):
        """Divisor b must be non-zero."""
        b = params.get("b", 1)
        return b != 0

    def can_invert(self) -> bool:
        return False


@register_technique
class FloorDivInverseA(MethodBlock):
    """Find range of a given floor(a/b) = q."""

    def __init__(self):
        super().__init__()
        self.name = "floor_div_inverse_a"
        self.input_type = "integer"
        self.output_type = "sequence"
        self.difficulty = 3
        self.tags = ["algebra", "floor", "inverse"]

    def generate_parameters(self, input_value: Optional[Any] = None, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters."""
        if input_value is not None and input_value > 1000:
            b = random.randint(20, 100)
            q = random.randint(10, 50)
        else:
            b = random.randint(5, 15)
            q = random.randint(3, 10)
        return {"b": b, "q": q}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Find range: q*b <= a < (q+1)*b."""
        b, q = params.get("b", 10), params.get("q", 10)

        a_min = q * b
        a_max = (q + 1) * b - 1

        description = f"If floor(a/{b}) = {q}, then {a_min} <= a <= {a_max}"

        return MethodResult(
            value=[a_min, a_max],
            description=description,
            params=params
        )

    def validate_params(self, params, prev_value=None):
        """Divisor b must be positive."""
        b = params.get("b", 1)
        return b > 0

    def can_invert(self) -> bool:
        return False


@register_technique
class HermiteIdentity(MethodBlock):
    """Apply Hermite's identity: Sigma floor(x+(i-1)/n) = floor(nx) for i=1..n."""

    def __init__(self):
        super().__init__()
        self.name = "hermite_identity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "floor", "hermite", "identity"]

    def generate_parameters(self, target_output: Optional[Any] = None) -> Dict[str, Any]:
        """Generate parameters."""
        n = random.randint(3, 8)
        x = random.random() * 10
        return {"n": n, "x": x}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Verify Hermite's identity."""
        n = params.get("n", 10)
        x = params.get("x", 5)

        lhs = sum(math.floor(x + (i-1)/n) for i in range(1, n+1))
        rhs = math.floor(n * x)

        return MethodResult(
            value=rhs,
            description=f"Hermite: Sigma floor(x+(i-1)/{n}) = floor({n}*{x:.2f}) = {rhs} (verified: {lhs}=={rhs})",
            params=params, metadata={"n": n, "x": x}
        )

    def can_invert(self) -> bool:
        return False
