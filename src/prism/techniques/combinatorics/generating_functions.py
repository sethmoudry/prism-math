"""
Generating function methods for combinatorics.

This module contains:
- GeneratingFunction (coefficient extraction, fibonacci, catalan)
- CountingViaAlgebra (algebraic counting methods)
- BinomialTheoremAnalogy
"""

import random
import math
from typing import Any, Dict, Optional
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class GeneratingFunction(MethodBlock):
    """
    Apply generating function techniques to extract integer coefficients.

    Operations:
    - coefficient: Extract [x^k] coefficient from (1+x)^n
    - fibonacci: Fibonacci via generating function x/(1-x-x^2)
    - catalan: Catalan via (1-sqrt(1-4x))/(2x)
    """
    def __init__(self):
        super().__init__()
        self.name = "generating_function"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "generating_functions", "algebra"]

    def validate_params(self, params, prev_value=None):
        """Validate generating function parameters."""
        operation = params.get("operation")
        n = params.get("n")
        k = params.get("k")
        if operation in ["coefficient", "binomial_coeff"]:
            return n is not None and k is not None and 0 <= k <= n
        elif operation in ["fibonacci", "catalan"]:
            return n is not None and n >= 0
        return True

    def generate_parameters(self, input_value=None):
        operation = random.choice(["coefficient", "fibonacci", "catalan"])
        if operation == "coefficient":
            n = random.randint(5, 20)
            k = random.randint(0, n)
            return {"operation": operation, "n": n, "k": k}
        else:
            n = random.randint(1, 15)
            return {"operation": operation, "n": n}

    def compute(self, input_value, params):
        operation = params.get("operation", "coefficient")
        n = params.get("n", 10)
        # Bound n to prevent slow computations
        n = min(abs(n) if n else 10, 50)

        if operation == "coefficient":
            k = params.get("k", n // 2)
            k = min(abs(k) if k else 5, n)
            # [x^k] (1+x)^n = C(n,k)
            result = math.comb(n, k)
            desc = f"[x^{k}] (1+x)^{n} = C({n},{k}) = {result}"
        elif operation == "fibonacci":
            # F_n from generating function
            def fib(m):
                if m <= 1:
                    return m
                a, b = 0, 1
                for _ in range(m - 1):
                    a, b = b, a + b
                return b
            result = fib(n)
            desc = f"F_{n} = {result} (via generating function x/(1-x-x^2))"
        elif operation == "catalan":
            # C_n = (1/(n+1)) * C(2n, n)
            result = math.comb(2 * n, n) // (n + 1)
            desc = f"C_{n} = {result} (via generating function)"
        else:
            result = 1
            desc = "Unknown operation"

        return MethodResult(value=result, description=desc, params=params)

    def can_invert(self) -> bool:
        return False


@register_technique
class CountingViaAlgebra(MethodBlock):
    """Use generating functions and algebraic methods for counting.

    Operations:
    - coefficient_extraction: Extract [x^k] from (1+x)^n or other generating functions
    - partition_generating: Count integer partitions using generating functions
    - fibonacci_gf: Use generating function for Fibonacci numbers
    - catalan_gf: Catalan numbers via generating function
    - stars_and_bars: Count non-negative integer solutions to x1+...+xk=n
    """
    def __init__(self):
        super().__init__()
        self.name = "counting_via_algebra"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "generating_functions", "algebra"]

    def validate_params(self, params, prev_value=None):
        """Validate counting parameters based on operation."""
        operation = params.get("operation")
        if operation in ["coefficient_extraction", "binomial_expansion"]:
            n = params.get("n")
            k = params.get("k")
            return n is not None and k is not None and 0 <= k <= n
        elif operation == "stars_and_bars":
            n = params.get("n")
            k = params.get("k")
            return n is not None and k is not None and n >= 0 and k >= 1
        elif operation in ["fibonacci_gf", "catalan_gf"]:
            n = params.get("n")
            return n is not None and n >= 0
        return True

    def generate_parameters(self, input_value=None):
        operation = random.choice(["coefficient_extraction", "stars_and_bars",
                                   "fibonacci_gf", "catalan_gf", "binomial_expansion"])
        if operation == "coefficient_extraction":
            n = random.randint(5, 20)
            k = random.randint(0, n)
            return {"operation": operation, "n": n, "k": k}
        elif operation == "stars_and_bars":
            n = random.randint(5, 15)
            k = random.randint(2, 6)
            return {"operation": operation, "n": n, "k": k}
        elif operation == "fibonacci_gf":
            n = random.randint(1, 20)
            return {"operation": operation, "n": n}
        elif operation == "catalan_gf":
            n = random.randint(1, 12)
            return {"operation": operation, "n": n}
        else:  # binomial_expansion
            n = random.randint(3, 10)
            k = random.randint(0, n)
            a, b = random.randint(1, 3), random.randint(1, 3)
            return {"operation": operation, "n": n, "k": k, "a": a, "b": b}

    def compute(self, input_value, params):
        operation = params.get("operation", "coefficient_extraction")

        if operation == "coefficient_extraction":
            # [x^k](1+x)^n = C(n,k)
            n = params.get("n", 10)
            k = params.get("k", 5)
            n = min(abs(n) if n else 10, 100)
            k = min(abs(k) if k else 5, n)
            result = math.comb(n, k)
            return MethodResult(
                value=result,
                description=f"Coefficient of x^{k} in (1+x)^{n} is C({n},{k}) = {result}",
                metadata={"operation": operation, "n": n, "k": k}
            )

        elif operation == "stars_and_bars":
            # Number of non-negative integer solutions to x1+...+xk=n is C(n+k-1, k-1)
            n = params.get("n", 10)
            k = params.get("k", 3)
            n = min(abs(n) if n else 10, 100)
            k = min(abs(k) if k else 3, 50)
            result = math.comb(n + k - 1, k - 1)
            return MethodResult(
                value=result,
                description=f"Non-negative integer solutions to x1+...+x{k}={n}: C({n}+{k}-1,{k}-1) = {result}",
                metadata={"operation": operation, "n": n, "k": k}
            )

        elif operation == "fibonacci_gf":
            # F_n via generating function x/(1-x-x^2)
            n = params.get("n", 10)
            n = min(abs(n) if n else 10, 50)
            if n <= 0:
                result = 0
            elif n == 1:
                result = 1
            else:
                a, b = 0, 1
                for _ in range(2, n + 1):
                    a, b = b, a + b
                result = b
            return MethodResult(
                value=result,
                description=f"Fibonacci F_{n} (from generating function x/(1-x-x^2)) = {result}",
                metadata={"operation": operation, "n": n}
            )

        elif operation == "catalan_gf":
            # C_n = (1/(n+1)) * C(2n, n), from generating function (1-sqrt(1-4x))/(2x)
            n = params.get("n", 5)
            n = min(abs(n) if n else 5, 20)
            result = math.comb(2 * n, n) // (n + 1)
            return MethodResult(
                value=result,
                description=f"Catalan number C_{n} = C(2*{n},{n})/({n}+1) = {result}",
                metadata={"operation": operation, "n": n}
            )

        else:  # binomial_expansion
            # [x^k](ax+b)^n = C(n,k) * a^k * b^(n-k)
            n = params.get("n", 5)
            k = params.get("k", 2)
            a = params.get("a", 2)
            b = params.get("b", 1)
            n = min(abs(n) if n else 5, 30)
            k = min(abs(k) if k else 2, n)
            result = math.comb(n, k) * (a ** k) * (b ** (n - k))
            return MethodResult(
                value=result,
                description=f"Coefficient of x^{k} in ({a}x+{b})^{n} = C({n},{k})*{a}^{k}*{b}^{n-k} = {result}",
                metadata={"operation": operation, "n": n, "k": k, "a": a, "b": b}
            )

    def can_invert(self) -> bool:
        return False


@register_technique
class BinomialTheoremAnalogy(MethodBlock):
    """Apply binomial theorem analogy to compute binomial-like expansions."""
    def __init__(self):
        super().__init__()
        self.name = "binomial_theorem_analogy"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "binomial"]

    def generate_parameters(self, input_value=None):
        n = random.randint(3, 10)
        k = random.randint(1, min(n, 5))
        return {"n": n, "k": k}

    def compute(self, input_value, params):
        n = params.get("n", 5)
        k = params.get("k", 2)
        result = math.comb(n, k)
        return MethodResult(
            value=result,
            description=f"Binomial theorem: C({n},{k}) = {result}",
            metadata={"n": n, "k": k}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class HookLengthFormula(MethodBlock):
    """Count standard Young tableaux using the hook length formula.

    For a partition lambda = (lambda_1, lambda_2, ..., lambda_k), the number
    of standard Young tableaux of shape lambda is:

    f^lambda = n! / product of all hook lengths

    where n = |lambda| = sum of parts, and the hook length at cell (i,j)
    is the number of cells directly to the right plus cells directly below
    plus 1 (for the cell itself).
    """

    def __init__(self):
        super().__init__()
        self.name = "hook_length_formula"
        self.input_type = "integer"
        self.output_type = "count"
        self.difficulty = 4
        self.tags = ["combinatorics", "young_tableaux", "hook_length", "counting"]

    def validate_params(self, params, prev_value=None):
        """Validate that partition parts are positive and non-increasing."""
        partition = params.get("partition")
        if partition is None:
            rows = params.get("rows")
            cols = params.get("cols")
            return rows is not None and cols is not None and rows >= 1 and cols >= 1

        if not isinstance(partition, (list, tuple)) or len(partition) == 0:
            return False
        for i, part in enumerate(partition):
            if part < 1:
                return False
            if i > 0 and partition[i] > partition[i-1]:
                return False
        return True

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        """Generate a rectangular partition for simplicity."""
        if input_value is not None and input_value >= 2:
            n = input_value
            rows = int(math.sqrt(n))
            while rows >= 1:
                if n % rows == 0:
                    cols = n // rows
                    break
                rows -= 1
            else:
                rows, cols = 1, n
        else:
            rows = random.randint(2, 5)
            cols = random.randint(2, 5)
        return {"rows": rows, "cols": cols}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute number of standard Young tableaux for rectangular partition."""
        rows = params.get("rows", 3)
        cols = params.get("cols", 3)

        rows = min(abs(rows) if rows else 3, 6)
        cols = min(abs(cols) if cols else 3, 6)

        n = rows * cols

        hook_product = 1
        for i in range(rows):
            for j in range(cols):
                hook = (cols - j - 1) + (rows - i - 1) + 1
                hook_product *= hook

        result = math.factorial(n) // hook_product

        return MethodResult(
            value=result,
            description=f"Standard Young tableaux for {rows}x{cols} rectangle: {result}",
            params=params,
            metadata={
                "rows": rows,
                "cols": cols,
                "n": n,
                "hook_product": hook_product,
                "formula": "n!/prod(hooks)"
            }
        )

    def can_invert(self) -> bool:
        return False
