"""
Advanced function algebra techniques.

Contains:
- Golden ratio identity (1)
- Generating functions (1)
- Fibonacci identity (1)
- Series approximation (1)
- Summation manipulation (1)
"""

import random
import math
from typing import Any, Dict

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class GoldenRatioIdentity(MethodBlock):
    """Use golden ratio properties: phi^2 = phi + 1."""

    def __init__(self):
        super().__init__()
        self.name = "golden_ratio_identity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "golden_ratio", "identity"]

    def generate_parameters(self, input_value=None, target_output=None) -> Dict[str, Any]:
        """Generate power of phi to compute."""
        if input_value is not None and input_value > 1000:
            n = random.randint(10, 15)
        else:
            n = random.randint(3, 8)
        return {"n": n}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Compute phi^n using identity phi^n = F_n*phi + F_(n-1)."""
        n = params.get("n", 10)

        phi = (1 + math.sqrt(5)) / 2

        fib = [0, 1]
        for i in range(2, n+1):
            fib.append(fib[-1] + fib[-2])

        result = fib[n] * phi + fib[n-1]
        result_int = int(round(result))

        desc = f"phi^{n} = {result_int} (using F_{n}*phi + F_{n-1})"

        return MethodResult(
            value=result_int,
            description=desc,
            params=params, metadata={"n": n, "fib": fib}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class OrdinaryGeneratingFunction(MethodBlock):
    """Use ordinary generating function."""

    def __init__(self):
        super().__init__()
        self.name = "ordinary_generating_function"
        self.input_type = "sequence"
        self.output_type = "polynomial"
        self.difficulty = 4
        self.tags = ["algebra", "generating_functions", "ogf"]

    def generate_parameters(self, target_output=None) -> Dict[str, Any]:
        """Generate sequence."""
        n = random.randint(4, 7)
        sequence = [random.randint(0, 5) for _ in range(n)]
        return {"sequence": sequence}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Create OGF: G(x) = Sigma a_n * x^n."""
        seq = params.get("sequence", 10)

        return MethodResult(
            value=seq,
            description=f"OGF G(x) = {' + '.join(f'{seq[i]}x^{i}' for i in range(len(seq)))}",
            params=params, metadata={"sequence": seq}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class FibonacciIdentity(MethodBlock):
    """Apply Fibonacci identities to compute values."""

    def __init__(self):
        super().__init__()
        self.name = "fibonacci_identity"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "fibonacci", "identities", "sequences"]

    def _fib(self, n):
        """Compute nth Fibonacci number (1-indexed: F_1=1, F_2=1)."""
        if n <= 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b

    def generate_parameters(self, input_value=None):
        """Generate parameters for a Fibonacci identity problem."""
        identity_type = random.choice([
            "cassini", "sum_first_n", "sum_odd", "sum_even", "sum_of_squares", "catalan", "docagne"
        ])

        if identity_type == "cassini":
            n = random.randint(3, 25)
            return {"identity_type": identity_type, "n": n}

        elif identity_type == "sum_first_n":
            n = random.randint(5, 20)
            return {"identity_type": identity_type, "n": n}

        elif identity_type == "sum_odd":
            n = random.randint(3, 15)
            return {"identity_type": identity_type, "n": n}

        elif identity_type == "sum_even":
            n = random.randint(3, 15)
            return {"identity_type": identity_type, "n": n}

        elif identity_type == "sum_of_squares":
            n = random.randint(4, 15)
            return {"identity_type": identity_type, "n": n}

        elif identity_type == "catalan":
            n = random.randint(5, 15)
            r = random.randint(1, n - 2)
            return {"identity_type": identity_type, "n": n, "r": r}

        else:
            m = random.randint(5, 15)
            n = random.randint(2, m - 2)
            return {"identity_type": identity_type, "m": m, "n": n}

    def compute(self, input_value, params):
        """Apply the Fibonacci identity to compute the result."""
        identity_type = params.get("identity_type", "cassini")

        if identity_type == "cassini":
            n = params.get("n", 10)
            f_n_minus_1 = self._fib(n - 1)
            f_n = self._fib(n)
            f_n_plus_1 = self._fib(n + 1)
            left_side = f_n_minus_1 * f_n_plus_1 - f_n * f_n
            result = left_side
            desc = f"Cassini's identity: F_{{{n-1}}} * F_{{{n+1}}} - F_{{{n}}}^2 = {f_n_minus_1} * {f_n_plus_1} - {f_n}^2 = {result}"

        elif identity_type == "sum_first_n":
            n = params.get("n", 10)
            result = self._fib(n + 2) - 1
            desc = f"Sum of first {n} Fibonacci numbers: F_1 + ... + F_{n} = F_{{{n+2}}} - 1 = {result}"

        elif identity_type == "sum_odd":
            n = params.get("n", 5)
            result = self._fib(2 * n)
            desc = f"Sum of odd-indexed Fibonacci: F_1 + F_3 + ... + F_{{{2*n-1}}} = F_{{{2*n}}} = {result}"

        elif identity_type == "sum_even":
            n = params.get("n", 5)
            result = self._fib(2 * n + 1) - 1
            desc = f"Sum of even-indexed Fibonacci: F_2 + F_4 + ... + F_{{{2*n}}} = F_{{{2*n+1}}} - 1 = {result}"

        elif identity_type == "sum_of_squares":
            n = params.get("n", 10)
            f_n = self._fib(n)
            f_n_plus_1 = self._fib(n + 1)
            result = f_n * f_n_plus_1
            desc = f"Sum of squares: F_1^2 + ... + F_{n}^2 = F_{n} * F_{{{n+1}}} = {f_n} * {f_n_plus_1} = {result}"

        elif identity_type == "catalan":
            n = params.get("n", 10)
            r = params.get("r", 2)
            f_n = self._fib(n)
            f_n_plus_r = self._fib(n + r)
            f_n_minus_r = self._fib(n - r)
            left_side = f_n * f_n - f_n_plus_r * f_n_minus_r
            result = left_side
            desc = f"Catalan's identity: F_{n}^2 - F_{{{n+r}}} * F_{{{n-r}}} = {f_n}^2 - {f_n_plus_r} * {f_n_minus_r} = {result}"

        else:
            m = params.get("m", 10)
            n = params.get("n", 5)
            f_m = self._fib(m)
            f_n = self._fib(n)
            f_m_plus_1 = self._fib(m + 1)
            f_n_plus_1 = self._fib(n + 1)
            left_side = f_m * f_n_plus_1 - f_m_plus_1 * f_n
            result = left_side
            desc = f"d'Ocagne's identity: F_{m} * F_{{{n+1}}} - F_{{{m+1}}} * F_{n} = {f_m} * {f_n_plus_1} - {f_m_plus_1} * {f_n} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"identity_type": identity_type, **params}
        )

    def can_invert(self):
        return False


@register_technique
class SeriesApproximation(MethodBlock):
    """Compute Taylor/Maclaurin series approximations and partial sums."""

    def __init__(self):
        super().__init__()
        self.name = "series_approximation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "series", "approximation", "calculus"]

    def _factorial(self, n):
        """Compute factorial."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def generate_parameters(self, input_value=None):
        """Generate parameters for series approximation."""
        series_type = random.choice([
            "geometric_sum", "power_sum", "alternating", "harmonic_partial", "exponential_terms", "binomial_sum"
        ])

        if series_type == "geometric_sum":
            r = random.randint(2, 4)
            n = random.randint(3, 8)
            return {"series_type": series_type, "r": r, "n": n}

        elif series_type == "power_sum":
            k = random.choice([1, 2, 3])
            n = random.randint(5, 15)
            return {"series_type": series_type, "k": k, "n": n}

        elif series_type == "alternating":
            n = random.randint(5, 30)
            return {"series_type": series_type, "n": n}

        elif series_type == "harmonic_partial":
            n = random.randint(5, 20)
            return {"series_type": series_type, "n": n}

        elif series_type == "exponential_terms":
            x = random.randint(1, 3)
            terms = random.randint(4, 8)
            return {"series_type": series_type, "x": x, "terms": terms}

        else:
            n = random.randint(3, 10)
            return {"series_type": series_type, "n": n}

    def compute(self, input_value, params):
        """Compute the series approximation."""
        series_type = params.get("series_type", "geometric_sum")

        if series_type == "geometric_sum":
            r = params.get("r", 2)
            n = params.get("n", 5)
            if r == 1:
                result = n + 1
            else:
                result = (r ** (n + 1) - 1) // (r - 1)
            terms = " + ".join([f"{r}^{i}" for i in range(min(n+1, 4))]) + (" + ..." if n > 3 else "")
            desc = f"Geometric sum: 1 + {terms} = (r^{n+1} - 1)/(r-1) = {result}"

        elif series_type == "power_sum":
            k = params.get("k", 2)
            n = params.get("n", 10)

            if k == 1:
                result = n * (n + 1) // 2
                desc = f"Sum of 1 to {n}: n(n+1)/2 = {result}"
            elif k == 2:
                result = n * (n + 1) * (2 * n + 1) // 6
                desc = f"Sum of squares 1^2 + ... + {n}^2 = n(n+1)(2n+1)/6 = {result}"
            else:
                s = n * (n + 1) // 2
                result = s * s
                desc = f"Sum of cubes 1^3 + ... + {n}^3 = [n(n+1)/2]^2 = {result}"

        elif series_type == "alternating":
            n = params.get("n", 10)
            if n % 2 == 0:
                result = -(n // 2)
            else:
                result = (n + 1) // 2
            desc = f"Alternating sum 1 - 2 + 3 - ... +/- {n} = {result}"

        elif series_type == "harmonic_partial":
            n = params.get("n", 10)
            h_n = sum(1.0 / k for k in range(1, n + 1))
            result = int(h_n * n)
            desc = f"Scaled harmonic sum: floor(H_{n} * {n}) = floor({h_n:.4f} * {n}) = {result}"

        elif series_type == "exponential_terms":
            x = params.get("x", 2)
            terms = params.get("terms", 5)
            m_fact = self._factorial(terms - 1)
            total = 0
            for k in range(terms):
                coeff = m_fact // self._factorial(k)
                total += (x ** k) * coeff
            result = total
            desc = f"Scaled exponential partial sum (e^{x} series, {terms} terms, scaled by {terms-1}!) = {result}"

        else:
            n = params.get("n", 5)
            result = 2 ** n
            desc = f"Sum of binomial coefficients C({n},0) + C({n},1) + ... + C({n},{n}) = 2^{n} = {result}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"series_type": series_type, **params}
        )

    def can_invert(self):
        return False


@register_technique
class SummationManipulation(MethodBlock):
    """Manipulate sums using algebraic techniques."""

    def __init__(self):
        super().__init__()
        self.name = "summation_manipulation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["algebra", "summation", "manipulation"]

    def generate_parameters(self, input_value=None):
        """Generate parameters for summation manipulation."""
        manipulation_type = random.choice([
            "telescoping", "partial_fractions", "sum_split", "reverse_and_add",
            "gauss_pairing", "difference_of_sums", "double_sum"
        ])

        n = random.randint(5, 20)

        if manipulation_type == "telescoping":
            return {"manipulation_type": manipulation_type, "n": n, "formula": "reciprocal_product"}

        elif manipulation_type == "partial_fractions":
            d = random.randint(1, 3)
            return {"manipulation_type": manipulation_type, "n": n, "d": d}

        elif manipulation_type == "sum_split":
            k = random.randint(n // 3, 2 * n // 3)
            return {"manipulation_type": manipulation_type, "n": n, "k": k}

        elif manipulation_type == "reverse_and_add":
            return {"manipulation_type": manipulation_type, "n": n}

        elif manipulation_type == "gauss_pairing":
            return {"manipulation_type": manipulation_type, "n": n}

        elif manipulation_type == "difference_of_sums":
            return {"manipulation_type": manipulation_type, "n": n}

        else:
            m = random.randint(3, 8)
            return {"manipulation_type": manipulation_type, "n": m}

    def compute(self, input_value, params):
        """Apply summation manipulation technique."""
        manipulation_type = params.get("manipulation_type", "telescoping")
        n = params.get("n", 10)

        if manipulation_type == "telescoping":
            numerator = n
            result = n
            desc = f"Telescoping sum: sum(1/(k(k+1))) for k=1 to {n} = 1 - 1/{n+1} = {n}/{n+1}, numerator = {result}"

        elif manipulation_type == "partial_fractions":
            d = params.get("d", 1)
            harmonic_part = sum(1.0 / k for k in range(1, d + 1))
            tail_part = sum(1.0 / k for k in range(n + 1, n + d + 1))
            scaled = int(n * d * (harmonic_part - tail_part + 0.001))
            result = scaled if scaled > 0 else n
            desc = f"Partial fractions: sum(1/(k(k+{d}))) for k=1 to {n}, scaled result = {result}"

        elif manipulation_type == "sum_split":
            k = params.get("k", n // 2)
            sum_first = k * (k + 1) // 2
            sum_second = n * (n + 1) // 2 - sum_first
            result = sum_first + sum_second
            desc = f"Split sum: (1+...+{k}) + ({k+1}+...+{n}) = {sum_first} + {sum_second} = {result}"

        elif manipulation_type == "reverse_and_add":
            result = n * (n + 1) // 2
            desc = f"Reverse and add: S + S_rev = {n}*{n+1}, so S = {n}*{n+1}/2 = {result}"

        elif manipulation_type == "gauss_pairing":
            num_pairs = n // 2
            pair_sum = n + 1
            if n % 2 == 0:
                result = num_pairs * pair_sum
            else:
                result = num_pairs * pair_sum + (n + 1) // 2
            desc = f"Gauss pairing: {num_pairs} pairs of sum {pair_sum}" + (f" + middle {(n+1)//2}" if n % 2 == 1 else "") + f" = {result}"

        elif manipulation_type == "difference_of_sums":
            sum_squares = n * (n + 1) * (2 * n + 1) // 6
            sum_linear = n * (n + 1) // 2
            result = sum_squares - sum_linear
            desc = f"Difference of sums: sum(k^2) - sum(k) = {sum_squares} - {sum_linear} = {result}"

        else:
            total = 0
            for j in range(2, n + 1):
                for i in range(1, j):
                    total += i + j
            result = total
            desc = f"Double sum: sum over 1 <= i < j <= {n} of (i+j) = {result}"

        return MethodResult(
            value=result,
            description=desc,
            metadata={"manipulation_type": manipulation_type, **params}
        )

    def can_invert(self):
        return False
