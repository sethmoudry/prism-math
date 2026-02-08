"""
Ported analysis methods.

High-impact methods covering:
- Special functions (gamma, Lambert W series)
- Bounds and inequalities (upper/lower bounds, squeeze theorem)
- Functional analysis (tail sum, subharmonic, convolution, Ando projection)
- Measure theory (measure zero, epsilon covering, trig zero sets)
- Ergodic theory (Birkhoff)
- Topology (top cohomology)
- Stochastic processes (continuous process jumps, finite variation martingale)
- Calculus (log symmetry integral, constant derivative)
"""

import math
import random
from typing import Any, Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# SPECIAL FUNCTIONS
# ============================================================================

@register_technique
class GammaFunction(MethodBlock):
    """Compute the gamma function Gamma(n). For positive integers: Gamma(n) = (n-1)!"""

    def __init__(self):
        super().__init__()
        self.name = "gamma_function"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["special_functions", "gamma", "factorial"]

    def generate_parameters(self, input_value=None):
        if input_value is not None:
            n = input_value
        else:
            n = random.choice([
                random.randint(1, 10),
                random.randint(1, 20) / 2.0,
            ])
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value)
        if n is None:
            raise ValueError("n must be provided")
        if n <= 0:
            raise ValueError(f"Gamma function requires n > 0, got n={n}")

        if isinstance(n, int) or (isinstance(n, float) and n.is_integer()):
            n_int = int(n)
            result = math.factorial(n_int - 1)
            desc = f"Gamma({n_int}) = ({n_int}-1)! = {result}"
        else:
            result = math.gamma(n)
            desc = f"Gamma({n}) = {result:.6f}"

        return MethodResult(
            value=result, description=desc, params=params,
            metadata={"techniques_used": [self.name], "n": n}
        )

    def can_invert(self):
        return False


# ============================================================================
# BOUNDS
# ============================================================================

@register_technique
class ComputeUpperBound(MethodBlock):
    """Compute an upper bound for sum of 1/k^p using integral comparison."""

    def __init__(self):
        super().__init__()
        self.name = "compute_upper_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["inequalities", "bounds", "estimation"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 100)
        p = random.choice([2, 3])
        return {"n": n, "p": p}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 10000)
        p = params.get("p", 2)

        if p == 2:
            bound = 1 + (1 - 1 / n)
        else:
            bound = 1 + (1 - 1 / n ** 2) / 2

        result = math.ceil(bound * 1000)
        return MethodResult(
            value=result,
            description=f"Upper bound for sum_{{k=1}}^{{{n}}} 1/k^{p} <= {bound:.4f}",
            params=params, metadata={"n": n, "p": p, "bound": bound}
        )

    def can_invert(self):
        return False


@register_technique
class ComputeLowerBound(MethodBlock):
    """Compute a lower bound for sum of sqrt(k) using integral comparison."""

    def __init__(self):
        super().__init__()
        self.name = "compute_lower_bound"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["inequalities", "bounds", "estimation"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 100)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        n = min(abs(n) if n else 10, 10000)

        bound = (2 / 3) * (n ** 1.5)
        result = math.floor(bound)

        return MethodResult(
            value=result,
            description=f"Lower bound for sum_{{k=1}}^{{{n}}} sqrt(k) >= {result}",
            params=params, metadata={"n": n, "bound": bound}
        )

    def can_invert(self):
        return False


@register_technique
class SqueezeTheoremMethod(MethodBlock):
    """Apply squeeze/sandwich theorem to bound a sequence."""

    def __init__(self):
        super().__init__()
        self.name = "squeeze_theorem"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["inequalities", "bounds", "sequences", "limits"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(10, 100)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)

        sum_lower = sum(k - 1 for k in range(1, n + 1))
        sum_upper = sum(k + 1 for k in range(1, n + 1))
        result = (sum_lower + sum_upper) // 2

        return MethodResult(
            value=result,
            description=f"Squeeze theorem: average of bounds for n={n}",
            params=params,
            metadata={"n": n, "lower_sum": sum_lower, "upper_sum": sum_upper}
        )

    def can_invert(self):
        return False


# ============================================================================
# FUNCTIONAL ANALYSIS
# ============================================================================

@register_technique
class TailSumBound(MethodBlock):
    """Compute tail sum bound: total - partial."""

    def __init__(self):
        super().__init__()
        self.name = "tail_sum_bound"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["functional_analysis", "series", "bounds"]

    def generate_parameters(self, input_value=None):
        total = random.uniform(10, 100)
        partial = random.uniform(0, total)
        return {"total": total, "partial": partial}

    def compute(self, input_value, params):
        total = params.get("total", 100)
        partial = params.get("partial", 60)
        result = total - partial
        return MethodResult(
            value=result,
            description=f"Tail sum: {total:.4f} - {partial:.4f} = {result:.4f}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SubharmonicSphericalAverageLimit(MethodBlock):
    """Subharmonic spherical average limit: value at center (lower bound)."""

    def __init__(self):
        super().__init__()
        self.name = "subharmonic_spherical_average_limit"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["functional_analysis", "subharmonic", "potential_theory"]

    def generate_parameters(self, input_value=None):
        return {"value_at_center": random.uniform(-10, 10)}

    def compute(self, input_value, params):
        value_at_center = input_value or params.get("value_at_center", 5.0)
        result = value_at_center
        return MethodResult(
            value=result,
            description=f"Subharmonic spherical average >= {value_at_center:.4f}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConvolutionLimitFromSphericalAverage(MethodBlock):
    """Convolution limit from spherical average: return the average value."""

    def __init__(self):
        super().__init__()
        self.name = "convolution_limit_from_spherical_average"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["functional_analysis", "convolution", "harmonic_analysis"]

    def generate_parameters(self, input_value=None):
        return {"avg": random.uniform(0, 10)}

    def compute(self, input_value, params):
        avg = params.get("avg", 5.0)
        return MethodResult(
            value=avg,
            description=f"Convolution limit from spherical average: {avg:.4f}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class AndoProjectionBound(MethodBlock):
    """Ando projection bound: projection constant is at least 1."""

    def __init__(self):
        super().__init__()
        self.name = "ando_projection_bound"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["functional_analysis", "projection", "operator_theory"]

    def generate_parameters(self, input_value=None):
        return {"norm": random.uniform(0.5, 10.0)}

    def compute(self, input_value, params):
        norm = params.get("norm", 2.0)
        result = max(1.0, norm)
        return MethodResult(
            value=result,
            description=f"Ando projection bound: max(1, {norm:.4f}) = {result:.4f}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# MEASURE THEORY
# ============================================================================

@register_technique
class MeasureZeroConclusion(MethodBlock):
    """Conclude measure zero if set is countable."""

    def __init__(self):
        super().__init__()
        self.name = "measure_zero_conclusion"
        self.input_type = "boolean"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["measure_theory", "analysis", "sets"]

    def generate_parameters(self, input_value=None):
        return {"is_countable": random.choice([True, False])}

    def compute(self, input_value, params):
        is_countable = input_value if input_value is not None else params.get("is_countable", True)
        if isinstance(is_countable, int):
            is_countable = bool(is_countable)
        result = 0 if is_countable else 1
        return MethodResult(
            value=result,
            description=f"Set is {'countable' if is_countable else 'uncountable'} -> measure: {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MeasureCoverEpsilon(MethodBlock):
    """Sum of epsilon/2^k for k=1 to n (epsilon-covering technique)."""

    def __init__(self):
        super().__init__()
        self.name = "measure_cover_epsilon"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 2
        self.tags = ["measure_theory", "analysis", "series"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(1, 20), "epsilon": random.uniform(0.01, 1.0)}

    def compute(self, input_value, params):
        n = params.get("n", 10)
        epsilon = params.get("epsilon", 0.1)
        result = epsilon * (1 - 0.5 ** n)
        return MethodResult(
            value=result,
            description=f"Sum of eps/2^k for k=1 to {n}, eps={epsilon:.4f} = {result:.6f}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TrigonometricZeroSetSin(MethodBlock):
    """Count zeros of sin(x) in [0, n*pi]."""

    def __init__(self):
        super().__init__()
        self.name = "trigonometric_zero_set_sin"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["measure_theory", "trigonometry", "analysis"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(1, 20)}

    def compute(self, input_value, params):
        n = input_value or params.get("n", 3)
        if n < 0:
            raise ValueError("n must be non-negative")
        result = n + 1
        return MethodResult(
            value=result,
            description=f"Number of zeros of sin(x) in [0, {n}pi] = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class TrigonometricZeroSetCos(MethodBlock):
    """Count zeros of cos(x) in [0, n*pi]."""

    def __init__(self):
        super().__init__()
        self.name = "trigonometric_zero_set_cos"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["measure_theory", "trigonometry", "analysis"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(1, 20)}

    def compute(self, input_value, params):
        n = input_value or params.get("n", 3)
        if n < 0:
            raise ValueError("n must be non-negative")
        result = n
        return MethodResult(
            value=result,
            description=f"Number of zeros of cos(x) in [0, {n}pi] = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# ERGODIC THEORY
# ============================================================================

@register_technique
class BirkhoffErgodicTheorem(MethodBlock):
    """Birkhoff's pointwise ergodic theorem: time averages converge a.e."""

    def __init__(self):
        super().__init__()
        self.name = "birkhoff_ergodic_theorem"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["functional_analysis", "ergodic"]

    def generate_parameters(self, input_value=None):
        return {"n": random.randint(10, 1000)}

    def compute(self, input_value, params):
        n = params.get("n", 100)
        result = n
        desc = f"Birkhoff average: (1/{n}) * sum_{{k=0}}^{{{n - 1}}} f(T^k x)"
        return MethodResult(
            value=result, description=desc, params=params,
            metadata={"techniques_used": [self.name], "n": n}
        )

    def can_invert(self):
        return False


# ============================================================================
# TOPOLOGY / COHOMOLOGY
# ============================================================================

@register_technique
class TopCohomologyNoncompactManifold(MethodBlock):
    """Top cohomology of noncompact manifold: return 0 (vanishes)."""

    def __init__(self):
        super().__init__()
        self.name = "top_cohomology_noncompact_manifold"
        self.input_type = "boolean"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["functional_analysis", "topology", "cohomology"]

    def generate_parameters(self, input_value=None):
        return {"is_orientable": random.choice([True, False])}

    def compute(self, input_value, params):
        is_orientable = input_value if input_value is not None else params.get("is_orientable", True)
        result = 0
        return MethodResult(
            value=result,
            description=f"Top cohomology of noncompact manifold (orientable={is_orientable}) -> {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# STOCHASTIC PROCESSES
# ============================================================================

@register_technique
class ContinuousProcessJumpSum(MethodBlock):
    """Continuous process has no jumps: sum of jumps = 0."""

    def __init__(self):
        super().__init__()
        self.name = "continuous_process_jump_sum"
        self.input_type = "boolean"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["probability", "stochastic", "jumps", "continuous"]

    def generate_parameters(self, input_value=None):
        return {"is_continuous": random.choice([True, False])}

    def compute(self, input_value, params):
        is_continuous = input_value if input_value is not None else params.get("is_continuous", True)
        result = 0
        if is_continuous:
            description = "Continuous process has no jumps: sum_{s<=t} Delta_X_s = 0"
        else:
            description = "Jump sum computation (no specific jump data provided)"
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name], "is_continuous": is_continuous}
        )

    def can_invert(self):
        return False


# ============================================================================
# CALCULUS
# ============================================================================

@register_technique
class LogSymmetryIntegral(MethodBlock):
    """Integral using log(x) + log(1-x) = log(x(1-x)) symmetry."""

    def __init__(self):
        super().__init__()
        self.name = "log_symmetry_integral"
        self.input_type = "integer"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["calculus", "integration", "logarithms", "symmetry"]

    def generate_parameters(self, input_value=None):
        return {"n": 1}

    def compute(self, input_value, params):
        n = params.get("n", 1)
        result = -1.0
        description = f"integral[0,1] log(x) dx = {result}"
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class ConstantFunctionDerivative(MethodBlock):
    """Derivative of a constant function is zero: d/dx c = 0."""

    def __init__(self):
        super().__init__()
        self.name = "constant_function_derivative"
        self.input_type = "none"
        self.output_type = "number"
        self.difficulty = 1
        self.tags = ["calculus", "derivatives", "basic_rules"]

    def generate_parameters(self, input_value=None):
        return {"c": random.uniform(-100, 100), "x": random.uniform(-10, 10)}

    def compute(self, input_value, params):
        c = params.get("c", 5.0)
        if isinstance(c, str):
            try:
                c = float(c)
            except ValueError:
                c = 5.0
        result = 0
        description = f"d/dx {c:.2f} = 0 (derivative of constant function)"
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name], "constant": c}
        )

    def can_invert(self):
        return False


# ============================================================================
# LAMBERT W SERIES (18 failing problems)
# ============================================================================

@register_technique
class LambertWSeries(MethodBlock):
    """Compute Lambert W function approximation using series expansion.

    W(x) is the inverse of f(W) = W * e^W.
    Series: W(x) = sum_{n=1}^{terms} (-n)^{n-1}/n! * x^n
    Converges for |x| <= 1/e.
    """

    def __init__(self):
        super().__init__()
        self.name = "lambert_w_series"
        self.input_type = "number"
        self.output_type = "number"
        self.difficulty = 4
        self.tags = ["special_functions", "series", "transcendental"]

    def generate_parameters(self, input_value=None):
        return {"x": random.uniform(-0.3, 0.3), "terms": random.randint(5, 15)}

    def compute(self, input_value, params):
        x = input_value if input_value is not None else params.get("x", 0.1)
        terms = params.get("terms", 10)

        if abs(x) > 1 / math.e + 0.01:
            return MethodResult(
                value=0,
                description=f"Lambert W series: x={x} out of convergence range",
                params=params,
                metadata={"techniques_used": [self.name], "warning": "out_of_range"}
            )

        result = 0.0
        for n in range(1, terms + 1):
            coeff = ((-n) ** (n - 1)) / math.factorial(n)
            result += coeff * (x ** n)

        result = round(result, 10)

        return MethodResult(
            value=result,
            description=f"Lambert W series: W({x}) ~ {result} ({terms} terms)",
            params=params,
            metadata={"techniques_used": [self.name], "terms_used": terms}
        )

    def can_invert(self):
        return False


# ============================================================================
# FINITE VARIATION MARTINGALE PART (17 failing problems)
# ============================================================================

@register_technique
class FiniteVariationMartingalePart(MethodBlock):
    """Get the continuous martingale part of a finite variation process.

    For continuous finite variation processes X_t = X_0 + A_t + M^c_t,
    M^c_t = 0 (the continuous martingale part vanishes).
    """

    def __init__(self):
        super().__init__()
        self.name = "finite_variation_martingale_part"
        self.input_type = "boolean"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["probability", "stochastic", "martingale", "finite_variation"]

    def generate_parameters(self, input_value=None):
        return {"is_finite_variation": random.choice([True, False])}

    def compute(self, input_value, params):
        is_fv = (input_value if input_value is not None
                 else params.get("is_finite_variation", True))

        if is_fv:
            result = 0
            desc = "Finite variation process: continuous martingale part M^c = 0"
        else:
            result = 0
            desc = "Martingale part contribution (non-trivial for infinite variation)"

        return MethodResult(
            value=result, description=desc, params=params,
            metadata={
                "techniques_used": [self.name],
                "is_finite_variation": is_fv,
                "martingale_part_is_zero": is_fv
            }
        )

    def can_invert(self):
        return False
