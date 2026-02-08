"""
Ported advanced algebra, geometry, and miscellaneous methods.

Methods covering:
- Abstract algebra (compositum degree)
- Iterative processes (Collatz)
- Algebraic geometry (projective, hyperoval, arc, Grothendieck, Krull)
- Number theory (quadratic forms, Euler phi, divisibility)
- Trigonometry (arctan diophantine)
- Sequences (sum of cubes AP)
- Misc utilities (injective evaluation, time calculations)
"""

import math
import random
from typing import Any, Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


# ============================================================================
# ABSTRACT ALGEBRA
# ============================================================================

@register_technique
class CompositumDegree(MethodBlock):
    """Compute degree of compositum of two field extensions over Q."""

    def __init__(self):
        super().__init__()
        self.name = "compositum_degree"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["abstract_algebra", "field_theory", "compositum"]

    def generate_parameters(self, input_value=None):
        d1 = random.choice([2, 3, 4, 5, 6])
        d2 = random.choice([2, 3, 4, 5, 6])
        coprime = random.choice([True, False])
        if not coprime:
            divisors = [d for d in range(2, d1 + 1) if d1 % d == 0]
            if divisors:
                factor = random.choice(divisors)
                d2 = factor * random.randint(1, 3)
        return {"degree1": d1, "degree2": d2, "coprime": coprime}

    def compute(self, input_value, params):
        d1 = params.get("degree1", 2)
        d2 = params.get("degree2", 3)
        coprime = params.get("coprime", True)
        if isinstance(d1, str):
            d1 = int(d1)
        if isinstance(d2, str):
            d2 = int(d2)
        g = math.gcd(d1, d2)
        if g == 1:
            result = d1 * d2
            case = "coprime"
        elif coprime:
            result = d1 * d2
            case = "independent"
        else:
            result = (d1 * d2) // g
            case = "dependent"
        return MethodResult(
            value=result,
            description=f"[KL:Q] = {result} for [K:Q]={d1}, [L:Q]={d2} ({case})",
            params=params,
            metadata={"degree1": d1, "degree2": d2, "gcd": g, "case": case}
        )

    def can_invert(self):
        return False


# ============================================================================
# ITERATIVE PROCESSES
# ============================================================================

@register_technique
class CollatzProcess(MethodBlock):
    """Define Collatz process: n -> n/2 if even, 3n+1 if odd."""

    def __init__(self):
        super().__init__()
        self.name = "collatz_process"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["iterative_process", "collatz", "sequences"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1, 1000)
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        if n <= 0:
            return MethodResult(
                value=1, description=f"Collatz undefined for n <= 0, using 1",
                params=params, metadata={"n": n, "next": 1}
            )
        next_val = n // 2 if n % 2 == 0 else 3 * n + 1
        return MethodResult(
            value=next_val, description=f"Collatz: {n} -> {next_val}",
            params=params, metadata={"n": n, "next": next_val}
        )

    def can_invert(self):
        return False


# ============================================================================
# ALGEBRAIC GEOMETRY
# ============================================================================

@register_technique
class HyperovalSizeProjective(MethodBlock):
    """Compute hyperoval size in PG(2,q) for q = 2^n. Size = q + 2."""

    def __init__(self):
        super().__init__()
        self.name = "hyperoval_size_projective"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["finite_geometry", "projective_geometry", "hyperoval"]

    def generate_parameters(self, input_value=None):
        n = random.choice([1, 2, 3, 4, 5])
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 2)
        q = 2 ** n
        result = q + 2
        return MethodResult(
            value=result,
            description=f"Hyperoval size in PG(2,{q}): {q} + 2 = {result}",
            params=params, metadata={"techniques_used": [self.name], "q": q, "n": n}
        )

    def can_invert(self):
        return False


@register_technique
class ProjectiveLineCurve(MethodBlock):
    """Compute genus of projective line P^1. Always 0."""

    def __init__(self):
        super().__init__()
        self.name = "projective_line_curve"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebraic_geometry", "curves", "genus"]

    def generate_parameters(self, input_value=None):
        return {"genus": random.choice([0, 0, 0, 0, 1, 2, 3])}

    def compute(self, input_value, params):
        genus = input_value or params.get("genus", 0)
        result = 0
        if genus == 0:
            description = f"Genus of projective line P^1 = {result}"
        else:
            description = f"Input genus={genus}, but P^1 has genus {result}"
        return MethodResult(
            value=result, description=description, params=params,
            metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class MaxArcSizeAffineWithTranslation(MethodBlock):
    """Max arc size in AG(2,q) with translation constraint. q = 2^n, size = q."""

    def __init__(self):
        super().__init__()
        self.name = "max_arc_size_affine_with_translation"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["finite_geometry", "projective_geometry", "hyperoval", "arc"]

    def generate_parameters(self, input_value=None):
        n = random.choice([1, 2, 3, 4, 5])
        return {"n": n}

    def compute(self, input_value, params):
        n = params.get("n", 2)
        q = 2 ** n
        hyperoval_size = q + 2
        result = hyperoval_size - 2  # = q
        return MethodResult(
            value=result,
            description=f"Max arc in AG(2,{q}) with translation: {hyperoval_size}-2 = {result}",
            params=params,
            metadata={"techniques_used": [self.name], "q": q, "n": n,
                       "hyperoval_size": hyperoval_size}
        )

    def can_invert(self):
        return False


@register_technique
class MaxArcSizeAffine(MethodBlock):
    """Max arc size in AG(2,q). Size = q + 1."""

    def __init__(self):
        super().__init__()
        self.name = "max_arc_size_affine"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["finite_geometry", "projective_geometry", "arc"]

    def generate_parameters(self, input_value=None):
        q = random.choice([2, 3, 4, 5, 7, 8, 9, 11, 13, 16])
        return {"q": q}

    def compute(self, input_value, params):
        q = params.get("q", 4)
        result = q + 1
        return MethodResult(
            value=result,
            description=f"Max arc size in AG(2,{q}): {q} + 1 = {result}",
            params=params, metadata={"techniques_used": [self.name], "q": q}
        )

    def can_invert(self):
        return False


@register_technique
class GrothendieckVectorBundleSplitting(MethodBlock):
    """Number of line bundle summands = rank (Grothendieck's theorem on P^1)."""

    def __init__(self):
        super().__init__()
        self.name = "grothendieck_vector_bundle_splitting"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebraic_geometry", "vector_bundles", "grothendieck"]

    def generate_parameters(self, input_value=None):
        return {"rank": random.randint(1, 10)}

    def compute(self, input_value, params):
        rank = input_value or params.get("rank", 3)
        if rank <= 0:
            raise ValueError("rank must be positive")
        result = rank
        return MethodResult(
            value=result,
            description=f"Vector bundle rank {rank} on P^1 splits into {result} line bundles",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class SubfieldDegreeFromSubgroupIndex(MethodBlock):
    """Compute subfield degree = field_degree / subgroup_index (Galois theory)."""

    def __init__(self):
        super().__init__()
        self.name = "subfield_degree_from_subgroup_index"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebraic_geometry", "galois_theory", "field_theory"]

    def generate_parameters(self, input_value=None):
        field_degree = random.choice([6, 8, 12, 16, 20, 24])
        divisors = [d for d in range(1, field_degree + 1) if field_degree % d == 0]
        subgroup_index = random.choice(divisors)
        return {"field_degree": field_degree, "subgroup_index": subgroup_index}

    def compute(self, input_value, params):
        field_degree = params.get("field_degree", 12)
        subgroup_index = params.get("subgroup_index", 3)
        if subgroup_index == 0:
            raise ValueError("subgroup_index cannot be zero")
        if field_degree % subgroup_index != 0:
            raise ValueError(
                f"subgroup_index {subgroup_index} must divide field_degree {field_degree}")
        result = field_degree // subgroup_index
        return MethodResult(
            value=result,
            description=f"Subfield degree: {field_degree}/{subgroup_index} = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class KrullDimensionFromTranscendenceDegree(MethodBlock):
    """Krull dimension from transcendence degree (integral domain case)."""

    def __init__(self):
        super().__init__()
        self.name = "krull_dimension_from_transcendence_degree"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["algebraic_geometry", "commutative_algebra", "dimension_theory"]

    def generate_parameters(self, input_value=None):
        return {
            "trans_deg": random.randint(0, 10),
            "is_integral": random.choice([True, False])
        }

    def compute(self, input_value, params):
        trans_deg = params.get("trans_deg", 2)
        is_integral = params.get("is_integral", True)
        if trans_deg < 0:
            raise ValueError("transcendence degree must be non-negative")
        result = trans_deg if is_integral else trans_deg + 1
        ring_type = "integral domain" if is_integral else "non-integral ring"
        return MethodResult(
            value=result,
            description=f"Krull dim from trans_deg={trans_deg}, {ring_type} = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


@register_technique
class LargestNForInjectiveEvaluation(MethodBlock):
    """Find largest n where polynomial evaluation is injective."""

    def __init__(self):
        super().__init__()
        self.name = "largest_n_for_injective_evaluation"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["algebra", "polynomials", "injectivity"]

    def generate_parameters(self, input_value=None):
        return {"bound": random.randint(10, 1000)}

    def compute(self, input_value, params):
        bound = input_value or params.get("bound", 100)
        if bound <= 0:
            raise ValueError("Bound must be positive")
        result = bound - 1
        return MethodResult(
            value=result,
            description=f"Largest n for injective evaluation with bound {bound} = {result}",
            params=params, metadata={"techniques_used": [self.name]}
        )

    def can_invert(self):
        return False


# ============================================================================
# NUMBER THEORY
# ============================================================================

@register_technique
class QuadraticFormSolutionCount(MethodBlock):
    """Count integer solutions to quadratic forms ax^2 + bxy + cy^2 = n."""

    def __init__(self):
        super().__init__()
        self.name = "quadratic_form_solution_count"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["number_theory", "quadratic_forms", "sums_of_squares"]

    def generate_parameters(self, input_value=None):
        n = input_value if input_value else random.randint(1, 200)
        form_type = random.choice(["sum_two_squares", "general_diagonal", "with_cross_term"])
        if form_type == "sum_two_squares":
            a, b, c = 1, 0, 1
        elif form_type == "general_diagonal":
            a = random.choice([1, 2, 3])
            b = 0
            c = random.choice([1, 2, 3])
        else:
            a = random.choice([1, 2])
            b = random.choice([1, 2])
            c = random.choice([1, 2])
        return {"a": a, "b": b, "c": c, "n": n}

    def compute(self, input_value, params):
        a = params.get("a", 1)
        b = params.get("b", 0)
        c = params.get("c", 1)
        n = params.get("n", input_value) if input_value else params.get("n", 10)
        count = 0
        max_val = int((n / min(a, c))**0.5) + 1 if min(a, c) > 0 else int(n**0.5) + 1
        for x in range(-max_val, max_val + 1):
            for y in range(-max_val, max_val + 1):
                if a * x * x + b * x * y + c * y * y == n:
                    count += 1
        if a == 1 and b == 0 and c == 1:
            form_str = "x^2 + y^2 = "
        elif b == 0:
            form_str = f"{a}x^2 + {c}y^2 = "
        else:
            form_str = f"{a}x^2 + {b}xy + {c}y^2 = "
        return MethodResult(
            value=count,
            description=f"Solutions to {form_str}{n}: {count}",
            metadata={"a": a, "b": b, "c": c, "n": n, "count": count}
        )

    def can_invert(self):
        return True


@register_technique
class EulerPhiValues(MethodBlock):
    """Count n where phi(n) <= max_phi."""

    def __init__(self):
        super().__init__()
        self.name = "euler_phi_values"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["number_theory", "totient", "euler_phi"]

    def generate_parameters(self, input_value=None):
        max_phi = random.choice([6, 8, 10, 12])
        return {"max_phi": max_phi}

    def _totient(self, n):
        """Compute Euler's totient function."""
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result

    def compute(self, input_value, params):
        max_phi = params.get("max_phi", 10)
        upper_bound = max(100, 2 * max_phi * max_phi)
        valid_n = []
        for n in range(1, upper_bound + 1):
            phi_n = self._totient(n)
            if phi_n <= max_phi:
                valid_n.append(n)
        return MethodResult(
            value=len(valid_n),
            description=f"Count of n where phi(n) <= {max_phi}: {len(valid_n)}",
            metadata={"max_phi": max_phi, "count": len(valid_n)}
        )

    def can_invert(self):
        return False


@register_technique
class DivisibilityPowerPlusOne(MethodBlock):
    """Count n <= max_n such that n divides base^n + 1."""

    def __init__(self):
        super().__init__()
        self.name = "divisibility_power_plus_one"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["number_theory", "divisibility", "counting"]

    def generate_parameters(self, input_value=None):
        base = random.choice([2, 3, 5])
        max_n = random.choice([100, 500, 1000, 5000])
        return {"base": base, "max_n": max_n}

    def compute(self, input_value, params):
        base = params.get("base", 2)
        max_n = params.get("max_n", 1000)
        count = 0
        for n in range(1, max_n + 1):
            if (pow(base, n, n) + 1) % n == 0:
                count += 1
        return MethodResult(
            value=count,
            description=f"Count of n <= {max_n} where n | {base}^n + 1 = {count}",
            params=params,
            metadata={"count": count, "base": base, "max_n": max_n}
        )

    def can_invert(self):
        return False


# ============================================================================
# TRIGONOMETRY
# ============================================================================

@register_technique
class ArctanDiophantine(MethodBlock):
    """Count solutions to arctan(1/a) + arctan(1/b) = pi/4 and variants."""

    def __init__(self):
        super().__init__()
        self.name = "arctan_diophantine"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["trigonometry", "diophantine", "deep_insight", "arctan"]
        self.max_search = 20

    def generate_parameters(self, input_value=None):
        max_val = random.choice([10, 20, 50, 100])
        return {"max_val": max_val, "num_terms": 2}

    def compute(self, input_value, params):
        max_val = params.get("max_val", 20)
        # Count ordered pairs (a,b) with arctan(1/a) + arctan(1/b) = pi/4
        # Using identity: arctan(1/a) + arctan(1/b) = pi/4 iff (a-1)(b-1)=2
        count = 0
        for a in range(1, max_val + 1):
            for b in range(1, max_val + 1):
                if (a - 1) * (b - 1) == 2:
                    count += 1
        return MethodResult(
            value=count,
            description=f"Pairs (a,b) with arctan(1/a)+arctan(1/b)=pi/4, max={max_val}: {count}",
            params=params, metadata={"techniques_used": [self.name], "max_val": max_val}
        )

    def can_invert(self):
        return False


# ============================================================================
# SEQUENCES
# ============================================================================

@register_technique
class SumOfCubesAPExpansion(MethodBlock):
    """Compute sum of cubes of first n terms of an arithmetic progression."""

    def __init__(self):
        super().__init__()
        self.name = "sum_of_cubes_ap_expansion"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["sequences", "arithmetic_progression", "algebra", "cubes"]

    def generate_parameters(self, input_value=None):
        return {
            "a": random.randint(1, 5),
            "d": random.randint(1, 3),
            "n": random.randint(2, 10)
        }

    def compute(self, input_value, params):
        a = params.get("a", 1)
        d = params.get("d", 1)
        n = params.get("n", 5)
        terms = [a + i * d for i in range(n)]
        sum_of_cubes = sum(term ** 3 for term in terms)
        terms_str = " + ".join(f"{term}^3" for term in terms)
        return MethodResult(
            value=sum_of_cubes,
            description=f"Sum of cubes of AP({a}, {d}, {n}): {terms_str} = {sum_of_cubes}",
            params=params,
            metadata={"techniques_used": [self.name], "terms": terms}
        )

    def can_invert(self):
        return False


# ============================================================================
# MISC UTILITIES
# ============================================================================

@register_technique
class CurrentTimeFromPast(MethodBlock):
    """Calculate current time given past time and elapsed duration."""

    def __init__(self):
        super().__init__()
        self.name = "current_time_from_past"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["word_problems", "time", "arithmetic"]

    def generate_parameters(self, input_value=None):
        return {
            "past_hours": random.randint(0, 23),
            "past_minutes": random.randint(0, 59),
            "elapsed_hours": random.randint(0, 12),
            "elapsed_minutes": random.randint(0, 59)
        }

    def compute(self, input_value, params):
        past_h = params.get("past_hours", 10)
        past_m = params.get("past_minutes", 30)
        elapsed_h = params.get("elapsed_hours", 2)
        elapsed_m = params.get("elapsed_minutes", 15)
        past_total = past_h * 60 + past_m
        elapsed_total = elapsed_h * 60 + elapsed_m
        current_total = past_total + elapsed_total
        current_h = (current_total // 60) % 24
        current_m = current_total % 60
        result = current_h * 100 + current_m
        return MethodResult(
            value=result,
            description=f"From {past_h}:{past_m:02d} + {elapsed_h}h {elapsed_m}m = {current_h}:{current_m:02d}",
            params=params,
            metadata={"techniques_used": [self.name],
                       "current_hours": current_h, "current_minutes": current_m}
        )

    def can_invert(self):
        return False


@register_technique
class FutureTime(MethodBlock):
    """Calculate future time given current time and duration to add."""

    def __init__(self):
        super().__init__()
        self.name = "future_time"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 2
        self.tags = ["word_problems", "time", "arithmetic"]

    def generate_parameters(self, input_value=None):
        return {
            "current_hours": random.randint(0, 23),
            "current_minutes": random.randint(0, 59),
            "add_hours": random.randint(0, 12),
            "add_minutes": random.randint(0, 59)
        }

    def compute(self, input_value, params):
        curr_h = params.get("current_hours", 10)
        curr_m = params.get("current_minutes", 30)
        add_h = params.get("add_hours", 3)
        add_m = params.get("add_minutes", 45)
        current_total = curr_h * 60 + curr_m
        add_total = add_h * 60 + add_m
        future_total = current_total + add_total
        future_h = (future_total // 60) % 24
        future_m = future_total % 60
        result = future_h * 100 + future_m
        return MethodResult(
            value=result,
            description=f"From {curr_h}:{curr_m:02d} + {add_h}h {add_m}m = {future_h}:{future_m:02d}",
            params=params,
            metadata={"techniques_used": [self.name],
                       "future_hours": future_h, "future_minutes": future_m}
        )

    def can_invert(self):
        return False
