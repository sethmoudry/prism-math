"""
Deep Insight Techniques: Functional Equations (Cauchy & Related)

Techniques for problems involving functional equations and transformations:
- Cauchy Additive-Multiplicative
- Jensen Functional
- Multiplicative to Additive
"""

import random
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass

from sympy import factorial

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


def generate_uuid() -> str:
    """Generate a unique identifier."""
    import uuid as uuid_lib
    return str(uuid_lib.uuid4())


def create_problem_dict(relationship: str, answer: int, techniques: list,
                       uuid: str, metadata: dict) -> dict:
    """Create a problem dictionary."""
    return {
        "problem": relationship,
        "answer": answer,
        "techniques": techniques,
        "uuid": uuid,
        "metadata": metadata
    }


@register_technique
class DeepCauchyAdditiveMultiplicative(MethodBlock):
    """
    Surface: f(m) + f(n) = f(m + n + mn) for positive integers m, n
    Hidden: Let g(n) = f(n-1). Then g(ab) = g(a) + g(b) - so g is additive over multiplication.
    """

    def __init__(self):
        super().__init__()
        self.name = "cauchy_additive_multiplicative"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["functional_equation", "deep_insight", "counting"]

    def generate_parameters(self, input_value=None):
        from sympy import isprime
        question_type = random.choice(["count_functions", "extremal", "existence"])

        if question_type == "count_functions":
            bound = random.choice([10, 20, 50, 100])
            limit = random.choice([10, 20, 50, 100])
            return {"question_type": question_type, "bound": bound, "limit": limit}
        elif question_type == "extremal":
            k = random.randint(5, 20)
            return {"question_type": question_type, "k": k}
        else:
            target = random.choice([100, 1000, 2024, 2025])
            value = random.randint(1, 50)
            return {"question_type": question_type, "target": target, "value": value}

    def compute(self, input_value, params):
        from sympy import isprime
        question_type = params.get("question_type", "compute")

        if question_type == "count_functions":
            bound, limit = params.get("bound", 10), params.get("limit", 100)
            primes_needed = [p for p in range(2, limit + 2) if isprime(p)]
            count = 1
            for p in primes_needed:
                max_power = 0
                temp = 1
                while temp <= limit + 1:
                    max_power += 1
                    temp *= p
                max_power -= 1
                if max_power > 0:
                    max_gp = bound // max_power
                else:
                    max_gp = bound
                count *= (max_gp + 1)
            answer = count
            description = f"Count of functions f with f(n) <= {bound} for n <= {limit}: {count}"

        elif question_type == "extremal":
            k = params.get("k", 5)
            answer = 2**(k+1) - 1
            description = f"Smallest n with f(n) > {k} for all valid f: {answer}"

        else:
            target, value = params.get("target", 10), params.get("value", 10)
            n = target + 1
            factors = []
            temp = n
            for p in range(2, int(n**0.5) + 1):
                while temp % p == 0:
                    factors.append(p)
                    temp //= p
            if temp > 1:
                factors.append(temp)
            omega = len(factors)
            exists = (value >= omega) or (value == 0)
            answer = 1 if exists else 0
            description = f"Existence of f with f({target}) = {value}: {'Yes' if exists else 'No'}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "g(k)=f(k-1) is completely additive"}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                params = self.generate_parameters()
                result = self.compute(None, params)
                if result.value == wrapped_target:
                    return params
            except Exception:
                continue
        return None

    def generate(self, target_answer: Optional[int] = None):
        try:
            if target_answer is not None:
                params = self._find_params_for_answer(target_answer)
                if params is None:
                    return None
            else:
                params = self.generate_parameters()
        except Exception:
            return None
        result = self.compute(None, params)
        question_type = params.get("question_type", "compute")

        base_statement = (
            "Let $f: \\mathbb{Z}_{\\geq 1} \\to \\mathbb{Z}_{\\geq 1}$ satisfy "
            "$f(m) + f(n) = f(m + n + mn)$ for all positive integers $m, n$."
        )

        if question_type == "count_functions":
            bound, limit = params.get("bound", 10), params.get("limit", 100)
            question = f"How many such functions $f$ satisfy $f(n) \\leq {bound}$ for all $n \\leq {limit}$?"
        elif question_type == "extremal":
            k = params.get("k", 5)
            question = f"Find the smallest positive integer $n$ such that $f(n) > {k}$ for every such function $f$."
        else:
            target, value = params.get("target", 10), params.get("value", 10)
            question = f"Does there exist such a function with $f({target}) = {value}$? Output 1 for yes, 0 for no."

        relationship = f"{base_statement}\n\n{question}"

        return create_problem_dict(
            relationship=relationship,
            answer=result.value,
            techniques=[self.name],
            uuid=generate_uuid(),
            metadata=result.metadata
        )

    def can_invert(self):
        return False


@register_technique
class DeepJensenFunctional(MethodBlock):
    """
    Surface: f((x+y)/2) = (f(x)+f(y))/2 for all reals (Jensen's equation - midpoint convexity)
    Hidden: For continuous f, all solutions are affine: f(x) = ax + b
    """

    def __init__(self):
        super().__init__()
        self.name = "jensen_functional"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["functional_equation", "deep_insight", "convexity"]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["count_functions", "evaluate", "general_form"])

        if question_type == "count_functions":
            a = random.randint(0, 10)
            b = random.randint(0, 10)
            return {"question_type": question_type, "f0": a, "f1": b}
        elif question_type == "evaluate":
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
            n = random.choice([10, 50, 100, 1000])
            return {"question_type": question_type, "f0": a, "f1": b, "n": n}
        else:
            f0 = random.randint(-10, 10)
            f1 = random.randint(-10, 10)
            return {"question_type": question_type, "f0": f0, "f1": f1}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "count_functions":
            answer = 1
            description = f"Continuous functions with f(0)={params['f0']}, f(1)={params['f1']}: {answer}"
        elif question_type == "evaluate":
            a = params.get("f1", 5) - params.get("f0", 1)
            b = params.get("f0", 1)
            n = params.get("n", 10)
            fn = a * n + b
            answer = abs(fn)
            description = f"f({n}) = {a}*{n} + {b} = {fn}, |f({n})| = {answer}"
        else:
            a = params.get("f1", 5) - params.get("f0", 1)
            answer = abs(a)
            description = f"Coefficient of n: f(1) - f(0) = {params['f1']} - {params['f0']} = {a}, |a| = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "continuous Jensen solutions are affine"}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                params = self.generate_parameters()
                result = self.compute(None, params)
                if result.value == wrapped_target:
                    return params
            except Exception:
                continue
        return None

    def generate(self, target_answer: Optional[int] = None):
        try:
            if target_answer is not None:
                params = self._find_params_for_answer(target_answer)
                if params is None:
                    return None
            else:
                params = self.generate_parameters()
        except Exception:
            return None
        result = self.compute(None, params)
        question_type = params.get("question_type", "compute")

        base_statement = (
            "Let $f: \\mathbb{R} \\to \\mathbb{R}$ be a continuous function satisfying "
            "$f\\left(\\frac{x+y}{2}\\right) = \\frac{f(x)+f(y)}{2}$ for all real numbers $x, y$."
        )

        if question_type == "count_functions":
            f0, f1 = params.get("f0", 1), params.get("f1", 5)
            question = f"How many such functions satisfy $f(0) = {f0}$ and $f(1) = {f1}$?"
        elif question_type == "evaluate":
            f0, f1, n = params.get("f0", 1), params.get("f1", 5), params.get("n", 10)
            question = f"If $f(0) = {f0}$ and $f(1) = {f1}$, what is the absolute value of $f({n})$?"
        else:
            f0, f1 = params.get("f0", 1), params.get("f1", 5)
            question = f"If $f(0) = {f0}$ and $f(1) = {f1}$, what is the absolute value of the coefficient of $n$ when $f(n)$ is expressed in the form $f(n) = An + B$?"

        relationship = f"{base_statement}\n\n{question}"

        return create_problem_dict(
            relationship=relationship,
            answer=result.value,
            techniques=[self.name],
            uuid=generate_uuid(),
            metadata=result.metadata
        )

    def can_invert(self):
        return False


@register_technique
class DeepMultiplicativeToAdditive(MethodBlock):
    """
    Surface: f(xy) = f(x) + f(y) for positive reals
    Hidden: f(x) = c * log(x) for some constant c (Cauchy's logarithmic equation)
    """

    def __init__(self):
        super().__init__()
        self.name = "multiplicative_to_additive"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["functional_equation", "deep_insight", "logarithm"]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["evaluate_power", "count_constraints", "multi_prime"])

        if question_type == "evaluate_power":
            base = random.choice([2, 3, 5])
            f_base = random.randint(1, 10)
            power = random.randint(4, 10)
            return {"question_type": question_type, "base": base, "f_base": f_base, "power": power}
        elif question_type == "count_constraints":
            B = random.choice([5, 10, 20])
            return {"question_type": question_type, "B": B}
        else:
            f2 = random.randint(1, 5)
            f3 = random.randint(1, 5)
            i = random.randint(2, 4)
            j = random.randint(2, 4)
            return {"question_type": question_type, "f2": f2, "f3": f3, "i": i, "j": j}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "evaluate_power":
            base = params.get("base", 2)
            f_base = params.get("f_base", 2)
            power = params.get("power", 2)
            answer = power * f_base
            description = f"f({base}^{power}) = {power} * f({base}) = {power} * {f_base} = {answer}"
        elif question_type == "count_constraints":
            B = params.get("B", 10)
            answer = (B + 1) ** 2
            description = f"f(2), f(3) each in [0,{B}]: ({B}+1)^2 = {answer} combinations"
        else:
            f2, f3 = params.get("f2", 1), params.get("f3", 5)
            i, j = params.get("i", 2), params.get("j", 3)
            answer = i * f2 + j * f3
            description = f"f(2^{i} * 3^{j}) = {i}*{f2} + {j}*{f3} = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "f(x) = c*log(x)"}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target
        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                params = self.generate_parameters()
                result = self.compute(None, params)
                if result.value == wrapped_target:
                    return params
            except Exception:
                continue
        return None

    def generate(self, target_answer: Optional[int] = None):
        try:
            if target_answer is not None:
                params = self._find_params_for_answer(target_answer)
                if params is None:
                    return None
            else:
                params = self.generate_parameters()
        except Exception:
            return None
        result = self.compute(None, params)
        question_type = params.get("question_type", "compute")

        base_statement = (
            "Let $f: \\mathbb{R}^+ \\to \\mathbb{R}$ satisfy "
            "$f(xy) = f(x) + f(y)$ for all positive real numbers $x, y$."
        )

        if question_type == "evaluate_power":
            base, f_base, power = params.get("base", 2), params.get("f_base", 2), params.get("power", 2)
            question = f"If $f({base}) = {f_base}$, what is $f({base}^{{{power}}})$?"
        elif question_type == "count_constraints":
            B = params.get("B", 10)
            question = f"Assuming $f$ takes only integer values, how many such functions satisfy $f(2) \\in [0, {B}]$ and $f(3) \\in [0, {B}]$?"
        else:
            f2, f3, i, j = params.get("f2", 1), params.get("f3", 5), params.get("i", 2), params.get("j", 3)
            question = f"If $f(2) = {f2}$ and $f(3) = {f3}$, what is $f(2^{{{i}}} \\cdot 3^{{{j}}})$?"

        relationship = f"{base_statement}\n\n{question}"

        return create_problem_dict(
            relationship=relationship,
            answer=result.value,
            techniques=[self.name],
            uuid=generate_uuid(),
            metadata=result.metadata
        )

    def can_invert(self):
        return False
