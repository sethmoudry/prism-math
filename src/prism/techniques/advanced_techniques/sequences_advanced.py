"""
Deep Insight Techniques: Sequences (Advanced)

Techniques for sequence-related problems requiring deep insights:
- Pell equations (x^2 - Dy^2 = 1)
- Collatz-like iterations
"""

import random
import math
from typing import Dict, Any, Optional
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
class DeepPellEquation(MethodBlock):
    """
    Surface: x^2 - Dy^2 = 1 for non-square D
    Hidden: Solutions form a group, fundamental solution generates all
    """

    def __init__(self):
        super().__init__()
        self.name = "pell_equation"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["pell", "diophantine", "deep_insight", "sequences"]

    def _is_perfect_square(self, n):
        sqrt_n = int(n**0.5)
        return sqrt_n * sqrt_n == n

    def _find_fundamental_solution(self, D):
        for x in range(1, 100000):
            for y in range(1, 10000):
                if x*x - D*y*y == 1:
                    return (x, y)
                if x*x - D*y*y < 1:
                    break
        return (0, 0)

    def _nth_solution(self, D, n):
        x1, y1 = self._find_fundamental_solution(D)
        if x1 == 0:
            return (0, 0)
        if n == 1:
            return (x1, y1)
        xn, yn = x1, y1
        for _ in range(n - 1):
            xn_new = x1 * xn + D * y1 * yn
            yn_new = x1 * yn + y1 * xn
            xn, yn = xn_new, yn_new
        return (xn, yn)

    def validate_params(self, params, prev_value=None):
        D = params.get("D")
        if D is None or D <= 0:
            return False
        sqrt_D = int(D ** 0.5)
        return sqrt_D * sqrt_D != D

    def generate_parameters(self, input_value=None):
        non_squares = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 19]
        question_type = random.choice(["fundamental", "nth_solution", "nth_mod_m"])

        if question_type == "fundamental":
            D = random.choice(non_squares)
            return {"question_type": question_type, "D": D}
        elif question_type == "nth_solution":
            D = random.choice([2, 3, 5, 7])
            n = random.randint(2, 5)
            return {"question_type": question_type, "D": D, "n": n}
        else:
            D = random.choice([2, 3, 5, 7])
            n = random.randint(3, 10)
            m = random.choice([100, 1000, 10000])
            return {"question_type": question_type, "D": D, "n": n, "m": m}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "fundamental":
            D = params.get("D", 10)
            x1, y1 = self._find_fundamental_solution(D)
            answer = x1
            description = f"Fundamental solution to x^2 - {D}y^2 = 1: ({x1}, {y1}), x = {answer}"
        elif question_type == "nth_solution":
            D, n = params.get("D", 10), params.get("n", 10)
            xn, yn = self._nth_solution(D, n)
            answer = xn
            description = f"{n}th solution to x^2 - {D}y^2 = 1: ({xn}, {yn}), x = {answer}"
        else:
            D, n, m = params.get("D", 10), params.get("n", 10), params.get("m", 10)
            xn, yn = self._nth_solution(D, n)
            answer = xn % m
            description = f"x_{n} mod {m} for Pell equation x^2 - {D}y^2 = 1: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "Pell equation solution structure"}
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

        if question_type == "fundamental":
            D = params.get("D", 10)
            relationship = (
                f"Consider the Pell equation $x^2 - {D}y^2 = 1$ in positive integers.\n\n"
                f"Find the $x$-coordinate of the fundamental (smallest positive) solution."
            )
        elif question_type == "nth_solution":
            D, n = params.get("D", 10), params.get("n", 10)
            relationship = (
                f"Consider the Pell equation $x^2 - {D}y^2 = 1$ in positive integers. "
                f"Let $(x_1, y_1), (x_2, y_2), \\ldots$ be the solutions in increasing order of $x$-coordinates.\n\n"
                f"Find $x_{{{n}}}$."
            )
        else:
            D, n, m = params.get("D", 10), params.get("n", 10), params.get("m", 10)
            relationship = (
                f"Consider the Pell equation $x^2 - {D}y^2 = 1$ in positive integers. "
                f"Let $(x_1, y_1), (x_2, y_2), \\ldots$ be the solutions in increasing order of $x$-coordinates.\n\n"
                f"Compute $x_{{{n}}} \\bmod {m}$."
            )

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
class DeepCollatzLike(MethodBlock):
    """
    Surface: Iterative function f(n) = n/2 if even, 3n+1 if odd (Collatz-like)
    Hidden: Stopping time patterns, density arguments
    """

    def __init__(self):
        super().__init__()
        self.name = "collatz_like"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["collatz", "iteration", "deep_insight", "sequences"]

    def _stopping_time(self, n, variant="standard"):
        steps = 0
        while n != 1 and steps < 10000:
            if variant == "standard":
                n = n // 2 if n % 2 == 0 else 3 * n + 1
            elif variant == "compressed":
                n = n // 2 if n % 2 == 0 else (3 * n + 1) // 2
            steps += 1
        return steps if n == 1 else -1

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["count_stopping_time", "exact_stopping_time", "max_with_stopping_time"])
        variant = random.choice(["standard", "compressed"])

        if question_type == "count_stopping_time":
            N = random.randint(50, 200)
            k = random.randint(5, 15)
            return {"question_type": question_type, "variant": variant, "N": N, "k": k}
        elif question_type == "exact_stopping_time":
            k = random.randint(3, 10)
            return {"question_type": question_type, "variant": variant, "k": k}
        else:
            N = random.randint(100, 500)
            k = random.randint(3, 10)
            return {"question_type": question_type, "variant": variant, "N": N, "k": k}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")
        variant = params.get("variant", 10)

        if question_type == "count_stopping_time":
            N, k = params.get("N", 50), params.get("k", 5)
            count = 0
            for n in range(1, N + 1):
                if self._stopping_time(n, variant) <= k:
                    count += 1
            answer = count
            description = f"Values in [1,{N}] with stopping time <= {k}: {count}"
        elif question_type == "exact_stopping_time":
            k = params.get("k", 5)
            for n in range(1, 10000):
                if self._stopping_time(n, variant) == k:
                    answer = n
                    break
            else:
                answer = 0
            description = f"Smallest value with stopping time exactly {k}: {answer}"
        else:
            N, k = params.get("N", 50), params.get("k", 5)
            answer = 0
            for n in range(1, N + 1):
                if self._stopping_time(n, variant) == k:
                    answer = n
            description = f"Largest value <= {N} with stopping time {k}: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "variant": variant}
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
        variant = params.get("variant", 10)

        if variant == "standard":
            rule_text = "$f(n) = n/2$ if $n$ is even, and $f(n) = 3n+1$ if $n$ is odd"
        else:
            rule_text = "$f(n) = n/2$ if $n$ is even, and $f(n) = (3n+1)/2$ if $n$ is odd"

        if question_type == "count_stopping_time":
            N, k = params.get("N", 50), params.get("k", 5)
            relationship = (
                f"Consider the function {rule_text}. "
                f"The stopping time of a positive integer $n$ is the number of times $f$ must be applied to reach $1$.\n\n"
                f"How many integers in $\\{{1, 2, \\ldots, {N}\\}}$ have stopping time at most ${k}$?"
            )
        elif question_type == "exact_stopping_time":
            k = params.get("k", 5)
            relationship = (
                f"Consider the function {rule_text}. "
                f"The stopping time of a positive integer $n$ is the number of times $f$ must be applied to reach $1$.\n\n"
                f"Find the smallest positive integer with stopping time exactly ${k}$."
            )
        else:
            N, k = params.get("N", 50), params.get("k", 5)
            relationship = (
                f"Consider the function {rule_text}. "
                f"The stopping time of a positive integer $n$ is the number of times $f$ must be applied to reach $1$.\n\n"
                f"Find the largest integer in $\\{{1, 2, \\ldots, {N}\\}}$ with stopping time exactly ${k}$. "
                f"If no such integer exists, output $0$."
            )

        return create_problem_dict(
            relationship=relationship,
            answer=result.value,
            techniques=[self.name],
            uuid=generate_uuid(),
            metadata=result.metadata
        )

    def can_invert(self):
        return False
