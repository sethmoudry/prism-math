"""
Deep Insight Technique: GCD Euclidean (Fibonacci worst case)
"""

import random
from typing import Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepGCDEuclidean(MethodBlock):
    """
    Surface: gcd(a, b) computed via Euclidean algorithm
    Hidden: Consecutive Fibonacci numbers F_n, F_{n+1} require exactly n-2 steps.
    """

    def __init__(self):
        super().__init__()
        self.name = "gcd_euclidean"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "gcd", "fibonacci", "deep_insight"]

    def _euclidean_steps(self, a, b):
        steps = 0
        while b > 0:
            a, b = b, a % b
            steps += 1
        return steps

    def _fibonacci(self, n):
        if n <= 0:
            return 0
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["smallest_pair", "max_steps_sum", "count_pairs"])

        if question_type == "smallest_pair":
            k = random.randint(3, 10)
            return {"question_type": question_type, "k": k}
        elif question_type == "max_steps_sum":
            S = random.choice([50, 100, 200, 500, 1000])
            return {"question_type": question_type, "S": S}
        else:
            k = random.randint(3, 6)
            N = random.choice([20, 50, 100])
            return {"question_type": question_type, "k": k, "N": N}

    def validate_params(self, params, prev_value=None):
        question_type = params.get("question_type", "compute")
        if question_type in ["smallest_pair", "count_pairs"]:
            k = params.get("k")
            if k is None:
                return False
            try:
                return int(k) >= 1
            except (ValueError, TypeError):
                return False
        return True

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "smallest_pair":
            k = params.get("k", 5)
            a = self._fibonacci(k + 1)
            b = self._fibonacci(k + 2)
            answer = a + b
            description = f"Smallest pair requiring {k} steps: ({a}, {b}), sum = {answer}"
        elif question_type == "max_steps_sum":
            S = params.get("S", 50)
            k = 2
            while self._fibonacci(k + 2) <= S:
                k += 1
            k -= 1
            answer = k - 1 if k >= 2 else 1
            description = f"Max steps for pairs summing to <= {S}: {answer}"
        else:
            k = params.get("k", 5)
            N = params.get("N", 50)
            count = 0
            for a in range(1, N + 1):
                for b in range(a + 1, N + 1):
                    if self._euclidean_steps(b, a) == k:
                        count += 1
            answer = count
            description = f"Pairs (a,b) with a<b<={N} requiring {k} steps: {count}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "Fibonacci pairs are worst case"}
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

        if question_type == "smallest_pair":
            k = params.get("k", 5)
            relationship = (
                f"The Euclidean algorithm computes $\\gcd(a, b)$ by repeated division. "
                f"Find the sum $a + b$ of the smallest pair of positive integers $(a, b)$ with $a < b$ "
                f"such that the algorithm requires exactly ${k}$ division steps."
            )
        elif question_type == "max_steps_sum":
            S = params.get("S", 50)
            relationship = (
                f"The Euclidean algorithm computes $\\gcd(a, b)$ by repeated division. "
                f"What is the maximum number of division steps required for any pair $(a, b)$ "
                f"of positive integers satisfying $a + b \\leq {S}$?"
            )
        else:
            k, N = params.get("k", 5), params.get("N", 50)
            relationship = (
                f"The Euclidean algorithm computes $\\gcd(a, b)$ by repeated division. "
                f"How many pairs $(a, b)$ of positive integers with $1 \\leq a < b \\leq {N}$ "
                f"require exactly ${k}$ division steps?"
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
