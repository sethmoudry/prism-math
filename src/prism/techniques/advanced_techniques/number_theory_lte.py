"""
Deep Insight Technique: Lifting the Exponent Lemma
"""

import random
from typing import Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepLiftingExponent(MethodBlock):
    """
    Surface: v_p(a^n - b^n) = p-adic valuation of a^n - b^n
    Hidden: LTE lemma: v_p(a^n - b^n) = v_p(a - b) + v_p(n) when p | (a-b), p does not divide a, p odd
    """

    def __init__(self):
        super().__init__()
        self.name = "lifting_exponent"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["number_theory", "lifting_exponent", "valuation", "deep_insight"]

    def _p_adic_valuation(self, n, p):
        if n == 0:
            return float('inf')
        val = 0
        while n % p == 0:
            val += 1
            n //= p
        return val

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["compute_valuation", "largest_power", "find_n_for_valuation"])

        if question_type == "compute_valuation":
            p = random.choice([3, 5, 7])
            a = random.randint(2, 20)
            b = random.randint(1, a - 1)
            while (a - b) % p != 0:
                a = random.randint(2, 20)
                b = random.randint(1, a - 1)
            n = random.randint(2, 10)
            return {"question_type": question_type, "p": p, "a": a, "b": b, "n": n}
        elif question_type == "largest_power":
            p = random.choice([2, 3, 5])
            a = random.randint(3, 15)
            b = random.randint(1, a - 1)
            while (a - b) % p != 0:
                a = random.randint(3, 15)
                b = random.randint(1, a - 1)
            n = random.randint(3, 8)
            return {"question_type": question_type, "p": p, "a": a, "b": b, "n": n}
        else:
            p = random.choice([3, 5, 7])
            k = random.randint(1, 3)
            return {"question_type": question_type, "p": p, "k": k}

    def validate_params(self, params, prev_value=None):
        p = params.get("p")
        if p is None:
            return False
        try:
            return int(p) >= 2
        except (ValueError, TypeError):
            return False

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "compute_valuation":
            p, a, b, n = params.get("p", 5), params.get("a", 10), params.get("b", 10), params.get("n", 10)
            val_diff = self._p_adic_valuation(a - b, p)
            val_n = self._p_adic_valuation(n, p)
            answer = val_diff + val_n
            description = f"v_{p}({a}^{n} - {b}^{n}) = v_{p}({a-b}) + v_{p}({n}) = {val_diff} + {val_n} = {answer}"
        elif question_type == "largest_power":
            p, a, b, n = params.get("p", 5), params.get("a", 10), params.get("b", 10), params.get("n", 10)
            val_diff = self._p_adic_valuation(a - b, p)
            val_n = self._p_adic_valuation(n, p)
            answer = val_diff + val_n
            description = f"Largest k where {p}^k | ({a}^{n} - {b}^{n}): k = {answer}"
        else:
            p, k = params.get("p", 5), params.get("k", 5)
            mod = p ** k
            order = 1
            power = 2
            while power % mod != 1:
                power = (power * 2) % mod
                order += 1
                if order > mod:
                    break
            answer = order
            description = f"Smallest n where v_{p}(2^n - 1) >= {k}: ord_{{{p}^{k}}}(2) = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "LTE_lemma"}
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

        if question_type == "compute_valuation":
            p, a, b, n = params.get("p", 5), params.get("a", 10), params.get("b", 10), params.get("n", 10)
            relationship = (
                f"For a prime $p$ and integer $n$, let $v_p(n)$ denote the largest integer $k$ "
                f"such that $p^k$ divides $n$.\n\n"
                f"Find $v_{{{p}}}({a}^{{{n}}} - {b}^{{{n}}})$."
            )
        elif question_type == "largest_power":
            p, a, b, n = params.get("p", 5), params.get("a", 10), params.get("b", 10), params.get("n", 10)
            relationship = (
                f"For a prime $p$ and integer $n$, let $v_p(n)$ denote the largest integer $k$ "
                f"such that $p^k$ divides $n$.\n\n"
                f"What is the largest value of $k$ such that ${p}^k$ divides ${a}^{{{n}}} - {b}^{{{n}}}$?"
            )
        else:
            p, k = params.get("p", 5), params.get("k", 5)
            relationship = (
                f"For a prime $p$ and positive integer $n$, let $v_p(n)$ denote the $p$-adic valuation "
                f"of $n$ (the largest power of $p$ dividing $n$).\n\n"
                f"Find the smallest positive integer $n$ such that $v_{{{p}}}(2^n - 1) \\geq {k}$."
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
