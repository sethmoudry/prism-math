"""
Deep Insight Technique: Fermat's Little Theorem
"""

import random
from typing import Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepFermatLittle(MethodBlock):
    """
    Surface: a^p = a (mod p) for prime p (Fermat's Little Theorem)
    Hidden: a^(p-1) = 1 (mod p) when gcd(a,p)=1, exponents reduce mod p-1
    """

    def __init__(self):
        super().__init__()
        self.name = "fermat_little"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "fermat", "modular_exponentiation", "deep_insight"]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["huge_exponent", "tower_exponent", "count_valid_a"])

        if question_type == "huge_exponent":
            p = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
            a = random.randint(2, p - 1)
            exp = random.randint(100, 10000)
            return {"question_type": question_type, "a": a, "exp": exp, "p": p}
        elif question_type == "tower_exponent":
            p = random.choice([7, 11, 13, 17, 19, 23])
            a = random.randint(2, p - 1)
            b = random.randint(2, 10)
            c = random.randint(2, 10)
            return {"question_type": question_type, "a": a, "b": b, "c": c, "p": p}
        else:
            n = random.choice([6, 10, 12, 15, 20, 21])
            k = random.randint(2, 5)
            return {"question_type": question_type, "n": n, "k": k}

    def validate_params(self, params, prev_value=None):
        p = params.get("p")
        if p is not None:
            try:
                p_val = int(p)
                return p_val >= 2
            except (ValueError, TypeError):
                return False
        n = params.get("n")
        if n is not None:
            try:
                return int(n) >= 2
            except (ValueError, TypeError):
                return False
        return True

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "huge_exponent":
            a, exp, p = params.get("a", 10), params.get("exp", 100), params.get("p", 5)
            reduced_exp = exp % (p - 1)
            answer = pow(a, reduced_exp, p)
            description = f"{a}^{exp} = {a}^{reduced_exp} = {answer} (mod {p})"
        elif question_type == "tower_exponent":
            a, b, c, p = params.get("a", 10), params.get("b", 10), params.get("c", 10), params.get("p", 5)
            reduced_exp = pow(b, c, p - 1)
            answer = pow(a, reduced_exp, p)
            description = f"{a}^({b}^{c}) = {a}^{reduced_exp} = {answer} (mod {p})"
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            count = 0
            for a in range(1, n):
                if pow(a, k, n) == a % n:
                    count += 1
            answer = count
            description = f"Count of a in [1,{n-1}] where a^{k} = a (mod {n}): {count}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "fermat_little_theorem"}
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

        if question_type == "huge_exponent":
            a, exp, p = params.get("a", 10), params.get("exp", 100), params.get("p", 5)
            relationship = (
                f"By Fermat's Little Theorem, for prime $p$ and integer $a$ with $\\gcd(a, p) = 1$, "
                f"we have $a^{{p-1}} \\equiv 1 \\pmod{{p}}$.\n\n"
                f"Find ${a}^{{{exp}}} \\bmod {p}$."
            )
        elif question_type == "tower_exponent":
            a, b, c, p = params.get("a", 10), params.get("b", 10), params.get("c", 10), params.get("p", 5)
            relationship = (
                f"For a prime $p$ and integer $a$ with $\\gcd(a, p) = 1$, "
                f"Fermat's Little Theorem states $a^{{p-1}} \\equiv 1 \\pmod{{p}}$.\n\n"
                f"Find ${a}^{{{b}^{{{c}}}}} \\bmod {p}$."
            )
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"For a positive integer $n$ and exponent $k$, an integer $a$ is called "
                f"$k$-stable modulo $n$ if $a^{{{k}}} \\equiv a \\pmod{{n}}$.\n\n"
                f"How many integers $a$ with $1 \\leq a < {n}$ are ${k}$-stable modulo ${n}$?"
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
