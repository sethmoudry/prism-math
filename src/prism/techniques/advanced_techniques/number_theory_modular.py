"""
Deep Insight Technique: Modular Order (primitive roots)
"""

import random
import math
from typing import Dict, Optional
from sympy import totient

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepModularOrder(MethodBlock):
    """
    Surface: ord_n(a) = smallest k > 0 with a^k = 1 (mod n)
    Hidden: ord_n(a) divides phi(n). Primitive roots have order exactly p-1.
    """

    def __init__(self):
        super().__init__()
        self.name = "modular_order"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["number_theory", "modular", "primitive_root", "deep_insight"]

    def validate_params(self, params, prev_value=None):
        p = params.get("p")
        if p is None or p <= 1:
            return False
        question_type = params.get("question_type", "")
        if question_type == "count_order_d":
            d = params.get("d")
            if d is None or d <= 0:
                return False
            return (p - 1) % d == 0
        return True

    def _multiplicative_order(self, a, n):
        if math.gcd(a, n) != 1:
            return 0
        order = 1
        current = a % n
        while current != 1:
            current = (current * a) % n
            order += 1
            if order > n:
                return 0
        return order

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["count_order_d", "smallest_primitive_root", "primitive_root_count"])

        if question_type == "count_order_d":
            p = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
            divs = [d for d in range(1, p) if (p-1) % d == 0]
            d = random.choice(divs)
            return {"question_type": question_type, "p": p, "d": d}
        elif question_type == "smallest_primitive_root":
            p = random.choice([7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
            return {"question_type": question_type, "p": p}
        else:
            p = random.choice([7, 11, 13, 17, 19, 23, 29, 31])
            return {"question_type": question_type, "p": p}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "count_order_d":
            p, d = params.get("p", 5), params.get("d", 5)
            answer = int(totient(d))
            description = f"Elements of order {d} mod {p}: phi({d}) = {answer}"
        elif question_type == "smallest_primitive_root":
            p = params.get("p", 5)
            for a in range(2, p):
                if self._multiplicative_order(a, p) == p - 1:
                    answer = a
                    break
            else:
                answer = 0
            description = f"Smallest primitive root mod {p}: {answer}"
        else:
            p = params.get("p", 5)
            answer = int(totient(p - 1))
            description = f"Number of primitive roots mod {p}: phi({p-1}) = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type}
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

        if question_type == "count_order_d":
            p, d = params.get("p", 5), params.get("d", 5)
            relationship = (
                f"For a prime $p$ and integer $a$ with $\\gcd(a, p) = 1$, the multiplicative order "
                f"$\\text{{ord}}_p(a)$ is the smallest positive integer $k$ such that $a^k \\equiv 1 \\pmod{{p}}$.\n\n"
                f"How many integers $a$ with $1 \\leq a \\leq {p-1}$ satisfy $\\text{{ord}}_{{{p}}}(a) = {d}$?"
            )
        elif question_type == "smallest_primitive_root":
            p = params.get("p", 5)
            relationship = (
                f"A primitive root modulo a prime $p$ is an integer $a$ such that the powers "
                f"$a, a^2, \\ldots, a^{{p-1}}$ produce all nonzero residues modulo $p$.\n\n"
                f"Find the smallest primitive root modulo ${p}$."
            )
        else:
            p = params.get("p", 5)
            relationship = (
                f"A primitive root modulo a prime $p$ is an integer $a$ with $1 \\leq a \\leq p-1$ "
                f"such that the multiplicative order of $a$ modulo $p$ equals $p-1$.\n\n"
                f"How many primitive roots modulo ${p}$ exist?"
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
