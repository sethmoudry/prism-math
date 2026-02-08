"""
Deep Insight Technique: Chinese Remainder Theorem
"""

import random
import math
from typing import Dict, Optional

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepChineseRemainder(MethodBlock):
    """
    Surface: System of congruences x = a_i (mod m_i)
    Hidden: CRT: Unique solution mod lcm(m_i) when gcd(m_i, m_j) | (a_i - a_j) for all i,j
    """

    def __init__(self):
        super().__init__()
        self.name = "chinese_remainder"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "crt", "congruences", "deep_insight"]

    def validate_params(self, params, prev_value=None):
        question_type = params.get("question_type", "")
        m1 = params.get("m1")
        m2 = params.get("m2")
        if m1 is None or m2 is None:
            return False
        if m1 <= 0 or m2 <= 0:
            return False
        if question_type in ("count_solutions", "smallest_solution"):
            return math.gcd(m1, m2) == 1
        return True

    def _extended_gcd(self, a, b):
        if b == 0:
            return a, 1, 0
        gcd_val, x1, y1 = self._extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return gcd_val, x, y

    def _crt_two(self, a1, m1, a2, m2):
        g, p, q = self._extended_gcd(m1, m2)
        if (a2 - a1) % g != 0:
            return None, None
        lcm_val = (m1 * m2) // g
        x = (a1 + m1 * ((a2 - a1) // g) * p) % lcm_val
        return x, lcm_val

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["count_solutions", "smallest_solution", "existence"])

        if question_type == "count_solutions":
            m1 = random.choice([3, 4, 5, 7])
            m2 = random.choice([4, 5, 7, 9, 11])
            while math.gcd(m1, m2) != 1:
                m2 = random.choice([4, 5, 7, 9, 11])
            a1 = random.randint(0, m1 - 1)
            a2 = random.randint(0, m2 - 1)
            N = random.choice([100, 500, 1000])
            return {"question_type": question_type, "a1": a1, "m1": m1, "a2": a2, "m2": m2, "N": N}
        elif question_type == "smallest_solution":
            m1 = random.choice([3, 5, 7])
            m2 = random.choice([4, 8, 9])
            while math.gcd(m1, m2) != 1:
                m2 = random.choice([4, 8, 9])
            a1 = random.randint(1, m1 - 1)
            a2 = random.randint(1, m2 - 1)
            return {"question_type": question_type, "a1": a1, "m1": m1, "a2": a2, "m2": m2}
        else:
            m1 = random.choice([6, 8, 10, 12])
            m2 = random.choice([8, 10, 12, 15])
            a1 = random.randint(0, m1 - 1)
            a2 = random.randint(0, m2 - 1)
            return {"question_type": question_type, "a1": a1, "m1": m1, "a2": a2, "m2": m2}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "count_solutions":
            a1, m1, a2, m2, N = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10), params.get("N", 50)
            x0, M = self._crt_two(a1, m1, a2, m2)
            if x0 is None:
                answer = 0
            else:
                if x0 == 0:
                    x0 = M
                count = (N - x0) // M + 1 if x0 <= N else 0
                answer = max(0, count)
            description = f"Solutions in [1,{N}] to x={a1}(mod {m1}), x={a2}(mod {m2}): {answer}"
        elif question_type == "smallest_solution":
            a1, m1, a2, m2 = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10)
            x0, M = self._crt_two(a1, m1, a2, m2)
            if x0 is None:
                answer = 0
            else:
                answer = x0 if x0 > 0 else x0 + M
            description = f"Smallest positive x: x={a1}(mod {m1}), x={a2}(mod {m2}): {answer}"
        else:
            a1, m1, a2, m2 = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10)
            g = math.gcd(m1, m2)
            exists = ((a2 - a1) % g == 0)
            answer = 1 if exists else 0
            description = f"Solution exists for x={a1}(mod {m1}), x={a2}(mod {m2}): {'Yes' if exists else 'No'}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "chinese_remainder_theorem"}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target

        moduli_pairs = [
            (3, 4), (3, 5), (3, 7), (3, 8), (3, 11), (3, 13), (3, 16),
            (4, 5), (4, 7), (4, 9), (4, 11), (4, 13), (4, 15),
            (5, 4), (5, 7), (5, 8), (5, 9), (5, 11), (5, 12), (5, 13),
            (7, 4), (7, 8), (7, 9), (7, 11), (7, 12), (7, 13), (7, 15),
            (8, 9), (8, 11), (8, 13), (8, 15),
            (9, 11), (9, 13), (9, 16),
            (11, 13), (11, 16),
        ]

        for m1, m2 in moduli_pairs:
            if math.gcd(m1, m2) != 1:
                continue
            for a1 in range(1, m1):
                for a2 in range(1, m2):
                    x0, M = self._crt_two(a1, m1, a2, m2)
                    if x0 is None:
                        continue
                    answer = x0 if x0 > 0 else x0 + M
                    if answer == wrapped_target:
                        return {"question_type": "smallest_solution", "a1": a1, "m1": m1, "a2": a2, "m2": m2}

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

        if question_type == "count_solutions":
            a1, m1, a2, m2, N = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10), params.get("N", 50)
            relationship = (
                f"Consider the system of congruences:\n"
                f"$$x \\equiv {a1} \\pmod{{{m1}}}$$\n"
                f"$$x \\equiv {a2} \\pmod{{{m2}}}$$\n\n"
                f"How many positive integer solutions $x$ with $1 \\leq x \\leq {N}$ exist?"
            )
        elif question_type == "smallest_solution":
            a1, m1, a2, m2 = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10)
            relationship = (
                f"Find the smallest positive integer $x$ satisfying:\n"
                f"$$x \\equiv {a1} \\pmod{{{m1}}}$$\n"
                f"$$x \\equiv {a2} \\pmod{{{m2}}}$$"
            )
        else:
            a1, m1, a2, m2 = params.get("a1", 10), params.get("m1", 10), params.get("a2", 10), params.get("m2", 10)
            relationship = (
                f"Does there exist an integer $x$ satisfying both:\n"
                f"$$x \\equiv {a1} \\pmod{{{m1}}}$$\n"
                f"$$x \\equiv {a2} \\pmod{{{m2}}}$$\n\n"
                f"Output $1$ if yes, $0$ if no."
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
