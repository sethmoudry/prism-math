"""
Deep Insight Technique: Quadratic Residues (Legendre symbol)
"""

import random
from typing import Dict, Optional
from sympy import isprime

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepQuadraticResidue(MethodBlock):
    """
    Surface: Legendre symbol (a/p) = a^((p-1)/2) mod p
    Hidden: Quadratic reciprocity law: (p/q)(q/p) = (-1)^((p-1)(q-1)/4)
    """

    def __init__(self):
        super().__init__()
        self.name = "quadratic_residue"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["number_theory", "quadratic_residue", "legendre", "deep_insight"]

    def _legendre_symbol(self, a, p):
        if p == 2:
            return 1
        a = a % p
        if a == 0:
            return 0
        result = pow(a, (p - 1) // 2, p)
        return -1 if result == p - 1 else result

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["count_qr_for_a", "smallest_prime_both_nonresidue", "count_primes_qr_equals_1"])

        if question_type == "count_qr_for_a":
            a = random.choice([2, 3, 5, 7])
            N = random.choice([50, 100, 200])
            return {"question_type": question_type, "a": a, "N": N}
        elif question_type == "smallest_prime_both_nonresidue":
            a = random.choice([2, 3, 5])
            b = random.choice([2, 3, 5, 7])
            while a == b:
                b = random.choice([2, 3, 5, 7])
            return {"question_type": question_type, "a": a, "b": b}
        else:
            a = random.choice([2, 3, 5, 7, 11])
            N = random.choice([30, 50, 100])
            return {"question_type": question_type, "a": a, "N": N}

    def validate_params(self, params, prev_value=None):
        N = params.get("N")
        if N is not None:
            try:
                return int(N) >= 3
            except (ValueError, TypeError):
                return False
        return True

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "count_qr_for_a":
            a, N = params.get("a", 10), params.get("N", 50)
            count = 0
            for p in range(3, N + 1):
                if isprime(p):
                    if self._legendre_symbol(a, p) == 1:
                        count += 1
            answer = count
            description = f"Primes p <= {N} where {a} is QR: {count}"
        elif question_type == "smallest_prime_both_nonresidue":
            a, b = params.get("a", 10), params.get("b", 10)
            for p in range(3, 1000):
                if isprime(p):
                    if self._legendre_symbol(a, p) == -1 and self._legendre_symbol(b, p) == -1:
                        answer = p
                        break
            else:
                answer = 0
            description = f"Smallest prime where ({a}/p) = ({b}/p) = -1: {answer}"
        else:
            a, N = params.get("a", 10), params.get("N", 50)
            count = 0
            for p in range(3, N + 1):
                if isprime(p):
                    if self._legendre_symbol(a, p) == 1:
                        count += 1
            answer = count
            description = f"Primes p <= {N} where ({a}/p) = 1: {count}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "quadratic_reciprocity"}
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

        if question_type == "count_qr_for_a":
            a, N = params.get("a", 10), params.get("N", 50)
            relationship = (
                f"For an odd prime $p$ and integer $a$, the Legendre symbol $(a/p)$ equals $1$ "
                f"if $a$ is a quadratic residue modulo $p$, and $-1$ otherwise.\n\n"
                f"For how many primes $p \\leq {N}$ is ${a}$ a quadratic residue modulo $p$?"
            )
        elif question_type == "smallest_prime_both_nonresidue":
            a, b = params.get("a", 10), params.get("b", 10)
            relationship = (
                f"For an odd prime $p$ and integer $a$, the Legendre symbol $(a/p)$ equals $1$ "
                f"if $a$ is a quadratic residue modulo $p$, and $-1$ if $a$ is a non-residue.\n\n"
                f"Find the smallest odd prime $p$ such that both ${a}$ and ${b}$ are "
                f"quadratic non-residues modulo $p$."
            )
        else:
            a, N = params.get("a", 10), params.get("N", 50)
            relationship = (
                f"For an odd prime $p$ and integer $a$ with $\\gcd(a, p) = 1$, the Legendre symbol "
                f"$(a/p)$ equals $a^{{(p-1)/2}} \\bmod p$, which is $1$ if $a$ is a quadratic residue "
                f"and $-1$ otherwise.\n\n"
                f"Count the number of primes $p \\leq {N}$ for which $({a}/p) = 1$."
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
