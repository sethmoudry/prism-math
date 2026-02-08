"""
Deep Insight Technique: Wilson's Theorem
"""

import random
from typing import Dict, Optional
from sympy import isprime

from ..base import MethodBlock, MethodResult
from ..registry import register_technique
from .number_theory_base import generate_uuid, create_problem_dict


@register_technique
class DeepWilson(MethodBlock):
    """
    Surface: (p-1)! = -1 (mod p) for prime p (Wilson's theorem)
    Hidden: Wilson's theorem CHARACTERIZES primes: n is prime iff (n-1)! = -1 (mod n)
    """

    def __init__(self):
        super().__init__()
        self.name = "wilson"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["number_theory", "wilson", "factorial", "deep_insight"]

    def _factorial_mod(self, n, mod):
        if n > mod:
            return 0
        result = 1
        for i in range(1, n):
            result = (result * i) % mod
        return result

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["compute_factorial_mod", "count_wilson_property", "verify_primality"])

        if question_type == "compute_factorial_mod":
            n = random.choice([6, 7, 8, 10, 11, 12, 13, 15, 17, 19, 20])
            return {"question_type": question_type, "n": n}
        elif question_type == "count_wilson_property":
            N = random.choice([20, 30, 50, 100])
            return {"question_type": question_type, "N": N}
        else:
            n = random.choice([7, 11, 13, 15, 17, 21, 23, 25, 29])
            return {"question_type": question_type, "n": n}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "compute_factorial_mod":
            n = params.get("n", 10)
            if n == 1:
                answer = 0
            else:
                result = self._factorial_mod(n, n)
                if result == n - 1:
                    answer = n - 1
                else:
                    answer = result
            description = f"({n-1})! mod {n} = {answer}"
        elif question_type == "count_wilson_property":
            N = params.get("N", 50)
            count = 0
            for n in range(2, N + 1):
                if isprime(n):
                    count += 1
            answer = count
            description = f"Count n <= {N} where (n-1)! = -1 (mod n): {count} (primes)"
        else:
            n = params.get("n", 10)
            if n <= 1:
                is_prime_wilson = 0
            else:
                fact_mod = self._factorial_mod(n, n)
                is_prime_wilson = 1 if (fact_mod == n - 1 or fact_mod == -1 % n) else 0
            answer = is_prime_wilson
            description = f"Is {n} prime by Wilson? {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "wilson_theorem"}
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

        if question_type == "compute_factorial_mod":
            n = params.get("n", 10)
            relationship = (
                f"Wilson's Theorem states that for a prime $p$, we have $(p-1)! \\equiv -1 \\pmod{{p}}$.\n\n"
                f"Find the remainder when $({n-1})!$ is divided by ${n}$."
            )
        elif question_type == "count_wilson_property":
            N = params.get("N", 50)
            relationship = (
                f"Wilson's Theorem states that an integer $n > 1$ is prime if and only if "
                f"$(n-1)! \\equiv -1 \\pmod{{n}}$.\n\n"
                f"For how many integers $n$ with $2 \\leq n \\leq {N}$ does "
                f"$(n-1)! \\equiv -1 \\pmod{{n}}$ hold?"
            )
        else:
            n = params.get("n", 10)
            relationship = (
                f"Wilson's Theorem characterizes primes: an integer $n > 1$ is prime if and only if "
                f"$(n-1)! \\equiv -1 \\pmod{{n}}$.\n\n"
                f"Is ${n}$ prime according to Wilson's criterion? Output $1$ for yes, $0$ for no."
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
