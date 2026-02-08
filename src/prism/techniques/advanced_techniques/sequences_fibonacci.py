"""
Deep Insight Techniques: Sequences (Fibonacci & Recurrences)

Techniques for sequence-related problems requiring deep insights:
- Fibonacci divisibility (gcd(F_m, F_n) = F_{gcd(m,n)})
- Linear recurrences (characteristic polynomials, periodicity)
- Beatty sequences (floor(n*alpha) for irrational alpha)
"""

import random
import math
from typing import Dict, Any, Optional
from sympy import fibonacci, factorial, floor as sym_floor

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
class DeepFibonacciDivisibility(MethodBlock):
    """
    Surface: Fibonacci sequence F_n (F_1 = 1, F_2 = 1, F_n = F_{n-1} + F_{n-2})
    Hidden: gcd(F_m, F_n) = F_{gcd(m,n)}, F_n divides F_{kn} for all k
    """

    def __init__(self):
        super().__init__()
        self.name = "fibonacci_divisibility"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["fibonacci", "gcd", "divisibility", "deep_insight", "sequences"]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["gcd_fibonacci", "smallest_divisible", "divides_check"])

        if question_type == "gcd_fibonacci":
            m = random.randint(20, 100)
            n = random.randint(20, 100)
            return {"question_type": question_type, "m": m, "n": n}
        elif question_type == "smallest_divisible":
            d = random.choice([2, 3, 5, 7, 11, 13])
            return {"question_type": question_type, "d": d}
        else:
            a = random.randint(3, 15)
            if random.random() < 0.5:
                k = random.randint(2, 5)
                b = a * k
            else:
                b = random.randint(a + 1, a * 3)
                while b % a == 0:
                    b = random.randint(a + 1, a * 3)
            return {"question_type": question_type, "a": a, "b": b}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "gcd_fibonacci":
            m, n = params.get("m", 10), params.get("n", 10)
            g = math.gcd(m, n)
            answer = int(fibonacci(g))
            description = f"gcd(F_{m}, F_{n}) = F_{g} = {answer}"
        elif question_type == "smallest_divisible":
            d = params.get("d", 5)
            for n in range(1, 10000):
                if int(fibonacci(n)) % d == 0:
                    if n > 1:
                        answer = n
                        break
            else:
                answer = 0
            description = f"Smallest n > 1 where {d} | F_n: {answer}"
        else:
            a, b = params.get("a", 10), params.get("b", 10)
            if b % a == 0:
                answer = 1
            else:
                answer = 0
            description = f"Does F_{a} divide F_{b}? {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "gcd(F_m, F_n) = F_{gcd(m,n)}"}
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

        if question_type == "gcd_fibonacci":
            m, n = params.get("m", 10), params.get("n", 10)
            relationship = (
                f"Let $F_n$ denote the $n$-th Fibonacci number, defined by $F_1 = 1$, $F_2 = 1$, "
                f"and $F_n = F_{{n-1}} + F_{{n-2}}$ for $n \\geq 3$.\n\n"
                f"Compute $\\gcd(F_{{{m}}}, F_{{{n}}})$."
            )
        elif question_type == "smallest_divisible":
            d = params.get("d", 5)
            relationship = (
                f"Let $F_n$ denote the $n$-th Fibonacci number, defined by $F_1 = 1$, $F_2 = 1$, "
                f"and $F_n = F_{{n-1}} + F_{{n-2}}$ for $n \\geq 3$.\n\n"
                f"Find the smallest integer $n > 1$ such that ${d}$ divides $F_n$."
            )
        else:
            a, b = params.get("a", 10), params.get("b", 10)
            relationship = (
                f"Let $F_n$ denote the $n$-th Fibonacci number, defined by $F_1 = 1$, $F_2 = 1$, "
                f"and $F_n = F_{{n-1}} + F_{{n-2}}$ for $n \\geq 3$.\n\n"
                f"Does $F_{{{a}}}$ divide $F_{{{b}}}$? Output $1$ for yes, $0$ for no."
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
class DeepLinearRecurrence(MethodBlock):
    """
    Surface: a_n = c_1*a_{n-1} + c_2*a_{n-2} + ...
    Hidden: Characteristic polynomial roots determine closed form
    """

    def __init__(self):
        super().__init__()
        self.name = "deep_linear_recurrence"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["recurrence", "characteristic_polynomial", "deep_insight", "sequences"]

    def validate_params(self, params, prev_value=None):
        question_type = params.get("question_type", "")
        if question_type in ("value_mod_m", "period_mod_m"):
            m = params.get("m")
            if m is None or m <= 0:
                return False
        c1 = params.get("c1")
        c2 = params.get("c2")
        if c1 is None and c2 is None:
            return False
        return True

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["value_mod_m", "period_mod_m", "large_index"])

        if question_type == "value_mod_m":
            c1 = random.randint(1, 5)
            c2 = random.randint(1, 5)
            a0 = random.randint(0, 10)
            a1 = random.randint(0, 10)
            N = random.randint(50, 200)
            m = random.choice([7, 11, 13, 17, 1000])
            return {"question_type": question_type, "c1": c1, "c2": c2, "a0": a0, "a1": a1, "N": N, "m": m}
        elif question_type == "period_mod_m":
            c1 = random.randint(1, 3)
            c2 = random.randint(1, 3)
            a0 = random.randint(0, 5)
            a1 = random.randint(1, 5)
            m = random.choice([5, 7, 10, 12])
            return {"question_type": question_type, "c1": c1, "c2": c2, "a0": a0, "a1": a1, "m": m}
        else:
            c1 = random.randint(1, 3)
            c2 = random.randint(1, 3)
            a0 = random.randint(0, 5)
            a1 = random.randint(1, 5)
            N = random.randint(100, 500)
            m = random.choice([100, 1000, 10000])
            return {"question_type": question_type, "c1": c1, "c2": c2, "a0": a0, "a1": a1, "N": N, "m": m}

    def _compute_recurrence(self, c1, c2, a0, a1, N, m):
        if N == 0:
            return a0 % m
        if N == 1:
            return a1 % m
        if N < 10000:
            prev2, prev1 = a0, a1
            for _ in range(2, N + 1):
                curr = (c1 * prev1 + c2 * prev2) % m
                prev2, prev1 = prev1, curr
            return prev1
        prev2, prev1 = a0, a1
        for _ in range(2, min(N + 1, 100000)):
            curr = (c1 * prev1 + c2 * prev2) % m
            prev2, prev1 = prev1, curr
        return prev1

    def _find_period(self, c1, c2, a0, a1, m):
        seen = {(a0 % m, a1 % m): 0}
        prev2, prev1 = a0 % m, a1 % m
        for i in range(1, m * m + 100):
            curr = (c1 * prev1 + c2 * prev2) % m
            state = (prev1, curr)
            if state in seen:
                return i - seen[state]
            seen[state] = i
            prev2, prev1 = prev1, curr
        return 0

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type in ["value_mod_m", "large_index"]:
            c1, c2, a0, a1 = params.get("c1", 10), params.get("c2", 10), params.get("a0", 10), params.get("a1", 10)
            N, m = params.get("N", 50), params.get("m", 10)
            answer = self._compute_recurrence(c1, c2, a0, a1, N, m)
            description = f"a_{N} mod {m} = {answer} for recurrence a_n = {c1}*a_{{n-1}} + {c2}*a_{{n-2}}"
        else:
            c1, c2, a0, a1 = params.get("c1", 10), params.get("c2", 10), params.get("a0", 10), params.get("a1", 10)
            m = params.get("m", 10)
            answer = self._find_period(c1, c2, a0, a1, m)
            description = f"Period of recurrence mod {m}: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "Characteristic polynomial and periodicity", "raw_answer": answer}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target

        if target > 10000:
            raw_targets = [target]

            for raw_target in raw_targets:
                m = raw_target + random.randint(10000, 100000)
                constructions = [
                    (1, 1, 0, raw_target, 2),
                    (1, 0, 0, raw_target, 1),
                    (2, 1, raw_target, 0, 2),
                    (1, 1, raw_target, 0, 3),
                ]
                for c1, c2, a0, a1, N in constructions:
                    computed = self._compute_recurrence(c1, c2, a0, a1, N, m)
                    if computed == target:
                        return {"question_type": "value_mod_m", "c1": c1, "c2": c2, "a0": a0, "a1": a1, "N": N, "m": m}

        moduli = [7, 11, 13, 17, 100, 1000, 10000]
        if target >= 10000:
            if target < 100000:
                moduli.extend([100000])
            if target < 1000000:
                moduli.extend([1000000])

        for m in moduli:
            if target >= m:
                continue
            for c1 in [1, 2, 3, 5]:
                for c2 in [1, 2, 3, 5]:
                    for a0 in range(0, min(11, target + 1)):
                        for a1 in range(0, min(11, target + 1)):
                            prev2, prev1 = a0 % m, a1 % m
                            max_n = min(500, m) if m > 10000 else min(201, m * 2)
                            for N in range(2, max_n):
                                curr = (c1 * prev1 + c2 * prev2) % m
                                if curr == target:
                                    return {"question_type": "value_mod_m", "c1": c1, "c2": c2, "a0": a0, "a1": a1, "N": N, "m": m}
                                prev2, prev1 = prev1, curr

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

        if question_type in ["value_mod_m", "large_index"]:
            c1, c2, a0, a1 = params.get("c1", 10), params.get("c2", 10), params.get("a0", 10), params.get("a1", 10)
            N, m = params.get("N", 50), params.get("m", 10)
            relationship = (
                f"Consider the sequence defined by $a_0 = {a0}$, $a_1 = {a1}$, and "
                f"$a_n = {c1} \\cdot a_{{n-1}} + {c2} \\cdot a_{{n-2}}$ for $n \\geq 2$.\n\n"
                f"Compute $a_{{{N}}} \\bmod {m}$."
            )
        else:
            c1, c2, a0, a1 = params.get("c1", 10), params.get("c2", 10), params.get("a0", 10), params.get("a1", 10)
            m = params.get("m", 10)
            relationship = (
                f"Consider the sequence defined by $a_0 = {a0}$, $a_1 = {a1}$, and "
                f"$a_n = {c1} \\cdot a_{{n-1}} + {c2} \\cdot a_{{n-2}}$ for $n \\geq 2$.\n\n"
                f"Find the smallest positive integer $T$ such that $a_{{n+T}} \\equiv a_n \\pmod{{{m}}}$ for all $n \\geq 0$."
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
class DeepBeattySequence(MethodBlock):
    """
    Surface: floor(n*alpha) for irrational alpha
    Hidden: Beatty's theorem: if 1/alpha + 1/beta = 1, sequences partition positive integers
    """

    def __init__(self):
        super().__init__()
        self.name = "beatty_sequence"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["beatty", "sequences", "irrational", "deep_insight"]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["membership", "nth_term", "count_below"])
        alpha_choices = [
            ("phi", (1 + 5**0.5) / 2),
            ("sqrt2", 2**0.5),
            ("sqrt3", 3**0.5),
            ("e", 2.718281828),
        ]

        if question_type == "membership":
            alpha_name, alpha_val = random.choice(alpha_choices)
            k = random.randint(50, 300)
            return {"question_type": question_type, "alpha_name": alpha_name, "alpha_val": alpha_val, "k": k}
        elif question_type == "nth_term":
            alpha_name, alpha_val = random.choice(alpha_choices)
            n = random.randint(20, 100)
            return {"question_type": question_type, "alpha_name": alpha_name, "alpha_val": alpha_val, "n": n}
        else:
            alpha_name, alpha_val = random.choice(alpha_choices)
            N = random.randint(100, 500)
            return {"question_type": question_type, "alpha_name": alpha_name, "alpha_val": alpha_val, "N": N}

    def _alpha_latex(self, alpha_name):
        if alpha_name == "phi":
            return "\\varphi = \\frac{1 + \\sqrt{5}}{2}"
        elif alpha_name == "sqrt2":
            return "\\sqrt{2}"
        elif alpha_name == "sqrt3":
            return "\\sqrt{3}"
        elif alpha_name == "e":
            return "e"
        return "\\alpha"

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "membership":
            alpha_val, k = params.get("alpha_val", 10), params.get("k", 5)
            n_low = k / alpha_val
            n_high = (k + 1) / alpha_val
            if int(sym_floor(n_high)) > int(sym_floor(n_low)):
                answer = 1
            else:
                answer = 0
            description = f"Is {k} in floor(n*{params['alpha_name']})? {answer}"
        elif question_type == "nth_term":
            alpha_val, n = params.get("alpha_val", 10), params.get("n", 10)
            answer = int(sym_floor(n * alpha_val))
            description = f"{n}th term of floor(n*{params['alpha_name']}): {answer}"
        else:
            alpha_val, N = params.get("alpha_val", 10), params.get("N", 50)
            answer = int(sym_floor(N / alpha_val))
            description = f"Terms <= {N} in floor(n*{params['alpha_name']}): {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "Beatty sequence properties"}
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
        alpha_latex = self._alpha_latex(params.get("alpha_name", 10))

        if question_type == "membership":
            k = params.get("k", 5)
            relationship = (
                f"Consider the sequence $a_n = \\lfloor n \\cdot {alpha_latex} \\rfloor$ for positive integers $n$.\n\n"
                f"Does the value ${k}$ appear in this sequence? Output $1$ for yes, $0$ for no."
            )
        elif question_type == "nth_term":
            n = params.get("n", 10)
            relationship = (
                f"Consider the sequence $a_n = \\lfloor n \\cdot {alpha_latex} \\rfloor$ for positive integers $n$.\n\n"
                f"Compute $a_{{{n}}}$."
            )
        else:
            N = params.get("N", 50)
            relationship = (
                f"Consider the sequence $a_n = \\lfloor n \\cdot {alpha_latex} \\rfloor$ for positive integers $n$.\n\n"
                f"How many terms of this sequence are less than or equal to ${N}$?"
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
