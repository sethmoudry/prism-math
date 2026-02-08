"""
Deep Insight Techniques: Combinatorics (Advanced)

Techniques for combinatorial problems requiring deep insights:
- Derangements
- Burnside's lemma (necklaces, bracelets)
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
class DeepDerangements(MethodBlock):
    """
    Surface: Permutations with no fixed points (derangements)
    Hidden: D_n = (n-1)(D_{n-1} + D_{n-2}), D_n approx n!/e
    """

    def __init__(self):
        super().__init__()
        self.name = "derangements"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "derangements", "permutations", "deep_insight"]

    def _derangements(self, n):
        if n == 0:
            return 1
        if n == 1:
            return 0
        if n == 2:
            return 1
        d_prev2, d_prev1 = 1, 0
        for i in range(2, n + 1):
            d_curr = (i - 1) * (d_prev1 + d_prev2)
            d_prev2, d_prev1 = d_prev1, d_curr
        return d_prev1

    def generate_parameters(self, input_value=None):
        question_type = random.choice([
            "count_derangements", "probability", "ratio_to_factorial",
            "derangements_plus_k", "derangements_offset"
        ])

        if question_type == "count_derangements":
            n = random.randint(4, 12)
            return {"question_type": question_type, "n": n}
        elif question_type == "probability":
            n = random.randint(5, 15)
            return {"question_type": question_type, "n": n}
        elif question_type == "ratio_to_factorial":
            n = random.randint(6, 12)
            return {"question_type": question_type, "n": n}
        elif question_type == "derangements_plus_k":
            n = random.randint(6, 12)
            k = random.randint(1, min(n-2, 5))
            return {"question_type": question_type, "n": n, "k": k}
        else:
            n = random.randint(4, 8)
            offset = random.randint(-50, 200)
            return {"question_type": question_type, "n": n, "offset": offset}

    def compute(self, input_value, params):
        from math import comb
        question_type = params.get("question_type", "compute")

        if question_type == "count_derangements":
            n = params.get("n", 10)
            answer = self._derangements(n)
            description = f"Derangements of {n} elements: D_{n} = {answer}"
        elif question_type == "probability":
            n = params.get("n", 10)
            d_n = self._derangements(n)
            n_fact = int(factorial(n))
            prob = d_n / n_fact
            answer = int(prob * 100000)
            description = f"Probability of derangement for n={n}: {prob:.5f}, encoded as {answer}"
        elif question_type == "ratio_to_factorial":
            n = params.get("n", 10)
            d_n = self._derangements(n)
            answer = d_n
            description = f"D_{n} = {d_n}, n! = {int(factorial(n))}, ratio approx 1/e"
        elif question_type == "derangements_plus_k":
            n = params.get("n", 10)
            k = params.get("k", 5)
            d_n_minus_k = self._derangements(n - k)
            answer = comb(n, k) * d_n_minus_k
            description = f"Permutations of {n} elements with exactly {k} fixed points: C({n},{k}) * D({n-k}) = {answer}"
        else:
            n = params.get("n", 10)
            offset = params.get("offset", 10)
            d_n = self._derangements(n)
            answer = d_n + offset
            description = f"D({n}) with offset {offset}: {d_n} + {offset} = {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target
        from math import comb

        for n in range(2, 20):
            if self._derangements(n) == target:
                return {"question_type": "count_derangements", "n": n}

        for n in range(3, 20):
            for k in range(1, n):
                try:
                    d_val = self._derangements(n - k)
                    answer = comb(n, k) * d_val
                    if answer == wrapped_target:
                        return {"question_type": "derangements_plus_k", "n": n, "k": k}
                    if answer > target * 10:
                        break
                except (OverflowError, ValueError):
                    continue

        for n in range(2, 15):
            d_n = self._derangements(n)
            offset = target - d_n
            if -200 <= offset <= 200:
                return {"question_type": "derangements_offset", "n": n, "offset": offset}

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

        if question_type == "count_derangements":
            n = params.get("n", 10)
            relationship = (
                f"A derangement is a permutation where no element appears in its original position. "
                f"How many derangements are there of a set of ${n}$ distinct elements?"
            )
        elif question_type == "probability":
            n = params.get("n", 10)
            relationship = (
                f"A derangement is a permutation where no element appears in its original position. "
                f"What is the probability that a uniformly random permutation of ${n}$ elements is a derangement? "
                f"Express your answer as an integer by computing $\\lfloor 10^5 \\cdot P \\rfloor$ where $P$ is the probability."
            )
        elif question_type == "ratio_to_factorial":
            n = params.get("n", 10)
            relationship = (
                f"A derangement is a permutation where no element appears in its original position. "
                f"Let $D_n$ denote the number of derangements of $n$ elements. Find $D_{{{n}}}$."
            )
        elif question_type == "derangements_plus_k":
            n = params.get("n", 10)
            k = params.get("k", 5)
            relationship = (
                f"A derangement is a permutation where no element appears in its original position. "
                f"How many permutations of ${n}$ distinct elements have exactly ${k}$ fixed points?"
            )
        else:
            n = params.get("n", 10)
            offset = params.get("offset", 10)
            d_n = self._derangements(n)
            if offset > 0:
                relationship = (
                    f"A derangement is a permutation where no element appears in its original position. "
                    f"Let $D_n$ denote the number of derangements of $n$ elements. "
                    f"Given that $D_{{{n}}} = {d_n}$, compute $D_{{{n}}} + {offset}$."
                )
            elif offset < 0:
                relationship = (
                    f"A derangement is a permutation where no element appears in its original position. "
                    f"Let $D_n$ denote the number of derangements of $n$ elements. "
                    f"Given that $D_{{{n}}} = {d_n}$, compute $D_{{{n}}} - {abs(offset)}$."
                )
            else:
                relationship = (
                    f"A derangement is a permutation where no element appears in its original position. "
                    f"Let $D_n$ denote the number of derangements of $n$ elements. Find $D_{{{n}}}$."
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
class DeepBurnside(MethodBlock):
    """
    Surface: Count distinct objects under group action (necklaces, colorings)
    Hidden: Burnside's lemma: |X/G| = (1/|G|) * sum_{g in G} |X^g|
    """

    def __init__(self):
        super().__init__()
        self.name = "burnside"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["combinatorics", "burnside", "group_theory", "deep_insight"]

    def _gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def _necklaces(self, n, k):
        total = 0
        for r in range(n):
            total += k ** self._gcd(n, r)
        return total // n

    def _necklaces_with_reflection(self, n, k):
        rot_count = sum(k ** self._gcd(n, i) for i in range(n)) // n
        if n % 2 == 1:
            refl_count = n * (k ** ((n + 1) // 2))
        else:
            refl_count = (n // 2) * (k ** (n // 2 + 1)) + (n // 2) * (k ** (n // 2))
        total = (rot_count * n + refl_count) // (2 * n)
        return total

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["necklace_rotation", "necklace_reflection", "bracelet"])

        if question_type == "necklace_rotation":
            n = random.randint(4, 10)
            k = random.randint(2, 5)
            return {"question_type": question_type, "n": n, "k": k}
        elif question_type == "necklace_reflection":
            n = random.randint(4, 8)
            k = random.randint(2, 4)
            return {"question_type": question_type, "n": n, "k": k}
        else:
            n = random.randint(4, 8)
            k = random.randint(2, 4)
            return {"question_type": question_type, "n": n, "k": k}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "necklace_rotation":
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._necklaces(n, k)
            description = f"Necklaces with {n} beads, {k} colors (rotation only): {answer}"
        elif question_type == "necklace_reflection":
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._necklaces_with_reflection(n, k)
            description = f"Necklaces with {n} beads, {k} colors (rotation + reflection): {answer}"
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._necklaces_with_reflection(n, k)
            description = f"Bracelets with {n} beads, {k} colors: {answer}"

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

        if question_type == "necklace_rotation":
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"A necklace is a circular arrangement of ${n}$ beads, where each bead can be colored "
                f"with one of ${k}$ colors. Two necklaces are considered the same if one can be obtained "
                f"from the other by rotation. How many distinct necklaces are there?"
            )
        elif question_type == "necklace_reflection":
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"A necklace is a circular arrangement of ${n}$ beads, where each bead can be colored "
                f"with one of ${k}$ colors. Two necklaces are considered the same if one can be obtained "
                f"from the other by rotation or reflection. How many distinct necklaces are there?"
            )
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"A bracelet is a circular arrangement of ${n}$ beads, where each bead can be colored "
                f"with one of ${k}$ colors. Two bracelets are identical if one can be obtained from "
                f"the other by rotation or flipping. How many distinct bracelets exist?"
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
