"""
Deep Insight Techniques: Functional Equations (Cyclic & Involutions)

Techniques for problems involving functional equations and transformations:
- Idempotent functions
- Involution functions
- Cyclic functional equations
"""

import random
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass

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
class DeepIdempotent(MethodBlock):
    """
    Surface: f(f(x)) = f(x) for all x (idempotent/projection)
    Hidden: f projects onto its fixed points. Count = sum of C(n,k) * k^(n-k)
    """

    def __init__(self):
        super().__init__()
        self.name = "idempotent"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["functional_equation", "deep_insight", "combinatorics"]

    def _count_idempotent(self, n):
        from math import comb
        total = 0
        for k in range(1, n + 1):
            total += comb(n, k) * (k ** (n - k))
        return total

    def _count_idempotent_k_fixed(self, n, k):
        from math import comb
        return comb(n, k) * (k ** (n - k))

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["total_count", "fixed_count", "partial_constraint"])

        if question_type == "total_count":
            n = random.randint(3, 6)
            return {"question_type": question_type, "n": n}
        elif question_type == "fixed_count":
            n = random.randint(4, 7)
            k = random.randint(2, n - 1)
            return {"question_type": question_type, "n": n, "k": k}
        else:
            n = random.randint(4, 6)
            a = random.randint(1, n)
            return {"question_type": question_type, "n": n, "a": a}

    def compute(self, input_value, params):
        from math import comb
        question_type = params.get("question_type", "compute")

        if question_type == "total_count":
            n = params.get("n", 10)
            answer = self._count_idempotent(n)
            description = f"Idempotent f: [1..{n}] -> [1..{n}]: {answer}"
        elif question_type == "fixed_count":
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._count_idempotent_k_fixed(n, k)
            description = f"Idempotent f: [1..{n}] -> [1..{n}] with {k} fixed points: {answer}"
        else:
            n, a = params.get("n", 10), params.get("a", 10)
            total = 0
            for k in range(1, n + 1):
                total += comb(n - 1, k - 1) * (k ** (n - k))
            answer = total
            description = f"Idempotent f with f(1)={a}, f({a})={a}: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "f projects onto fixed points"}
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

        base_statement = (
            "A function $f: \\{1, 2, \\ldots, n\\} \\to \\{1, 2, \\ldots, n\\}$ is called idempotent if "
            "$f(f(x)) = f(x)$ for all $x$."
        )

        if question_type == "total_count":
            n = params.get("n", 10)
            question = f"How many idempotent functions are there from $\\{{1, 2, \\ldots, {n}\\}}$ to itself?"
        elif question_type == "fixed_count":
            n, k = params.get("n", 10), params.get("k", 5)
            question = f"How many idempotent functions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ have exactly ${k}$ fixed points?"
        else:
            n, a = params.get("n", 10), params.get("a", 10)
            question = f"How many idempotent functions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ satisfy $f(1) = {a}$?"

        relationship = f"{base_statement}\n\n{question}"

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
class DeepInvolution(MethodBlock):
    """
    Surface: f(f(x)) = x for all x (involution)
    Hidden: f pairs up non-fixed elements, fixed points are isolated.
    """

    def __init__(self):
        super().__init__()
        self.name = "involution"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["functional_equation", "deep_insight", "combinatorics", "involution"]

    def _double_factorial(self, n):
        if n <= 0:
            return 1
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result

    def _count_involutions(self, n):
        from math import comb
        total = 0
        for k in range(n + 1):
            if (n - k) % 2 == 0:
                pairings = self._double_factorial(n - k - 1) if n - k > 0 else 1
                total += comb(n, k) * pairings
        return total

    def _count_involutions_k_fixed(self, n, k):
        if (n - k) % 2 == 1:
            return 0
        from math import comb
        pairings = self._double_factorial(n - k - 1) if n - k > 0 else 1
        return comb(n, k) * pairings

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["total_count", "fixed_count", "partial_constraint"])

        if question_type == "total_count":
            n = random.randint(4, 8)
            return {"question_type": question_type, "n": n}
        elif question_type == "fixed_count":
            n = random.randint(5, 8)
            k = random.choice([i for i in range(0, n + 1) if (n - i) % 2 == 0])
            return {"question_type": question_type, "n": n, "k": k}
        else:
            n = random.randint(5, 7)
            a = random.randint(1, n)
            return {"question_type": question_type, "n": n, "a": a}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "total_count":
            n = params.get("n", 10)
            answer = self._count_involutions(n)
            description = f"Involutions on [1..{n}]: {answer}"
        elif question_type == "fixed_count":
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._count_involutions_k_fixed(n, k)
            description = f"Involutions on [1..{n}] with {k} fixed points: {answer}"
        else:
            n, a = params.get("n", 10), params.get("a", 10)
            if a == 1:
                answer = self._count_involutions(n - 1)
            else:
                answer = self._count_involutions(n - 2)
            description = f"Involutions with f(1)={a}: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "involutions pair elements"}
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

        base_statement = (
            "A function $f: \\{1, 2, \\ldots, n\\} \\to \\{1, 2, \\ldots, n\\}$ is called an involution if "
            "$f(f(x)) = x$ for all $x$."
        )

        if question_type == "total_count":
            n = params.get("n", 10)
            question = f"How many involutions are there from $\\{{1, 2, \\ldots, {n}\\}}$ to itself?"
        elif question_type == "fixed_count":
            n, k = params.get("n", 10), params.get("k", 5)
            question = f"How many involutions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ have exactly ${k}$ fixed points?"
        else:
            n, a = params.get("n", 10), params.get("a", 10)
            question = f"How many involutions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ satisfy $f(1) = {a}$?"

        relationship = f"{base_statement}\n\n{question}"

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
class DeepCyclicFunctional(MethodBlock):
    """
    Surface: f(f(f(x))) = x for all x (3-cycle functional equation)
    Hidden: Elements partition into fixed points (f(x)=x) and 3-cycles (x->y->z->x).
    """

    def __init__(self):
        super().__init__()
        self.name = "cyclic_functional"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["functional_equation", "deep_insight", "permutation", "cycles"]

    def _count_3cycles(self, n):
        from math import comb, factorial
        total = 0
        for k in range(n // 3 + 1):
            fixed = n - 3 * k
            ways_to_choose = comb(n, 3 * k)
            if k == 0:
                ways_to_partition = 1
            else:
                ways_to_partition = factorial(3 * k) // ((3 ** k) * factorial(k))
            total += ways_to_choose * ways_to_partition
        return total

    def _count_3cycles_k_fixed(self, n, k):
        if (n - k) % 3 != 0:
            return 0
        num_3cycles = (n - k) // 3
        from math import comb, factorial
        ways_to_choose = comb(n, k)
        if num_3cycles == 0:
            ways_to_partition = 1
        else:
            ways_to_partition = factorial(n - k) // ((3 ** num_3cycles) * factorial(num_3cycles))
        return ways_to_choose * ways_to_partition

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["total_count", "no_fixed", "k_fixed"])

        if question_type == "total_count":
            n = random.choice([6, 9, 12])
            return {"question_type": question_type, "n": n}
        elif question_type == "no_fixed":
            n = random.choice([6, 9, 12])
            return {"question_type": question_type, "n": n}
        else:
            n = random.choice([6, 9, 12])
            k = random.choice([i for i in range(0, n + 1) if (n - i) % 3 == 0])
            return {"question_type": question_type, "n": n, "k": k}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "total_count":
            n = params.get("n", 10)
            answer = self._count_3cycles(n)
            description = f"Functions f: [1..{n}] -> [1..{n}] with f^3=id: {answer}"
        elif question_type == "no_fixed":
            n = params.get("n", 10)
            answer = self._count_3cycles_k_fixed(n, 0)
            description = f"Functions f: [1..{n}] -> [1..{n}] with f^3=id, no fixed points: {answer}"
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._count_3cycles_k_fixed(n, k)
            description = f"Functions f: [1..{n}] -> [1..{n}] with f^3=id, {k} fixed points: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type, "insight": "elements partition into fixed points and 3-cycles"}
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

        base_statement = (
            "A function $f: \\{1, 2, \\ldots, n\\} \\to \\{1, 2, \\ldots, n\\}$ satisfies "
            "$f(f(f(x))) = x$ for all $x$ (that is, $f^3 = \\text{id}$)."
        )

        if question_type == "total_count":
            n = params.get("n", 10)
            question = f"How many such functions are there from $\\{{1, 2, \\ldots, {n}\\}}$ to itself?"
        elif question_type == "no_fixed":
            n = params.get("n", 10)
            question = f"How many such functions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ have no fixed points (i.e., $f(x) \\neq x$ for all $x$)?"
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            question = f"How many such functions $f: \\{{1, \\ldots, {n}\\}} \\to \\{{1, \\ldots, {n}\\}}$ have exactly ${k}$ fixed points?"

        relationship = f"{base_statement}\n\n{question}"

        return create_problem_dict(
            relationship=relationship,
            answer=result.value,
            techniques=[self.name],
            uuid=generate_uuid(),
            metadata=result.metadata
        )

    def can_invert(self):
        return False
