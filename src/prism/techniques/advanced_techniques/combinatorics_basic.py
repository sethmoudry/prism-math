"""
Deep Insight Techniques: Combinatorics (Basic)

Techniques for combinatorial problems requiring deep insights:
- Pigeonhole principle
- Catalan numbers (triangulations, Dyck paths, binary trees)
- Stirling numbers (set partitions, Bell numbers)
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
class DeepPigeonhole(MethodBlock):
    """
    Surface: n items in k boxes => some box has >= ceil(n/k) items
    Hidden: This bound is TIGHT - achievable by even distribution.
    """

    def __init__(self):
        super().__init__()
        self.name = "deep_pigeonhole"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "pigeonhole", "deep_insight"]

    def validate_params(self, params, prev_value=None):
        question_type = params.get("question_type", "")
        if question_type == "residue_classes":
            m = params.get("m")
            return m is not None and m > 0
        elif question_type == "sum_constraint":
            k = params.get("k")
            return k is not None and k > 0
        elif question_type == "divisibility":
            n = params.get("n")
            return n is not None and n > 0
        elif question_type == "min_items":
            boxes = params.get("boxes")
            items_per_box = params.get("items_per_box")
            return boxes is not None and boxes > 0 and items_per_box is not None and items_per_box > 0
        return True

    def generate_parameters(self, input_value=None):
        question_type = random.choice([
            "residue_classes", "sum_constraint", "divisibility", "min_items"
        ])

        if question_type == "residue_classes":
            N = random.choice([100, 1000, 10000])
            m = random.randint(5, 20)
            return {"question_type": question_type, "N": N, "m": m}
        elif question_type == "sum_constraint":
            k = random.randint(10, 50)
            return {"question_type": question_type, "k": k}
        elif question_type == "divisibility":
            n = random.randint(10, 50)
            return {"question_type": question_type, "n": n}
        else:
            boxes = random.randint(2, 50)
            items_per_box = random.randint(2, 10)
            return {"question_type": question_type, "boxes": boxes, "items_per_box": items_per_box}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "residue_classes":
            N, m = params.get("N", 50), params.get("m", 10)
            answer = m + 1
            description = f"Minimum integers from [1,{N}] to guarantee two with same residue mod {m}: {answer}"
        elif question_type == "sum_constraint":
            k = params.get("k", 5)
            answer = k + 1
            description = f"Minimum from [1,{2*k}] to guarantee pair summing to {2*k+1}: {answer}"
        elif question_type == "divisibility":
            n = params.get("n", 10)
            answer = n + 1
            description = f"Minimum integers to guarantee two with quotient power of 2: {answer}"
        elif question_type == "min_overlap":
            items = params.get("items", 10)
            boxes = params.get("boxes", 5)
            answer = 0
            description = f"Minimum items guaranteed in some box: {answer}"
        else:
            boxes = params.get("boxes", 5)
            items_per_box = params.get("items_per_box", 2)
            answer = (items_per_box - 1) * boxes + 1
            description = f"Minimum items in {boxes} boxes to guarantee {items_per_box} in same box: {answer}"

        return MethodResult(
            value=answer,
            description=description,
            params=params,
            metadata={"question_type": question_type}
        )

    def _find_params_for_answer(self, target: int) -> Optional[Dict]:
        wrapped_target = target
        if target < 0:
            return None

        if target == 0:
            items = random.randint(1, 10)
            boxes = random.randint(items + 1, items + 20)
            return {"question_type": "min_overlap", "items": items, "boxes": boxes}

        if target == 1 or target < 2:
            return None

        for items_per_box in range(2, min(100, target + 1)):
            if (target - 1) % (items_per_box - 1) == 0:
                boxes = (target - 1) // (items_per_box - 1)
                if boxes >= 2:
                    return {"question_type": "min_items", "boxes": boxes, "items_per_box": items_per_box}

        if target >= 2:
            m = target - 1
            if target <= 50:
                N = 100
            elif target <= 500:
                N = 1000
            else:
                N = 10000
            return {"question_type": "residue_classes", "N": N, "m": m}

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

        if question_type == "residue_classes":
            N, m = params.get("N", 50), params.get("m", 10)
            relationship = (
                f"What is the minimum number of integers that must be chosen from "
                f"$\\{{1, 2, \\ldots, {N}\\}}$ to guarantee that at least two of them "
                f"are congruent modulo ${m}$?"
            )
        elif question_type == "sum_constraint":
            k = params.get("k", 5)
            relationship = (
                f"What is the minimum number of integers that must be chosen from "
                f"$\\{{1, 2, \\ldots, {2*k}\\}}$ to guarantee that at least two of them "
                f"sum to ${2*k + 1}$?"
            )
        elif question_type == "divisibility":
            n = params.get("n", 10)
            relationship = (
                f"What is the minimum number of integers that must be chosen from "
                f"$\\{{1, 2, \\ldots, {2*n}\\}}$ to guarantee that among the chosen integers, "
                f"there exist two whose quotient (larger divided by smaller) is a power of $2$?"
            )
        elif question_type == "min_overlap":
            items = params.get("items", 10)
            boxes = params.get("boxes", 5)
            relationship = (
                f"What is the minimum number of items that must be in some box when "
                f"distributing ${items}$ items into ${boxes}$ boxes?"
            )
        else:
            boxes = params.get("boxes", 5)
            items_per_box = params.get("items_per_box", 2)
            relationship = (
                f"What is the minimum number of items that must be placed into ${boxes}$ boxes "
                f"to guarantee that at least one box contains at least ${items_per_box}$ items?"
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
class DeepCatalan(MethodBlock):
    """
    Surface: Count valid parentheses, binary trees, triangulations, Dyck paths
    Hidden: All counted by Catalan numbers C_n = C(2n,n)/(n+1)
    """

    def __init__(self):
        super().__init__()
        self.name = "deep_catalan"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 4
        self.tags = ["combinatorics", "catalan", "deep_insight"]

    def _catalan(self, n):
        if n <= 1:
            return 1
        numerator = int(factorial(2 * n))
        denominator = int(factorial(n) * factorial(n + 1))
        return numerator // denominator

    def generate_parameters(self, input_value=None):
        question_type = random.choice([
            "triangulation", "dyck_paths", "binary_trees", "parentheses", "mountain_ranges"
        ])
        n = random.randint(3, 10)
        return {"question_type": question_type, "n": n}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")
        n = params.get("n", 10)
        answer = self._catalan(n)

        descriptions = {
            "triangulation": f"Triangulations of {n+2}-gon: C_{n} = {answer}",
            "dyck_paths": f"Dyck paths of length 2*{n}: C_{n} = {answer}",
            "binary_trees": f"Binary trees with {n} internal nodes: C_{n} = {answer}",
            "parentheses": f"Valid parentheses with {n} pairs: C_{n} = {answer}",
            "mountain_ranges": f"Mountain ranges with {n} upstrokes: C_{n} = {answer}"
        }

        return MethodResult(
            value=answer,
            description=descriptions.get(question_type, f"Catalan({n}) = {answer}"),
            params=params,
            metadata={"question_type": question_type, "n": n, "catalan": answer}
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
        n = params.get("n", 10)

        if question_type == "triangulation":
            relationship = (
                f"A convex polygon can be divided into triangles by drawing non-intersecting diagonals. "
                f"How many ways can a convex ${n+2}$-gon be triangulated?"
            )
        elif question_type == "dyck_paths":
            relationship = (
                f"A Dyck path is a lattice path from $(0, 0)$ to $(2n, 0)$ using steps $(1, 1)$ (up) "
                f"and $(1, -1)$ (down) that never goes below the $x$-axis. "
                f"How many Dyck paths are there from $(0, 0)$ to $({2*n}, 0)$?"
            )
        elif question_type == "binary_trees":
            relationship = (
                f"A full binary tree is a rooted tree where each internal node has exactly two children. "
                f"How many structurally distinct full binary trees have exactly ${n}$ internal nodes?"
            )
        elif question_type == "parentheses":
            relationship = (
                f"A sequence of ${2*n}$ parentheses is valid if at every position, "
                f"the number of opening parentheses is at least the number of closing parentheses. "
                f"How many valid sequences use exactly ${n}$ pairs of parentheses?"
            )
        else:
            relationship = (
                f"A mountain range is a path from $(0, 0)$ to $(2n, 0)$ using steps "
                f"$(1, 1)$ (upstroke) and $(1, -1)$ (downstroke) that never goes below the $x$-axis. "
                f"How many such mountain ranges exist with exactly ${n}$ upstrokes?"
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
class DeepStirling(MethodBlock):
    """
    Surface: Partition set of n elements into exactly k non-empty subsets
    Hidden: Stirling numbers of second kind S(n,k)
    """

    def __init__(self):
        super().__init__()
        self.name = "stirling"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 5
        self.tags = ["combinatorics", "stirling", "partitions", "deep_insight"]

    def _stirling_second(self, n, k):
        if n == 0 and k == 0:
            return 1
        if n == 0 or k == 0 or k > n:
            return 0
        if k == 1 or k == n:
            return 1

        table = [[0 for _ in range(k + 1)] for _ in range(n + 1)]
        table[0][0] = 1

        for i in range(1, n + 1):
            for j in range(1, min(i + 1, k + 1)):
                if j == 1:
                    table[i][j] = 1
                elif j == i:
                    table[i][j] = 1
                else:
                    table[i][j] = j * table[i-1][j] + table[i-1][j-1]

        return table[n][k]

    def generate_parameters(self, input_value=None):
        question_type = random.choice(["partition_count", "bell_number", "onto_functions"])

        if question_type == "partition_count":
            n = random.randint(5, 12)
            k = random.randint(2, min(n, 6))
            return {"question_type": question_type, "n": n, "k": k}
        elif question_type == "bell_number":
            n = random.randint(4, 10)
            return {"question_type": question_type, "n": n}
        else:
            n = random.randint(5, 10)
            k = random.randint(2, min(n, 5))
            return {"question_type": question_type, "n": n, "k": k}

    def compute(self, input_value, params):
        question_type = params.get("question_type", "compute")

        if question_type == "partition_count":
            n, k = params.get("n", 10), params.get("k", 5)
            answer = self._stirling_second(n, k)
            description = f"Partitions of {n} elements into {k} non-empty subsets: S({n},{k}) = {answer}"
        elif question_type == "bell_number":
            n = params.get("n", 10)
            bell = sum(self._stirling_second(n, k) for k in range(n + 1))
            answer = bell
            description = f"Bell number B_{n} = sum of S({n},k) = {answer}"
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            stirling = self._stirling_second(n, k)
            answer = int(factorial(k)) * stirling
            description = f"Onto functions from {n}-set to {k}-set: {k}! * S({n},{k}) = {answer}"

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

        if question_type == "partition_count":
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"In how many ways can a set of ${n}$ distinct elements be partitioned into "
                f"exactly ${k}$ non-empty, unlabeled subsets?"
            )
        elif question_type == "bell_number":
            n = params.get("n", 10)
            relationship = (
                f"The Bell number $B_n$ counts the number of ways to partition a set of $n$ elements "
                f"into any number of non-empty subsets. Find $B_{{{n}}}$."
            )
        else:
            n, k = params.get("n", 10), params.get("k", 5)
            relationship = (
                f"A function $f: A \\to B$ is onto (surjective) if every element of $B$ is the image "
                f"of at least one element of $A$. "
                f"How many onto functions exist from a set of ${n}$ elements to a set of ${k}$ elements?"
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
