"""
Probability and information theory methods.

This module contains:
- Probability (various probability scenarios)
- ConditionalProbability
- EntropyCompute (Shannon entropy)
- ConditionalExpectationSymmetric
"""

import random
import math
from typing import Any, Dict, Optional
from fractions import Fraction
from ..base import MethodBlock, MethodResult
from ..registry import register_technique


@register_technique
class Probability(MethodBlock):
    """Compute probability P(event) for various scenarios.

    Scenarios:
    - dice_sum: Probability of specific sum/outcome with dice
    - coins: Probability of k heads in n flips
    - cards_same_suit: Probability of drawing specific cards
    - at_least_one: Probability problems with complement
    """
    def __init__(self):
        super().__init__()
        self.name = "probability"
        self.input_type = "integer"
        self.output_type = "fraction"
        self.difficulty = 3
        self.tags = ["combinatorics", "probability", "counting"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        scenarios = ["dice_sum", "coins", "cards_same_suit", "at_least_one"]
        scenario = random.choice(scenarios)
        if scenario == "dice_sum":
            target_sum = random.randint(2, 12)
            return {"scenario": "dice_sum", "target_sum": target_sum}
        elif scenario == "coins":
            n = input_value if input_value is not None else random.randint(3, 10)
            k = random.randint(0, n)
            return {"scenario": "coins", "n": n, "k": k}
        elif scenario == "cards_same_suit":
            return {"scenario": "cards_same_suit"}
        else:
            n = input_value if input_value is not None else random.randint(2, 6)
            return {"scenario": "at_least_one", "n": n, "target": 6}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        scenario = params.get("scenario", "coins")

        if scenario == "dice_sum":
            target_sum = params.get("target_sum", 7)
            count = 0
            for d1 in range(1, 7):
                d2 = target_sum - d1
                if 1 <= d2 <= 6:
                    count += 1
            prob = Fraction(count, 36)
            result = (prob.numerator, prob.denominator)
            description = f"P(sum of two dice = {target_sum}) = {count}/36 = {prob}"

        elif scenario == "coins":
            n = params.get("n", input_value if input_value is not None else 5)
            k = params.get("k", 2)
            numerator = math.comb(n, k)
            denominator = 2 ** n
            prob = Fraction(numerator, denominator)
            result = (prob.numerator, prob.denominator)
            description = f"P(exactly {k} heads in {n} flips) = C({n},{k})/2^{n} = {numerator}/{denominator} = {prob}"

        elif scenario == "cards_same_suit":
            numerator = 4 * math.comb(13, 2)
            denominator = math.comb(52, 2)
            prob = Fraction(numerator, denominator)
            result = (prob.numerator, prob.denominator)
            description = f"P(two cards same suit) = 4*C(13,2)/C(52,2) = {numerator}/{denominator} = {prob}"

        elif scenario == "at_least_one":
            n = params.get("n", input_value if input_value is not None else 3)
            target = params.get("target", 6)
            numerator = 6 ** n - 5 ** n
            denominator = 6 ** n
            prob = Fraction(numerator, denominator)
            result = (prob.numerator, prob.denominator)
            description = f"P(at least one {target} in {n} dice) = 1 - (5/6)^{n} = {numerator}/{denominator} = {prob}"

        else:
            prob = Fraction(1, 2)
            result = (1, 2)
            description = f"Probability = 1/2"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"scenario": scenario, "probability": str(prob)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ConditionalProbability(MethodBlock):
    """Compute conditional probability P(A|B) = P(A and B) / P(B)."""
    def __init__(self):
        super().__init__()
        self.name = "conditional_probability"
        self.input_type = "integer"
        self.output_type = "fraction"
        self.difficulty = 3
        self.tags = ["combinatorics", "probability", "conditional"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        scenario = random.choice(["dice", "coins", "cards"])
        if scenario == "dice":
            threshold = random.randint(7, 10)
            target_sum = random.randint(threshold, 12)
            return {"scenario": "dice", "threshold": threshold, "target_sum": target_sum}
        elif scenario == "coins":
            n = input_value if input_value is not None else random.randint(4, 8)
            k = random.randint(1, n - 1)
            m = random.randint(k, n)
            return {"scenario": "coins", "n": n, "k": k, "m": m}
        else:
            return {"scenario": "cards"}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        scenario = params.get("scenario", "coins")

        if scenario == "dice":
            threshold = params.get("threshold", 7)
            target_sum = params.get("target_sum", 7)
            def count_sum(s):
                count = 0
                for d1 in range(1, 7):
                    d2 = s - d1
                    if 1 <= d2 <= 6:
                        count += 1
                return count
            p_b = sum(count_sum(s) for s in range(threshold, 13))
            p_a_and_b = count_sum(target_sum) if target_sum >= threshold else 0
            prob = Fraction(p_a_and_b, p_b) if p_b > 0 else Fraction(0)
            result = (prob.numerator, prob.denominator)
            description = f"P(sum={target_sum} | sum>={threshold}) = {p_a_and_b}/{p_b} = {prob}"

        elif scenario == "coins":
            n = params.get("n", input_value if input_value is not None else 5)
            k = params.get("k", 2)
            m = params.get("m", 3)
            p_b_num = sum(math.comb(n, i) for i in range(k, n + 1))
            p_a_and_b_num = math.comb(n, m) if m >= k else 0
            prob = Fraction(p_a_and_b_num, p_b_num) if p_b_num > 0 else Fraction(0)
            result = (prob.numerator, prob.denominator)
            description = f"P(exactly {m} heads | at least {k} heads in {n} flips) = {p_a_and_b_num}/{p_b_num} = {prob}"

        else:
            prob = Fraction(1, 2)
            result = (1, 2)
            description = f"P(hearts | red card) = 13/26 = {prob}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"scenario": scenario, "probability": str(prob)}
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class EntropyCompute(MethodBlock):
    """Compute Shannon entropy H(X) = -sum p(x) * log2(p(x)) in bits."""

    def __init__(self):
        super().__init__()
        self.name = "entropy_compute"
        self.input_type = "none"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "information_theory", "entropy", "probability"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        distribution_type = random.choice(["uniform", "binary", "custom"])

        if distribution_type == "uniform":
            n = random.randint(2, 16)
            probabilities = [1.0 / n] * n
            return {"distribution_type": "uniform", "n": n, "probabilities": probabilities}

        elif distribution_type == "binary":
            p_numerator = random.randint(1, 9)
            p_denominator = 10
            p = p_numerator / p_denominator
            probabilities = [p, 1 - p]
            return {
                "distribution_type": "binary",
                "p_numerator": p_numerator,
                "p_denominator": p_denominator,
                "probabilities": probabilities
            }

        else:
            n = random.randint(3, 5)
            raw_values = [random.randint(1, 10) for _ in range(n)]
            total = sum(raw_values)
            probabilities = [v / total for v in raw_values]
            return {"distribution_type": "custom", "n": n, "probabilities": probabilities}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        probabilities = params.get("probabilities", [0.5, 0.5])
        distribution_type = params.get("distribution_type", "binary")

        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy += -p * math.log2(p)

        result = int(round(entropy * 1000))

        if distribution_type == "uniform":
            n = params.get("n", len(probabilities))
            description = f"H(X) for uniform distribution on {n} outcomes: {entropy:.6f} bits -> {result}"
        elif distribution_type == "binary":
            p_num = params.get("p_numerator", 5)
            p_den = params.get("p_denominator", 10)
            description = f"H(X) for binary distribution p={p_num}/{p_den}: {entropy:.6f} bits -> {result}"
        else:
            n = len(probabilities)
            description = f"H(X) for distribution on {n} outcomes: {entropy:.6f} bits -> {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={
                "entropy_bits": entropy,
                "entropy_scaled": result,
                "distribution_type": distribution_type,
                "num_outcomes": len(probabilities),
                "formula": "-sum p*log2(p)"
            }
        )

    def can_invert(self) -> bool:
        return False


@register_technique
class ConditionalExpectationSymmetric(MethodBlock):
    """Compute conditional expectation E[X|Y] for symmetric distributions."""

    def __init__(self):
        super().__init__()
        self.name = "conditional_expectation_symmetric"
        self.input_type = "integer"
        self.output_type = "integer"
        self.difficulty = 3
        self.tags = ["combinatorics", "probability", "expectation", "conditional"]

    def generate_parameters(self, input_value: Optional[Any] = None) -> Dict[str, Any]:
        distribution_case = random.choice(["independent", "symmetric_joint"])

        if distribution_case == "independent":
            a = random.randint(0, 20)
            b = a + random.randint(10, 50)
            c = random.randint(0, 20)
            d = c + random.randint(10, 50)
            return {"case": "independent", "X_min": a, "X_max": b, "Y_min": c, "Y_max": d}
        else:
            mu_X = random.randint(20, 100)
            mu_Y = random.randint(20, 100)
            y_value = random.randint(0, 200)
            return {"case": "symmetric_joint", "mu_X": mu_X, "mu_Y": mu_Y, "y_value": y_value}

    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        case = params.get("case", "independent")

        if case == "independent":
            X_min = params.get("X_min", 0)
            X_max = params.get("X_max", 10)
            expectation = (X_min + X_max) / 2.0
            result = int(round(expectation * 100))
            description = f"E[X|Y] = E[X] = ({X_min}+{X_max})/2 = {expectation:.2f} (independent) -> {result}"
        else:
            mu_X = params.get("mu_X", 50)
            y_value = params.get("y_value", 50)
            expectation = mu_X
            result = int(round(expectation * 100))
            description = f"E[X|Y={y_value}] = mu_X = {mu_X} (symmetric joint distribution) -> {result}"

        return MethodResult(
            value=result,
            description=description,
            params=params,
            metadata={"case": case, "expectation": expectation, "expectation_scaled": result}
        )

    def can_invert(self) -> bool:
        return False
