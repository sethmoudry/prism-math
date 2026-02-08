"""
Deep Insights Techniques - Problems requiring structural discovery.

Unlike shallow problems ("compute f(15) given f(1)=3"), deep problems ask questions
that REQUIRE discovering the hidden mathematical structure to answer.

Categories:
- functional_equations: Cauchy, Jensen, multiplicative-to-additive, idempotent, involution, cyclic
- combinatorics: Catalan, Stirling, derangements, Burnside, pigeonhole
- sequences: Fibonacci divisibility, linear recurrence, Beatty, Pell, Collatz
- number_theory: GCD Euclidean, modular order, quadratic residues, LTE, CRT, Fermat, Wilson

Total: 24 registered techniques
"""

# Functional equations (6 techniques) - split into cauchy and cyclic files
from .functional_equations_cauchy import (
    DeepCauchyAdditiveMultiplicative,
    DeepJensenFunctional,
    DeepMultiplicativeToAdditive,
)
from .functional_equations_cyclic import (
    DeepIdempotent,
    DeepInvolution,
    DeepCyclicFunctional,
)

# Combinatorics (5 techniques) - split into basic and advanced files
from .combinatorics_basic import (
    DeepPigeonhole,
    DeepCatalan,
    DeepStirling,
)
from .combinatorics_advanced import (
    DeepDerangements,
    DeepBurnside,
)

# Sequences (5 techniques) - split into fibonacci and advanced files
from .sequences_fibonacci import (
    DeepFibonacciDivisibility,
    DeepLinearRecurrence,
    DeepBeattySequence,
)
from .sequences_advanced import (
    DeepPellEquation,
    DeepCollatzLike,
)

# Number theory (8 techniques)
from .number_theory import (
    DeepGCDEuclidean,
    DeepModularOrder,
    DeepQuadraticResidue,
    DeepLiftingExponent,
    DeepChineseRemainder,
    DeepFermatLittle,
    DeepWilson,
)

# List of all deep insight technique classes
DEEP_INSIGHT_TECHNIQUES = [
    # Functional equations
    DeepCauchyAdditiveMultiplicative,
    DeepJensenFunctional,
    DeepMultiplicativeToAdditive,
    DeepIdempotent,
    DeepInvolution,
    DeepCyclicFunctional,
    # Combinatorics
    DeepPigeonhole,
    DeepCatalan,
    DeepStirling,
    DeepDerangements,
    DeepBurnside,
    # Sequences
    DeepFibonacciDivisibility,
    DeepLinearRecurrence,
    DeepBeattySequence,
    DeepPellEquation,
    DeepCollatzLike,
    # Number theory
    DeepGCDEuclidean,
    DeepModularOrder,
    DeepQuadraticResidue,
    DeepLiftingExponent,
    DeepChineseRemainder,
    DeepFermatLittle,
    DeepWilson,
]


def generate_deep_insight_problem():
    """Generate a random deep insight problem."""
    import random
    technique_class = random.choice(DEEP_INSIGHT_TECHNIQUES)
    technique = technique_class()
    return technique.generate()


def generate_deep_insight_problems(count=10):
    """Generate multiple deep insight problems."""
    problems = []
    for _ in range(count):
        problems.append(generate_deep_insight_problem())
    return problems


__all__ = [
    # Functional equations
    "DeepCauchyAdditiveMultiplicative",
    "DeepJensenFunctional",
    "DeepMultiplicativeToAdditive",
    "DeepIdempotent",
    "DeepInvolution",
    "DeepCyclicFunctional",
    # Combinatorics
    "DeepPigeonhole",
    "DeepCatalan",
    "DeepStirling",
    "DeepDerangements",
    "DeepBurnside",
    # Sequences
    "DeepFibonacciDivisibility",
    "DeepLinearRecurrence",
    "DeepBeattySequence",
    "DeepPellEquation",
    "DeepCollatzLike",
    # Number theory
    "DeepGCDEuclidean",
    "DeepModularOrder",
    "DeepQuadraticResidue",
    "DeepLiftingExponent",
    "DeepChineseRemainder",
    "DeepFermatLittle",
    "DeepWilson",
    # Utility
    "DEEP_INSIGHT_TECHNIQUES",
    "generate_deep_insight_problem",
    "generate_deep_insight_problems",
]
