"""
Deep Insight Techniques: Number Theory

Re-exports all number theory techniques from split modules:
- GCD Euclidean (Fibonacci worst case)
- Modular order (primitive roots)
- Quadratic residues (Legendre symbol)
- Lifting the exponent lemma
- Chinese Remainder Theorem
- Fermat's Little Theorem
- Wilson's Theorem
"""

from .number_theory_gcd import DeepGCDEuclidean
from .number_theory_modular import DeepModularOrder
from .number_theory_quadratic import DeepQuadraticResidue
from .number_theory_lte import DeepLiftingExponent
from .number_theory_crt import DeepChineseRemainder
from .number_theory_fermat import DeepFermatLittle
from .number_theory_wilson import DeepWilson

__all__ = [
    "DeepGCDEuclidean",
    "DeepModularOrder",
    "DeepQuadraticResidue",
    "DeepLiftingExponent",
    "DeepChineseRemainder",
    "DeepFermatLittle",
    "DeepWilson",
]
