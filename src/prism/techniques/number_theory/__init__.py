"""
Number theory techniques module.

Re-exports all number theory techniques from submodules:
- divisibility: GCD, LCM, divisors, totient
- primes: Prime factorization, counting, testing
- modular: Modular arithmetic, CRT, Fermat/Euler
- sequences: Fibonacci, Lucas, Fermat numbers
- valuation: p-adic and Legendre valuations
- quadratic: Quadratic residues, Legendre/Jacobi symbols
- misc: Digit operations, special functions
"""

# Divisibility techniques
from .divisibility import (
    Totient,
    TotientInverse,
    DivisorCount,
    DivisorCountInverse,
    DivisorSum,
    DivisorSumInverse,
    GCDCompute,
    GCDInverseA,
    GCDInverseB,
    LCMCompute,
    LCMInverseA,
)

# Prime techniques
from .primes import (
    PrimeFactorization,
    PrimeCountingFunction,
    SmallestPrimeDivisor,
    PrimesBelow,
    MinPrimeSatisfyingConstraint,
    CheckPrimePowerExponent,
    SquareFreeDivisors,
    FilterByPrimeFactorCount,
    PrimesOfForm2kPlus1,
    DistinctPrimeDivisorsEqualsB,
    AllowedPrimeDivisors,
    NumberFromPrimePowers,
    Zsigmondy,
)

# Modular arithmetic techniques
from .modular import (
    ModularExponentiation,
    ModExpInverseBase,
    ModExpInverseExp,
    ModInverse,
    ChineseRemainderSolve,
    ChineseRemainderInverseResidue,
    FermatReduce,
    EulerReduce,
    SolveLinearCongruence,
)

# Modular order and primitive root techniques
from .modular_order import (
    MultiplicativeOrder,
    MultiplicativeOrderInverseA,
    PrimitiveRoot,
    MultiplicativeOrderModPrime,
    ModByPowerOfTen,
    ModByPrime,
    ModularPowerResult,
    ModularArithmeticAnalysis,
)

# Sequence techniques
from .sequences import (
    Fibonacci,
    FibonacciInverse,
    Lucas,
    LucasInverse,
    FibonacciGCD,
    FermatNumber,
    GoldenRatioPower,
    ExtractPQFromGoldenPower,
    LargePower,
    LargeFibonacci,
    LargePowerTower,
)

# Valuation techniques
from .valuation import (
    LegendreValuation,
    LegendreInverse,
    PadicValuation,
    PadicInverse,
    KummerValuation,
    KummerInverseN,
    KummerInverseK,
    LiftingExponentValuation,
    LiftingExponentInverseN,
    LTEEvenPower,
    MaxValuationUnderConstraint,
)

# Quadratic residue techniques
from .quadratic import (
    LegendreSymbol,
    SqrtModP,
    JacobiSymbol,
    CountQuadResidues,
    QuadraticResidueProductModP,
    CubicResidueSet,
    QuadraticReciprocity,
    FermatTwoSquares,
    SumOfSquaresCount,
    HenselLift,
    SolveQuadraticCongruence,
)

# Miscellaneous techniques
from .misc import (
    DigitSumBaseB,
    SumOfDigits,
    DigitSumInverse,
    PerfectSquaresUpTo,
    IsPerfectSquare,
    IsPerfectCube,
    CarmichaelLambda,
    CeilLog2Large,
    WilsonTheoremMod,
    FactorCommonTerm,
    CountLinearCombinationRange,
    BaseRepresentation,
    BaseDigitSumInverse,
    BaseDigitCount,
    GCDIteration,
    BaseDigitSum,
    DigitSum,
)

# Ported number theory techniques
from .ported_nt import (
    LcmMultiplicativeOrderBackward,
    SieveCount,
    CarmichaelConditionPrimes,
    FermatConditionPrimes,
)

__all__ = [
    # Divisibility
    "Totient",
    "TotientInverse",
    "DivisorCount",
    "DivisorCountInverse",
    "DivisorSum",
    "DivisorSumInverse",
    "GCDCompute",
    "GCDInverseA",
    "GCDInverseB",
    "LCMCompute",
    "LCMInverseA",
    # Primes
    "PrimeFactorization",
    "PrimeCountingFunction",
    "SmallestPrimeDivisor",
    "PrimesBelow",
    "MinPrimeSatisfyingConstraint",
    "CheckPrimePowerExponent",
    "SquareFreeDivisors",
    "FilterByPrimeFactorCount",
    "PrimesOfForm2kPlus1",
    "DistinctPrimeDivisorsEqualsB",
    "AllowedPrimeDivisors",
    "NumberFromPrimePowers",
    "Zsigmondy",
    # Modular
    "ModularExponentiation",
    "ModExpInverseBase",
    "ModExpInverseExp",
    "ModInverse",
    "ChineseRemainderSolve",
    "ChineseRemainderInverseResidue",
    "FermatReduce",
    "EulerReduce",
    "SolveLinearCongruence",
    "MultiplicativeOrder",
    "MultiplicativeOrderInverseA",
    "PrimitiveRoot",
    "MultiplicativeOrderModPrime",
    "ModByPowerOfTen",
    "ModByPrime",
    "ModularPowerResult",
    "ModularArithmeticAnalysis",
    # Sequences
    "Fibonacci",
    "FibonacciInverse",
    "Lucas",
    "LucasInverse",
    "FibonacciGCD",
    "FermatNumber",
    "GoldenRatioPower",
    "ExtractPQFromGoldenPower",
    "LargePower",
    "LargeFibonacci",
    "LargePowerTower",
    # Valuation
    "LegendreValuation",
    "LegendreInverse",
    "PadicValuation",
    "PadicInverse",
    "KummerValuation",
    "KummerInverseN",
    "KummerInverseK",
    "LiftingExponentValuation",
    "LiftingExponentInverseN",
    "LTEEvenPower",
    "MaxValuationUnderConstraint",
    # Quadratic
    "LegendreSymbol",
    "SqrtModP",
    "JacobiSymbol",
    "CountQuadResidues",
    "QuadraticResidueProductModP",
    "CubicResidueSet",
    "QuadraticReciprocity",
    "FermatTwoSquares",
    "SumOfSquaresCount",
    "HenselLift",
    "SolveQuadraticCongruence",
    # Misc
    "DigitSumBaseB",
    "SumOfDigits",
    "DigitSumInverse",
    "PerfectSquaresUpTo",
    "IsPerfectSquare",
    "IsPerfectCube",
    "CarmichaelLambda",
    "CeilLog2Large",
    "WilsonTheoremMod",
    "FactorCommonTerm",
    "CountLinearCombinationRange",
    "BaseRepresentation",
    "BaseDigitSumInverse",
    "BaseDigitCount",
    "GCDIteration",
    "BaseDigitSum",
    "DigitSum",
    # Ported NT
    "LcmMultiplicativeOrderBackward",
    "SieveCount",
    "CarmichaelConditionPrimes",
    "FermatConditionPrimes",
]
