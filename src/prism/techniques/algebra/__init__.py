"""
Algebra techniques for composable problem generation.

Contains 62+ algebra method blocks organized into modules:
- vieta.py: Vieta's formulas (3) + Newton's identities (2)
- polynomials.py: Polynomial operations (6) + Cyclotomic (3) + Laurent (1)
- equations.py: Recurrences (5) + Functional equations (4)
- quadratics.py: Quadratic functions (5) + Exponential/Diophantine (2)
- inequalities.py: Optimization/inequalities (6) + Floor/ceiling (5)
- functions.py: Series (5) + Advanced (7)
- misc.py: Matrix operations (6) + Curves/Cubics (2)
- stubs.py: Stub techniques (11) + Utilities (2)
- expressions.py: Expression manipulation (10)
"""

# Vieta and Newton's identities
from .vieta import (
    VietaSum,
    VietaProduct,
    VietaInverseCoeffs,
    NewtonPowerSum,
    NewtonInverseElemSym,
)

# Polynomial-related techniques
from .polynomials import (
    PolynomialEvaluation,
    PolynomialEvaluationInverseX,
    PolynomialDivision,
    PolynomialDifference,
    PolynomialFactorization,
    PolynomialRoots,
    CyclotomicEval,
    CyclotomicInverseN,
    CyclotomicFactorization,
    LaurentPolynomial,
)

# Equation-solving techniques
from .equations import (
    LinearRecurrence,
    LinearRecurrenceInverseN,
    CharPolynomial,
    BinetFormula,
    Recurrence,
    FunctionalEqCauchy,
    FunctionalEqMultiplicative,
    FunctionalEqPower,
    FunctionalEqAddMult,
)

# Quadratic functions and exponential/diophantine equations
from .quadratics import (
    QuadraticVertex,
    QuadraticAxisOfSymmetry,
    QuadraticDiscriminant,
    QuadraticRootCount,
    QuadraticFactoring,
    ExponentialEquation,
    DiophantineEquations,
)

# Inequality and optimization techniques
from .inequalities import (
    ArithmeticGeometricMeanMin,
    ArithmeticGeometricMeanInverseVars,
    CauchySchwarzBound,
    HolderInequality,
    JensenInequality,
    LagrangeOptimize,
    FloorSum,
    FloorSumInverseN,
    FloorDiv,
    FloorDivInverseA,
    HermiteIdentity,
)

# Series, sequence, and function techniques
from .functions import (
    ArithmeticSum,
    ArithmeticSumInverseN,
    SumRange,
    GeometricSum,
    GeometricSumInverseN,
    Telescoping,
    PartialFractions,
)

# Advanced function techniques
from .advanced_functions import (
    GoldenRatioIdentity,
    OrdinaryGeneratingFunction,
    FibonacciIdentity,
    SeriesApproximation,
    SummationManipulation,
)

# Matrix operations and curves/cubics
from .misc import (
    MatrixPower,
    MatrixPowerInverseN,
    MatrixDet,
    MatrixTrace,
    Determinant2x2,
    VectorDotProductFromDiagonals,
    CurveIntersection,
    NamedCubicsProperties,
)

# Stub techniques and utilities
from .stubs import (
    AlgebraicManipulation,
    AlgebraicSimplification,
    BinomialCoefficient,
    BinomialCoefficients,
    FunctionalEquation,
    PolynomialEvalExtended,
    RootsOfUnitySum,
    Substitution,
    Summation,
    SummationOfLengths,
    CoefficientSolution,
    CompareLhsRhs,
)

# Expression manipulation
from .expressions import (
    SquareSumSimplify,
    ExpandSquareSum,
    ExpandDifference,
    SeparateExpressionParts,
    ExtractMN,
    ExtractMNSum,
    QuotientOfSums,
    SimplifyDotProduct,
    ExtractSqrtABCParameters,
)

# Ported algebra methods
from .ported_algebra import (
    AlgebraicExpand,
    EvaluateFunctionComposition,
    ComputeFunctionIterate,
    ApplyFunctionalCondition,
    DefinePiecewiseFunction,
    ExpressInForm,
    VietaAdvanced,
    ComplexNumbersMethod,
    ContradictionArgument,
    FloorSystemSolutions,
    SequenceSum,
    SineMethod,
    ArccosineMethod,
    AbsoluteValueAnalysis,
    RandomInteger,
    PowerTower,
    FactorialTower,
    InequalityConstraints,
)

# Ported algebra methods batch 2 (advanced/geometry/misc)
from .ported_algebra_2 import (
    CompositumDegree,
    CollatzProcess,
    HyperovalSizeProjective,
    ProjectiveLineCurve,
    MaxArcSizeAffineWithTranslation,
    MaxArcSizeAffine,
    GrothendieckVectorBundleSplitting,
    SubfieldDegreeFromSubgroupIndex,
    KrullDimensionFromTranscendenceDegree,
    LargestNForInjectiveEvaluation,
    QuadraticFormSolutionCount,
    EulerPhiValues,
    DivisibilityPowerPlusOne,
    ArctanDiophantine,
    SumOfCubesAPExpansion,
    CurrentTimeFromPast,
    FutureTime,
)

# Export all classes
__all__ = [
    # Vieta
    "VietaSum",
    "VietaProduct",
    "VietaInverseCoeffs",
    # Newton
    "NewtonPowerSum",
    "NewtonInverseElemSym",
    # Polynomials
    "PolynomialEvaluation",
    "PolynomialEvaluationInverseX",
    "PolynomialDivision",
    "PolynomialDifference",
    "PolynomialFactorization",
    "PolynomialRoots",
    "CyclotomicEval",
    "CyclotomicInverseN",
    "CyclotomicFactorization",
    "LaurentPolynomial",
    # Recurrences
    "LinearRecurrence",
    "LinearRecurrenceInverseN",
    "CharPolynomial",
    "BinetFormula",
    "Recurrence",
    # Functional equations
    "FunctionalEqCauchy",
    "FunctionalEqMultiplicative",
    "FunctionalEqPower",
    "FunctionalEqAddMult",
    "FunctionalEquation",
    # Quadratic
    "QuadraticVertex",
    "QuadraticAxisOfSymmetry",
    "QuadraticDiscriminant",
    "QuadraticRootCount",
    "QuadraticFactoring",
    # Equations
    "ExponentialEquation",
    "DiophantineEquations",
    # Inequalities/Optimization
    "ArithmeticGeometricMeanMin",
    "ArithmeticGeometricMeanInverseVars",
    "CauchySchwarzBound",
    "HolderInequality",
    "JensenInequality",
    "LagrangeOptimize",
    # Floor/Ceiling
    "FloorSum",
    "FloorSumInverseN",
    "FloorDiv",
    "FloorDivInverseA",
    "HermiteIdentity",
    # Series
    "ArithmeticSum",
    "ArithmeticSumInverseN",
    "SumRange",
    "GeometricSum",
    "GeometricSumInverseN",
    # Advanced
    "Telescoping",
    "PartialFractions",
    "GoldenRatioIdentity",
    "OrdinaryGeneratingFunction",
    "FibonacciIdentity",
    "SeriesApproximation",
    "SummationManipulation",
    # Matrix
    "MatrixPower",
    "MatrixPowerInverseN",
    "MatrixDet",
    "MatrixTrace",
    "Determinant2x2",
    "VectorDotProductFromDiagonals",
    # Curves/Cubics
    "CurveIntersection",
    "NamedCubicsProperties",
    # Stubs/Misc
    "AlgebraicManipulation",
    "AlgebraicSimplification",
    "BinomialCoefficient",
    "BinomialCoefficients",
    "PolynomialEvalExtended",
    "RootsOfUnitySum",
    "Substitution",
    "Summation",
    "SummationOfLengths",
    # Expression manipulation
    "SquareSumSimplify",
    "ExpandSquareSum",
    "ExpandDifference",
    "SeparateExpressionParts",
    "ExtractMN",
    "ExtractMNSum",
    "QuotientOfSums",
    "SimplifyDotProduct",
    "ExtractSqrtABCParameters",
    # Utility
    "CoefficientSolution",
    "CompareLhsRhs",
    # Ported Algebra
    "AlgebraicExpand",
    "EvaluateFunctionComposition",
    "ComputeFunctionIterate",
    "ApplyFunctionalCondition",
    "DefinePiecewiseFunction",
    "ExpressInForm",
    "VietaAdvanced",
    "ComplexNumbersMethod",
    "ContradictionArgument",
    "FloorSystemSolutions",
    "SequenceSum",
    "SineMethod",
    "ArccosineMethod",
    "AbsoluteValueAnalysis",
    "RandomInteger",
    "PowerTower",
    "FactorialTower",
    "InequalityConstraints",
    # Ported Algebra 2
    "CompositumDegree",
    "CollatzProcess",
    "HyperovalSizeProjective",
    "ProjectiveLineCurve",
    "MaxArcSizeAffineWithTranslation",
    "MaxArcSizeAffine",
    "GrothendieckVectorBundleSplitting",
    "SubfieldDegreeFromSubgroupIndex",
    "KrullDimensionFromTranscendenceDegree",
    "LargestNForInjectiveEvaluation",
    "QuadraticFormSolutionCount",
    "EulerPhiValues",
    "DivisibilityPowerPlusOne",
    "ArctanDiophantine",
    "SumOfCubesAPExpansion",
    "CurrentTimeFromPast",
    "FutureTime",
]
