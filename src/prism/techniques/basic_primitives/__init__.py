"""
Basic primitive mathematical methods for composable problem generation.

This module implements fundamental mathematical operations that are commonly used
as building blocks in more complex problem generation. These primitives cover:
- Basic arithmetic operations (sum, factorial, division, negate, subtract)
- Geometry utilities (midpoint, triangle area, centroid)
- Number theory utilities (divisors, perfect squares, polynomials)
- Integer math utilities (floor operations, counting)
"""

from .arithmetic import (
    Sum,
    Factorial,
    CeilDiv,
    DigitCount,
    Log2,
    BinaryRepresentation,
    BinaryPopcount,
    ClosestInteger,
    FloorTechnique,
    Power,
    Product,
    FloorDivision,
    Modulo,
    PowerOfTwo,
    PowerOfBase,
    NearestMultiple,
)
from .arithmetic_2 import (
    Negate,
    RoundToNearest,
    Add,
    Subtract,
    Multiply,
    AddOne,
    SubtractOne,
    Divide,
    Difference,
    HalfValue,
    AbsoluteDifference,
    AbsoluteValue,
    MinValue,
    MaxValue,
)
from .arithmetic_3 import (
    CompareValues,
    FilterOdd,
    FilterEven,
    RangeIntegers,
    ProductRange,
    Mod,
    Calculate100aPlusB,
    AddNumeratorDenominator,
    ExtractNumeratorDenominator,
    SumNumeratorDenominator,
    ConstantZero,
    ConstantOne,
)
from .geometry import (
    Midpoint,
    TriangleAreaHeron,
    PerpendicularBisector,
    LineIntersection,
    TriangleArea,
    TriangleCentroid,
    TriangleInequality,
    IsAcuteAngle,
    DivideArea,
)
from .number_theory import (
    DivisorList,
    DivisorPairs,
    PerfectSquareCheck,
    DivisibilityCheck,
    IsEven,
    IsOdd,
    IsPrime,
    ReduceFraction,
    DivisibleBy3,
    Divides,
)
from .number_theory_2 import (
    RationalReduce,
    DivisorsOf,
    FilterPrimes,
    BinaryExponentiation,
    FloorLog2,
    FloorSqrt,
    SqrtComputation,
    CeilSqrt,
    FloorCbrt,
)
from .sequences import (
    TriangularNumber,
    ArithmeticSequence,
    CountIntegerRange,
    CountSolutionsInInterval,
    Tribonacci,
    ArithmeticProgression,
    CollatzExtremal,
    DigitSumExtremal,
)
from .algebra import (
    PolynomialDiscriminant,
    PolynomialDegree,
    SubstitutionTransform,
    EvaluateAtZero,
    TestConstantPolynomial,
)
from .validators import (
    IsOpenSet,
    IsConnectedSpace,
    CandidateTest,
    CheckConvergenceInProbability,
    CheckIdenticalDistribution,
    CheckIndependenceProduct,
    ValidateTripleCondition,
    CheckConstructionExists,
)

from .ported_utils import (
    InductionSimplified,
    LessThan,
    VerifyEquation,
    Square,
    EqualTo,
    Powerset,
    GreaterThan,
    Gcd,
    FactorInteger,
    ParityCheck,
    IntegerLog,
    EquationSolutionCheck,
)

from .ported_utils_2 import (
    GreaterThanOrEqual,
    DeepStirling,
    ModularReduction,
    ExtractMnSum,
    BoardSize,
    GridSize,
    TotalSum,
    TotalPeople,
    SideLength,
    TotalVertices,
    MaxVal,
)

__all__ = [
    # Arithmetic
    'Sum', 'Factorial', 'CeilDiv', 'DigitCount', 'Log2',
    'BinaryRepresentation', 'BinaryPopcount', 'ClosestInteger',
    'FloorTechnique', 'Power', 'Product', 'FloorDivision', 'Modulo',
    'PowerOfTwo', 'PowerOfBase', 'NearestMultiple',
    'Negate', 'RoundToNearest', 'Add', 'Subtract', 'Multiply',
    'AddOne', 'SubtractOne', 'Divide', 'Difference', 'HalfValue',
    'AbsoluteDifference', 'AbsoluteValue', 'MinValue', 'MaxValue',
    'CompareValues', 'FilterOdd', 'FilterEven', 'RangeIntegers',
    'ProductRange', 'Mod', 'Calculate100aPlusB',
    'AddNumeratorDenominator', 'ExtractNumeratorDenominator',
    'SumNumeratorDenominator', 'ConstantZero', 'ConstantOne',
    # Geometry
    'Midpoint', 'TriangleAreaHeron', 'PerpendicularBisector',
    'LineIntersection', 'TriangleArea', 'TriangleCentroid',
    'TriangleInequality', 'IsAcuteAngle', 'DivideArea',
    # Number theory
    'DivisorList', 'DivisorPairs', 'PerfectSquareCheck',
    'DivisibilityCheck', 'IsEven', 'IsOdd', 'IsPrime',
    'ReduceFraction', 'DivisibleBy3', 'Divides',
    'RationalReduce', 'DivisorsOf', 'FilterPrimes',
    'BinaryExponentiation', 'FloorLog2', 'FloorSqrt',
    'SqrtComputation', 'CeilSqrt', 'FloorCbrt',
    # Sequences
    'TriangularNumber', 'ArithmeticSequence',
    'CountIntegerRange', 'CountSolutionsInInterval',
    'Tribonacci', 'ArithmeticProgression',
    'CollatzExtremal', 'DigitSumExtremal',
    # Algebra
    'PolynomialDiscriminant', 'PolynomialDegree',
    'SubstitutionTransform', 'EvaluateAtZero', 'TestConstantPolynomial',
    # Validators
    'IsOpenSet', 'IsConnectedSpace', 'CandidateTest',
    'CheckConvergenceInProbability', 'CheckIdenticalDistribution',
    'CheckIndependenceProduct', 'ValidateTripleCondition',
    'CheckConstructionExists',
    # Ported utils
    'InductionSimplified', 'LessThan', 'VerifyEquation', 'Square',
    'EqualTo', 'Powerset', 'GreaterThan', 'Gcd', 'FactorInteger',
    'ParityCheck', 'IntegerLog', 'EquationSolutionCheck',
    # Ported utils 2 - genuine methods
    'GreaterThanOrEqual', 'DeepStirling', 'ModularReduction', 'ExtractMnSum',
    # Ported utils 2 - passthrough stubs
    'BoardSize', 'GridSize', 'TotalSum', 'TotalPeople',
    'SideLength', 'TotalVertices', 'MaxVal',
]
