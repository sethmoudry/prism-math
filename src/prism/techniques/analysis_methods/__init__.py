"""
Analysis and limits methods for composable problem generation.

This module implements mathematical analysis and limit-related operations:
- Limit computations (integral expressions, ratios, sequences)
- Asymptotic analysis (tail sums, correction terms)
- Convergence analysis (dominated convergence, pointwise limits)
- Bound computations (integrand bounds, supremum)
"""

from .limits import (
    LimitIntegralExpression,
    AnalyzeLimit,
    ComposeLimits,
    LimitOfReciprocal,
    LimitAtInfinity,
    LimitExistenceConclusion,
    LimitImpliesConstant,
    LimitCombinationFromLiminfLimsup,
    LiminfConclusion,
    LimitOfProduct,
)
from .limits_2 import (
    LimitAsNGoesToInfinity,
    NthRootLimitBounds,
    NestedSqrtLimitSequence,
    LimitPowerSeries,
    RatioAnalysisLimit,
    SequenceExtraction,
    DominantTermSelection,
)
from .asymptotics import (
    AsymptoticTailSum,
    CorrectionTerm,
    AsymptoticIntegralBehavior,
    SineIntegralAsymptotic,
    CosineIntegralAsymptotic,
    SubstituteAsymptotics,
    MatchAsymptoticCoefficient,
)
from .asymptotics_2 import (
    AsymptoticExpansionSi,
    AsymptoticExpansionCi,
    AsymptoticLeadingTerm,
    LogarithmicLimitTransform,
    AsymptoticRatioComparison,
    ExponentComparison,
)
from .convergence import (
    SupremumLimit,
    DominatedConvergenceLimit,
    PointwiseLimitAnalysis,
    IntegrandDominatedBound,
    ConcludeNormLimit,
    MutualSingularityLimit,
    IntegralConvergenceTest,
    SupremumNormBound,
)
from .convergence_2 import (
    InfimumOfConvergenceRegion,
    SupremumOfBallIntegrals,
    FunctionSupremum,
    WeakStarNullConvergence,
)
from .series import (
    CombineTerms,
    HarmonicSumMinimal,
    AlternatingHarmonicSeriesSum,
    PowerSeriesExpansionLog,
    HarmonicSeriesDivergence,
    LogarithmExpansion,
)
from .misc import (
    MaxValueFromCases,
    MaxRatioOverConfigurations,
    CharacteristicEquationRoots,
    ParticularSolutionUndeterminedCoeffs,
    ConvexFunctionProperties,
    IsZeroRing,
    ItoIntegralExpectation,
    TestFeasibilityK,
)

# Ported analysis methods
from .ported_analysis import (
    GammaFunction,
    ComputeUpperBound,
    ComputeLowerBound,
    SqueezeTheoremMethod,
    TailSumBound,
    SubharmonicSphericalAverageLimit,
    ConvolutionLimitFromSphericalAverage,
    AndoProjectionBound,
    MeasureZeroConclusion,
    MeasureCoverEpsilon,
    TrigonometricZeroSetSin,
    TrigonometricZeroSetCos,
    BirkhoffErgodicTheorem,
    TopCohomologyNoncompactManifold,
    ContinuousProcessJumpSum,
    LogSymmetryIntegral,
    ConstantFunctionDerivative,
    LambertWSeries,
    FiniteVariationMartingalePart,
)

__all__ = [
    # Limits
    'LimitIntegralExpression', 'AnalyzeLimit', 'ComposeLimits',
    'LimitOfReciprocal', 'LimitAtInfinity', 'LimitExistenceConclusion',
    'LimitImpliesConstant', 'LimitCombinationFromLiminfLimsup',
    'LiminfConclusion', 'LimitOfProduct',
    'LimitAsNGoesToInfinity', 'NthRootLimitBounds', 'NestedSqrtLimitSequence',
    'LimitPowerSeries', 'RatioAnalysisLimit', 'SequenceExtraction',
    'DominantTermSelection',
    # Asymptotics
    'AsymptoticTailSum', 'CorrectionTerm', 'AsymptoticIntegralBehavior',
    'SineIntegralAsymptotic', 'CosineIntegralAsymptotic',
    'SubstituteAsymptotics', 'MatchAsymptoticCoefficient',
    'AsymptoticExpansionSi', 'AsymptoticExpansionCi', 'AsymptoticLeadingTerm',
    'LogarithmicLimitTransform', 'AsymptoticRatioComparison', 'ExponentComparison',
    # Convergence
    'SupremumLimit', 'DominatedConvergenceLimit', 'PointwiseLimitAnalysis',
    'IntegrandDominatedBound', 'ConcludeNormLimit', 'MutualSingularityLimit',
    'IntegralConvergenceTest', 'SupremumNormBound',
    'InfimumOfConvergenceRegion', 'SupremumOfBallIntegrals',
    'FunctionSupremum', 'WeakStarNullConvergence',
    # Series
    'CombineTerms', 'HarmonicSumMinimal', 'AlternatingHarmonicSeriesSum',
    'PowerSeriesExpansionLog', 'HarmonicSeriesDivergence', 'LogarithmExpansion',
    # Misc
    'MaxValueFromCases', 'MaxRatioOverConfigurations',
    'CharacteristicEquationRoots', 'ParticularSolutionUndeterminedCoeffs',
    'ConvexFunctionProperties', 'IsZeroRing', 'ItoIntegralExpectation',
    'TestFeasibilityK',
    # Ported Analysis
    'GammaFunction',
    'ComputeUpperBound',
    'ComputeLowerBound',
    'SqueezeTheoremMethod',
    'TailSumBound',
    'SubharmonicSphericalAverageLimit',
    'ConvolutionLimitFromSphericalAverage',
    'AndoProjectionBound',
    'MeasureZeroConclusion',
    'MeasureCoverEpsilon',
    'TrigonometricZeroSetSin',
    'TrigonometricZeroSetCos',
    'BirkhoffErgodicTheorem',
    'TopCohomologyNoncompactManifold',
    'ContinuousProcessJumpSum',
    'LogSymmetryIntegral',
    'ConstantFunctionDerivative',
    'LambertWSeries',
    'FiniteVariationMartingalePart',
]
