"""
Combinatorics techniques for mathematical problem solving.

This package contains combinatorial methods organized into:
- counting_basic.py: Basic counting, binomial coefficients, Catalan numbers, lattice paths
- counting_advanced.py: Pigeonhole principle, bijections
- special_numbers_stirling_bell.py: Stirling numbers, Bell numbers
- special_numbers_derangements_partitions.py: Derangements, partitions, explosive growth
- graphs_basic.py: Basic graph theory (chromatic, spanning trees, Eulerian, Hamiltonian, etc.)
- graphs_flow.py: Flow bounds and Platonic Eulerian circuits
- generating_functions.py: Generating function methods, hook length formula
- probability.py: Probability and information theory methods
- topology.py: Topological complexity methods
- misc.py: Counting constraints, permutation methods, multiset permutations, etc.
"""

# Re-export from counting_basic.py
from .counting_basic import (
    Catalan,
    CatalanInverse,
    Binomial,
    BinomialInverseN,
    BinomialInverseK,
    BinomialSum,
    LatticePaths,
    LatticePathsInverseM,
    Ballot,
    DyckPaths,
    Counting,
    CountingPrinciples,
)

# Re-export from counting_advanced.py
from .counting_advanced import (
    Pigeonhole,
    PigeonholeGeneralized,
    DoubleCount,
    Bijection,
    CountingBijection,
)

# Re-export from special_numbers_stirling_bell.py
from .special_numbers_stirling_bell import (
    # Helper functions
    stirling_first,
    stirling_second,
    # Stirling numbers
    StirlingFirst,
    StirlingSecond,
    StirlingSecondInverseN,
    Bell,
)

# Re-export from special_numbers_derangements_partitions.py
from .special_numbers_derangements_partitions import (
    # Helper functions
    derangement,
    partition_count,
    partition_k_parts,
    # Derangements
    Derangement,
    DerangementInverse,
    Subfactorial,
    # Partitions
    Partition,
    PartitionInverse,
    PartitionKParts,
    IntegerPartitionsIntoK,
    # Large values (explosive growth)
    LargeFactorial,
    LargeBinomial,
    LargeCatalan,
    LargeStirling,
    # Binary trees
    BinaryTreeCountRecurrence,
)

# Re-export from graphs_basic.py
from .graphs_basic import (
    ChromaticPoly,
    GraphColoring,
    SpanningTrees,
    EulerianPaths,
    HallMarriage,
    GraphTheory,
    HamiltonianPath,
    TuranBound,
)

# Re-export from graphs_flow.py
from .graphs_flow import (
    SeymourFlowBound,
    MaxPowerOfTwoBelowBound,
    MaxMSatisfyingFlowBound,
    PlatonicEulerianCircuits,
)

# Re-export from generating_functions.py
from .generating_functions import (
    GeneratingFunction,
    CountingViaAlgebra,
    BinomialTheoremAnalogy,
    HookLengthFormula,
)

# Re-export from probability.py
from .probability import (
    Probability,
    ConditionalProbability,
    EntropyCompute,
    ConditionalExpectationSymmetric,
)

# Re-export from topology.py
from .topology import (
    TopologicalComplexityGraph,
    BouquetOfCircles,
    KleitmanRothschildHeightBound,
    EssentialVertexCount,
    FigureEightTopology,
)

# Re-export from misc.py
from .misc import (
    # Counting constraints
    CountingConstraints,
    CountingIntersections,
    CountingPaths,
    CountingRegions,
    # Permutation methods
    PermutationDecomposition,
    PermutationWithRepetition,
    PermutationOrder,
    # Multiset permutations
    MultisetPermutationTotal,
    MultisetPermutationAvoidingPattern,
    # Rectangle counting
    MinAreaFunction,
    SumMinAreas,
    # Legendre valuation
    LegendreValuationFactorial,
    # Utility techniques
    PlaceValueSum,
    FactorialBaseToDecimal,
    CountOdd,
    CountEven,
    CircularArrangementsWithAdjacentConstraint,
    # Helper functions
    multinomial_coefficient,
    count_non_overlapping_placements,
)

# Re-export from ported_comb.py
from .ported_comb import (
    LinearityOfExpectation,
    ProbabilisticMethod,
    OptionalStoppingTheorem,
    InclusionExclusion,
    DerangementViaInclusionExclusion,
    ExtremalConstruction,
    GraphArboricityBound,
    ChromaticNumberBound,
    RamseyBound,
)

# Re-export from ported_comb_2.py
from .ported_comb_2 import (
    TriangularNumbers,
    ArithmeticSumCubesExpansion,
    CyclotomicSubsetCount,
    CountCoprimePairs,
    MultiplicativePermutation,
    ComputeNimValue,
    IdentifyWinningPosition,
    DetermineOptimalMove,
    DefineGameStrategy,
    SimulateJumps,
    ConstructIsomorphism,
    MinPointsForMonochromatic,
)

__all__ = [
    # counting_basic.py
    "Catalan",
    "CatalanInverse",
    "Binomial",
    "BinomialInverseN",
    "BinomialInverseK",
    "BinomialSum",
    "LatticePaths",
    "LatticePathsInverseM",
    "Ballot",
    "DyckPaths",
    "Counting",
    "CountingPrinciples",
    # counting_advanced.py
    "Pigeonhole",
    "PigeonholeGeneralized",
    "DoubleCount",
    "Bijection",
    "CountingBijection",
    # special_numbers_stirling_bell.py
    "stirling_first",
    "stirling_second",
    "StirlingFirst",
    "StirlingSecond",
    "StirlingSecondInverseN",
    "Bell",
    # special_numbers_derangements_partitions.py
    "derangement",
    "partition_count",
    "partition_k_parts",
    "Derangement",
    "DerangementInverse",
    "Subfactorial",
    "Partition",
    "PartitionInverse",
    "PartitionKParts",
    "IntegerPartitionsIntoK",
    "LargeFactorial",
    "LargeBinomial",
    "LargeCatalan",
    "LargeStirling",
    "BinaryTreeCountRecurrence",
    # graphs_basic.py
    "ChromaticPoly",
    "GraphColoring",
    "SpanningTrees",
    "EulerianPaths",
    "HallMarriage",
    "GraphTheory",
    "HamiltonianPath",
    "TuranBound",
    # graphs_flow.py
    "SeymourFlowBound",
    "MaxPowerOfTwoBelowBound",
    "MaxMSatisfyingFlowBound",
    "PlatonicEulerianCircuits",
    # generating_functions.py
    "GeneratingFunction",
    "CountingViaAlgebra",
    "BinomialTheoremAnalogy",
    "HookLengthFormula",
    # probability.py
    "Probability",
    "ConditionalProbability",
    "EntropyCompute",
    "ConditionalExpectationSymmetric",
    # topology.py
    "TopologicalComplexityGraph",
    "BouquetOfCircles",
    "KleitmanRothschildHeightBound",
    "EssentialVertexCount",
    "FigureEightTopology",
    # misc.py
    "CountingConstraints",
    "CountingIntersections",
    "CountingPaths",
    "CountingRegions",
    "PermutationDecomposition",
    "PermutationWithRepetition",
    "PermutationOrder",
    "MultisetPermutationTotal",
    "MultisetPermutationAvoidingPattern",
    "MinAreaFunction",
    "SumMinAreas",
    "LegendreValuationFactorial",
    "PlaceValueSum",
    "FactorialBaseToDecimal",
    "CountOdd",
    "CountEven",
    "CircularArrangementsWithAdjacentConstraint",
    "multinomial_coefficient",
    "count_non_overlapping_placements",
    # ported_comb.py
    "LinearityOfExpectation",
    "ProbabilisticMethod",
    "OptionalStoppingTheorem",
    "InclusionExclusion",
    "DerangementViaInclusionExclusion",
    "ExtremalConstruction",
    "GraphArboricityBound",
    "ChromaticNumberBound",
    "RamseyBound",
    # ported_comb_2.py
    "TriangularNumbers",
    "ArithmeticSumCubesExpansion",
    "CyclotomicSubsetCount",
    "CountCoprimePairs",
    "MultiplicativePermutation",
    "ComputeNimValue",
    "IdentifyWinningPosition",
    "DetermineOptimalMove",
    "DefineGameStrategy",
    "SimulateJumps",
    "ConstructIsomorphism",
    "MinPointsForMonochromatic",
]
