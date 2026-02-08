"""
Registry for all available method blocks.

The registry maintains:
- All available method implementations
- Type compatibility information
- Categorization by tags, difficulty, input/output types
"""

from typing import List, Dict, Type, Optional, Set
from .base import MethodBlock


# Magnitude behavior profiles for steering answer distribution
# Methods not listed default to "preserving" (roughly maintain magnitude)
# Auto-classified - 97.7% coverage
MAGNITUDE_PROFILES = {
    "reducing": [
        # Number theory - valuations, counts, small outputs
        "legendre_valuation", "padic_valuation", "kummer_valuation",
        "divisor_count", "digit_sum",
        "digit_sum_base_b", "sum_of_digits", "digit_product",
        "gcd_compute", "modular_exponentiation", "primitive_root", "totient",
        "multiplicative_order", "ord_p", "hamming_weight",
        "liouville", "omega", "big_omega", "radical",
        "legendre_valuation_factorial",  # Legendre's formula: v_p(n!)
        "lte_even_power",  # Lifting the Exponent lemma
        # Modular arithmetic - outputs bounded by modulus
        "mod_inverse", "chinese_remainder_solve", "hensel_lift", "euler_reduce",
        "fermat_reduce", "legendre_symbol", "jacobi_symbol",
        "count_quad_residues",
        "quadratic_reciprocity",
        # Floor/ceiling operations - typically reduce
        "floor_div", "floor_sum", "integer_sqrt",
        # Counting/combinatorics that produce small counts
        "ballot", "eulerian_number", "lah_number",
        "turan_bound",
        # Sequence indices - output small compared to value
        "pisano_period", "zeckendorf", "binet_formula",
        # Geometry - ratios and normalized values
        "cross_ratio", "power_of_point",
        # Special functions - beta produces small fractions
        "beta_function",
        # Inverse operations (finding small parameters)
        "divisor_count_inverse", "totient_inverse", "digit_sum_inverse",
        "binomial_inverse_k", "binomial_inverse_n", "catalan_inverse",
        "fibonacci_inverse", "derangement_inverse",
        "geometric_sum_inverse_n", "arithmetic_sum_inverse_n",
        # Functional equations that solve for small parameters
        "functional_eq_multiplicative", "functional_eq_involution",
        # Other reducing operations
        "hermite_identity", "cyclotomic_eval",
        "cyclotomic_field_degree",
        "cyclotomic_inverse_n",  # Find n from cyclotomic value
        # Inverse/optimization operations that find small parameters
        "mod_exp_inverse_base", "mod_exp_inverse_exp",
        "golden_ratio_identity",
        "matrix_rank",
        # More inverse operations
        "stirling_inverse", "bell_inverse", "partition_inverse",
        "motzkin_inverse", "schroder_inverse", "delannoy_inverse",
        # === Auto-classified reducing techniques (ratio < 0.5) ===
        "and_operation", "angle_bisector_theorem", "angle_constraints_tilings",
        "arcsine", "arithmetic_geometric_mean_min", "automorphism_composition",
        "automorphism_order", "base_digit_count", "base_digit_product",
        "base_digit_sum", "binary_popcount", "bound_by_integral",
        "brauer_group_trivial", "brianchon_theorem", "caratheodory_coefficient_bound",
        "cauchy_schwarz_bound", "ceil_div", "ceil_sqrt", "centroid_properties",
        "chinese_remainder", "circumcenter_centroid_properties", "circumcircle_properties",
        "circumradius_inradius_formulas", "coefficient_solution", "collatz_count",
        "coloring_game_guarantee", "commutator", "compare_inequality", "compare_towers",
        "compare_values", "compute_discrete_log", "compute_modular_exponent",
        "construct_isomorphism", "convert_to_radians", "convex_hull_analysis",
        "coprime_check", "count_distinct_perimeters", "count_integer_triangles",
        "count_pythagorean_triples", "counting_principles", "cyclic_quadrilaterals",
        "determinant_3x3", "digit_count", "digit_sum_count", "digit_sum_squares",
        "digit_sum_to_one", "diophantine_equations", "distinct_prime_divisors_equals_b",
        "divide", "divides", "divisibility_check", "divisibility_power_general",
        "double_coset_count", "element_order", "euler_og_distance", "factoring",
        "fermat_little", "fermat_two_squares", "find_counterexample_for_n",
        "find_function_fixed_point", "find_m_satisfying_digit_sum_equation",
        "find_primitive_root", "floor_cbrt", "floor_division", "floor_log2",
        "floor_sqrt", "floor_sum_inverse_n", "gaussian_integral_half", "gcd",
        "gcd_euclidean", "gcd_iteration", "gcd_iteration_count", "geometric_series",
        "greater_than", "greater_than_or_equal", "harmonic_bundle", "holder_inequality",
        "imaginary_part", "inequality_solving", "integer_log", "is_acute_angle",
        "is_divisible", "is_even", "is_identity_automorphism", "is_nonzero",
        "is_perfect_cube", "is_perfect_power", "is_perfect_square", "jensen_inequality",
        "kleitman_rothschild_height_bound", "kummer_inverse_k", "lattice_paths_inverse_m",
        "lattice_points_on_segment", "less_than", "less_than_or_equal",
        "lifting_exponent_valuation", "logical_and", "martingale_verification",
        "max_m_satisfying_flow_bound", "max_power_leq", "max_power_of_two_below_bound",
        "max_valuation_under_constraint", "maximize_gcd_sum", "min_points_for_monochromatic",
        "min_prime_by_residue", "min_triangle_perimeter", "min_triangles_to_cover_grid",
        "min_value", "minimize_distance_plane", "mobius", "mod", "mod_99991",
        "modexp", "modular_arithmetic_analysis", "modular_inverse", "modular_power_result",
        "modulo", "multiplicative_order_inverse_a", "multiplicative_order_mod_prime",
        "nine_point_properties", "not_equal_to", "nth_root_bound",
        "particular_solution_undetermined_coeffs", "perfect_power_condition",
        "perfect_square_check", "perfect_squares_up_to", "permutation_representation_character",
        "perpendicular_to_line", "pigeonhole", "pigeonhole_generalized",
        "prime_counting_function", "prime_gap_inequality_check", "primes_below",
        "primes_of_form_2k_plus_1", "product_range", "pythagorean_triple_generator",
        "quadratic_residue", "quadratic_root_count", "real_part", "roots_of_unity",
        "roots_of_unity_ops", "similar_triangles", "slope_collinearity",
        "smallest_prime_divisor", "sobolev_embedding", "sobolev_embedding_limit",
        "solution_counting", "solve_exponential_inequality", "solve_inequality_range",
        "solve_linear_congruence", "solve_quadratic_congruence", "sqrt_mod_p",
        "square_free_divisors", "stewart_theorem", "stirling_second_inverse_n",
        "subgroup_order", "sum_divisible_backward", "supremum_property_check",
        "symmetry_rotation", "test_feasibility_k", "trapezoid_height",
        "triangle_inequality", "trig_identities_synthetic", "trigonometric_identity",
        "uniqueness_fourier_transform", "verify_coaction_property", "vieta_sum",
        "wilson", "zsigmondy",
    ],
    "amplifying": [
        # Sequences - exponential growth
        "fibonacci", "lucas", "tribonacci", "linear_recurrence",
        "fibonacci_gcd", "fermat_number",
        # EXPLOSIVE GROWTH - guaranteed large intermediate values
        "large_factorial", "large_binomial", "large_catalan", "large_stirling",
        "large_power", "large_fibonacci", "large_power_tower",
        # Complex number and polynomial operations
        "complex_numbers", "polynomial_roots",
        # Combinatorics - factorial growth
        "lcm_compute", "divisor_sum", "geometric_sum", "catalan", "binomial",
        "factorial", "subfactorial", "double_factorial",
        "stirling_second", "bell_number", "partition_count",
        "derangement", "multinomial", "central_binomial",
        "motzkin", "schroder", "delannoy",
        "bell", "stirling_first", "dyck_paths",
        # Special functions - gamma grows like factorial
        "gamma_function",
        # Sums - accumulate values
        "arithmetic_sum", "sequence_sum", "binomial_sum",
        "sum_of_powers", "harmonic_sum",
        # Exponential/power operations
        "functional_eq_add_mult", "power_tower", "functional_eq_power",
        # Products
        "carmichael_lambda", "jordan_totient",
        # Geometric - areas, volumes scale up
        "area_heron", "circumradius", "triangle_area",
        "shoelace_area", "tetrahedron_volume",
        # Geometry starters - produce geometric objects for chain extension
        "random_point", "random_point_pair", "random_triangle",
        "random_circle_pair", "random_point_quadruple", "random_hexagon",
        # input=none techniques that must be starters (can't be reducers)
        "chinese_remainder", "compare_values", "construct_isomorphism", "divide",
        "fermat_little", "gcd_euclidean", "greater_than", "less_than", "min_value",
        "mod", "quadratic_residue", "wilson", "arithmetic_sequence", "polynomial_roots",
        "divisibility_check",
        # Polynomial evaluations at large points
        "polynomial_evaluation", "characteristic_polynomial",
        "cyclotomic_factorization",
        # Matrix operations that can grow
        "matrix_det", "matrix_trace", "matrix_power",
        # Vieta/Newton - products and power sums
        "vieta_product", "newton_power_sum",
        # Counting operations
        "double_count", "sieve_count", "sum_of_squares_count",
        "quadratic_form_solution_count",
        # === Auto-classified amplifying techniques (ratio > 2.0) ===
        "algebraic_expand", "algebraic_manipulation", "apollonius_theorem",
        "apply_functional_condition", "area_sine_formula", "arithmetic_progression_analysis",
        "arithmetic_sum_cubes_expansion", "base_digit_sum_inverse", "base_representation",
        "binary_exponentiation", "binary_tree_count_recurrence", "binomial_coefficient",
        "binomial_coefficients", "binomial_theorem_analogy", "burnside",
        "calculate_100a_plus_b", "ceil_log2_large", "ceva_menelaus_combined",
        "compositum_degree", "compute_function_iterate", "compute_lower_bound",
        "constrained_polygon_perimeter", "coordinate_geometry_area", "count_coprime_pairs",
        "counting", "counting_bijection", "counting_intersections", "counting_regions",
        "counting_regions_space", "counting_via_algebra", "cutting_plane_equation",
        "cyclic_quadrilateral", "cyclotomic_subset_count", "cylinder_geometry",
        "define_piecewise_function", "derangement_via_inclusion_exclusion", "derangements",
        "distance_squared", "divisibility_power_plus_one", "evaluate_function_composition",
        "express_in_form", "floor_system_solutions", "functional_eq_cauchy",
        "functional_equation", "generating_function", "graph_coloring", "hall_marriage",
        "hook_length_formula", "hyperoval_size_projective", "induction", "inradius",
        "integer_partitions_into_k", "isogonal_conjugation", "kummer_inverse_n",
        "lattice_convex_polygon", "lattice_paths", "legendre_inverse",
        "line_covering_threshold", "linearity_of_expectation", "max_arc_size_affine_with_translation",
        "max_rectangle_area", "max_triangle_area", "medial_triangle_properties",
        "min_area_function", "multiplicative_permutation", "multiply",
        "named_cubics_properties", "optimal_box_volume", "padic_inverse",
        "parallelogram_properties", "partition", "partition_k_parts", "pell_equation",
        "permutation_decomposition", "permutation_with_repetition", "plane_geometry",
        "poles_and_polars", "polynomial_discriminant", "power_of_base", "power_of_two",
        "power_point_coordinate", "powerset", "probabilistic_method", "product_mod",
        "quadratic_discriminant", "ramsey_bound", "reflection_principle",
        "regular_polygon_properties", "simulate_jumps", "square", "square_sum_simplify",
        "squeeze_theorem", "stirling", "substitution", "sum_min_areas",
        "sum_min_areas_rectangles", "sum_of_angles", "sum_of_cubes_ap_expansion",
        "sum_range", "summation", "summation_of_lengths", "trapezoid_area",
        "triangular_number", "triangular_numbers", "vieta", "volume_calculation",
        "volume_from_bounds", "volume_rectangular_prism",
    ],
    # All other methods default to "preserving"
}


class MethodRegistry:
    """Central registry of all available method blocks.

    Maintains a graph of methods organized by input/output types,
    tags, and difficulty levels to enable efficient querying and
    composition of method chains.
    """

    _methods: Dict[str, MethodBlock] = {}
    _by_input_type: Dict[str, List[str]] = {}
    _by_output_type: Dict[str, List[str]] = {}
    _compatibility_graph: Dict[str, Set[str]] = {}  # output_type -> set of method names that accept it

    @classmethod
    def register(cls, method: MethodBlock) -> None:
        """Register a method block.

        Args:
            method: The method instance to register

        Note:
            Automatically updates internal indices for fast lookup
        """
        name = method.name

        # Store method
        cls._methods[name] = method

        # Index by input type
        if method.input_type not in cls._by_input_type:
            cls._by_input_type[method.input_type] = []
        cls._by_input_type[method.input_type].append(name)

        # Index by output type
        if method.output_type not in cls._by_output_type:
            cls._by_output_type[method.output_type] = []
        cls._by_output_type[method.output_type].append(name)

        # Update compatibility graph
        if method.output_type not in cls._compatibility_graph:
            cls._compatibility_graph[method.output_type] = set()

        # Find methods that can accept this output type
        for other_name, other_method in cls._methods.items():
            if is_compatible(method.output_type, other_method.input_type):
                cls._compatibility_graph[method.output_type].add(other_name)
            if is_compatible(other_method.output_type, method.input_type):
                if other_method.output_type not in cls._compatibility_graph:
                    cls._compatibility_graph[other_method.output_type] = set()
                cls._compatibility_graph[other_method.output_type].add(name)

    @classmethod
    def get(cls, name: str) -> MethodBlock:
        """Get method by name.

        Args:
            name: The method name

        Returns:
            The method instance

        Raises:
            KeyError: If method not found
        """
        return cls._methods[name]

    @classmethod
    def get_all(cls) -> List[MethodBlock]:
        """Get all registered methods.

        Returns:
            List of all method instances
        """
        return list(cls._methods.values())

    @classmethod
    def get_compatible_next(cls, output_type: str) -> List[MethodBlock]:
        """Get all methods that can accept this output type as input.

        Args:
            output_type: The output type from the previous method

        Returns:
            List of compatible methods
        """
        if output_type in cls._compatibility_graph:
            method_names = cls._compatibility_graph[output_type]
            return [cls._methods[name] for name in method_names]
        return []

    @classmethod
    def get_by_tag(cls, tag: str) -> List[MethodBlock]:
        """Get all methods with a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            List of methods with this tag
        """
        return [t for t in cls._methods.values() if t.has_tag(tag)]

    @classmethod
    def get_seed_methods(cls) -> List[MethodBlock]:
        """Get methods that can start a chain.

        Seed methods have input_type in ['none', 'seed', 'params'].

        Returns:
            List of seed methods
        """
        seed_types = {'none', 'seed', 'params'}
        return [t for t in cls._methods.values() if t.input_type in seed_types]

    @classmethod
    def get_terminal_methods(cls) -> List[MethodBlock]:
        """Get methods that can end a chain.

        Terminal methods have output_type == 'answer'.

        Returns:
            List of terminal methods
        """
        return [t for t in cls._methods.values() if t.output_type == 'answer']

    @classmethod
    def get_by_input_type(cls, input_type: str) -> List[MethodBlock]:
        """Get methods that accept a specific input type.

        Args:
            input_type: The input type to filter by

        Returns:
            List of methods accepting this input type
        """
        if input_type in cls._by_input_type:
            names = cls._by_input_type[input_type]
            return [cls._methods[name] for name in names]
        return []

    @classmethod
    def get_by_output_type(cls, output_type: str) -> List[MethodBlock]:
        """Get methods that produce a specific output type.

        Args:
            output_type: The output type to filter by

        Returns:
            List of methods producing this output type
        """
        if output_type in cls._by_output_type:
            names = cls._by_output_type[output_type]
            return [cls._methods[name] for name in names]
        return []

    @classmethod
    def get_by_difficulty(cls, min_diff: int = 1, max_diff: int = 5) -> List[MethodBlock]:
        """Get methods within a difficulty range.

        Args:
            min_diff: Minimum difficulty (inclusive)
            max_diff: Maximum difficulty (inclusive)

        Returns:
            List of methods in the difficulty range
        """
        return [t for t in cls._methods.values()
                if min_diff <= t.difficulty <= max_diff]

    @classmethod
    def register_alias(cls, alias_name: str, target_name: str) -> None:
        """Register an alias that maps to an existing method.

        Creates a shallow copy of the target method with the alias name,
        so both names resolve independently in get_all() and get().

        Args:
            alias_name: The new alias name (e.g., 'bell_number')
            target_name: The existing method name (e.g., 'bell')

        Note:
            Silently skips if the target doesn't exist or alias already exists.
        """
        if target_name not in cls._methods or alias_name in cls._methods:
            return
        import copy
        alias_method = copy.copy(cls._methods[target_name])
        alias_method.name = alias_name
        cls.register(alias_method)

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        cls._methods = {}
        cls._by_input_type = {}
        cls._by_output_type = {}
        cls._compatibility_graph = {}


# Type compatibility rules
TYPE_COMPATIBILITY = {
    'integer': ['valuation', 'factorization', 'modular', 'divisor_sum', 'totient', 'gcd'],
    'sequence_value': ['valuation', 'modular', 'gcd', 'integer'],
    'polynomial': ['evaluation', 'root_analysis', 'modular', 'integer'],
    'count': ['integer', 'modular', 'valuation', 'factorization'],
    'geometric_value': ['integer', 'gcd', 'count', 'modular'],
    'valuation': ['integer', 'comparison', 'answer', 'count'],
    'modular': ['answer', 'integer', 'count'],
    'factorization': ['divisor_count', 'totient', 'integer'],
    'divisor_sum': ['modular', 'answer', 'integer'],
    'totient': ['modular', 'answer', 'integer'],
    'gcd': ['factorization', 'answer', 'integer'],
    'evaluation': ['modular', 'answer', 'integer'],
    'root_analysis': ['count', 'answer'],
    'divisor_count': ['answer', 'integer'],
    'optimization': ['answer', 'integer'],
    'comparison': ['answer', 'integer'],
    'matrix': ['integer', 'evaluation', 'modular'],
    'number': ['integer', 'modular', 'factorization'],
    'answer': ['integer', 'modular', 'valuation'],
    'sequence': ['integer', 'modular', 'count'],
    # Geometric types - expanded with more intermediate paths
    'triangle': ['geometric_value', 'integer', 'triangle', 'point'],
    'point': ['geometric_value', 'integer', 'point', 'point_pair'],
    'point_pair': ['geometric_value', 'integer', 'point_pair'],
    'polygon': ['geometric_value', 'integer', 'polygon', 'point'],
    # Combinatorics types
    'permutation': ['integer', 'count'],
    'partition': ['integer', 'count'],
    'subset': ['integer', 'count'],
    # Backward generation types
    'triple': ['integer'],  # Triple converts to integer
    # Rational number types
    'rational': ['rational', 'answer', 'integer'],  # Rational can chain to rational or terminal
    # Constraint satisfaction types
    'constraint_system': ['constraint_system', 'integer', 'answer'],  # Can validate or convert to integer
    'geometric_object': ['integer', 'geometric_value', 'answer'],  # Extract properties
    # Complex number types
    'complex': ['complex', 'integer', 'answer'],  # Complex can chain to complex or extract to integer
    'complex_set': ['complex_set', 'integer', 'answer'],  # Set operations or size extraction
    # List types - allows list-producing techniques to connect to integer-consuming ones
    'list': ['integer', 'count', 'answer'],  # List can extract to integer via len/sum/first
    'params': ['integer', 'answer'],  # Params dict can extract to integer value
    # Boolean types - allows boolean-output techniques to connect to integer-consuming reducers
    'bool': ['integer', 'count', 'answer'],
    'boolean': ['integer', 'count', 'answer'],
    # Python type aliases
    'int': ['integer', 'count', 'answer', 'modular', 'valuation'],
    'float': ['integer', 'count', 'answer'],
    # Geometric types that need path to integer reducers
    'angle': ['integer', 'geometric_value'],
    'circle': ['integer', 'geometric_value'],
    'line': ['integer', 'geometric_value'],
    'trig_value': ['integer', 'geometric_value'],
    'geometric': ['integer', 'count', 'answer', 'geometric_value', 'point'],
    # Catch-all for 'any' type
    'any': ['integer', 'count', 'answer'],
    # Additional geometric types
    'point_triple': ['integer', 'geometric_value', 'count'],
    'dict': ['integer', 'count', 'answer'],
    # Geometry starters
    'point_quadruple': ['geometric_value', 'integer', 'count'],
    'hexagon': ['geometric_value', 'integer', 'count'],
    'circle_pair': ['geometric_value', 'integer', 'line'],
    # TYPE_COMPATIBILITY orphan fixes - types that were dead ends
    'vector': ['integer', 'geometric_value', 'count'],
    'tuple': ['integer', 'count', 'answer'],
    'function': ['integer', 'evaluation'],
    'real': ['integer', 'float', 'answer'],
    'set': ['integer', 'count', 'answer'],
    'fraction': ['integer', 'rational', 'answer'],
    'symbolic_expression': ['integer', 'evaluation', 'answer'],
    'word_problem': ['integer', 'answer'],
    'tetrahedron': ['integer', 'geometric_value'],
    'sphere_pair': ['integer', 'geometric_value'],
    'expression': ['integer', 'answer'],
    'graph': ['integer', 'count', 'answer'],
}


def is_compatible(output_type: str, input_type: str) -> bool:
    """Check if an output type can feed into an input type.

    Args:
        output_type: The type produced by one technique
        input_type: The type required by another technique

    Returns:
        True if the types are compatible

    Note:
        Uses TYPE_COMPATIBILITY mapping to determine compatibility
    """
    if output_type == input_type:
        return True

    if output_type in TYPE_COMPATIBILITY:
        return input_type in TYPE_COMPATIBILITY[output_type]

    return False


def register_method(cls: Type[MethodBlock]) -> Type[MethodBlock]:
    """Decorator to auto-register method classes.

    Usage:
        @register_method
        class MyMethod(MethodBlock):
            ...

    Args:
        cls: The method class to register

    Returns:
        The same class (for use as a decorator)
    """
    instance = cls()
    MethodRegistry.register(instance)
    return cls


# Alias for backward compatibility
register_technique = register_method
