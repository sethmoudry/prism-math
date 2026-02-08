# fmt: off
"""Known-value tests for algebra methods (part 2 of 2).

Covers: inequalities, floor/ceiling, series, advanced functions,
matrix operations, curves/cubics, expressions, stubs, utilities.
"""

KNOWN_VALUES = {
    # ================================================================
    # OPTIMIZATION / INEQUALITIES
    # ================================================================
    "arithmetic_geometric_mean_min": [
        {"input": None, "params": {"product": 16, "n": 2}, "expected": 8},
        {"input": None, "params": {"product": 27, "n": 3}, "expected": 9},
    ],
    "arithmetic_geometric_mean_inverse_vars": [
        {"input": None, "params": {"product": 16, "n": 2}, "expected": [4.0, 4.0]},
    ],
    "cauchy_schwarz_bound": [
        {"input": None, "params": {"a": [1, 2], "b": [3, 4]}, "expected": 11},
        {"input": None, "params": {"a": [1, 0], "b": [0, 1]}, "expected": 1},
    ],
    "holder_inequality": [
        {"input": None, "params": {"a": [1, 2], "b": [3, 4], "p": 2, "q": 2}, "expected": 11},
    ],
    "jensen_inequality": [
        {"input": None, "params": {"values": [2, 4, 6], "func": "square"}, "expected": 16},
        {"input": None, "params": {"values": [3, 3, 3], "func": "square"}, "expected": 9},
    ],
    "lagrange_optimize": [
        {"input": None, "params": {"sum": 10, "objective": "product"}, "expected": 25},
        {"input": None, "params": {"sum": 6, "objective": "product"}, "expected": 9},
    ],

    # ================================================================
    # FLOOR / CEILING
    # ================================================================
    "floor_sum": [
        {"input": None, "params": {"n": 5, "a": 3, "b": 1, "m": 4}, "expected": 7},
    ],
    "floor_sum_inverse_n": [
        {"input": None, "params": {"a": 1, "b": 0, "m": 3, "target": 10}, "expected": 10},
    ],
    "floor_div": [
        {"input": None, "params": {"a": 17, "b": 5}, "expected": 3},
        {"input": None, "params": {"a": 100, "b": 7}, "expected": 14},
    ],
    "floor_div_inverse_a": [
        {"input": None, "params": {"b": 5, "q": 3}, "expected": [15, 19]},
    ],
    "hermite_identity": [
        {"input": None, "params": {"n": 3, "x": 2.5}, "expected": 7},
        {"input": None, "params": {"n": 4, "x": 1.0}, "expected": 4},
    ],

    # ================================================================
    # SERIES
    # ================================================================
    "arithmetic_sum": [
        {"input": None, "params": {"a": 1, "d": 1, "n": 10}, "expected": 55},
        {"input": None, "params": {"a": 1, "d": 1, "n": 100}, "expected": 5050},
    ],
    "arithmetic_sum_inverse_n": [
        {"input": None, "params": {"a": 1, "d": 1, "target": 55}, "expected": 10},
    ],
    "sum_range": [
        {"input": None, "params": {"start": 1, "end": 10}, "expected": 55},
        {"input": None, "params": {"start": 5, "end": 10}, "expected": 45},
    ],
    "geometric_sum": [
        {"input": None, "params": {"a": 1, "r": 2, "n": 5}, "expected": 31},
        {"input": None, "params": {"a": 2, "r": 3, "n": 4}, "expected": 80},
    ],
    "geometric_sum_inverse_n": [
        {"input": None, "params": {"a": 1, "r": 2, "target": 63}, "expected": 6},
    ],
    "telescoping": [
        {"input": None, "params": {"n": 10, "type": "harmonic"}, "expected": 10},
    ],
    "partial_fractions": [
        {"input": None, "params": {"numerator": [1], "denominator": [1, 1, 0]}, "expected": [-1, 1]},
    ],

    # ================================================================
    # ADVANCED FUNCTIONS
    # ================================================================
    "golden_ratio_identity": [
        {"input": None, "params": {"n": 5}, "expected": 11},
    ],
    "ordinary_generating_function": [
        {"input": None, "params": {"sequence": [1, 2, 3]}, "expected": [1, 2, 3]},
    ],
    "fibonacci_identity": [
        {"input": None, "params": {"identity_type": "cassini", "n": 5}, "expected": -1},
        {"input": None, "params": {"identity_type": "sum_first_n", "n": 6}, "expected": 20},
        {"input": None, "params": {"identity_type": "sum_of_squares", "n": 5}, "expected": 40},
        {"input": None, "params": {"identity_type": "cassini", "n": 4}, "expected": 1},
    ],
    "series_approximation": [
        {"input": None, "params": {"series_type": "geometric_sum", "r": 2, "n": 4}, "expected": 31},
        {"input": None, "params": {"series_type": "power_sum", "k": 1, "n": 10}, "expected": 55},
        {"input": None, "params": {"series_type": "power_sum", "k": 2, "n": 5}, "expected": 55},
        {"input": None, "params": {"series_type": "power_sum", "k": 3, "n": 5}, "expected": 225},
        {"input": None, "params": {"series_type": "alternating", "n": 5}, "expected": 3},
        {"input": None, "params": {"series_type": "binomial_sum", "n": 5}, "expected": 32},
    ],
    "summation_manipulation": [
        {"input": None, "params": {"manipulation_type": "telescoping", "n": 10}, "expected": 10},
        {"input": None, "params": {"manipulation_type": "gauss_pairing", "n": 10}, "expected": 55},
        {"input": None, "params": {"manipulation_type": "reverse_and_add", "n": 10}, "expected": 55},
        {"input": None, "params": {"manipulation_type": "difference_of_sums", "n": 5}, "expected": 40},
    ],

    # ================================================================
    # MATRIX OPERATIONS
    # ================================================================
    "matrix_power": [
        {"input": None, "params": {"matrix": [[1, 1], [1, 0]], "n": 3}, "expected": [[3, 2], [2, 1]]},
    ],
    "matrix_power_inverse_n": [
        {"input": None, "params": {"matrix": [[2, 0], [0, 3]], "M_n": [[8, 0], [0, 27]], "true_n": 3}, "expected": 3},
    ],
    "matrix_det": [
        {"input": None, "params": {"matrix": [[1, 2], [3, 4]]}, "expected": -2},
        {"input": None, "params": {"matrix": [[1, 0], [0, 1]]}, "expected": 1},
        {"input": None, "params": {"matrix": [[2, 3], [1, 4]]}, "expected": 5},
        {"input": None, "params": {"matrix": [[5, 6], [7, 8]]}, "expected": -2},
    ],
    "matrix_trace": [
        {"input": None, "params": {"matrix": [[1, 2], [3, 4]]}, "expected": 5},
        {"input": None, "params": {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, "expected": 15},
        {"input": None, "params": {"matrix": [[5, 0, 0], [0, 3, 0], [0, 0, 7]]}, "expected": 15},
    ],
    "determinant_2x2": [
        {"input": None, "params": {"matrix": [[3, 5], [2, 7]]}, "expected": 11},
        {"input": None, "params": {"matrix": [[1, 0], [0, 1]]}, "expected": 1},
    ],
    "vector_dot_product_from_diagonals": [
        {"input": None, "params": {"matrix": [[1, 2], [3, 4]]}, "expected": 14},
    ],

    # ================================================================
    # CURVES / CUBICS
    # ================================================================
    "curve_intersection": [
        {"input": None, "params": {"curve_type": "two_lines", "a1": 1, "b1": 0, "c1": 2, "a2": 0, "b2": 1, "c2": 3, "output": "x"}, "expected": 2},
        {"input": None, "params": {"curve_type": "line_parabola", "m": 5, "b": -6, "output": "count"}, "expected": 2},
    ],
    "named_cubics_properties": [
        {"input": None, "params": {"a": 0, "b": 0, "c": 0}, "expected": 0},
        {"input": None, "params": {"a": 3, "b": 0, "c": 0}, "expected": 2},
    ],

    # ================================================================
    # EXPRESSION MANIPULATION
    # ================================================================
    "square_sum_simplify": [
        {"input": None, "params": {"a": 3, "b": 4}, "expected": 49},
        {"input": None, "params": {"a": 1, "b": 1}, "expected": 4},
    ],
    "expand_square_sum": [
        {"input": None, "params": {"a": 3, "b": 4}, "expected": 49},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 100},
    ],
    "expand_difference": [
        {"input": None, "params": {"a": 7, "b": 3}, "expected": 16},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 0},
    ],
    "separate_expression_parts": [
        {"input": None, "params": {"coeffs": [3, -2, 1]}, "expected": ["3x^2", "-2x", "1"]},
    ],
    "extract_m_n": [
        {"input": None, "params": {"m": 3, "n": 7}, "expected": (3, 7)},
    ],
    "extract_m_n_sum": [
        {"input": None, "params": {"m": 3, "n": 7}, "expected": 10},
        {"input": None, "params": {"m": -5, "n": 8}, "expected": 3},
    ],
    "quotient_of_sums": [
        {"input": None, "params": {"seq_a": [1, 2, 3], "seq_b": [4, 5, 6]}, "expected": 0.4},
    ],
    "simplify_dot_product": [
        {"input": None, "params": {"v1": [1, 2, 3], "v2": [4, 5, 6]}, "expected": 32},
        {"input": None, "params": {"v1": [1, 0], "v2": [0, 1]}, "expected": 0},
    ],
    "extract_sqrt_abc_parameters": [
        {"input": None, "params": {"a": 2, "b": 3, "c": 5}, "expected": (2, 3, 5)},
    ],

    # ================================================================
    # STUBS / MISC
    # ================================================================
    "algebraic_manipulation": [
        {"input": None, "params": {"n": 5, "k": 3}, "expected": 120},
        {"input": None, "params": {"n": 10, "k": 2}, "expected": 90},
    ],
    "algebraic_simplification": [
        {"input": None, "params": {"operation": "difference_of_squares", "a": 10, "b": 3}, "expected": 91},
        {"input": None, "params": {"operation": "rationalize_denominator", "a": 4, "b": 9}, "expected": 5},
        {"input": None, "params": {"operation": "complete_the_square", "a": 1, "b": 6, "c": 5, "output": "k"}, "expected": -4},
        {"input": None, "params": {"operation": "factor_sum_cubes", "a": 2, "b": 3}, "expected": 35},
        {"input": None, "params": {"operation": "simplify_fraction", "numerator": 12, "denominator": 18, "output": "numerator"}, "expected": 2},
    ],
    "binomial_coefficient": [
        {"input": None, "params": {"n": 10, "k": 3}, "expected": 120},
        {"input": None, "params": {"n": 5, "k": 0}, "expected": 1},
        {"input": None, "params": {"n": 6, "k": 6}, "expected": 1},
    ],
    "binomial_coefficients": [
        {"input": None, "params": {"n": 5, "k": 2}, "expected": 10},
        {"input": None, "params": {"n": 10, "k": 5}, "expected": 252},
    ],
    "functional_equation": [
        {"input": None, "params": {"n": 5, "a": 3}, "expected": 15},
        {"input": None, "params": {"n": 10, "a": 7}, "expected": 70},
    ],
    "roots_of_unity": [
        {"input": None, "params": {"n": 4, "k": 4}, "expected": 0},
        {"input": None, "params": {"n": 4, "k": 1}, "expected": 1},
    ],
    "substitution": [
        {"input": None, "params": {"n": 3, "a": 1, "b": 2, "c": 1, "expr_type": "quadratic"}, "expected": 16},
        {"input": None, "params": {"n": 4, "a": 5, "b": 3, "expr_type": "linear"}, "expected": 23},
        {"input": None, "params": {"n": 3, "a": 1, "b": 0, "c": 0, "d": 0, "expr_type": "cubic"}, "expected": 27},
    ],
    "summation": [
        {"input": None, "params": {"n": 10}, "expected": 55},
        {"input": None, "params": {"n": 100}, "expected": 5050},
    ],
    "summation_of_lengths": [
        {"input": None, "params": {"a": 2, "r": 3, "n": 4}, "expected": 80},
        {"input": None, "params": {"a": 1, "r": 2, "n": 5}, "expected": 31},
    ],
    "polynomial_eval_extended": [
        {"input": None, "params": {"coeffs": [1, 2, 3], "x": 2}, "expected": 11},
        {"input": None, "params": {"coeffs": [2, 0, 1], "x": 3}, "expected": 19},
    ],
    "coefficient_solution": [
        {"input": None, "params": {"x": 2, "b": 5, "value": 15}, "expected": 5},
        {"input": None, "params": {"x": 3, "b": 0, "value": 12}, "expected": 4},
    ],
    "compare_lhs_rhs": [
        {"input": None, "params": {"lhs": 10, "rhs": 7}, "expected": 3},
        {"input": None, "params": {"lhs": 5, "rhs": 5}, "expected": 0},
    ],
}
