# fmt: off
"""Known-value tests for algebra methods (part 1 of 2).

Covers: vieta, newton, polynomials, cyclotomic, laurent, equations,
recurrences, functional equations, quadratics, exponential, diophantine.
"""

KNOWN_VALUES = {
    # ================================================================
    # VIETA'S FORMULAS
    # ================================================================
    "vieta_sum": [
        {"input": None, "params": {"coeffs": [1, -5, 6], "degree": 2}, "expected": 5},
        {"input": None, "params": {"coeffs": [1, -6, 11, -6], "degree": 3}, "expected": 6},
        {"input": None, "params": {"coeffs": [2, 0, 8], "degree": 2}, "expected": 0},
        {"input": None, "params": {"coeffs": [100, -500, 600], "degree": 2}, "expected": 5},
    ],
    "vieta_product": [
        {"input": None, "params": {"coeffs": [1, -5, 6], "degree": 2}, "expected": 6},
        {"input": None, "params": {"coeffs": [1, -6, 11, -6], "degree": 3}, "expected": 6},
        {"input": None, "params": {"coeffs": [1, 0, -4], "degree": 2}, "expected": -4},
    ],
    "vieta_inverse_coeffs": [
        {"input": None, "params": {"root_sum": 5, "root_product": 6}, "expected": [1, -5, 6]},
        {"input": None, "params": {"root_sum": 7, "root_product": 12}, "expected": [1, -7, 12]},
        {"input": None, "params": {"root_sum": 0, "root_product": -9}, "expected": [1, 0, -9]},
    ],

    # ================================================================
    # NEWTON'S IDENTITIES
    # ================================================================
    "newton_power_sum": [
        {"input": None, "params": {"roots": [1, 2, 3], "k": 1}, "expected": 6},
        {"input": None, "params": {"roots": [1, 2, 3], "k": 2}, "expected": 14},
        {"input": None, "params": {"roots": [2, -2], "k": 3}, "expected": 0},
        {"input": None, "params": {"roots": [1, 1, 1], "k": 4}, "expected": 3},
    ],
    "newton_inverse_elemsym": [
        {"input": None, "params": {"power_sums": [5, 13], "n": 2}, "expected": [5, 6]},
        {"input": None, "params": {"power_sums": [0, 8], "n": 2}, "expected": [0, -4]},
    ],

    # ================================================================
    # POLYNOMIAL OPERATIONS
    # ================================================================
    "polynomial_evaluation": [
        {"input": None, "params": {"coeffs": [1, 2, 1], "x": 2}, "expected": 9},
        {"input": None, "params": {"coeffs": [2, 3, 5], "x": 0}, "expected": 5},
        {"input": None, "params": {"coeffs": [1, 0, 1], "x": -2}, "expected": 5},
        {"input": None, "params": {"coeffs": [2, 3, 4, 5], "x": 2}, "expected": 41},
        {"input": None, "params": {"coeffs": [1, 2, 3, 4, 5], "x": 0}, "expected": 5},
    ],
    "polynomial_evaluation_inverse_x": [
        {"input": None, "params": {"coeffs": [2, 3], "y": 11}, "expected": 4},
        {"input": None, "params": {"coeffs": [1, 0], "y": 7}, "expected": 7},
        {"input": None, "params": {"coeffs": [5, -10], "y": 15}, "expected": 5},
    ],
    "polynomial_division": [
        {"input": None, "params": {"p_coeffs": [1, 0, -1], "q_coeffs": [1, -1]}, "expected": [1, 1]},
    ],
    "polynomial_difference": [
        {"input": None, "params": {"p_coeffs": [1, 0, -1], "q_coeffs": [1, 1]}, "expected": [1, -1, -2]},
    ],
    "polynomial_factorization": [
        {"input": None, "params": {"coeffs": [1, -5, 6]}, "expected": "(x - 3)*(x - 2)"},
    ],
    "polynomial_roots": [
        {"input": None, "params": {"coeffs": [1, -5, 6], "degree": 2, "operation": "count_real_roots"}, "expected": 2},
        {"input": None, "params": {"coeffs": [1, -5, 6], "degree": 2, "operation": "sum_of_roots"}, "expected": 5},
        {"input": None, "params": {"coeffs": [1, -5, 6], "degree": 2, "operation": "product_of_roots"}, "expected": 6},
    ],

    # ================================================================
    # CYCLOTOMIC POLYNOMIALS
    # ================================================================
    "cyclotomic_eval": [
        {"input": None, "params": {"n": 1, "x": 5}, "expected": 4},
        {"input": None, "params": {"n": 2, "x": 3}, "expected": 4},
        {"input": None, "params": {"n": 3, "x": 2}, "expected": 7},
        {"input": None, "params": {"n": 4, "x": 1}, "expected": 2},
        {"input": None, "params": {"n": 4, "x": 2}, "expected": 5},
        {"input": None, "params": {"n": 6, "x": 2}, "expected": 3},
    ],
    "cyclotomic_inverse_n": [
        {"input": None, "params": {"x": 2, "target_n": 3}, "expected": 3},
        {"input": None, "params": {"x": 3, "target_n": 1}, "expected": 1},
    ],
    "cyclotomic_factorization": [
        {"input": None, "params": {"n": 6}, "expected": 4},
        {"input": None, "params": {"n": 4}, "expected": 3},
        {"input": None, "params": {"n": 7}, "expected": 2},
    ],

    # ================================================================
    # LAURENT POLYNOMIALS
    # ================================================================
    "laurent_polynomial": [
        {"input": None, "params": {"coeffs": {-1: 2, 0: 3, 1: 1}, "x": 2}, "expected": 6},
    ],

    # ================================================================
    # RECURRENCES
    # ================================================================
    "linear_recurrence": [
        {"input": None, "params": {"coeffs": [1, 1], "initial": [0, 1], "n": 6}, "expected": 8},
        {"input": None, "params": {"coeffs": [2, 1], "initial": [1, 1], "n": 4}, "expected": 17},
        {"input": None, "params": {"coeffs": [1, 1], "initial": [0, 0], "n": 5}, "expected": 0},
    ],
    "linear_recurrence_inverse_n": [
        {"input": None, "params": {"coeffs": [1, 1], "initial": [0, 1], "target": 20}, "expected": 8},
    ],
    "characteristic_polynomial": [
        {"input": None, "params": {"coeffs": [1, 1]}, "expected": 2},
    ],
    "binet_formula": [
        {"input": None, "params": {"f0": 0, "f1": 1, "n": 10}, "expected": 55},
        {"input": None, "params": {"f0": 1, "f1": 1, "n": 6}, "expected": 13},
    ],
    "recurrence": [
        {"input": None, "params": {"recurrence_type": "first_order", "c": 2, "d": 1, "a0": 1, "n": 3}, "expected": 15},
        {"input": None, "params": {"recurrence_type": "first_order", "c": 1, "d": 5, "a0": 0, "n": 4}, "expected": 20},
        {"input": None, "params": {"recurrence_type": "fibonacci", "a0": 1, "a1": 1, "n": 7}, "expected": 21},
        {"input": None, "params": {"recurrence_type": "lucas", "a0": 2, "a1": 1, "n": 5}, "expected": 11},
        {"input": None, "params": {"recurrence_type": "geometric", "r": 2, "a0": 3, "n": 5}, "expected": 96},
        {"input": None, "params": {"recurrence_type": "second_order", "p": 1, "q": 1, "a0": 1, "a1": 1, "n": 6}, "expected": 13},
    ],

    # ================================================================
    # FUNCTIONAL EQUATIONS
    # ================================================================
    "functional_eq_cauchy": [
        {"input": None, "params": {"c": 3, "x": 7}, "expected": 21},
        {"input": None, "params": {"c": 1, "x": 100}, "expected": 100},
    ],
    "functional_eq_multiplicative": [
        {"input": None, "params": {"n": 8, "f_p": 1}, "expected": 3},
        {"input": None, "params": {"n": 16, "f_p": 1}, "expected": 4},
        {"input": None, "params": {"n": 27, "f_p": 1}, "expected": 3},
    ],
    "functional_eq_power": [
        {"input": None, "params": {"base": 2, "n": 3, "c": 2}, "expected": 6},
        {"input": None, "params": {"base": 3, "n": 4, "c": 1}, "expected": 4},
    ],
    "functional_eq_add_mult": [
        {"input": None, "params": {"a": 2, "x": 4}, "expected": 16},
        {"input": None, "params": {"a": 3, "x": 3}, "expected": 27},
    ],

    # ================================================================
    # QUADRATIC FUNCTIONS
    # ================================================================
    "quadratic_vertex": [
        {"input": None, "params": {"a": 1, "b": -4, "c": 3}, "expected": (2, -1)},
        {"input": None, "params": {"a": 2, "b": -8, "c": 6}, "expected": (2, -2)},
    ],
    "quadratic_axis_of_symmetry": [
        {"input": None, "params": {"a": 1, "b": -6}, "expected": 3},
        {"input": None, "params": {"a": 2, "b": -4}, "expected": 1},
    ],
    "quadratic_discriminant": [
        {"input": None, "params": {"a": 1, "b": 5, "c": 6}, "expected": 1},
        {"input": None, "params": {"a": 1, "b": 4, "c": 4}, "expected": 0},
        {"input": None, "params": {"a": 1, "b": 1, "c": 1}, "expected": -3},
    ],
    "quadratic_root_count": [
        {"input": None, "params": {"a": 1, "b": 5, "c": 6}, "expected": 2},
        {"input": None, "params": {"a": 1, "b": 4, "c": 4}, "expected": 1},
        {"input": None, "params": {"a": 1, "b": 1, "c": 1}, "expected": 0},
    ],
    "factoring": [
        {"input": None, "params": {"a": 1, "b": 5, "c": 6}, "expected": 2},
        {"input": None, "params": {"a": 1, "b": 4, "c": 4}, "expected": 1},
        {"input": None, "params": {"a": 1, "b": 1, "c": 1}, "expected": 0},
    ],
    "complete_square": [
        {"input": None, "params": {"a": 1, "b": 6, "c": 5}, "expected": (1, -3, -4)},
        {"input": None, "params": {"a": 1, "b": -4, "c": 3}, "expected": (1, 2, -1)},
    ],
    "quadratic_formula": [
        {"input": None, "params": {"a": 1, "b": -5, "c": 6}, "expected": (3, 2)},
        {"input": None, "params": {"a": 1, "b": -2, "c": 1}, "expected": (1,)},
    ],

    # ================================================================
    # EXPONENTIAL EQUATIONS
    # ================================================================
    "exponential_equation": [
        {"input": None, "params": {"equation_type": "find_exponent", "base": 2, "result": 8, "answer": 3}, "expected": 3},
        {"input": None, "params": {"equation_type": "find_base", "exponent": 2, "result": 16, "answer": 4}, "expected": 4},
        {"input": None, "params": {"equation_type": "evaluate_power", "base": 3, "exponent": 4}, "expected": 81},
        {"input": None, "params": {"equation_type": "compare_powers", "a": 2, "b": 5, "c": 3, "d": 3}, "expected": 1},
        {"input": None, "params": {"equation_type": "power_difference", "base": 2, "x": 5, "y": 3}, "expected": 24},
        {"input": None, "params": {"equation_type": "nested_power", "a": 2, "b": 3, "c": 2}, "expected": 64},
    ],

    # ================================================================
    # DIOPHANTINE EQUATIONS
    # ================================================================
    "diophantine_equations": [
        {"input": None, "params": {"a": 2, "b": 3, "c": 12}, "expected": 3},
        {"input": None, "params": {"a": 1, "b": 1, "c": 5}, "expected": 6},
    ],

    # ================================================================
    # RATIONALIZE DENOMINATOR
    # ================================================================
    "rationalize_denominator": [
        {"input": None, "params": {"numerator": 1, "radicand": 2, "denom_type": "simple"},
         "expected": {"coeff": 1, "radicand": 2, "denom": 2}},
    ],

    # ================================================================
    # SOLVE EQUATION
    # ================================================================
    "solve_equation": [
        {"input": None, "params": {"equation_type": "linear", "a": 3, "b": -9}, "expected": [3]},
        {"input": None, "params": {"equation_type": "quadratic", "a": 1, "b": -5, "c": 6}, "expected": [3, 2]},
    ],
}
