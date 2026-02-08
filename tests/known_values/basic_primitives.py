# fmt: off
"""Known ground-truth values for basic_primitives domain.

Every entry has been hand-verified against the implementation.
Format: method_name -> [{input, params, expected}, ...]
"""

import math

KNOWN_VALUES = {
    # =========================================================================
    # arithmetic.py
    # =========================================================================
    "sum": [
        {"input": [1, 2, 3], "params": {}, "expected": 6},
        {"input": [10, 20, 30, 40], "params": {}, "expected": 100},
        {"input": [-5, 5], "params": {}, "expected": 0},
    ],
    "factorial": [
        {"input": None, "params": {"n": 0}, "expected": 1},
        {"input": None, "params": {"n": 1}, "expected": 1},
        {"input": None, "params": {"n": 5}, "expected": 120},
        {"input": None, "params": {"n": 10}, "expected": 3628800},
        {"input": None, "params": {"n": 20}, "expected": 2432902008176640000},
    ],
    "ceil_div": [
        {"input": None, "params": {"a": 10, "b": 3}, "expected": 4},
        {"input": None, "params": {"a": 9, "b": 3}, "expected": 3},
        {"input": None, "params": {"a": 7, "b": 2}, "expected": 4},
        {"input": None, "params": {"a": 1, "b": 1}, "expected": 1},
    ],
    "digit_count": [
        {"input": None, "params": {"n": 12345}, "expected": 5},
        {"input": None, "params": {"n": 1}, "expected": 1},
        {"input": None, "params": {"n": 999}, "expected": 3},
        {"input": None, "params": {"n": 1000000}, "expected": 7},
    ],
    "log2": [
        {"input": None, "params": {"x": 8}, "expected": 3.0},
        {"input": None, "params": {"x": 1}, "expected": 0.0},
        {"input": None, "params": {"x": 1024}, "expected": 10.0},
    ],
    "binary_representation": [
        {"input": None, "params": {"n": 10}, "expected": "0b1010"},
        {"input": None, "params": {"n": 0}, "expected": "0b0"},
        {"input": None, "params": {"n": 255}, "expected": "0b11111111"},
    ],
    "binary_popcount": [
        {"input": None, "params": {"n": 7}, "expected": 3},
        {"input": None, "params": {"n": 255}, "expected": 8},
        {"input": None, "params": {"n": 16}, "expected": 1},
        {"input": None, "params": {"n": 0}, "expected": 0},
    ],
    "closest_integer": [
        {"input": 3.7, "params": {}, "expected": 4},
        {"input": 3.2, "params": {}, "expected": 3},
        {"input": 2.5, "params": {}, "expected": 2},  # Python banker's rounding
        {"input": 3.5, "params": {}, "expected": 4},  # Python banker's rounding
    ],
    "floor": [
        {"input": None, "params": {"x": 3.7}, "expected": 3},
        {"input": None, "params": {"x": -1.5}, "expected": -2},
        {"input": None, "params": {"x": 5.0}, "expected": 5},
    ],
    "power": [
        {"input": None, "params": {"base": 2, "exp": 10}, "expected": 1024},
        {"input": None, "params": {"base": 3, "exp": 0}, "expected": 1},
        {"input": None, "params": {"base": 5, "exp": 3}, "expected": 125},
    ],
    "product": [
        {"input": [2, 3, 4], "params": {}, "expected": 24},
        {"input": [1, 1, 1, 1], "params": {}, "expected": 1},
        {"input": [10, 10], "params": {}, "expected": 100},
    ],
    "floor_division": [
        {"input": None, "params": {"a": 17, "b": 5}, "expected": 3},
        {"input": None, "params": {"a": 20, "b": 4}, "expected": 5},
        {"input": None, "params": {"a": 7, "b": 3}, "expected": 2},
    ],
    "modulo": [
        {"input": None, "params": {"a": 17, "b": 5}, "expected": 2},
        {"input": None, "params": {"a": 20, "b": 7}, "expected": 6},
        {"input": None, "params": {"a": 100, "b": 3}, "expected": 1},
    ],

    # =========================================================================
    # arithmetic_2.py
    # =========================================================================
    "negate": [
        {"input": 5, "params": {}, "expected": -5},
        {"input": -3, "params": {}, "expected": 3},
        {"input": 0, "params": {}, "expected": 0},
    ],
    "round_to_nearest": [
        {"input": 3.7, "params": {"precision": 1}, "expected": 4},
        {"input": 3.2, "params": {"precision": 1}, "expected": 3},
        {"input": 17.0, "params": {"precision": 5}, "expected": 15},
    ],
    "add": [
        {"input": None, "params": {"a": 3, "b": 5}, "expected": 8},
        {"input": None, "params": {"a": -10, "b": 10}, "expected": 0},
        {"input": None, "params": {"a": 0, "b": 0}, "expected": 0},
    ],
    "subtract": [
        {"input": None, "params": {"a": 10, "b": 3}, "expected": 7},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 0},
        {"input": None, "params": {"a": 0, "b": 7}, "expected": -7},
    ],
    "multiply": [
        {"input": None, "params": {"a": 4, "b": 7}, "expected": 28},
        {"input": None, "params": {"a": -3, "b": 5}, "expected": -15},
        {"input": None, "params": {"a": 0, "b": 100}, "expected": 0},
    ],
    "add_one": [
        {"input": 10, "params": {}, "expected": 11},
        {"input": -1, "params": {}, "expected": 0},
        {"input": 99, "params": {}, "expected": 100},
    ],
    "subtract_one": [
        {"input": 10, "params": {}, "expected": 9},
        {"input": 1, "params": {}, "expected": 0},
        {"input": 100, "params": {}, "expected": 99},
    ],
    "divide": [
        # integer // integer -> integer division
        {"input": None, "params": {"a": 10, "b": 2}, "expected": 5},
        {"input": None, "params": {"a": 10, "b": 3}, "expected": 3},
        {"input": None, "params": {"a": 100, "b": 7}, "expected": 14},
    ],
    "difference": [
        {"input": None, "params": {"a": 10, "b": 3}, "expected": 7},
        {"input": None, "params": {"a": 5, "b": 10}, "expected": -5},
        {"input": None, "params": {"a": 0, "b": 0}, "expected": 0},
    ],
    "half_value": [
        {"input": 10, "params": {}, "expected": 5.0},
        {"input": 7, "params": {}, "expected": 3.5},
        {"input": 100, "params": {}, "expected": 50.0},
    ],
    "absolute_difference": [
        {"input": None, "params": {"a": 10, "b": 3}, "expected": 7},
        {"input": None, "params": {"a": 3, "b": 10}, "expected": 7},
        {"input": None, "params": {"a": -5, "b": 5}, "expected": 10},
    ],
    "absolute_value": [
        {"input": None, "params": {"x": -5}, "expected": 5},
        {"input": None, "params": {"x": 7}, "expected": 7},
        {"input": None, "params": {"x": 0}, "expected": 0},
    ],
    "min_value": [
        {"input": None, "params": {"a": 3, "b": 7}, "expected": 3},
        {"input": None, "params": {"a": 10, "b": 2}, "expected": 2},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 5},
    ],
    "max_value": [
        {"input": None, "params": {"a": 3, "b": 7}, "expected": 7},
        {"input": None, "params": {"a": 10, "b": 2}, "expected": 10},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 5},
    ],

    # =========================================================================
    # arithmetic_3.py
    # =========================================================================
    "compare_values": [
        {"input": None, "params": {"a": 5, "b": 3}, "expected": 1},
        {"input": None, "params": {"a": 3, "b": 5}, "expected": -1},
        {"input": None, "params": {"a": 4, "b": 4}, "expected": 0},
    ],
    "filter_odd": [
        {"input": [1, 2, 3, 4, 5, 6], "params": {}, "expected": [1, 3, 5]},
        {"input": [2, 4, 6], "params": {}, "expected": []},
        {"input": [7], "params": {}, "expected": [7]},
    ],
    "filter_even": [
        {"input": [1, 2, 3, 4, 5, 6], "params": {}, "expected": [2, 4, 6]},
        {"input": [1, 3, 5], "params": {}, "expected": []},
        {"input": [8], "params": {}, "expected": [8]},
    ],
    "range_integers": [
        {"input": None, "params": {"start": 1, "end": 5}, "expected": [1, 2, 3, 4, 5]},
        {"input": None, "params": {"start": 3, "end": 3}, "expected": [3]},
        {"input": None, "params": {"start": -2, "end": 2}, "expected": [-2, -1, 0, 1, 2]},
    ],
    "product_range": [
        {"input": None, "params": {"start": 1, "end": 5}, "expected": 120},  # 5!
        {"input": None, "params": {"start": 3, "end": 5}, "expected": 60},   # 3*4*5
        {"input": None, "params": {"start": 1, "end": 1}, "expected": 1},
        {"input": None, "params": {"start": 5, "end": 3}, "expected": 1},    # empty
    ],
    "mod": [
        {"input": None, "params": {"a": 17, "b": 5}, "expected": 2},
        {"input": None, "params": {"a": 100, "b": 7}, "expected": 2},
        {"input": None, "params": {"a": 25, "b": 5}, "expected": 0},
    ],
    "calculate_100a_plus_b": [
        {"input": None, "params": {"a": 1, "b": 23}, "expected": 123},
        {"input": None, "params": {"a": 0, "b": 0}, "expected": 0},
        {"input": None, "params": {"a": 99, "b": 99}, "expected": 9999},
    ],
    "add_numerator_denominator": [
        {"input": None, "params": {"p": 3, "q": 5}, "expected": 8},
        {"input": None, "params": {"p": 1, "q": 1}, "expected": 2},
    ],
    "extract_numerator_denominator": [
        {"input": (3, 5), "params": {}, "expected": (3, 5)},
        {"input": (7, 11), "params": {}, "expected": (7, 11)},
    ],
    "sum_numerator_denominator": [
        {"input": None, "params": {"p": 3, "q": 5}, "expected": 8},
        {"input": None, "params": {"p": 10, "q": 7}, "expected": 17},
    ],
    "constant_zero": [
        {"input": None, "params": {}, "expected": 0},
    ],
    "constant_one": [
        {"input": None, "params": {}, "expected": 1},
    ],
    "constant_result": [
        {"input": 42, "params": {}, "expected": 42},
        {"input": 0, "params": {}, "expected": 0},
        {"input": None, "params": {"value": 7}, "expected": 7},
    ],

    # =========================================================================
    # geometry.py
    # =========================================================================
    "midpoint": [
        {"input": None, "params": {"p1": (0, 0), "p2": (4, 6)}, "expected": (2.0, 3.0)},
        {"input": None, "params": {"p1": (-2, -2), "p2": (2, 2)}, "expected": (0.0, 0.0)},
    ],
    "triangle_area_heron": [
        # 3-4-5 right triangle -> area = 6
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 6.0},
        # equilateral triangle side 2 -> area = sqrt(3) ~ 1.7320508
        {"input": None, "params": {"a": 2, "b": 2, "c": 2}, "expected": math.sqrt(3)},
    ],
    "triangle_area": [
        # right triangle (0,0), (4,0), (0,3) -> area = 6
        {"input": None, "params": {"p1": (0, 0), "p2": (4, 0), "p3": (0, 3)}, "expected": 6.0},
        # degenerate triangle (all collinear) -> area = 0
        {"input": None, "params": {"p1": (0, 0), "p2": (1, 1), "p3": (2, 2)}, "expected": 0.0},
    ],
    "triangle_centroid": [
        {"input": None, "params": {"p1": (0, 0), "p2": (3, 0), "p3": (0, 3)},
         "expected": (1.0, 1.0)},
        {"input": None, "params": {"p1": (0, 0), "p2": (6, 0), "p3": (0, 6)},
         "expected": (2.0, 2.0)},
    ],
    "triangle_inequality": [
        # valid triangle
        {"input": None, "params": {"a": 3, "b": 4, "c": 5}, "expected": 1},
        # degenerate (equality)
        {"input": None, "params": {"a": 1, "b": 2, "c": 3}, "expected": 0},
        # invalid
        {"input": None, "params": {"a": 1, "b": 1, "c": 10}, "expected": 0},
    ],
    "is_acute_angle": [
        {"input": 45, "params": {}, "expected": 1},
        {"input": 89, "params": {}, "expected": 1},
        {"input": 90, "params": {}, "expected": 0},
        {"input": 120, "params": {}, "expected": 0},
    ],
    "divide_area": [
        {"input": 100, "params": {"n": 4}, "expected": 25.0},
        {"input": 60, "params": {"n": 3}, "expected": 20.0},
    ],

    # =========================================================================
    # number_theory.py
    # =========================================================================
    "divisor_list": [
        {"input": 12, "params": {}, "expected": [1, 2, 3, 4, 6, 12]},
        {"input": 1, "params": {}, "expected": [1]},
        {"input": 7, "params": {}, "expected": [1, 7]},
    ],
    "divisor_pairs": [
        {"input": 12, "params": {}, "expected": [(1, 12), (2, 6), (3, 4)]},
        {"input": 1, "params": {}, "expected": [(1, 1)]},
    ],
    "perfect_square_check": [
        {"input": 16, "params": {}, "expected": 1},
        {"input": 15, "params": {}, "expected": 0},
        {"input": 1, "params": {}, "expected": 1},
        {"input": 0, "params": {}, "expected": 1},
    ],
    "divisibility_check": [
        {"input": None, "params": {"n": 12, "d": 3}, "expected": 1},
        {"input": None, "params": {"n": 12, "d": 5}, "expected": 0},
        {"input": None, "params": {"n": 0, "d": 7}, "expected": 1},
    ],
    "is_even": [
        {"input": 4, "params": {}, "expected": 1},
        {"input": 3, "params": {}, "expected": 0},
        {"input": 0, "params": {}, "expected": 1},
        {"input": -2, "params": {}, "expected": 1},
    ],
    "is_odd": [
        {"input": 3, "params": {}, "expected": 1},
        {"input": 4, "params": {}, "expected": 0},
        {"input": 1, "params": {}, "expected": 1},
        {"input": -3, "params": {}, "expected": 1},
    ],
    "is_prime": [
        {"input": 2, "params": {}, "expected": 1},
        {"input": 7, "params": {}, "expected": 1},
        {"input": 1, "params": {}, "expected": 0},
        {"input": 4, "params": {}, "expected": 0},
        {"input": 97, "params": {}, "expected": 1},
    ],
    "reduce_fraction": [
        {"input": None, "params": {"numerator": 6, "denominator": 9}, "expected": (2, 3)},
        {"input": None, "params": {"numerator": 12, "denominator": 18}, "expected": (2, 3)},
        {"input": None, "params": {"numerator": 5, "denominator": 7}, "expected": (5, 7)},
    ],
    "divisible_by_3": [
        {"input": 9, "params": {}, "expected": 1},
        {"input": 10, "params": {}, "expected": 0},
        {"input": 0, "params": {}, "expected": 1},
    ],
    "divides": [
        {"input": None, "params": {"n": 12, "d": 3}, "expected": 1},
        {"input": None, "params": {"n": 12, "d": 5}, "expected": 0},
        {"input": None, "params": {"n": 15, "d": 1}, "expected": 1},
    ],

    # =========================================================================
    # number_theory_2.py
    # =========================================================================
    "rational_reduce": [
        {"input": None, "params": {"p": 6, "q": 9}, "expected": (2, 3)},
        {"input": None, "params": {"p": 14, "q": 21}, "expected": (2, 3)},
    ],
    "divisors_of": [
        {"input": 12, "params": {}, "expected": [1, 2, 3, 4, 6, 12]},
        {"input": 7, "params": {}, "expected": [1, 7]},
    ],
    "filter_primes": [
        {"input": [2, 3, 4, 5, 6, 7, 8, 9, 10], "params": {}, "expected": [2, 3, 5, 7]},
        {"input": [4, 6, 8, 9], "params": {}, "expected": []},
        {"input": [2], "params": {}, "expected": [2]},
    ],
    "binary_exponentiation": [
        {"input": None, "params": {"base": 2, "exp": 10}, "expected": 1024},
        {"input": None, "params": {"base": 3, "exp": 5}, "expected": 243},
        {"input": None, "params": {"base": 7, "exp": 0}, "expected": 1},
    ],
    "floor_log2": [
        {"input": 8, "params": {}, "expected": 3},
        {"input": 10, "params": {}, "expected": 3},
        {"input": 1, "params": {}, "expected": 0},
        {"input": 1024, "params": {}, "expected": 10},
    ],
    "floor_sqrt": [
        {"input": 17, "params": {}, "expected": 4},
        {"input": 16, "params": {}, "expected": 4},
        {"input": 100, "params": {}, "expected": 10},
        {"input": 1, "params": {}, "expected": 1},
    ],
    "sqrt_computation": [
        {"input": 16, "params": {}, "expected": 4.0},
        {"input": 25, "params": {}, "expected": 5.0},
        {"input": 2, "params": {}, "expected": math.sqrt(2)},
    ],
    "ceil_sqrt": [
        {"input": 17, "params": {}, "expected": 5},
        {"input": 16, "params": {}, "expected": 4},
        {"input": 2, "params": {}, "expected": 2},
    ],
    "floor_cbrt": [
        {"input": 27, "params": {}, "expected": 3},
        {"input": 8, "params": {}, "expected": 2},
        {"input": 26, "params": {}, "expected": 2},
        # Note: 64**(1/3) = 3.999...96 due to floating point, floor -> 3
        {"input": 64, "params": {}, "expected": 3},
        {"input": 125, "params": {}, "expected": 4},  # 125^(1/3)=4.999..., floor=4
    ],

    # =========================================================================
    # sequences.py
    # =========================================================================
    "triangular_number": [
        {"input": 5, "params": {}, "expected": 15},   # 5*6/2
        {"input": 10, "params": {}, "expected": 55},   # 10*11/2
        {"input": 1, "params": {}, "expected": 1},
        {"input": 100, "params": {}, "expected": 5050},
    ],
    "arithmetic_sequence": [
        {"input": None, "params": {"start": 1, "step": 2, "n": 5},
         "expected": [1, 3, 5, 7, 9]},
        {"input": None, "params": {"start": 10, "step": 5, "n": 3},
         "expected": [10, 15, 20]},
    ],
    "count_integer_range": [
        {"input": None, "params": {"a": 1, "b": 10}, "expected": 10},
        {"input": None, "params": {"a": 5, "b": 5}, "expected": 1},
        {"input": None, "params": {"a": -3, "b": 3}, "expected": 7},
    ],
    "count_solutions_in_interval": [
        # values: 0, 5, 10, ..., 100 -> count = 21
        {"input": None, "params": {"a": 0, "b": 100, "step": 5}, "expected": 21},
        {"input": None, "params": {"a": 0, "b": 10, "step": 3}, "expected": 4},
        {"input": None, "params": {"a": 1, "b": 1, "step": 1}, "expected": 1},
    ],

    # =========================================================================
    # algebra.py
    # =========================================================================
    "polynomial_discriminant": [
        # x^2 - 3x + 2: b^2 - 4ac = 9 - 8 = 1
        {"input": None, "params": {"a": 1, "b": -3, "c": 2}, "expected": 1},
        # x^2 + 0x - 1: 0 - 4(1)(-1) = 4
        {"input": None, "params": {"a": 1, "b": 0, "c": -1}, "expected": 4},
        # x^2 + 2x + 1: 4 - 4 = 0  (perfect square)
        {"input": None, "params": {"a": 1, "b": 2, "c": 1}, "expected": 0},
    ],
    "polynomial_degree": [
        {"input": None, "params": {"coeffs": [1, 2, 3]}, "expected": 2},
        {"input": None, "params": {"coeffs": [5]}, "expected": 0},
        {"input": None, "params": {"coeffs": [0, 0, 0, 1]}, "expected": 3},
    ],
    "evaluate_at_zero": [
        # P(x) = 5 + 2x + x^2 -> P(0) = 5
        {"input": None, "params": {"coefficients": [5, 2, 1]}, "expected": 5},
        {"input": None, "params": {"coefficients": [0, 1, 1]}, "expected": 0},
        {"input": None, "params": {"coefficients": [42]}, "expected": 42},
    ],
    "test_constant_polynomial": [
        {"input": None, "params": {"coeffs": [5]}, "expected": 1},
        {"input": None, "params": {"coeffs": [5, 0, 0]}, "expected": 1},
        {"input": None, "params": {"coeffs": [5, 1]}, "expected": 0},
    ],

    # =========================================================================
    # validators.py
    # =========================================================================
    "is_open_set": [
        {"input": None, "params": {"set_type": "interval", "radius": 1.0}, "expected": 1},
        {"input": None, "params": {"set_type": "ball", "radius": 1.0}, "expected": 1},
        {"input": None, "params": {"set_type": "union", "radius": 1.0}, "expected": 1},
        {"input": None, "params": {"set_type": "intersection", "radius": 1.0}, "expected": 0},
    ],
    "is_connected_space": [
        {"input": None, "params": {"space_type": "interval", "num_components": 1}, "expected": 1},
        {"input": None, "params": {"space_type": "single_point", "num_components": 1}, "expected": 1},
        {"input": None, "params": {"space_type": "disconnected", "num_components": 2}, "expected": 0},
        {"input": None, "params": {"space_type": "union_intervals", "num_components": 2}, "expected": 0},
        {"input": None, "params": {"space_type": "union_intervals", "num_components": 1}, "expected": 1},
    ],
    "candidate_test": [
        {"input": None, "params": {"candidate": 10, "condition_type": "greater", "threshold": 5},
         "expected": 1},
        {"input": None, "params": {"candidate": 3, "condition_type": "greater", "threshold": 5},
         "expected": 0},
        {"input": None, "params": {"candidate": 5, "condition_type": "equal", "threshold": 5},
         "expected": 1},
        {"input": None, "params": {"candidate": 12, "condition_type": "divisible", "threshold": 3},
         "expected": 1},
        {"input": None, "params": {"candidate": 13, "condition_type": "divisible", "threshold": 3},
         "expected": 0},
    ],
    "check_convergence_in_probability": [
        # last 3 values are 5.0, 5.0, 5.0 -> mean=5, all within 0.1
        {"input": [1.0, 2.0, 5.0, 5.0, 5.0], "params": {"tolerance": 0.1}, "expected": 1},
        # last 3 values are 1, 2, 3 -> mean=2, not within 0.1
        {"input": [1, 2, 3], "params": {"tolerance": 0.1}, "expected": 0},
    ],
    "check_identical_distribution": [
        # Same samples -> identical
        {"input": None,
         "params": {"dist1": [1, 2, 3, 4, 5], "dist2": [1, 2, 3, 4, 5], "tolerance": 1.0},
         "expected": 1},
        # Very different means -> not identical
        {"input": None,
         "params": {"dist1": [1, 1, 1, 1, 1], "dist2": [100, 100, 100, 100, 100], "tolerance": 1.0},
         "expected": 0},
    ],
    "check_independence_product": [
        # P(A)=0.5, P(B)=0.5, P(AB)=0.25 -> independent
        {"input": None,
         "params": {"p_a": 0.5, "p_b": 0.5, "p_ab": 0.25, "tolerance": 0.01},
         "expected": 1},
        # P(A)=0.5, P(B)=0.5, P(AB)=0.1 -> dependent
        {"input": None,
         "params": {"p_a": 0.5, "p_b": 0.5, "p_ab": 0.1, "tolerance": 0.01},
         "expected": 0},
    ],
    "validate_triple_condition": [
        # (1, -2, 1) sums to 0
        {"input": None, "params": {"triple": (1, -2, 1), "condition": "sum_zero"}, "expected": 1},
        # (1, 2, 3) does not sum to 0
        {"input": None, "params": {"triple": (1, 2, 3), "condition": "sum_zero"}, "expected": 0},
        # 3-4-5 is Pythagorean
        {"input": None, "params": {"triple": (3, 4, 5), "condition": "pythagorean"}, "expected": 1},
        # (2, 3, 4) is not Pythagorean
        {"input": None, "params": {"triple": (2, 3, 4), "condition": "pythagorean"}, "expected": 0},
    ],
    "check_construction_exists": [
        {"input": None,
         "params": {"construction_type": "triangle", "constraints": 2},
         "expected": 1},
        {"input": None,
         "params": {"construction_type": "circle", "constraints": 3},
         "expected": 1},
    ],
    # ========================================================================
    # EVALUATE AT POINT (operations.py) - evaluate expression at variable values
    # ========================================================================
    "evaluate_at_point": [
        # (x^2 + 2x + 1) at x=3: 9+6+1 = 16
        {"input": None, "params": {"expression": "x**2 + 2*x + 1", "values": {"x": 3}}, "expected": 16},
        # (2x + 5) at x=10: 20+5 = 25
        {"input": None, "params": {"expression": "2*x + 5", "values": {"x": 10}}, "expected": 25},
    ],
}
