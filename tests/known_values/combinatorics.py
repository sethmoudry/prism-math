# fmt: off
"""
Known-value ground-truth tests for combinatorics techniques.

Every method exported from prism.techniques.combinatorics has at least one
deterministic test case.  Expected values are hand-verified or computed from
well-known sequences (OEIS, textbook identities).

Format:
    method_name -> [{"input": ..., "params": {...}, "expected": value}, ...]

The audit runner calls  method.compute(input, params)  and compares
result.value against expected.
"""

KNOWN_VALUES = {
    # ===================================================================
    # counting_basic.py  (12 techniques)
    # ===================================================================

    # -- Catalan --
    "catalan": [
        {"input": None, "params": {"n": 1}, "expected": 1},
        {"input": None, "params": {"n": 2}, "expected": 2},
        {"input": None, "params": {"n": 3}, "expected": 5},
        {"input": None, "params": {"n": 4}, "expected": 14},
        {"input": None, "params": {"n": 5}, "expected": 42},
        {"input": None, "params": {"n": 6}, "expected": 132},
    ],
    "catalan_inverse": [
        {"input": 14, "params": {"target": 14}, "expected": 4},
        {"input": 42, "params": {"target": 42}, "expected": 5},
        {"input": 132, "params": {"target": 132}, "expected": 6},
    ],

    # -- Binomial --
    "binomial": [
        {"input": 5,  "params": {"n": 5, "k": 2},  "expected": 10},
        {"input": 10, "params": {"n": 10, "k": 3}, "expected": 120},
        {"input": 10, "params": {"n": 10, "k": 7}, "expected": 120},
        {"input": 0,  "params": {"n": 0, "k": 0},  "expected": 1},
        {"input": 10, "params": {"n": 10, "k": 0}, "expected": 1},
        {"input": 7,  "params": {"n": 7, "k": 7},  "expected": 1},
    ],
    "binomial_inverse_n": [
        {"input": 120, "params": {"target": 120, "k": 3}, "expected": 10},
    ],
    "binomial_inverse_k": [
        {"input": 20, "params": {"target": 20, "n": 6}, "expected": 3},
    ],
    "binomial_sum": [
        {"input": 5,  "params": {"n": 5},  "expected": 32},
        {"input": 10, "params": {"n": 10}, "expected": 1024},
    ],

    # -- Lattice paths / Ballot / Dyck --
    "lattice_paths": [
        {"input": 2, "params": {"m": 2, "n": 3}, "expected": 10},
        {"input": 3, "params": {"m": 3, "n": 3}, "expected": 20},
        {"input": 0, "params": {"m": 0, "n": 0}, "expected": 1},
    ],
    "lattice_paths_inverse_m": [
        {"input": 10, "params": {"target": 10, "n": 3}, "expected": 2},
    ],
    "ballot": [
        {"input": 5, "params": {"a": 5, "b": 3}, "expected": 14},
        {"input": 7, "params": {"a": 7, "b": 3}, "expected": 48},
    ],
    "dyck_paths": [
        {"input": 4, "params": {"n": 4}, "expected": 14},
        {"input": 5, "params": {"n": 5}, "expected": 42},
        {"input": 3, "params": {"n": 3}, "expected": 5},
    ],

    # -- Counting / CountingPrinciples --
    "counting": [
        {"input": 5, "params": {"n": 5, "k": 2, "counting_type": "permutation"}, "expected": 20},
        {"input": 5, "params": {"n": 5, "k": 2, "counting_type": "combination"}, "expected": 10},
        {"input": 10, "params": {"n": 10, "k": 3, "counting_type": "permutation"}, "expected": 720},
    ],
    "counting_principles": [
        {"input": None, "params": {"outcomes_per_stage": [3, 4, 5], "use_addition": False}, "expected": 60},
        {"input": None, "params": {"outcomes_per_stage": [3, 4, 5], "use_addition": True}, "expected": 12},
    ],

    # ===================================================================
    # counting_advanced.py  (5 techniques)
    # ===================================================================

    "pigeonhole": [
        {"input": 10, "params": {"n_holes": 10, "n_pigeons": 25}, "expected": 3},
        {"input": 5,  "params": {"n_holes": 5, "n_pigeons": 20},  "expected": 4},
    ],
    "pigeonhole_generalized": [
        {"input": 50, "params": {"k_holes": 7, "n_pigeons": 50}, "expected": 8},
    ],
    "double_count": [
        {"input": 10, "params": {"n": 10, "object": "edges_K_n"}, "expected": 45},
        {"input": 6,  "params": {"n": 6, "object": "edges_K_n"},  "expected": 15},
    ],
    "bijection": [
        {"input": 5, "params": {"n": 5, "bijection_type": "binary_strings"}, "expected": 32},
        {"input": 5, "params": {"n": 5, "bijection_type": "compositions"}, "expected": 16},
        {"input": 4, "params": {"n": 4, "bijection_type": "dyck_paths"}, "expected": 14},
        {"input": 3, "params": {"n": 3, "m": 4, "bijection_type": "lattice_paths"}, "expected": 35},
    ],
    "counting_bijection": [
        {"input": 10, "params": {"n": 10, "k": 3}, "expected": 120},
    ],

    # ===================================================================
    # special_numbers_stirling_bell.py  (4 techniques)
    # ===================================================================

    "stirling_first": [
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 50},
        {"input": 4, "params": {"n": 4, "k": 1}, "expected": 6},
        {"input": 4, "params": {"n": 4, "k": 3}, "expected": 6},
    ],
    "stirling_second": [
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 15},
        {"input": 5, "params": {"n": 5, "k": 3}, "expected": 25},
        {"input": 4, "params": {"n": 4, "k": 2}, "expected": 7},
        {"input": 7, "params": {"n": 7, "k": 1}, "expected": 1},
        {"input": 7, "params": {"n": 7, "k": 7}, "expected": 1},
    ],
    "stirling_second_inverse_n": [
        {"input": 15, "params": {"target": 15, "k": 2}, "expected": 5},
    ],
    "bell": [
        {"input": 3, "params": {"n": 3}, "expected": 5},
        {"input": 4, "params": {"n": 4}, "expected": 15},
        {"input": 5, "params": {"n": 5}, "expected": 52},
    ],

    # ===================================================================
    # special_numbers_derangements_partitions.py  (13 techniques)
    # ===================================================================

    # -- Derangements --
    "derangement": [
        {"input": 4, "params": {"n": 4}, "expected": 9},
        {"input": 5, "params": {"n": 5}, "expected": 44},
        {"input": 6, "params": {"n": 6}, "expected": 265},
        {"input": 1, "params": {"n": 1}, "expected": 0},
        {"input": 3, "params": {"n": 3}, "expected": 2},
    ],
    "derangement_inverse": [
        {"input": 9,  "params": {"target": 9},  "expected": 4},
        {"input": 44, "params": {"target": 44}, "expected": 5},
    ],
    "subfactorial": [
        {"input": 4, "params": {"n": 4}, "expected": 9},
        {"input": 5, "params": {"n": 5}, "expected": 44},
    ],

    # -- Partitions --
    "partition": [
        {"input": 5,  "params": {"n": 5},  "expected": 7},
        {"input": 10, "params": {"n": 10}, "expected": 42},
        {"input": 4,  "params": {"n": 4},  "expected": 5},
    ],
    "partition_inverse": [
        {"input": 7,  "params": {"target": 7},  "expected": 5},
        {"input": 42, "params": {"target": 42}, "expected": 10},
    ],
    "partition_k_parts": [
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 2},
        {"input": 5, "params": {"n": 5, "k": 5}, "expected": 1},
        {"input": 6, "params": {"n": 6, "k": 3}, "expected": 3},
        {"input": 10, "params": {"n": 10, "k": 3}, "expected": 8},
    ],
    "integer_partitions_into_k": [
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 2},
        {"input": 6, "params": {"n": 6, "k": 3}, "expected": 3},
    ],

    # -- Large / explosive growth --
    "large_factorial": [
        {"input": None, "params": {"n": 12}, "expected": 479001600},
        {"input": None, "params": {"n": 15}, "expected": 1307674368000},
    ],
    "large_binomial": [
        {"input": None, "params": {"n": 60, "k": 30}, "expected": 118264581564861424},
    ],
    "large_catalan": [
        {"input": None, "params": {"n": 12}, "expected": 208012},
        {"input": None, "params": {"n": 15}, "expected": 9694845},
    ],
    "large_stirling": [
        {"input": None, "params": {"n": 15, "k": 2, "kind": "first"}, "expected": 283465647360},
        {"input": None, "params": {"n": 20, "k": 5, "kind": "second"}, "expected": 749206090500},
    ],

    # -- Binary tree counting --
    "binary_tree_count_recurrence": [
        {"input": 4, "params": {"n": 4}, "expected": 14},
        {"input": 5, "params": {"n": 5}, "expected": 42},
        {"input": 0, "params": {"n": 0}, "expected": 1},
    ],

    # ===================================================================
    # graphs_basic.py  (8 techniques)
    # ===================================================================

    "chromatic_poly": [
        {"input": 5, "params": {"graph_type": "tree", "n": 5, "k": 3}, "expected": 48},
        {"input": 4, "params": {"graph_type": "cycle", "n": 4, "k": 3}, "expected": 18},
        {"input": 3, "params": {"graph_type": "cycle", "n": 3, "k": 3}, "expected": 6},
    ],
    "graph_coloring": [
        # Computes k * (k-1)^(n-1) (tree / path formula)
        {"input": 5, "params": {"n": 5, "k": 3}, "expected": 48},
        {"input": 3, "params": {"n": 3, "k": 2}, "expected": 2},
    ],
    "spanning_trees": [
        {"input": 4, "params": {"graph_type": "complete", "n": 4}, "expected": 16},
        {"input": 5, "params": {"graph_type": "complete", "n": 5}, "expected": 125},
        {"input": 10, "params": {"graph_type": "cycle", "n": 10}, "expected": 10},
    ],
    "eulerian_paths": [
        {"input": 7, "params": {"graph_type": "cycle", "n": 7}, "expected": 7},
    ],
    "hall_marriage": [
        {"input": 4, "params": {"n": 4, "graph_type": "complete_bipartite"}, "expected": 24},
        {"input": 5, "params": {"n": 5, "graph_type": "complete_bipartite"}, "expected": 120},
    ],
    "graph_theory": [
        {"input": 5, "params": {"n": 5, "operation": "complete_graph_edges"}, "expected": 10},
        {"input": 4, "params": {"n": 4, "m": 3, "operation": "complete_bipartite_edges"}, "expected": 12},
        {"input": 7, "params": {"n": 7, "edges": 7, "operation": "sum_of_degrees"}, "expected": 14},
        {"input": 5, "params": {"n": 5, "operation": "spanning_trees"}, "expected": 125},
        {"input": 7, "params": {"n": 7, "operation": "cycle_edges"}, "expected": 7},
        {"input": 5, "params": {"n": 5, "operation": "path_edges"}, "expected": 4},
    ],
    "hamiltonian_path": [
        {"input": 5, "params": {"graph_type": "complete", "n": 5}, "expected": 60},
        {"input": 5, "params": {"graph_type": "complete_cycle", "n": 5}, "expected": 12},
        {"input": 3, "params": {"graph_type": "path_exists", "n": 3}, "expected": 1},
    ],
    "turan_bound": [
        # ex(6, K_3): 2 parts of size 3 each. (36 - 18)/2 = 9
        {"input": 6, "params": {"n": 6, "r": 3}, "expected": 9},
        # ex(9, K_4): 3 parts of size 3. (81 - 27)/2 = 27
        {"input": 9, "params": {"n": 9, "r": 4}, "expected": 27},
    ],

    # ===================================================================
    # graphs_flow.py  (4 techniques)
    # ===================================================================

    "seymour_flow_bound": [
        {"input": None, "params": {}, "expected": 6},
    ],
    "max_power_of_two_below_bound": [
        {"input": 6,  "params": {"n": 6},  "expected": 2},
        {"input": 15, "params": {"n": 15}, "expected": 3},
        {"input": 32, "params": {"n": 32}, "expected": 5},
        {"input": 1,  "params": {"n": 1},  "expected": 0},
    ],
    "max_m_satisfying_flow_bound": [
        {"input": None, "params": {
            "flow_upper_bound": 6,
            "require_nonzero_flow": True,
            "min_chromatic_for_flow_4": 2,
        }, "expected": 2},
    ],
    "platonic_eulerian_circuits": [
        # Tetrahedron has odd-degree vertices -> 0
        {"input": None, "params": {"solid": "tetrahedron"}, "expected": 0},
        # Octahedron (all even degree) -> 1
        {"input": None, "params": {"solid": "octahedron"}, "expected": 1},
        # Cube has odd-degree -> 0
        {"input": None, "params": {"solid": "cube"}, "expected": 0},
    ],

    # ===================================================================
    # generating_functions.py  (4 techniques)
    # ===================================================================

    "generating_function": [
        # [x^3](1+x)^7 = C(7,3) = 35
        {"input": 7, "params": {"operation": "coefficient", "n": 7, "k": 3}, "expected": 35},
        # Fibonacci F_10 = 55
        {"input": 10, "params": {"operation": "fibonacci", "n": 10}, "expected": 55},
        {"input": 5, "params": {"operation": "fibonacci", "n": 5}, "expected": 5},
        # Catalan C_5 = 42
        {"input": 5, "params": {"operation": "catalan", "n": 5}, "expected": 42},
    ],
    "counting_via_algebra": [
        # [x^3](1+x)^8 = C(8,3) = 56
        {"input": 8, "params": {"operation": "coefficient_extraction", "n": 8, "k": 3}, "expected": 56},
        # Stars and bars: x1+x2+x3=5 => C(7,2) = 21
        {"input": 5, "params": {"operation": "stars_and_bars", "n": 5, "k": 3}, "expected": 21},
        # Fibonacci F_10 = 55
        {"input": 10, "params": {"operation": "fibonacci_gf", "n": 10}, "expected": 55},
        # Catalan C_4 = 14
        {"input": 4, "params": {"operation": "catalan_gf", "n": 4}, "expected": 14},
        # [x^2](2x+1)^5 = C(5,2)*4*1 = 40
        {"input": 5, "params": {"operation": "binomial_expansion", "n": 5, "k": 2, "a": 2, "b": 1}, "expected": 40},
    ],
    "binomial_theorem_analogy": [
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 10},
        {"input": 10, "params": {"n": 10, "k": 4}, "expected": 210},
    ],
    "hook_length_formula": [
        # 2x2 rectangle: 4!/(3*2*2*1) = 24/12 = 2
        {"input": 4, "params": {"rows": 2, "cols": 2}, "expected": 2},
        # 2x3 rectangle: 6!/144 = 5
        {"input": 6, "params": {"rows": 2, "cols": 3}, "expected": 5},
        # 3x3 rectangle: 9!/8640 = 42
        {"input": 9, "params": {"rows": 3, "cols": 3}, "expected": 42},
    ],

    # ===================================================================
    # probability.py  (4 techniques)
    # ===================================================================

    "probability": [
        # P(sum=7 with 2 dice) = 6/36 = 1/6
        {"input": None, "params": {"scenario": "dice_sum", "target_sum": 7}, "expected": (1, 6)},
        # P(exactly 2 heads in 4 flips) = C(4,2)/16 = 6/16 = 3/8
        {"input": 4, "params": {"scenario": "coins", "n": 4, "k": 2}, "expected": (3, 8)},
        # P(two cards same suit) = 4*C(13,2)/C(52,2) = 312/1326 = 4/17
        {"input": None, "params": {"scenario": "cards_same_suit"}, "expected": (4, 17)},
        # P(at least one 6 in 3 dice) = 1-(5/6)^3 = 91/216
        {"input": 3, "params": {"scenario": "at_least_one", "n": 3, "target": 6}, "expected": (91, 216)},
    ],
    "conditional_probability": [
        # P(hearts | red card) = 1/2
        {"input": None, "params": {"scenario": "cards"}, "expected": (1, 2)},
    ],
    "entropy_compute": [
        # Uniform on 4 outcomes: H = log2(4) = 2.0 bits -> scaled 2000
        {"input": None, "params": {
            "distribution_type": "uniform", "n": 4,
            "probabilities": [0.25, 0.25, 0.25, 0.25],
        }, "expected": 2000},
        # Uniform on 8: H = 3.0 -> 3000
        {"input": None, "params": {
            "distribution_type": "uniform", "n": 8,
            "probabilities": [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        }, "expected": 3000},
    ],
    "conditional_expectation_symmetric": [
        # Independent, X in [0,10]: E[X] = 5.0 -> scaled 500
        {"input": None, "params": {
            "case": "independent", "X_min": 0, "X_max": 10, "Y_min": 0, "Y_max": 10,
        }, "expected": 500},
    ],

    # ===================================================================
    # topology.py  (5 techniques)
    # ===================================================================

    "topological_complexity_graph": [
        # Bouquet of 3 circles: 1 essential vertex -> TC = 2*1+1 = 3
        {"input": 3, "params": {
            "graph_type": "bouquet", "n_circles": 3, "n_essential_vertices": 1,
        }, "expected": 3},
        # No essential vertices -> TC = 1 (tree ~ interval)
        {"input": 0, "params": {
            "graph_type": "tree", "n_essential_vertices": 0,
        }, "expected": 1},
    ],
    "bouquet_of_circles": [
        # TC always 3 (Farber)
        {"input": 5, "params": {"n_circles": 5, "property": "tc"}, "expected": 3},
        # Degree of wedge point = 2n
        {"input": 4, "params": {"n_circles": 4, "property": "degree"}, "expected": 8},
        # Rank of pi_1 = n
        {"input": 3, "params": {"n_circles": 3, "property": "fundamental_group_rank"}, "expected": 3},
    ],
    "kleitman_rothschild_height_bound": [
        {"input": 100, "params": {"n": 100}, "expected": 3},
    ],
    "essential_vertex_count": [
        {"input": 5, "params": {"graph_type": "bouquet", "n_circles": 5}, "expected": 1},
        {"input": 3, "params": {"graph_type": "general", "n_essential_vertices": 3}, "expected": 3},
    ],
    "figure_eight_topology": [
        {"input": None, "params": {"property": "tc"}, "expected": 3},
        {"input": None, "params": {"property": "essential_vertices"}, "expected": 1},
        {"input": None, "params": {"property": "vertex_degree"}, "expected": 4},
        {"input": None, "params": {"property": "fundamental_group_rank"}, "expected": 2},
    ],

    # ===================================================================
    # misc.py  (16 techniques)
    # ===================================================================

    "counting_constraints": [
        # Binary strings of length 5 with no consecutive 1s = 13
        {"input": 5, "params": {"n": 5, "constraint_type": "no_consecutive"}, "expected": 13},
        # Derangement of 4 = 9
        {"input": 4, "params": {"n": 4, "constraint_type": "derangement"}, "expected": 9},
        # 4-digit ascending = C(9,4) = 126
        {"input": 4, "params": {"n": 4, "constraint_type": "ascending_digits"}, "expected": 126},
        # Write 6 as sum of 3 positive ints = C(5,2) = 10
        {"input": 6, "params": {"n": 6, "k": 3, "constraint_type": "sum_to_n"}, "expected": 10},
    ],
    "counting_intersections": [
        # 5 lines in general position: C(5,2) = 10
        {"input": 5, "params": {"n": 5}, "expected": 10},
    ],
    "counting_paths": [
        # Basic lattice paths (3,4) = C(7,4) = 35
        {"input": 3, "params": {"m": 3, "n": 4, "path_type": "basic"}, "expected": 35},
        # Catalan path n=4 = C_4 = 14
        {"input": 4, "params": {"m": 4, "n": 4, "path_type": "catalan"}, "expected": 14},
    ],
    "counting_regions": [
        # n lines in general position: 1 + n + C(n,2)
        # 5 lines: 1 + 5 + 10 = 16
        {"input": 5, "params": {"n": 5}, "expected": 16},
        {"input": 3, "params": {"n": 3}, "expected": 7},
    ],
    "permutation_decomposition": [
        # s(5,2) = 50  (unsigned Stirling first kind)
        {"input": 5, "params": {"n": 5, "k": 2}, "expected": 50},
    ],
    "permutation_with_repetition": [
        {"input": 3, "params": {"n": 3, "k": 4}, "expected": 81},
        {"input": 2, "params": {"n": 2, "k": 3}, "expected": 8},
    ],
    "permutation_order": [
        # LCM(2, 3) = 6
        {"input": None, "params": {"cycle_lengths": [2, 3]}, "expected": 6},
        # LCM(2, 3, 5) = 30
        {"input": None, "params": {"cycle_lengths": [2, 3, 5]}, "expected": 30},
        # LCM(4, 6) = 12
        {"input": None, "params": {"cycle_lengths": [4, 6]}, "expected": 12},
    ],
    "multiset_permutation_total": [
        # MISSISSIPPI-style: frequencies [2,3,1] -> 6!/(2!3!1!) = 60
        {"input": None, "params": {"frequencies": [2, 3, 1]}, "expected": 60},
        # [2,2,2] -> 6!/(2!2!2!) = 90
        {"input": None, "params": {"frequencies": [2, 2, 2]}, "expected": 90},
    ],
    "multiset_permutation_avoiding_pattern": [
        # freq=[2,2], pattern [1,1] of length 2:
        # total=6, IE gives at_least_one=5, result=1
        {"input": None, "params": {
            "frequencies": [2, 2],
            "pattern_frequencies": [1, 1],
            "pattern_length": 2,
        }, "expected": 1},
    ],
    "min_area_function": [
        # s <= N: result = s - 1
        {"input": None, "params": {"s": 5, "N": 50}, "expected": 4},
        # s > N: result = N * (s - N)
        {"input": None, "params": {"s": 60, "N": 50}, "expected": 500},
    ],
    "sum_min_areas": [
        # k=3, N=50: s in [2,3,4] -> (1)+(2)+(3) = 6
        {"input": None, "params": {"k": 3, "N": 50}, "expected": 6},
    ],
    "legendre_valuation_factorial": [
        # v_2(10!) = 5 + 2 + 1 = 8
        {"input": 10, "params": {"n": 10, "p": 2}, "expected": 8},
        # v_5(100!) = 20 + 4 = 24
        {"input": 100, "params": {"n": 100, "p": 5}, "expected": 24},
    ],
    "place_value_sum": [
        # [3,2,1] -> 3*10^0 + 2*10^1 + 1*10^2 = 3+20+100 = 123
        {"input": [3, 2, 1], "params": {"digits": [3, 2, 1]}, "expected": 123},
    ],
    "factorial_base_to_decimal": [
        # [1,1,0] -> 1*1! + 1*2! + 0*3! = 1+2+0 = 3
        {"input": [1, 1, 0], "params": {"factorial_base": [1, 1, 0]}, "expected": 3},
        # [0,1,1] -> 0*1! + 1*2! + 1*3! = 0+2+6 = 8
        {"input": [0, 1, 1], "params": {"factorial_base": [0, 1, 1]}, "expected": 8},
    ],
    "count_odd": [
        {"input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "params": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, "expected": 5},
    ],
    "count_even": [
        {"input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "params": {"numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, "expected": 5},
    ],
    "circular_arrangements_with_adjacent_constraint": [
        # n=5, 1 pair: units=4, (4-1)!*2^1 = 6*2 = 12
        {"input": 5, "params": {"n": 5, "pairs": [(1, 2)]}, "expected": 12},
        # n=6, 2 pairs: units=4, (4-1)!*2^2 = 6*4 = 24
        {"input": 6, "params": {"n": 6, "pairs": [(1, 2), (3, 4)]}, "expected": 24},
    ],
}
