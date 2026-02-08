# fmt: off
"""Known ground-truth values for advanced_techniques and operations domains.

Covers:
  - 6 topology methods from combinatorics/topology.py
  - 8 core operational methods from operations.py
  - 27+ alias operational methods from operations.py

Every entry has been hand-verified against the implementation.
Format: method_name -> [{input, params, expected}, ...]
"""

KNOWN_VALUES = {
    # =========================================================================
    # Topology methods  (combinatorics/topology.py)
    # =========================================================================
    "topological_complexity_graph": [
        # Bouquet with 1 essential vertex: TC = 2*1 + 1 = 3
        {"input": None,
         "params": {"graph_type": "bouquet", "n_circles": 3, "n_essential_vertices": 1},
         "expected": 3},
        # No essential vertices (tree) -> TC = 1
        {"input": None,
         "params": {"graph_type": "tree", "n_essential_vertices": 0},
         "expected": 1},
        # General graph with 2 essential vertices: TC = 2*2 + 1 = 5
        {"input": None,
         "params": {"graph_type": "general", "n_essential_vertices": 2},
         "expected": 5},
        # Interval -> TC = 1
        {"input": None,
         "params": {"graph_type": "interval", "n_essential_vertices": 0},
         "expected": 1},
    ],
    "bouquet_of_circles": [
        # TC is always 3 for a bouquet of circles (any n >= 1)
        {"input": None, "params": {"n_circles": 5, "property": "tc"}, "expected": 3},
        # Degree of wedge point = 2n
        {"input": None, "params": {"n_circles": 4, "property": "degree"}, "expected": 8},
        {"input": None, "params": {"n_circles": 3, "property": "degree"}, "expected": 6},
        # Fundamental group rank = n
        {"input": None, "params": {"n_circles": 7, "property": "fundamental_group_rank"},
         "expected": 7},
    ],
    "kleitman_rothschild_height_bound": [
        # Always returns 3 regardless of n (Kleitman-Rothschild theorem)
        {"input": None, "params": {"n": 100}, "expected": 3},
        {"input": None, "params": {"n": 10}, "expected": 3},
        {"input": None, "params": {"n": 1000}, "expected": 3},
    ],
    "essential_vertex_count": [
        # Bouquet always has 1 essential vertex
        {"input": None, "params": {"graph_type": "bouquet", "n_circles": 5}, "expected": 1},
        {"input": None, "params": {"graph_type": "bouquet", "n_circles": 2}, "expected": 1},
        # General graph with specified count
        {"input": None, "params": {"graph_type": "general", "n_essential_vertices": 3},
         "expected": 3},
    ],
    "figure_eight_topology": [
        # TC of figure-eight = 3
        {"input": None, "params": {"property": "tc"}, "expected": 3},
        # 1 essential vertex (the wedge point)
        {"input": None, "params": {"property": "essential_vertices"}, "expected": 1},
        # Wedge point has degree 4
        {"input": None, "params": {"property": "vertex_degree"}, "expected": 4},
        # Fundamental group rank = 2 (free group F_2)
        {"input": None, "params": {"property": "fundamental_group_rank"}, "expected": 2},
    ],

    # =========================================================================
    # Core operational methods  (operations.py)
    # =========================================================================

    # --- Compute ---
    "compute": [
        # GCD
        {"input": None, "params": {"operation": "gcd", "a": 12, "b": 8}, "expected": 4},
        {"input": None, "params": {"operation": "gcd", "a": 15, "b": 25}, "expected": 5},
        # LCM
        {"input": None, "params": {"operation": "lcm", "a": 4, "b": 6}, "expected": 12},
        # Mod
        {"input": None, "params": {"operation": "mod", "a": 17, "modulus": 5}, "expected": 2},
        # Arithmetic: add
        {"input": None,
         "params": {"operation": "arithmetic", "a": 10, "b": 3, "op_type": "add"},
         "expected": 13},
        # Arithmetic: multiply
        {"input": None,
         "params": {"operation": "arithmetic", "a": 7, "b": 6, "op_type": "multiply"},
         "expected": 42},
        # Arithmetic: power
        {"input": None,
         "params": {"operation": "arithmetic", "a": 2, "b": 10, "op_type": "power"},
         "expected": 1024},
    ],

    # --- Verify ---
    "verify": [
        # Positive constraint, value=5 -> True, passthrough 5
        {"input": 5, "params": {"condition_type": "constraint", "constraint": "positive"},
         "expected": 5},
        # Positive constraint, value=-3 -> False
        {"input": -3, "params": {"condition_type": "constraint", "constraint": "positive"},
         "expected": False},
        # Base case: value matches expected
        {"input": 42, "params": {"condition_type": "base_case", "expected": 42},
         "expected": 42},
        # Base case: value does NOT match expected -> False
        {"input": 42, "params": {"condition_type": "base_case", "expected": 99},
         "expected": False},
        # Boundary: lower bound, 5 >= 0 -> passthrough
        {"input": 5, "params": {"condition_type": "boundary", "boundary": "lower", "limit": 0},
         "expected": 5},
    ],

    # --- ControlFlow ---
    "control_flow": [
        # Extremal: minimum of list
        {"input": [3, 1, 4, 1, 5, 9], "params": {"flow_type": "extremal", "extremal_type": "minimum"},
         "expected": 1},
        # Extremal: maximum of list
        {"input": [3, 1, 4, 1, 5, 9], "params": {"flow_type": "extremal", "extremal_type": "maximum"},
         "expected": 9},
        # Pigeonhole: 10 pigeons, 9 holes -> guaranteed min 2
        {"input": 10, "params": {"flow_type": "pigeonhole", "holes": 9},
         "expected": 2},
        # Pigeonhole: 25 pigeons, 6 holes -> guaranteed min 5
        {"input": 25, "params": {"flow_type": "pigeonhole", "holes": 6},
         "expected": 5},
        # Case split: modular, 7 mod 3 = 1
        {"input": 7, "params": {"flow_type": "case_split", "criterion": "modular", "modulus": 3},
         "expected": 1},
    ],

    # --- ExtractCoefficients ---
    "extract_coefficients": [
        # List of coefficients: degree 0 -> first element
        {"input": [5, 2, 1], "params": {"expression_type": "polynomial", "degree": 0},
         "expected": 5},
        # List of coefficients: degree 1 -> second element
        {"input": [5, 2, 1], "params": {"expression_type": "polynomial", "degree": 1},
         "expected": 2},
        # Constant polynomial: degree 0 -> value, degree > 0 -> 0
        {"input": 42, "params": {"expression_type": "polynomial", "degree": 0},
         "expected": 42},
        {"input": 42, "params": {"expression_type": "polynomial", "degree": 1},
         "expected": 0},
    ],

    # --- ExtractResult ---
    "extract_result": [
        # Integer format
        {"input": 42, "params": {"format": "integer"}, "expected": 42},
        {"input": 3.7, "params": {"format": "integer"}, "expected": 4},
        # Mod format
        {"input": 123456, "params": {"format": "mod", "modulus": 1000}, "expected": 456},
        # Integer format with large number (raw value, no mod)
        {"input": 123456, "params": {"format": "integer"}, "expected": 123456},
    ],

    # =========================================================================
    # Alias operational methods  (operations.py)
    # =========================================================================

    # --- Compute aliases ---
    "compute_arithmetic": [
        {"input": None, "params": {"a": 5, "b": 3, "op_type": "add"}, "expected": 8},
        {"input": None, "params": {"a": 5, "b": 3, "op_type": "subtract"}, "expected": 2},
        {"input": None, "params": {"a": 5, "b": 3, "op_type": "multiply"}, "expected": 15},
    ],
    "mod_reduce": [
        {"input": None, "params": {"value": 17, "modulus": 5}, "expected": 2},
        {"input": None, "params": {"value": 100, "modulus": 7}, "expected": 2},
        {"input": None, "params": {"value": 10, "modulus": 10}, "expected": 0},
    ],
    "compute_gcd": [
        {"input": None, "params": {"values": [12, 8]}, "expected": 4},
        {"input": None, "params": {"values": [15, 25, 35]}, "expected": 5},
        {"input": None, "params": {"values": [7, 13]}, "expected": 1},
    ],
    "compute_lcm": [
        {"input": None, "params": {"values": [4, 6]}, "expected": 12},
        {"input": None, "params": {"values": [3, 5, 7]}, "expected": 105},
        {"input": None, "params": {"values": [2, 4, 8]}, "expected": 8},
    ],

    # --- Verify aliases ---
    "verify_constraint": [
        # Positive value passes, returns passthrough
        {"input": 10, "params": {"constraint": "positive"}, "expected": 10},
        # Non-negative, value=0 passes
        {"input": 0, "params": {"constraint": "non_negative"}, "expected": 0},
        # Integer check, value=5 passes
        {"input": 5, "params": {"constraint": "integer"}, "expected": 5},
    ],
    "verify_base_case": [
        # Value matches expected -> passthrough
        {"input": 1, "params": {"expected": 1, "n": 0}, "expected": 1},
        # Value does NOT match expected -> False
        {"input": 1, "params": {"expected": 2, "n": 0}, "expected": False},
    ],
    "check_boundary": [
        # Lower bound: 5 >= 0 -> passthrough
        {"input": 5, "params": {"boundary": "lower", "limit": 0}, "expected": 5},
        # Upper bound: 5 <= 10 -> passthrough
        {"input": 5, "params": {"boundary": "upper", "limit": 10}, "expected": 5},
        # Upper bound: 15 <= 10 -> False
        {"input": 15, "params": {"boundary": "upper", "limit": 10}, "expected": False},
    ],

    # --- Control flow aliases ---
    "extremal_argument": [
        {"input": [7, 2, 9, 1, 4], "params": {"extremal_type": "minimum"}, "expected": 1},
        {"input": [7, 2, 9, 1, 4], "params": {"extremal_type": "maximum"}, "expected": 9},
    ],
    "pigeonhole_apply": [
        # 10 pigeons, 3 holes -> ceil(10/3) = 4
        {"input": 10, "params": {"holes": 3}, "expected": 4},
        # 7 pigeons, 2 holes -> ceil(7/2) = 4
        {"input": 7, "params": {"holes": 2}, "expected": 4},
    ],

    # --- Extract aliases ---
    "extract_answer": [
        {"input": 42, "params": {}, "expected": 42},
        {"input": 3.7, "params": {}, "expected": 4},
        {"input": -1, "params": {}, "expected": -1},       # raw value, no mod
        {"input": 100001, "params": {}, "expected": 100001},  # raw value, no mod
    ],
}
