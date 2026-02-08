# fmt: off
"""
Known-value ground-truth tests for the analysis_methods domain.

Every method in prism/src/prism/techniques/analysis_methods/ has at least
one hand-verified test case.  Expected values were computed by hand or
confirmed via Python arithmetic (see inline comments).
"""
import math

_GAMMA = 0.5772156649  # Euler-Mascheroni constant

KNOWN_VALUES = {
    # ================================================================
    # limits.py
    # ================================================================

    # a/n + b/n^2  =>  1/10 + 2/100 = 0.12
    "limit_integral_expression": [
        {
            "input": None,
            "params": {"a": 1, "b": 2, "n": 10},
            "expected": 0.12,
        },
        {
            "input": None,
            "params": {"a": 5, "b": 0, "n": 100},
            "expected": 0.05,
        },
    ],

    # numerator / denominator (simple ratio)
    "analyze_limit": [
        {
            "input": None,
            "params": {"numerator": 10, "denominator": 5},
            "expected": 2.0,
        },
        {
            "input": None,
            "params": {"numerator": 7, "denominator": 2},
            "expected": 3.5,
        },
    ],

    # f_coeff * g_coeff * point
    "compose_limits": [
        {
            "input": None,
            "params": {"f_coeff": 2.0, "g_coeff": 3.0, "point": 1.0},
            "expected": 6.0,
        },
        {
            "input": None,
            "params": {"f_coeff": 5.0, "g_coeff": 2.0, "point": 0.5},
            "expected": 5.0,
        },
    ],

    # 1 / input_value
    "limit_of_reciprocal": [
        {
            "input": 4.0,
            "params": {},
            "expected": 0.25,
        },
        {
            "input": 0.5,
            "params": {},
            "expected": 2.0,
        },
    ],

    # coeff * x^power
    # power=0 => coeff, power<0 => coeff * x^power
    "limit_at_infinity": [
        {
            "input": None,
            "params": {"coeff": 5.0, "power": 0, "x": 100.0},
            "expected": 5.0,
        },
        {
            "input": None,
            "params": {"coeff": 3.0, "power": -2, "x": 100.0},
            "expected": 0.0003,
        },
    ],

    # 1 if left==right, else 0
    "limit_existence_conclusion": [
        {
            "input": None,
            "params": {"left_lim": 3.0, "right_lim": 3.0},
            "expected": 1,
        },
        {
            "input": None,
            "params": {"left_lim": 2.0, "right_lim": 5.0},
            "expected": 0,
        },
    ],

    # returns the input_value as the constant
    "limit_implies_constant": [
        {
            "input": 7.0,
            "params": {},
            "expected": 7.0,
        },
        {
            "input": None,
            "params": {"limit_val": 3.5},
            "expected": 3.5,
        },
    ],

    # liminf == limsup => that value, else average
    "limit_combination_from_liminf_limsup": [
        {
            "input": None,
            "params": {"liminf": 4.0, "limsup": 4.0},
            "expected": 4.0,
        },
        {
            "input": None,
            "params": {"liminf": 2.0, "limsup": 6.0},
            "expected": 4.0,
        },
    ],

    # min(sequence)
    "liminf_conclusion": [
        {
            "input": None,
            "params": {"sequence": [5, 3, 8, 1, 9]},
            "expected": 1,
        },
        {
            "input": None,
            "params": {"sequence": [-2, 0, 4, 7]},
            "expected": -2,
        },
    ],

    # f_limit * g_limit
    "limit_of_product": [
        {
            "input": None,
            "params": {"f_limit": 3.0, "g_limit": 4.0},
            "expected": 12.0,
        },
        {
            "input": None,
            "params": {"f_limit": 0.5, "g_limit": 6.0},
            "expected": 3.0,
        },
    ],

    # ================================================================
    # limits_2.py
    # ================================================================

    # last element of sequence
    "limit_as_n_goes_to_infinity": [
        {
            "input": None,
            "params": {"sequence": [3.0, 2.5, 2.1, 2.01, 2.001]},
            "expected": 2.001,
        },
    ],

    # max of last min(5,len) nth roots.
    # sequence=[2,4,8,16,32]: nth_roots all = 2.0
    "nth_root_limit_bounds": [
        {
            "input": None,
            "params": {"sequence": [2, 4, 8, 16, 32]},
            "expected": 2.0,
        },
        {
            "input": None,
            "params": {"sequence": [1, 1, 1, 1, 1]},
            "expected": 1.0,
        },
    ],

    # iterative sqrt(a+x), a=2, iter=30 => ~2.0
    # exact = (1+sqrt(9))/2 = 2.0
    "nested_sqrt_limit_sequence": [
        {
            "input": None,
            "params": {"a": 2.0, "iterations": 30},
            "expected": 2.0,
        },
        {
            "input": None,
            "params": {"a": 6.0, "iterations": 20},
            "expected": 3.0,
        },
    ],

    # sum(coeff_i * point^i)
    # [1,1,1] at 0.5 => 1 + 0.5 + 0.25 = 1.75
    "limit_power_series": [
        {
            "input": None,
            "params": {"coefficients": [1, 1, 1], "point": 0.5},
            "expected": 1.75,
        },
        {
            "input": None,
            "params": {"coefficients": [1, 0, 0], "point": 0.5},
            "expected": 1.0,
        },
    ],

    # average of last ratios.  Geometric seq => ratio = 2.0
    "ratio_analysis_limit": [
        {
            "input": None,
            "params": {"sequence": [1, 2, 4, 8, 16]},
            "expected": 2.0,
        },
    ],

    # sum of elements at given indices
    # [10,20,30,40,50] at [0,2,4] => 10+30+50 = 90
    "sequence_extraction": [
        {
            "input": None,
            "params": {
                "sequence": [10, 20, 30, 40, 50],
                "indices": [0, 2, 4],
            },
            "expected": 90,
        },
    ],

    # int(abs(max_by_abs(terms)))
    # [3,-7,2,5] => dominant=-7, result=7
    "dominant_term_selection": [
        {
            "input": None,
            "params": {"terms": [3.0, -7.0, 2.0, 5.0]},
            "expected": 7,
        },
        {
            "input": None,
            "params": {"terms": [1.0, -1.0, 0.5]},
            "expected": 1,
        },
    ],

    # ================================================================
    # asymptotics.py
    # ================================================================

    # total - partial_sum
    "asymptotic_tail_sum": [
        {
            "input": None,
            "params": {"total": 100, "partial_sum": 60},
            "expected": 40,
        },
    ],

    # exact - approximation
    "correction_term": [
        {
            "input": None,
            "params": {"approximation": 95, "exact": 100},
            "expected": 5,
        },
        {
            "input": None,
            "params": {"approximation": 102, "exact": 100},
            "expected": -2,
        },
    ],

    # coefficient * x^power
    "asymptotic_integral_behavior": [
        {
            "input": None,
            "params": {"coefficient": 2, "power": -1, "x": 10.0},
            "expected": 0.2,
        },
    ],

    # pi/2 - cos(x)/x - sin(x)/x^2
    "sine_integral_asymptotic": [
        {
            "input": 10.0,
            "params": {},
            "expected": math.pi / 2 - math.cos(10) / 10 - math.sin(10) / 100,
        },
    ],

    # sin(x)/x - cos(x)/x^2
    "cosine_integral_asymptotic": [
        {
            "input": 10.0,
            "params": {},
            "expected": math.sin(10) / 10 - math.cos(10) / 100,
        },
    ],

    # returns coefficient (ignores input and power)
    "substitute_asymptotics": [
        {
            "input": 5.0,
            "params": {"coefficient": 3.0, "power": 2.0},
            "expected": 3.0,
        },
        {
            "input": 100.0,
            "params": {"coefficient": 7.5, "power": -1.0},
            "expected": 7.5,
        },
    ],

    # value / x^power
    "match_asymptotic_coefficient": [
        {
            "input": None,
            "params": {"value": 100.0, "x": 10.0, "power": 2},
            "expected": 1.0,
        },
        {
            "input": None,
            "params": {"value": 50.0, "x": 5.0, "power": 1},
            "expected": 10.0,
        },
    ],

    # ================================================================
    # asymptotics_2.py
    # ================================================================

    # Same formula as sine_integral_asymptotic
    "asymptotic_expansion_si": [
        {
            "input": 10.0,
            "params": {},
            "expected": math.pi / 2 - math.cos(10) / 10 - math.sin(10) / 100,
        },
    ],

    # Same formula as cosine_integral_asymptotic
    "asymptotic_expansion_ci": [
        {
            "input": 10.0,
            "params": {},
            "expected": math.sin(10) / 10 - math.cos(10) / 100,
        },
    ],

    # coefficient * x^power
    "asymptotic_leading_term": [
        {
            "input": None,
            "params": {"coefficient": 2.0, "x": 10.0, "power": 1},
            "expected": 20.0,
        },
        {
            "input": None,
            "params": {"coefficient": 3.0, "x": 5.0, "power": -1},
            "expected": 0.6,
        },
    ],

    # int(log(2)/log(base))
    "logarithmic_limit_transform": [
        {
            "input": 1.0,
            "params": {"base": 2},
            "expected": 1,
        },
        {
            "input": 1.0,
            "params": {"base": math.e},
            "expected": 0,
        },
        {
            "input": 1.0,
            "params": {"base": 10},
            "expected": 0,
        },
    ],

    # int(min(max((a*x^p)/(b*x^q), 0), 99999))
    # a=2,p=3,b=1,q=1,x=10 => 2000/10 = 200
    "asymptotic_ratio_comparison": [
        {
            "input": None,
            "params": {"a": 2, "p": 3, "b": 1, "q": 1, "x": 10},
            "expected": 200,
        },
        {
            "input": None,
            "params": {"a": 3, "p": 2, "b": 1, "q": 2, "x": 10},
            "expected": 3,
        },
    ],

    # int(abs(n*log(a) - m*log(b)) * 100) % 100000
    # 2^10=1024, 3^6=729 => log_diff ~ 0.3398 => int(33.98) = 33
    "exponent_comparison": [
        {
            "input": None,
            "params": {"a": 2.0, "n": 10, "b": 3.0, "m": 6},
            "expected": 33,
        },
        {
            "input": None,
            "params": {"a": 2.0, "n": 10, "b": 2.0, "m": 10},
            "expected": 0,
        },
    ],

    # ================================================================
    # convergence.py
    # ================================================================

    # max(sequence)
    "supremum_limit": [
        {
            "input": None,
            "params": {"sequence": [1, 5, 3, 2, 4]},
            "expected": 5,
        },
    ],

    # last element of values
    "dominated_convergence_limit": [
        {
            "input": None,
            "params": {"values": [10, 7, 5, 3, 1]},
            "expected": 1,
        },
    ],

    # last element of fn_values
    "pointwise_limit_analysis": [
        {
            "input": None,
            "params": {"fn_values": [4, 3, 2, 1, 0.5]},
            "expected": 0.5,
        },
    ],

    # max_val * interval_length
    "integrand_dominated_bound": [
        {
            "input": None,
            "params": {"max_val": 3, "interval_length": 4},
            "expected": 12,
        },
    ],

    # last element of norm_sequence
    "conclude_norm_limit": [
        {
            "input": None,
            "params": {"norm_sequence": [5, 4, 3, 2, 1]},
            "expected": 1,
        },
        {
            "input": None,
            "params": {"norm_sequence": [10, 5, 2.5]},
            "expected": 2.5,
        },
    ],

    # x^(p - q)
    # p=2, q=1, x=0.01 => 0.01
    "mutual_singularity_limit": [
        {
            "input": None,
            "params": {"numerator_power": 2, "denominator_power": 1, "point": 0.01},
            "expected": 0.01,
        },
        {
            "input": None,
            "params": {"numerator_power": 1, "denominator_power": 1, "point": 0.01},
            "expected": 1.0,
        },
    ],

    # p>1 => 1/(p-1);  p<=1 => 0
    "integral_convergence_test": [
        {
            "input": None,
            "params": {"p": 2.0},
            "expected": 1.0,
        },
        {
            "input": None,
            "params": {"p": 3.0},
            "expected": 0.5,
        },
        {
            "input": None,
            "params": {"p": 0.5},
            "expected": 0,
        },
    ],

    # max(abs(v) for v in values)
    "supremum_norm_bound": [
        {
            "input": None,
            "params": {"values": [1, -3, 2, -5, 4]},
            "expected": 5,
        },
        {
            "input": None,
            "params": {"values": [0.1, -0.2, 0.3]},
            "expected": 0.3,
        },
    ],

    # ================================================================
    # convergence_2.py
    # ================================================================

    # min(region)
    "infimum_of_convergence_region": [
        {
            "input": None,
            "params": {"region": [-3, -1, 0, 2, 5]},
            "expected": -3,
        },
    ],

    # max(integrals)
    "supremum_of_ball_integrals": [
        {
            "input": None,
            "params": {"integrals": [1, 7, 3, 5, 2]},
            "expected": 7,
        },
    ],

    # max(values)
    "function_supremum": [
        {
            "input": None,
            "params": {"values": [2, 8, 4, 6, 1]},
            "expected": 8,
        },
    ],

    # 1 if tail_max < 0.1 else 0
    "weak_star_null_convergence": [
        {
            "input": None,
            "params": {"sequence": [0.05, 0.04, 0.03, 0.02, 0.01]},
            "expected": 1,
        },
        {
            "input": None,
            "params": {"sequence": [5, 4, 3, 2, 1]},
            "expected": 0,
        },
    ],

    # ================================================================
    # series.py
    # ================================================================

    # sum(terms)
    "combine_terms": [
        {
            "input": None,
            "params": {"terms": [1, 2, 3, 4, 5]},
            "expected": 15,
        },
        {
            "input": None,
            "params": {"terms": [-1, 1, -1, 1]},
            "expected": 0,
        },
    ],

    # H_n = sum(1/k, k=1..n)
    # H_5 = 1 + 1/2 + 1/3 + 1/4 + 1/5 = 137/60
    "harmonic_sum_minimal": [
        {
            "input": None,
            "params": {"n": 1},
            "expected": 1.0,
        },
        {
            "input": None,
            "params": {"n": 5},
            "expected": sum(1.0 / k for k in range(1, 6)),
        },
    ],

    # sum((-1)^(k+1)/k, k=1..n)
    # n=4: 1 - 1/2 + 1/3 - 1/4 = 7/12
    "alternating_harmonic_series_sum": [
        {
            "input": None,
            "params": {"n": 1},
            "expected": 1.0,
        },
        {
            "input": None,
            "params": {"n": 4},
            "expected": 1.0 - 0.5 + 1.0 / 3.0 - 0.25,
        },
    ],

    # Taylor series of log(1+x): sum((-1)^(n+1) * x^n / n, n=1..terms)
    "power_series_expansion_log": [
        {
            "input": None,
            "params": {"x": 0.5, "terms": 5},
            "expected": sum((-1) ** (n + 1) * 0.5 ** n / n for n in range(1, 6)),
        },
    ],

    # ln(n) + gamma
    "harmonic_series_divergence": [
        {
            "input": None,
            "params": {"n": 10},
            "expected": math.log(10) + _GAMMA,
        },
        {
            "input": None,
            "params": {"n": 1},
            "expected": math.log(1) + _GAMMA,
        },
    ],

    # same Taylor series as power_series_expansion_log but uses n_terms
    "logarithm_expansion": [
        {
            "input": None,
            "params": {"x": 0.5, "n_terms": 10},
            "expected": sum((-1) ** (n + 1) * 0.5 ** n / n for n in range(1, 11)),
        },
    ],

    # ================================================================
    # misc.py
    # ================================================================

    # max(cases)
    "max_value_from_cases": [
        {
            "input": None,
            "params": {"cases": [3, 1, 4, 1, 5, 9, 2, 6]},
            "expected": 9,
        },
    ],

    # max(numerator_i / denominator_i)
    # [10,20,15] / [5,8,3] => [2.0, 2.5, 5.0] => 5.0
    "max_ratio_over_configurations": [
        {
            "input": None,
            "params": {
                "numerators": [10.0, 20.0, 15.0],
                "denominators": [5.0, 8.0, 3.0],
            },
            "expected": 5.0,
        },
    ],

    # roots of r^2 + a*r + b = 0  via  (-a +/- sqrt(a^2-4b))/2
    # a=-5, b=6: disc=1, roots=[3.0, 2.0]
    "characteristic_equation_roots": [
        {
            "input": None,
            "params": {"a": -5, "b": 6},
            "expected": [3.0, 2.0],
        },
        {
            "input": None,
            "params": {"a": 0, "b": -4},
            "expected": [2.0, -2.0],
        },
    ],

    # c / b  (particular solution for y'' + ay' + by = c)
    "particular_solution_undetermined_coeffs": [
        {
            "input": None,
            "params": {"a": 1, "b": 5, "c": 10},
            "expected": 2.0,
        },
        {
            "input": None,
            "params": {"a": 0, "b": 4, "c": 12},
            "expected": 3.0,
        },
    ],

    # min of convex f(x) = ax^2 + bx + c is c - b^2/(4a)
    # a=1, b=-4, c=5 => 5 - 16/4 = 1.0
    "convex_function_properties": [
        {
            "input": None,
            "params": {"a": 1.0, "b": -4.0, "c": 5.0},
            "expected": 1.0,
        },
        {
            "input": None,
            "params": {"a": 2.0, "b": 0.0, "c": 3.0},
            "expected": 3.0,
        },
    ],

    # 1 if |value| < 1e-10 else 0
    "is_zero_ring": [
        {
            "input": 0,
            "params": {},
            "expected": 1,
        },
        {
            "input": 5,
            "params": {},
            "expected": 0,
        },
        {
            "input": None,
            "params": {"value": 0.0},
            "expected": 1,
        },
    ],

    # E[Ito integral] = 0 always
    "ito_integral_expectation": [
        {
            "input": None,
            "params": {"process_name": "f", "time_interval": 1.0},
            "expected": 0,
        },
    ],

    # 1 if min_val <= k <= max_val else 0
    "test_feasibility_k": [
        {
            "input": None,
            "params": {"k": 50, "min_val": 10, "max_val": 100},
            "expected": 1,
        },
        {
            "input": None,
            "params": {"k": 5, "min_val": 10, "max_val": 100},
            "expected": 0,
        },
        {
            "input": None,
            "params": {"k": 200, "min_val": 10, "max_val": 100},
            "expected": 0,
        },
    ],
}
