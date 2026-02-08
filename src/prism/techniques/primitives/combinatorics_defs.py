"""
Combinatorics decompositions: counting, permutations, partitions, probability.

Sections covered:
- 2. COMBINATORICS DECOMPOSITIONS
- 7. STATISTICS/PROBABILITY
- 16. COMBINATORICS (extended)
"""

from ..decomposition import Decomposition

COMBINATORICS_DECOMPOSITIONS = {
    # =========================================================================
    # 2. COMBINATORICS DECOMPOSITIONS
    # =========================================================================

    "catalan": Decomposition(
        expression="divide(binomial(multiply(2, n), n), add(n, constant_one()))",
        param_map={"n": "n"},
        notes="C_n = C(2n,n)/(n+1)"
    ),

    "multinomial": Decomposition(
        expression="divide(factorial(n), product_factorials(ks))",
        param_map={"n": "n", "ks": "ks"},
        notes="n! / (k1! * k2! * ... * km!)"
    ),

    "bell_number": Decomposition(
        expression="sum_range(k, 0, n, stirling_second(n, k))",
        param_map={"n": "n"},
        notes="B_n = sum S(n,k) for k=0..n"
    ),

    "partition_number": Decomposition(
        expression="count_partitions(n)",
        param_map={"n": "n"},
        notes="p(n) = number of integer partitions"
    ),

    "derangement": Decomposition(
        expression="floor(add(divide(factorial(n), 2.718281828), 0.5))",
        param_map={"n": "n"},
        notes="D_n = n!/e rounded - subfactorial"
    ),

    "subfactorial": Decomposition(
        expression="floor(add(divide(factorial(n), 2.718281828), 0.5))",
        param_map={"n": "n"},
        notes="!n = derangements of n elements"
    ),

    "combinations_with_rep": Decomposition(
        expression="binomial(add(n, subtract(k, 1)), k)",
        param_map={"n": "n", "k": "k"},
        notes="C(n+k-1, k) - multiset coefficient"
    ),

    "permutations_with_rep": Decomposition(
        expression="power(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="n^k - arrangements with repetition"
    ),

    "falling_factorial": Decomposition(
        expression="divide(factorial(n), factorial(subtract(n, k)))",
        param_map={"n": "n", "k": "k"},
        notes="(n)_k = n!/(n-k)! = n*(n-1)*...*(n-k+1)"
    ),

    "rising_factorial": Decomposition(
        expression="divide(factorial(add(n, subtract(k, 1))), factorial(subtract(n, 1)))",
        param_map={"n": "n", "k": "k"},
        notes="n^(k) = (n+k-1)!/(n-1)! = n*(n+1)*...*(n+k-1)"
    ),

    "stirling_first": Decomposition(
        expression="stirling_first_kind(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="s(n,k) - unsigned Stirling numbers of first kind"
    ),

    "stirling_second": Decomposition(
        expression="divide(sum_range(j, 0, k, multiply(power(negate(1), subtract(k, j)), multiply(binomial(k, j), power(j, n)))), factorial(k))",
        param_map={"n": "n", "k": "k"},
        notes="S(n,k) = (1/k!) * sum (-1)^(k-j) * C(k,j) * j^n"
    ),

    "eulerian_number": Decomposition(
        expression="sum_range(j, 0, k, multiply(power(negate(1), j), multiply(binomial(add(n, 1), j), power(subtract(add(k, 1), j), n))))",
        param_map={"n": "n", "k": "k"},
        notes="A(n,k) = sum (-1)^j * C(n+1,j) * (k+1-j)^n"
    ),

    "binomial_sum": Decomposition(
        expression="power(2, n)",
        param_map={"n": "n"},
        notes="Sum of C(n,k) for k=0..n = 2^n"
    ),

    "central_binomial": Decomposition(
        expression="binomial(multiply(2, n), n)",
        param_map={"n": "n"},
        notes="C(2n, n) - central binomial coefficient"
    ),

    "double_factorial_odd": Decomposition(
        expression="divide(factorial(multiply(2, n)), multiply(power(2, n), factorial(n)))",
        param_map={"n": "n"},
        notes="(2n-1)!! = (2n)!/(2^n * n!)"
    ),

    "double_factorial_even": Decomposition(
        expression="multiply(power(2, n), factorial(n))",
        param_map={"n": "n"},
        notes="(2n)!! = 2^n * n!"
    ),

    "motzkin": Decomposition(
        expression="sum_range(k, 0, floor(divide(n, 2)), multiply(binomial(n, multiply(2, k)), catalan(k)))",
        param_map={"n": "n"},
        notes="M_n = sum C(n, 2k) * C_k"
    ),

    "narayana": Decomposition(
        expression="divide(multiply(binomial(n, k), binomial(n, subtract(k, 1))), n)",
        param_map={"n": "n", "k": "k"},
        notes="N(n,k) = C(n,k)*C(n,k-1)/n - Narayana numbers"
    ),

    "delannoy": Decomposition(
        expression="sum_range(k, 0, min_value(m, n), multiply(multiply(binomial(m, k), binomial(n, k)), power(2, k)))",
        param_map={"m": "m", "n": "n"},
        notes="D(m,n) = sum C(m,k) * C(n,k) * 2^k"
    ),

    # =========================================================================
    # 7. STATISTICS/PROBABILITY
    # =========================================================================

    "mean": Decomposition(
        expression="divide(sum(values), length(values))",
        param_map={"values": "values"},
        notes="Average = sum / count"
    ),

    "variance_population": Decomposition(
        expression="subtract(mean_of_squares(values), power(mean(values), 2))",
        param_map={"values": "values"},
        notes="Var = E[X^2] - E[X]^2"
    ),

    "expected_value_uniform": Decomposition(
        expression="divide(add(a, b), 2)",
        param_map={"a": "a", "b": "b"},
        notes="E[U(a,b)] = (a+b)/2"
    ),

    "expected_value_binomial": Decomposition(
        expression="multiply(n, p)",
        param_map={"n": "n", "p": "p"},
        notes="E[Bin(n,p)] = n*p"
    ),

    "variance_binomial": Decomposition(
        expression="multiply(multiply(n, p), subtract(1, p))",
        param_map={"n": "n", "p": "p"},
        notes="Var[Bin(n,p)] = n*p*(1-p)"
    ),

    "probability_binomial": Decomposition(
        expression="multiply(binomial(n, k), multiply(power(p, k), power(subtract(1, p), subtract(n, k))))",
        param_map={"n": "n", "k": "k", "p": "p"},
        notes="P(X=k) = C(n,k) * p^k * (1-p)^(n-k)"
    ),

    "expected_value_geometric": Decomposition(
        expression="divide(1, p)",
        param_map={"p": "p"},
        notes="E[Geo(p)] = 1/p"
    ),

    "variance_geometric": Decomposition(
        expression="divide(subtract(1, p), power(p, 2))",
        param_map={"p": "p"},
        notes="Var[Geo(p)] = (1-p)/p^2"
    ),

    "expected_value_poisson": Decomposition(
        expression="lambda_param",
        param_map={"lambda_param": "lambda_param"},
        notes="E[Poi(lambda)] = lambda"
    ),

    "variance_poisson": Decomposition(
        expression="lambda_param",
        param_map={"lambda_param": "lambda_param"},
        notes="Var[Poi(lambda)] = lambda"
    ),

    "probability_poisson": Decomposition(
        expression="divide(multiply(power(lambda_param, k), power(2.718281828, negate(lambda_param))), factorial(k))",
        param_map={"lambda_param": "lambda_param", "k": "k"},
        notes="P(X=k) = (lambda^k * e^-lambda) / k!"
    ),

    "bayes_posterior": Decomposition(
        expression="divide(multiply(p_b_given_a, p_a), p_b)",
        param_map={"p_b_given_a": "p_b_given_a", "p_a": "p_a", "p_b": "p_b"},
        notes="P(A|B) = P(B|A) * P(A) / P(B)"
    ),

    # =========================================================================
    # 16. COMBINATORICS (extended)
    # =========================================================================

    "dyck_paths": Decomposition(
        expression="divide(binomial(multiply(2, n), n), add(n, constant_one()))",
        param_map={"n": "n"},
        notes="Dyck paths of length 2n = C_n = C(2n,n)/(n+1)"
    ),

    "binary_tree_count_recurrence": Decomposition(
        expression="divide(binomial(multiply(2, n), n), add(n, constant_one()))",
        param_map={"n": "n"},
        notes="Binary trees with n nodes = Catalan number C_n"
    ),

    "catalan_inverse": Decomposition(
        expression="search_catalan_n(target)",
        param_map={"target": "target"},
        notes="Find n where C_n = target (search-based)"
    ),

    "lattice_paths": Decomposition(
        expression="binomial(add(m, n), n)",
        param_map={"m": "m", "n": "n"},
        notes="Paths from (0,0) to (m,n) = C(m+n, n)"
    ),

    "lattice_paths_inverse_m": Decomposition(
        expression="search_m_for_lattice_paths(target, n)",
        param_map={"target": "target", "n": "n"},
        notes="Find m where C(m+n,n) = target"
    ),

    "ballot": Decomposition(
        expression="divide(multiply(subtract(a, b), binomial(add(a, b), a)), add(a, b))",
        param_map={"a": "a", "b": "b"},
        notes="Ballot problem: (a-b)/(a+b) * C(a+b, a)"
    ),

    "binomial_inverse_n": Decomposition(
        expression="search_n_for_binomial(target, k)",
        param_map={"target": "target", "k": "k"},
        notes="Find n where C(n,k) = target"
    ),

    "binomial_inverse_k": Decomposition(
        expression="search_k_for_binomial(target, n)",
        param_map={"target": "target", "n": "n"},
        notes="Find k where C(n,k) = target"
    ),

    "stirling_second_inverse_n": Decomposition(
        expression="search_n_for_stirling_second(target, k)",
        param_map={"target": "target", "k": "k"},
        notes="Find n where S(n,k) = target"
    ),

    "derangement_inverse": Decomposition(
        expression="search_n_for_derangement(target)",
        param_map={"target": "target"},
        notes="Find n where D_n = target"
    ),

    "derangement_via_inclusion_exclusion": Decomposition(
        expression="floor(add(divide(factorial(n), 2.71828), 0.5))",
        param_map={"n": "n"},
        notes="D_n via inclusion-exclusion = n! * sum(-1)^k/k!"
    ),

    "partition_inverse": Decomposition(
        expression="search_n_for_partition(target)",
        param_map={"target": "target"},
        notes="Find n where p(n) = target"
    ),

    "integer_partitions_into_k": Decomposition(
        expression="count_partitions_into_k(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="p(n,k) = partitions of n into exactly k parts"
    ),

    "spanning_trees": Decomposition(
        expression="power(n, subtract(n, 2))",
        param_map={"n": "n"},
        notes="Spanning trees of K_n = n^(n-2) (Cayley's formula)"
    ),

    "hall_marriage": Decomposition(
        expression="factorial(n)",
        param_map={"n": "n"},
        notes="Perfect matchings in K_{n,n} = n!"
    ),

    "chromatic_poly": Decomposition(
        expression="multiply(k, power(subtract(k, 1), subtract(n, 1)))",
        param_map={"n": "n", "k": "k"},
        notes="Chromatic polynomial of tree/path: k * (k-1)^(n-1)"
    ),

    "graph_coloring": Decomposition(
        expression="multiply(k, power(subtract(k, 1), subtract(n, 1)))",
        param_map={"n": "n", "k": "k"},
        notes="Proper colorings of tree with k colors: k * (k-1)^(n-1)"
    ),

    "eulerian_paths": Decomposition(
        expression="n",
        param_map={"n": "n"},
        notes="Eulerian paths in cycle graph C_n = n"
    ),

    "pigeonhole": Decomposition(
        expression="negate(floor(negate(divide(n_pigeons, n_holes))))",
        param_map={"n_pigeons": "n_pigeons", "n_holes": "n_holes"},
        notes="ceil(pigeons/holes) = minimum in some hole"
    ),

    "pigeonhole_generalized": Decomposition(
        expression="negate(floor(negate(divide(n_pigeons, k_holes))))",
        param_map={"n_pigeons": "n_pigeons", "k_holes": "k_holes"},
        notes="ceil(n/k) by generalized pigeonhole"
    ),

    "double_count": Decomposition(
        expression="divide(multiply(n, subtract(n, 1)), 2)",
        param_map={"n": "n"},
        notes="Edges in K_n = n(n-1)/2 by double counting"
    ),

    "counting_intersections": Decomposition(
        expression="divide(multiply(n, subtract(n, 1)), 2)",
        param_map={"n": "n"},
        notes="n lines in general position: C(n,2) = n(n-1)/2 intersections"
    ),

    "counting_regions": Decomposition(
        expression="add(add(1, n), divide(multiply(n, subtract(n, 1)), 2))",
        param_map={"n": "n"},
        notes="Regions from n lines: 1 + n + C(n,2) = 1 + n + n(n-1)/2"
    ),

    "counting_regions_space": Decomposition(
        expression="power(2, n)",
        param_map={"n": "n"},
        notes="Regions from n planes in 3D (max): 2^n"
    ),

    "permutation_with_repetition": Decomposition(
        expression="power(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="n^k - arrangements with repetition"
    ),

    "counting": Decomposition(
        expression="factorial(n)",
        param_map={"n": "n"},
        notes="n! - number of permutations"
    ),

    "counting_bijection": Decomposition(
        expression="binomial(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="C(n,k) - bijection count"
    ),

    "counting_principles": Decomposition(
        expression="divide(factorial(n), 2)",
        param_map={"n": "n"},
        notes="n!/2 - counting with symmetry"
    ),

    "permutation_order": Decomposition(
        expression="lcm_of_list(cycle_lengths)",
        param_map={"cycle_lengths": "cycle_lengths"},
        notes="Order of permutation = LCM of cycle lengths"
    ),

    "permutation_decomposition": Decomposition(
        expression="sum_factorials(n)",
        param_map={"n": "n"},
        notes="Sum of k! for k=1 to n"
    ),

    "circular_arrangements_with_adjacent_constraint": Decomposition(
        expression="multiply(factorial(subtract(subtract(n, num_pairs), 1)), power(2, num_pairs))",
        param_map={"n": "n", "num_pairs": "num_pairs"},
        notes="Circular arrangements with k adjacent pairs: (n-k-1)! * 2^k"
    ),

    "multiset_permutation_total": Decomposition(
        expression="divide(factorial(n), product_factorials(frequencies))",
        param_map={"n": "n", "frequencies": "frequencies"},
        notes="n! / (k1! * k2! * ... * km!) - multinomial"
    ),

    "legendre_valuation_factorial": Decomposition(
        expression="sum_legendre_terms(n, p)",
        param_map={"n": "n", "p": "p"},
        notes="v_p(n!) = sum(floor(n/p^k)) - Legendre's formula"
    ),

    "inclusion_exclusion": Decomposition(
        expression="inclusion_exclusion_union(set_sizes, universe_size)",
        param_map={"set_sizes": "set_sizes", "universe_size": "universe_size"},
        notes="PIE: |A union B union ...| via inclusion-exclusion"
    ),

    "inclusion_exclusion_principle": Decomposition(
        expression="subtract(sum(sets), intersection_sum(sets, n))",
        param_map={"sets": "sets", "n": "n"},
        notes="|A union B union C| via inclusion-exclusion"
    ),

    "sieve_count": Decomposition(
        expression="sieve_pie(n, primes)",
        param_map={"n": "n", "primes": "primes"},
        notes="Count integers in [1,n] coprime to primes via sieve"
    ),

    "linearity_of_expectation": Decomposition(
        expression="divide(multiply(multiply(n, p_num), 100), p_den)",
        param_map={"n": "n", "p_num": "p_num", "p_den": "p_den"},
        notes="E[X] = n * p (scaled by 100)"
    ),

    "expected_value": Decomposition(
        expression="multiply(subtract(n, 1), 50)",
        param_map={"n": "n"},
        notes="E[X] for uniform[0,n-1] = (n-1)/2 scaled by 100"
    ),

    "probabilistic_method": Decomposition(
        expression="multiply(binomial(n, k), probability)",
        param_map={"n": "n", "k": "k", "probability": "probability"},
        notes="Expected count of k-subsets with property"
    ),

    "binomial_theorem_analogy": Decomposition(
        expression="binomial(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="C(n,k) via binomial theorem"
    ),
}
