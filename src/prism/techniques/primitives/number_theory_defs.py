"""
Number theory decompositions: primes, divisors, modular arithmetic, GCD/LCM.

Sections covered:
- 1. NUMBER THEORY DECOMPOSITIONS
- 9. MODULAR OPERATIONS
- 13. GCD/LCM DECOMPOSITIONS
- 14. DIVISOR FUNCTIONS
- 15. NUMBER THEORY METHODS (extended)
"""

from ..decomposition import Decomposition

NUMBER_THEORY_DECOMPOSITIONS = {
    # =========================================================================
    # 1. NUMBER THEORY DECOMPOSITIONS
    # =========================================================================

    "euler_totient": Decomposition(
        expression="totient(n)",
        param_map={"n": "n"},
        notes="phi(n) - Euler's totient function"
    ),

    "divisor_count": Decomposition(
        expression="count_divisors(n)",
        param_map={"n": "n"},
        notes="d(n) = number of divisors"
    ),

    "prime_counting": Decomposition(
        expression="count_primes_leq(n)",
        param_map={"n": "n"},
        notes="pi(n) = count of primes <= n"
    ),

    "mobius": Decomposition(
        expression="mobius_function(n)",
        param_map={"n": "n"},
        notes="mu(n) - Mobius function"
    ),

    "legendre_symbol": Decomposition(
        expression="power(a, divide(subtract(p, 1), 2))",
        param_map={"a": "a", "p": "p"},
        notes="Legendre symbol (a/p) via Euler's criterion"
    ),

    "jacobi_symbol": Decomposition(
        expression="jacobi_compute(a, n)",
        param_map={"a": "a", "n": "n"},
        notes="Jacobi symbol (a|n)"
    ),

    "find_primitive_root": Decomposition(
        expression="primitive_root(p)",
        param_map={"p": "p"},
        notes="Find primitive root mod p"
    ),

    "multiplicative_order": Decomposition(
        expression="n_order(a, n)",
        param_map={"a": "a", "n": "n"},
        notes="Smallest k such that a^k = 1 mod n"
    ),

    "chinese_remainder": Decomposition(
        expression="chinese_remainder(remainders, moduli)",
        param_map={"remainders": "remainders", "moduli": "moduli"},
        notes="Chinese Remainder Theorem solution - PRIMITIVE"
    ),

    "is_perfect_square": Decomposition(
        expression="equal_to(multiply(floor(power(n, 0.5)), floor(power(n, 0.5))), n)",
        param_map={"n": "n"},
        notes="Check if n is a perfect square"
    ),

    "is_perfect_cube": Decomposition(
        expression="equal_to(power(floor(power(n, divide(1, 3))), 3), n)",
        param_map={"n": "n"},
        notes="Check if n is a perfect cube"
    ),

    "is_perfect_power": Decomposition(
        expression="check_perfect_power(n)",
        param_map={"n": "n"},
        notes="Check if n = a^b for some a,b > 1"
    ),

    "coprime_check": Decomposition(
        expression="equal_to(gcd(a, b), 1)",
        param_map={"a": "a", "b": "b"},
        notes="Check if gcd(a,b) = 1"
    ),

    "legendre_valuation": Decomposition(
        expression="sum_legendre_terms(n, p)",
        param_map={"n": "n", "p": "p"},
        notes="v_p(n!) = sum(floor(n/p^k))"
    ),

    "sum_of_divisors": Decomposition(
        expression="sum_divisors(n)",
        param_map={"n": "n"},
        notes="sigma(n) = sum of divisors of n"
    ),

    "radical": Decomposition(
        expression="product_distinct_primes(n)",
        param_map={"n": "n"},
        notes="rad(n) = product of distinct prime factors"
    ),

    "omega_small": Decomposition(
        expression="count_distinct_prime_factors(n)",
        param_map={"n": "n"},
        notes="omega(n) = number of distinct prime factors"
    ),

    "omega_big": Decomposition(
        expression="count_prime_factors_with_mult(n)",
        param_map={"n": "n"},
        notes="Omega(n) = number of prime factors with multiplicity"
    ),

    "liouville_lambda": Decomposition(
        expression="power(negate(1), count_prime_factors_with_mult(n))",
        param_map={"n": "n"},
        notes="lambda(n) = (-1)^Omega(n)"
    ),

    "is_squarefree": Decomposition(
        expression="equal_to(mobius_function(n), negate(mobius_function(n)))",
        param_map={"n": "n"},
        notes="n is squarefree iff mu(n) != 0"
    ),

    # =========================================================================
    # 9. MODULAR OPERATIONS
    # =========================================================================

    "mod_add": Decomposition(
        expression="mod(add(a, b), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a + b) mod m"
    ),

    "mod_sub": Decomposition(
        expression="mod(add(subtract(a, b), m), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a - b + m) mod m"
    ),

    "mod_mul": Decomposition(
        expression="mod(multiply(a, b), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a * b) mod m"
    ),

    "mod_pow": Decomposition(
        expression="mod(power(base, exp), m)",
        param_map={"base": "base", "exp": "exp", "m": "m"},
        notes="base^exp mod m"
    ),

    "mod_div": Decomposition(
        expression="mod(multiply(a, mod_inverse(b, m)), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a / b) mod m = a * b^(-1) mod m"
    ),

    "modular_add": Decomposition(
        expression="mod(add(a, b), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a+b) mod m"
    ),

    "modular_subtract": Decomposition(
        expression="mod(subtract(a, b), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a-b) mod m"
    ),

    "modular_multiply": Decomposition(
        expression="mod(multiply(a, b), m)",
        param_map={"a": "a", "b": "b", "m": "m"},
        notes="(a*b) mod m"
    ),

    "binomial_mod": Decomposition(
        expression="mod(binomial(n, k), m)",
        param_map={"n": "n", "k": "k", "m": "m"},
        notes="C(n,k) mod m"
    ),

    "factorial_mod": Decomposition(
        expression="mod(factorial(n), m)",
        param_map={"n": "n", "m": "m"},
        notes="n! mod m"
    ),

    "power_mod": Decomposition(
        expression="mod(power(base, exp), m)",
        param_map={"base": "base", "exp": "exp", "m": "m"},
        notes="base^exp mod m"
    ),

    "fermat_inverse": Decomposition(
        expression="mod(power(a, subtract(p, 2)), p)",
        param_map={"a": "a", "p": "p"},
        notes="a^(p-2) mod p for prime p (Fermat's little theorem)"
    ),

    # =========================================================================
    # 13. GCD/LCM DECOMPOSITIONS
    # =========================================================================

    "lcm_compute": Decomposition(
        expression="divide(multiply(a, b), gcd(a, b))",
        param_map={"a": "a", "b": "b"},
        notes="lcm(a,b) = a*b/gcd(a,b)"
    ),

    "gcd_extended": Decomposition(
        expression="extended_gcd(a, b)",
        param_map={"a": "a", "b": "b"},
        notes="Returns (gcd, x, y) such that ax + by = gcd"
    ),

    "gcd_three": Decomposition(
        expression="gcd(gcd(a, b), c)",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="gcd(a, b, c)"
    ),

    "lcm_three": Decomposition(
        expression="lcm(lcm(a, b), c)",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="lcm(a, b, c)"
    ),

    "bezout_coefficient_x": Decomposition(
        expression="bezout_x(a, b)",
        param_map={"a": "a", "b": "b"},
        notes="x such that ax + by = gcd(a,b)"
    ),

    "bezout_coefficient_y": Decomposition(
        expression="bezout_y(a, b)",
        param_map={"a": "a", "b": "b"},
        notes="y such that ax + by = gcd(a,b)"
    ),

    # =========================================================================
    # 14. DIVISOR FUNCTIONS
    # =========================================================================

    "divisor_sum": Decomposition(
        expression="sum_divisors(n)",
        param_map={"n": "n"},
        notes="sigma(n) = sum of divisors - calls sum_divisors PRIMITIVE"
    ),

    "proper_divisor_sum": Decomposition(
        expression="subtract(sum_divisors(n), n)",
        param_map={"n": "n"},
        notes="Sum of proper divisors = sigma(n) - n"
    ),

    "is_perfect_number": Decomposition(
        expression="equal_to(subtract(sum_divisors(n), n), n)",
        param_map={"n": "n"},
        notes="n is perfect if sum of proper divisors = n"
    ),

    "is_abundant": Decomposition(
        expression="compare_values(subtract(sum_divisors(n), n), n)",
        param_map={"n": "n"},
        notes="n is abundant if proper divisor sum > n"
    ),

    "is_deficient": Decomposition(
        expression="compare_values(n, subtract(sum_divisors(n), n))",
        param_map={"n": "n"},
        notes="n is deficient if proper divisor sum < n"
    ),

    # =========================================================================
    # 15. NUMBER THEORY METHODS (extended)
    # =========================================================================

    "padic_inverse": Decomposition(
        expression="power(p, v)",
        param_map={"p": "p", "v": "v"},
        notes="Smallest n with v_p(n) = v is p^v"
    ),

    "kummer_valuation": Decomposition(
        expression="divide(subtract(add(digit_sum_base(k, p), digit_sum_base(subtract(n, k), p)), digit_sum_base(n, p)), subtract(p, 1))",
        param_map={"n": "n", "k": "k", "p": "p"},
        notes="v_p(C(n,k)) = (S_p(k) + S_p(n-k) - S_p(n))/(p-1) by Kummer's theorem"
    ),

    "lifting_exponent_valuation": Decomposition(
        expression="padic_valuation(subtract(power(a, n), power(b, n)), p)",
        param_map={"a": "a", "b": "b", "n": "n", "p": "p"},
        notes="v_p(a^n - b^n) using LTE lemma"
    ),

    "divisor_count_inverse": Decomposition(
        expression="power(2, subtract(tau, 1))",
        param_map={"tau": "tau"},
        notes="For prime tau, smallest n with tau(n) = tau is 2^(tau-1)"
    ),

    "divisor_sum_k": Decomposition(
        expression="divisor_sigma(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="sigma_k(n) = sum of d^k over divisors d of n - calls divisor_sigma PRIMITIVE"
    ),

    "carmichael_lambda": Decomposition(
        expression="reduced_totient(n)",
        param_map={"n": "n"},
        notes="lambda(n) = Carmichael function = reduced totient"
    ),

    "gcd_compute": Decomposition(
        expression="gcd(a, b)",
        param_map={"a": "a", "b": "b"},
        notes="gcd(a, b) - greatest common divisor"
    ),

    "gcd_inverse_a": Decomposition(
        expression="g",
        param_map={"g": "g", "b": "b"},
        notes="Smallest a with gcd(a,b)=g is a=g when g|b"
    ),

    "gcd_inverse_b": Decomposition(
        expression="g",
        param_map={"g": "g", "a": "a"},
        notes="Smallest b with gcd(a,b)=g is b=g when g|a"
    ),

    "lcm_inverse_a": Decomposition(
        expression="divide(l, gcd(l, b))",
        param_map={"l": "l", "b": "b"},
        notes="Smallest a with lcm(a,b)=l requires a | l"
    ),

    "modular_exponentiation": Decomposition(
        expression="mod(power(a, n), m)",
        param_map={"a": "a", "n": "n", "m": "m"},
        notes="a^n mod m"
    ),

    "fermat_reduce": Decomposition(
        expression="mod(power(a, mod(n, subtract(p, 1))), p)",
        param_map={"a": "a", "n": "n", "p": "p"},
        notes="a^n mod p = a^(n mod (p-1)) mod p by Fermat's little theorem"
    ),

    "euler_reduce": Decomposition(
        expression="mod(power(a, mod(n, totient(m))), m)",
        param_map={"a": "a", "n": "n", "m": "m"},
        notes="a^n mod m = a^(n mod phi(m)) mod m by Euler's theorem"
    ),

    "chinese_remainder_inverse_residue": Decomposition(
        expression="mod(x, mod_i)",
        param_map={"x": "x", "mod_i": "mod_i"},
        notes="x mod m_i for CRT residue extraction"
    ),

    "fibonacci_gcd": Decomposition(
        expression="fibonacci(gcd(m, n))",
        param_map={"m": "m", "n": "n"},
        notes="gcd(F_m, F_n) = F_gcd(m,n)"
    ),

    "fermat_number": Decomposition(
        expression="add(power(2, power(2, n)), 1)",
        param_map={"n": "n"},
        notes="F_n = 2^(2^n) + 1 - Fermat number"
    ),

    "lucas_inverse": Decomposition(
        expression="floor(divide(log(L), log(phi)))",
        param_map={"L": "L"},
        notes="n where L_n = L, approximated via n = log(L)/log(phi)"
    ),

    "fibonacci_inverse": Decomposition(
        expression="floor(divide(log(multiply(F, sqrt(5))), log(phi)))",
        param_map={"F": "F"},
        notes="n where F_n = F, approximated via n = log(F*sqrt(5))/log(phi)"
    ),

    "legendre_symbol_value": Decomposition(
        expression="power(a, divide(subtract(p, 1), 2))",
        param_map={"a": "a", "p": "p"},
        notes="(a/p) = a^((p-1)/2) mod p by Euler's criterion"
    ),

    "count_quad_residues": Decomposition(
        expression="divide(subtract(p, 1), 2)",
        param_map={"p": "p"},
        notes="Number of QRs mod prime p is (p-1)/2"
    ),

    "sqrt_mod_p": Decomposition(
        expression="tonelli_shanks(a, p)",
        param_map={"a": "a", "p": "p"},
        notes="x where x^2 = a mod p"
    ),

    "digit_sum_base_b": Decomposition(
        expression="digit_sum_base(n, b)",
        param_map={"n": "n", "b": "b"},
        notes="Sum of digits of n in base b - calls digit_sum_base PRIMITIVE"
    ),

    "digit_sum_inverse": Decomposition(
        expression="smallest_with_digit_sum(s)",
        param_map={"s": "s"},
        notes="Smallest n with digit sum s (uses 9s and remainder)"
    ),

    "sum_of_digits": Decomposition(
        expression="sum_digits(n)",
        param_map={"n": "n"},
        notes="Sum of decimal digits of n"
    ),

    "fermat_two_squares": Decomposition(
        expression="two_squares_decomposition(p)",
        param_map={"p": "p"},
        notes="Find a,b where p = a^2 + b^2 for p = 1 mod 4"
    ),

    "sum_of_squares_count": Decomposition(
        expression="r2(n)",
        param_map={"n": "n"},
        notes="Count ways to write n = a^2 + b^2 with a,b >= 0"
    ),

    "hensel_lift": Decomposition(
        expression="hensel_sqrt(a, p, k)",
        param_map={"a": "a", "p": "p", "k": "k"},
        notes="x where x^2 = a mod p^k via Hensel lifting"
    ),

    "zsigmondy": Decomposition(
        expression="primitive_prime_divisor(a, n)",
        param_map={"a": "a", "n": "n"},
        notes="Primitive prime divisor of a^n - 1"
    ),

    "is_perfect_square_check": Decomposition(
        expression="equal_to(power(floor(sqrt(n)), 2), n)",
        param_map={"n": "n"},
        notes="n is perfect square iff floor(sqrt(n))^2 = n"
    ),

    "is_perfect_cube_check": Decomposition(
        expression="equal_to(power(round(power(n, divide(1, 3))), 3), n)",
        param_map={"n": "n"},
        notes="n is perfect cube iff round(n^(1/3))^3 = n"
    ),

    "modular_power_result": Decomposition(
        expression="mod(power(base, exp), m)",
        param_map={"base": "base", "exp": "exp", "m": "m"},
        notes="base^exp mod m"
    ),

    "wilson_theorem_mod": Decomposition(
        expression="subtract(p, 1)",
        param_map={"p": "p"},
        notes="(p-1)! mod p = p-1 by Wilson's theorem for prime p"
    ),

    "divisor_sigma_prime_power": Decomposition(
        expression="divide(subtract(power(p, multiply(add(e, 1), k)), 1), subtract(power(p, k), 1))",
        param_map={"p": "p", "e": "e", "k": "k"},
        notes="sigma_k(p^e) = (p^((e+1)k) - 1)/(p^k - 1)"
    ),

    "ceil_log2_large": Decomposition(
        expression="ceil(multiply(exp, log2(base)))",
        param_map={"base": "base", "exp": "exp"},
        notes="ceil(log2(base^exp)) = ceil(exp * log2(base))"
    ),

    "golden_ratio_power_p": Decomposition(
        expression="divide(add(fibonacci(n), multiply(2, fibonacci(subtract(n, 1)))), 2)",
        param_map={"n": "n"},
        notes="p in phi^n = p + q*sqrt(5): p = (F_n + 2*F_{n-1})/2"
    ),

    "golden_ratio_power_q": Decomposition(
        expression="divide(fibonacci(n), 2)",
        param_map={"n": "n"},
        notes="q coefficient in phi^n = p + q*sqrt(5): q = F_n/2"
    ),

    "euler_phi_values_count": Decomposition(
        expression="count_n_with_phi_leq(max_phi)",
        param_map={"max_phi": "max_phi"},
        notes="Count of n where phi(n) <= max_phi"
    ),

    "count_linear_combination_range": Decomposition(
        expression="subtract(floor_div(max_val, g), floor_div(subtract(min_val, 1), g))",
        param_map={"min_val": "min_val", "max_val": "max_val", "g": "g"},
        notes="Count distinct values in range achievable by linear combination"
    ),

    "max_valuation_under_constraint": Decomposition(
        expression="floor(divide(constraint, max_exp))",
        param_map={"constraint": "constraint", "max_exp": "max_exp"},
        notes="floor(constraint / max_exp) for valuation bounds"
    ),

    "prime_factorization": Decomposition(
        expression="prime_factorization(n)",
        param_map={"n": "n"},
        notes="Prime factorization {p: e} for n = product(p^e) - PRIMITIVE"
    ),

    "min_prime_by_residue": Decomposition(
        expression="min_prime_factor_in_residue_class(n, r, m, min_bound)",
        param_map={"n": "n", "r": "r", "m": "m", "min_bound": "min_bound"},
        notes="Smallest prime factor p of n with p = r mod m and p >= min_bound"
    ),

    "lte_even_power": Decomposition(
        expression="add(subtract(padic_valuation(add(power(p, k), 1), 2), 1), padic_valuation(n, 2))",
        param_map={"p": "p", "k": "k", "n": "n"},
        notes="v_2((p^(nk)-1)/(p^k-1)) = v_2(p^k+1) + v_2(n) - 1"
    ),

    "lcm_multiplicative_order_backward": Decomposition(
        expression="lcm(n_order(a, p), n_order(b, q))",
        param_map={"a": "a", "p": "p", "b": "b", "q": "q"},
        notes="lcm(ord_p(a), ord_q(b)) for backward generation"
    ),
}
