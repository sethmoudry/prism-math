# fmt: off
"""Known-value ground truth for number_theory methods.

Every expected value was computed by hand and cross-checked with sympy.
"""

KNOWN_VALUES = {
    # ========================================================================
    # divisibility.py
    # ========================================================================
    "totient": [
        {"input": None, "params": {"n": 12}, "expected": 4},
        {"input": None, "params": {"n": 100}, "expected": 40},
        {"input": None, "params": {"n": 7}, "expected": 6},
    ],
    "totient_inverse": [
        # smallest n with phi(n) = 4 is 5 (phi(5)=4)
        {"input": None, "params": {"phi": 4}, "expected": 5},
        # smallest n with phi(n) = 6 is 7 (phi(7)=6, prime)
        {"input": None, "params": {"phi": 6}, "expected": 7},
        # phi(n) = 2 -> smallest n is 3
        {"input": None, "params": {"phi": 2}, "expected": 3},
    ],
    "divisor_count": [
        {"input": None, "params": {"n": 12}, "expected": 6},
        {"input": None, "params": {"n": 60}, "expected": 12},
        {"input": None, "params": {"n": 1}, "expected": 1},
    ],
    "divisor_count_inverse": [
        # tau=3 is prime -> 2^(3-1) = 4; tau(4)=3
        {"input": None, "params": {"tau": 3}, "expected": 4},
        # tau=4 -> search finds 6 (divisors 1,2,3,6)
        {"input": None, "params": {"tau": 4}, "expected": 6},
    ],
    "divisor_sum": [
        # sigma_1(12) = 1+2+3+4+6+12 = 28
        {"input": None, "params": {"n": 12, "k": 1}, "expected": 28},
        # sigma_1(6) = 1+2+3+6 = 12 (perfect number)
        {"input": None, "params": {"n": 6, "k": 1}, "expected": 12},
    ],
    "divisor_sum_inverse": [
        # sigma=12: isprime(11) so shortcut returns 11 (sigma(11)=12)
        {"input": None, "params": {"sigma": 12}, "expected": 11},
        # sigma=15: isprime(14)=False, search finds 8 (sigma(8)=15)
        {"input": None, "params": {"sigma": 15}, "expected": 8},
    ],
    "gcd_compute": [
        {"input": None, "params": {"a": 48, "b": 18}, "expected": 6},
        {"input": None, "params": {"a": 100, "b": 75}, "expected": 25},
    ],
    "gcd_inverse_a": [
        # smallest a with gcd(a, 30) = 6 -> a=6 (30 % 6 == 0)
        {"input": None, "params": {"g": 6, "b": 30}, "expected": 6},
    ],
    "gcd_inverse_b": [
        # smallest b with gcd(24, b) = 6 -> b=6 (24 % 6 == 0)
        {"input": None, "params": {"g": 6, "a": 24}, "expected": 6},
    ],
    "lcm_compute": [
        {"input": None, "params": {"a": 12, "b": 18}, "expected": 36},
        {"input": None, "params": {"a": 4, "b": 6}, "expected": 12},
    ],
    "lcm_inverse_a": [
        # smallest a (divisor of 12) with lcm(a, 6)=12 -> a=4
        {"input": None, "params": {"l": 12, "b": 6}, "expected": 4},
    ],

    # ========================================================================
    # primes.py
    # ========================================================================
    "prime_factorization": [
        # returns count of distinct primes: 12 = 2^2*3 -> 2
        {"input": None, "params": {"n": 12}, "expected": 2},
        # 30 = 2*3*5 -> 3
        {"input": None, "params": {"n": 30}, "expected": 3},
    ],
    "prime_counting_function": [
        # pi(100) = 25
        {"input": None, "params": {"n": 100}, "expected": 25},
        # pi(10) = 4 (2,3,5,7)
        {"input": None, "params": {"n": 10}, "expected": 4},
    ],
    "smallest_prime_divisor": [
        # 91 = 7*13 -> smallest is 7
        {"input": None, "params": {"n": 91}, "expected": 7},
        {"input": None, "params": {"n": 15}, "expected": 3},
    ],
    "primes_below": [
        # primes below 20: 2,3,5,7,11,13,17,19 -> count=8
        {"input": None, "params": {"n": 20}, "expected": 8},
    ],
    "min_prime_satisfying_constraint": [
        # smallest prime > 100 is 101
        {"input": None, "params": {"n": 100}, "expected": 101},
        {"input": None, "params": {"n": 10}, "expected": 11},
    ],
    "check_prime_power_exponent": [
        # 8 = 2^3 -> exponent 3
        {"input": None, "params": {"n": 8}, "expected": 3},
        # 12 = 2^2*3 -> not prime power -> 0
        {"input": None, "params": {"n": 12}, "expected": 0},
        # 49 = 7^2 -> exponent 2
        {"input": None, "params": {"n": 49}, "expected": 2},
    ],
    "square_free_divisors": [
        # 12 = 2^2*3: squarefree divisors = {1,2,3,6} -> count 4
        {"input": None, "params": {"n": 12}, "expected": 4},
    ],
    "filter_by_prime_factor_count": [
        # n<=30 with exactly 2 distinct prime factors: 12 such numbers
        {"input": None, "params": {"upper": 30, "target_count": 2}, "expected": 12},
    ],
    "primes_of_form_2k_plus_1": [
        # odd primes <= 50: 3,5,7,11,13,17,19,23,29,31,37,41,43,47 -> 14
        {"input": None, "params": {"upper": 50}, "expected": 14},
    ],
    "distinct_prime_divisors_equals_b": [
        # n<=30 with omega(n)=1: prime powers -> 16
        {"input": None, "params": {"upper": 30, "b": 1}, "expected": 16},
    ],
    "allowed_prime_divisors": [
        # n<=20 with prime divs in {2,3}: 1,2,3,4,6,8,9,12,16,18 -> 10
        {"input": None, "params": {"upper": 20, "allowed": [2, 3]}, "expected": 10},
        # n<=30 with prime divs in {2,5}: 1,2,4,5,8,10,16,20,25 -> 9
        {"input": None, "params": {"upper": 30, "allowed": [2, 5]}, "expected": 9},
    ],
    "number_from_prime_powers": [
        # 2^3 * 3^2 = 72
        {"input": None, "params": {"powers": {2: 3, 3: 2}}, "expected": 72},
        # 2^4 * 5^2 = 400
        {"input": None, "params": {"powers": {2: 4, 5: 2}}, "expected": 400},
    ],

    # ========================================================================
    # modular.py
    # ========================================================================
    "modular_exponentiation": [
        # 2^10 mod 1000 = 1024 mod 1000 = 24
        {"input": None, "params": {"a": 2, "n": 10, "m": 1000}, "expected": 24},
        # 3^5 mod 7 = 243 mod 7 = 5
        {"input": None, "params": {"a": 3, "n": 5, "m": 7}, "expected": 5},
    ],
    "mod_exp_inverse_base": [
        # smallest a with a^2 = 4 mod 11 -> a=2
        {"input": None, "params": {"r": 4, "n": 2, "m": 11}, "expected": 2},
    ],
    "mod_exp_inverse_exp": [
        # smallest n with 2^n = 5 mod 11 -> n=4 (2^4=16=5 mod 11)
        {"input": None, "params": {"r": 5, "a": 2, "m": 11}, "expected": 4},
    ],
    "mod_inverse": [
        # 3^(-1) mod 7 = 5 (3*5=15=1 mod 7)
        {"input": None, "params": {"a": 3, "m": 7}, "expected": 5},
        # 7^(-1) mod 11 = 8 (7*8=56=1 mod 11)
        {"input": None, "params": {"a": 7, "m": 11}, "expected": 8},
    ],
    "chinese_remainder_solve": [
        # x = 2 mod 3, x = 3 mod 5 -> x = 8 (mod 15)
        {"input": None, "params": {"moduli": [3, 5], "residues": [2, 3]}, "expected": 8},
        # x = 1 mod 7, x = 4 mod 11 -> x = 15 (mod 77)
        {"input": None, "params": {"moduli": [7, 11], "residues": [1, 4]}, "expected": 15},
    ],
    "chinese_remainder_inverse_residue": [
        # 23 mod 7 = 2
        {"input": None, "params": {"x": 23, "moduli": [7, 11]}, "expected": 2},
    ],
    "fermat_reduce": [
        # 2^100 mod 13: gcd(2,13)=1, 100 mod 12=4, 2^4 mod 13=3
        {"input": None, "params": {"a": 2, "n": 100, "p": 13}, "expected": 3},
    ],
    "euler_reduce": [
        # 7^100 mod 100: phi(100)=40, 100 mod 40=20, 7^20 mod 100=1
        {"input": None, "params": {"a": 7, "n": 100, "m": 100}, "expected": 1},
    ],

    # ========================================================================
    # modular_order.py
    # ========================================================================
    "multiplicative_order": [
        # ord_7(2) = 3 (2^3=8=1 mod 7)
        {"input": None, "params": {"a": 2, "n": 7}, "expected": 3},
        # ord_7(3) = 6 (primitive root)
        {"input": None, "params": {"a": 3, "n": 7}, "expected": 6},
    ],
    "multiplicative_order_inverse_a": [
        # smallest a with ord_7(a)=3 -> a=2
        {"input": None, "params": {"ord": 3, "n": 7}, "expected": 2},
    ],
    "primitive_root": [
        # primitive root mod 7 = 3
        {"input": None, "params": {"p": 7}, "expected": 3},
        # primitive root mod 11 = 2
        {"input": None, "params": {"p": 11}, "expected": 2},
    ],
    "multiplicative_order_mod_prime": [
        # ord_7(2) = 3
        {"input": None, "params": {"a": 2, "p": 7}, "expected": 3},
    ],
    "mod_100000": [
        # 123456 mod 100000 = 23456
        {"input": None, "params": {"n": 123456, "modulus": 100000}, "expected": 23456},
    ],
    "mod_99991": [
        # 200000 mod 99991 = 18
        {"input": None, "params": {"n": 200000, "prime": 99991}, "expected": 18},
    ],
    "modular_power_result": [
        # 2^10 mod 1000 = 24
        {"input": None, "params": {"a": 2, "b": 10, "m": 1000}, "expected": 24},
    ],

    # ========================================================================
    # sequences.py
    # ========================================================================
    "fibonacci": [
        {"input": None, "params": {"n": 10}, "expected": 55},
        {"input": None, "params": {"n": 6}, "expected": 8},
        {"input": None, "params": {"n": 1}, "expected": 1},
    ],
    "fibonacci_inverse": [
        # F_10 = 55 -> n=10
        {"input": None, "params": {"F": 55}, "expected": 10},
        {"input": None, "params": {"F": 8}, "expected": 6},
    ],
    "lucas": [
        # L_5 = 11, L_10 = 123
        {"input": None, "params": {"n": 5}, "expected": 11},
        {"input": None, "params": {"n": 10}, "expected": 123},
    ],
    "lucas_inverse": [
        # L_10 = 123 -> n=10
        {"input": None, "params": {"L": 123}, "expected": 10},
        {"input": None, "params": {"L": 11}, "expected": 5},
    ],
    "fibonacci_gcd": [
        # gcd(F_12, F_8) = F_gcd(12,8) = F_4 = 3
        {"input": None, "params": {"m": 12, "n": 8}, "expected": 3},
    ],
    "fermat_number": [
        # F_0 = 3, F_1 = 5, F_2 = 17, F_3 = 257, F_4 = 65537
        {"input": None, "params": {"n": 0}, "expected": 3},
        {"input": None, "params": {"n": 4}, "expected": 65537},
    ],
    "golden_ratio_power": [
        # round(phi^n / sqrt(5)) = F(n)
        {"input": None, "params": {"n": 10}, "expected": 55},
        {"input": None, "params": {"n": 5}, "expected": 5},
    ],
    "extract_pq_from_golden_power": [
        # phi^10 = (L_10 + F_10*sqrt(5))/2 -> p=L_10=123, q=F_10=55
        {"input": None, "params": {"n": 10, "extract": "p"}, "expected": 123},
        {"input": None, "params": {"n": 10, "extract": "q"}, "expected": 55},
    ],
    "large_power": [
        # 2^20 = 1048576
        {"input": None, "params": {"base": 2, "exp": 20}, "expected": 1048576},
    ],
    "large_fibonacci": [
        # F(60) = 1548008755920
        {"input": None, "params": {"n": 60}, "expected": 1548008755920},
    ],
    "large_power_tower": [
        # 2^(2^4) = 2^16 = 65536
        {"input": None, "params": {"base": 2, "height_exp": 4}, "expected": 65536},
    ],

    # ========================================================================
    # valuation.py
    # ========================================================================
    "legendre_valuation": [
        # v_2(100!) = 50+25+12+6+3+1 = 97
        {"input": None, "params": {"n": 100, "p": 2}, "expected": 97},
        # v_5(100!) = 20+4 = 24
        {"input": None, "params": {"n": 100, "p": 5}, "expected": 24},
    ],
    "legendre_inverse": [
        # smallest n with v_2(n!) >= 10 -> n=12
        {"input": None, "params": {"v": 10, "p": 2}, "expected": 12},
        # smallest n with v_5(n!) >= 5 -> n=25
        {"input": None, "params": {"v": 5, "p": 5}, "expected": 25},
    ],
    "padic_valuation": [
        # v_2(48) = 4 (48 = 2^4 * 3)
        {"input": None, "params": {"n": 48, "p": 2}, "expected": 4},
        # v_3(81) = 4 (81 = 3^4)
        {"input": None, "params": {"n": 81, "p": 3}, "expected": 4},
    ],
    "padic_inverse": [
        # smallest n with v_2(n)=4 -> 2^4 = 16
        {"input": None, "params": {"v": 4, "p": 2}, "expected": 16},
        # smallest n with v_3(n)=3 -> 3^3 = 27
        {"input": None, "params": {"v": 3, "p": 3}, "expected": 27},
    ],
    "kummer_valuation": [
        # v_2(C(10,3)): digit sums in base 2: s(3)=2, s(7)=3, s(10)=2
        # v = (2+3-2)/1 = 3
        {"input": None, "params": {"n": 10, "k": 3, "p": 2}, "expected": 3},
    ],
    "kummer_inverse_n": [
        # smallest n>=5 with v_2(C(n,5))>=2 -> n=8
        {"input": None, "params": {"v": 2, "k": 5, "p": 2}, "expected": 8},
    ],
    "kummer_inverse_k": [
        # smallest k with v_2(C(20,k))>=2 -> k=1
        # s(1,2)=1, s(19,2)=s(10011_2)=3, s(20,2)=s(10100_2)=2
        # v = (1+3-2)/1 = 2 >= 2 -> k=1
        {"input": None, "params": {"v": 2, "n": 20, "p": 2}, "expected": 1},
    ],
    "lifting_exponent_valuation": [
        # v_3(5^3 - 2^3) = v_3(117) = v_3(9*13) = 2
        {"input": None, "params": {"a": 5, "b": 2, "n": 3, "p": 3}, "expected": 2},
    ],
    "lifting_exponent_inverse_n": [
        # smallest n with v_3(5^n - 2^n) >= 2 -> n=3
        {"input": None, "params": {"v": 2, "a": 5, "b": 2, "p": 3}, "expected": 3},
    ],
    "lte_even_power": [
        # v_2(3^4 - 5^4) = v_2(|81-625|) = v_2(544) = v_2(2^5*17) = 5
        {"input": None, "params": {"a": 3, "b": 5, "n": 4}, "expected": 5},
    ],
    "max_valuation_under_constraint": [
        # max v_2(n) for n<=1000: 2^9=512<=1000, 2^10=1024>1000 -> 9
        {"input": None, "params": {"upper": 1000, "p": 2}, "expected": 9},
    ],

    # ========================================================================
    # quadratic.py
    # ========================================================================
    "legendre_symbol": [
        # (2/7) = 1 (2 is a QR mod 7)
        {"input": None, "params": {"a": 2, "p": 7}, "expected": 1},
        # (3/7) = -1 (3 is a QNR mod 7)
        {"input": None, "params": {"a": 3, "p": 7}, "expected": -1},
    ],
    "sqrt_mod_p": [
        # x^2 = 2 mod 7 -> x=3 (3^2=9=2 mod 7)
        {"input": None, "params": {"a": 2, "p": 7}, "expected": 3},
        # x^2 = 4 mod 13 -> x=2
        {"input": None, "params": {"a": 4, "p": 13}, "expected": 2},
    ],
    "jacobi_symbol": [
        # (2/9) = 1
        {"input": None, "params": {"a": 2, "n": 9}, "expected": 1},
        # (5/21) = 1
        {"input": None, "params": {"a": 5, "n": 21}, "expected": 1},
    ],
    "count_quad_residues": [
        # QRs mod 7: (7-1)/2 = 3
        {"input": None, "params": {"p": 7}, "expected": 3},
    ],
    "quadratic_residue_product_mod_p": [
        # QRs mod 7 = {1,2,4}, product = 8 mod 7 = 1
        {"input": None, "params": {"p": 7}, "expected": 1},
    ],
    "cubic_residue_set": [
        # cubic residues mod 7: {0,1,6} -> count=3
        {"input": None, "params": {"p": 7}, "expected": 3},
    ],
    "quadratic_reciprocity": [
        # (3/5)*(5/3) = 1 = (-1)^((3-1)/2*(5-1)/2) -> law holds -> 1
        {"input": None, "params": {"p": 3, "q": 5}, "expected": 1},
    ],
    "fermat_two_squares": [
        # 5 = 1^2 + 2^2 -> returns a=1
        {"input": None, "params": {"p": 5}, "expected": 1},
        # 13 = 2^2 + 3^2 -> returns a=2
        {"input": None, "params": {"p": 13}, "expected": 2},
    ],
    "sum_of_squares_count": [
        # 25: 0^2+5^2, 3^2+4^2, 4^2+3^2, 5^2+0^2 -> 4 (counting a>=0)
        {"input": None, "params": {"n": 25}, "expected": 4},
    ],

    # ========================================================================
    # misc.py
    # ========================================================================
    "digit_sum_base_b": [
        # 255 in base 2 = 11111111, digit sum = 8
        {"input": None, "params": {"n": 255, "b": 2}, "expected": 8},
    ],
    "sum_of_digits": [
        # 12345 -> 1+2+3+4+5 = 15
        {"input": None, "params": {"n": 12345}, "expected": 15},
    ],
    "digit_sum_inverse": [
        # smallest n with digit sum 10 -> 19 (1+9=10)
        {"input": None, "params": {"s": 10}, "expected": 19},
        # smallest n with digit sum 9 -> 9
        {"input": None, "params": {"s": 9}, "expected": 9},
    ],
    "perfect_squares_up_to": [
        # floor(sqrt(100)) = 10
        {"input": None, "params": {"n": 100}, "expected": 10},
    ],
    "is_perfect_square": [
        {"input": None, "params": {"n": 49}, "expected": 1},
        {"input": None, "params": {"n": 50}, "expected": 0},
    ],
    "is_perfect_cube": [
        {"input": None, "params": {"n": 27}, "expected": 1},
        {"input": None, "params": {"n": 28}, "expected": 0},
    ],
    "carmichael_lambda": [
        # lambda(12) = 2
        {"input": None, "params": {"n": 12}, "expected": 2},
        # lambda(15) = lcm(phi(3),phi(5)) = lcm(2,4) = 4
        {"input": None, "params": {"n": 15}, "expected": 4},
    ],
    "ceil_log2_large": [
        # ceil(log2(1000)) = 10 via (999).bit_length() = 10
        {"input": None, "params": {"n": 1000}, "expected": 10},
        # ceil(log2(1024)) = 10 via (1023).bit_length() = 10
        {"input": None, "params": {"n": 1024}, "expected": 10},
    ],
    "wilson_theorem_mod": [
        # (p-1)! mod p = p-1 by Wilson's theorem
        {"input": None, "params": {"p": 7}, "expected": 6},
        {"input": None, "params": {"p": 11}, "expected": 10},
    ],
    "factor_common_term": [
        # gcd(12, 18, 24) = 6
        {"input": None, "params": {"terms": [12, 18, 24]}, "expected": 6},
    ],
    "count_linear_combination_range": [
        # integers in [1,20] representable as 3x+5y (x,y>=0) -> 16
        {"input": None, "params": {"n": 20, "a": 3, "b": 5}, "expected": 16},
    ],
}
