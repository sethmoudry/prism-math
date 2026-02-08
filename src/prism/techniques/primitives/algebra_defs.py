"""
Algebra decompositions: sequences, polynomials, matrices, functional equations, series.

Sections covered:
- 3. SEQUENCES DECOMPOSITIONS
- 4. FIGURATE NUMBERS
- 5. POLYNOMIAL OPERATIONS
- 17. ALGEBRA DECOMPOSITIONS (partial)
"""

from ..decomposition import Decomposition

ALGEBRA_DECOMPOSITIONS = {
    # =========================================================================
    # 3. SEQUENCES DECOMPOSITIONS
    # =========================================================================

    "lucas": Decomposition(
        expression="add(fibonacci(subtract(n, 1)), fibonacci(add(n, 1)))",
        param_map={"n": "n"},
        notes="L(n) = F(n-1) + F(n+1) - Lucas numbers"
    ),

    "tribonacci": Decomposition(
        expression="linear_recurrence([0, 0, 1], [1, 1, 1], n)",
        param_map={"n": "n"},
        notes="T(n) = T(n-1) + T(n-2) + T(n-3) - Tribonacci sequence"
    ),

    "pell": Decomposition(
        expression="linear_recurrence([0, 1], [2, 1], n)",
        param_map={"n": "n"},
        notes="P(n) = 2*P(n-1) + P(n-2) - Pell numbers"
    ),

    "pell_lucas": Decomposition(
        expression="linear_recurrence([2, 2], [2, 1], n)",
        param_map={"n": "n"},
        notes="Q(n) = 2*Q(n-1) + Q(n-2), Q(0)=Q(1)=2 - Pell-Lucas numbers"
    ),

    "harmonic": Decomposition(
        expression="sum_reciprocals(n)",
        param_map={"n": "n"},
        notes="H_n = 1 + 1/2 + 1/3 + ... + 1/n"
    ),

    "bernoulli": Decomposition(
        expression="bernoulli_number(n)",
        param_map={"n": "n"},
        notes="B_n - Bernoulli numbers"
    ),

    "fibonacci_mod": Decomposition(
        expression="mod(fibonacci(n), m)",
        param_map={"n": "n", "m": "m"},
        notes="F_n mod m"
    ),

    "lucas_mod": Decomposition(
        expression="mod(add(fibonacci(subtract(n, 1)), fibonacci(add(n, 1))), m)",
        param_map={"n": "n", "m": "m"},
        notes="L_n mod m"
    ),

    "jacobsthal": Decomposition(
        expression="divide(subtract(power(2, n), power(negate(1), n)), 3)",
        param_map={"n": "n"},
        notes="J_n = (2^n - (-1)^n) / 3"
    ),

    "mersenne": Decomposition(
        expression="subtract(power(2, n), 1)",
        param_map={"n": "n"},
        notes="M_n = 2^n - 1"
    ),

    "fermat": Decomposition(
        expression="add(power(2, power(2, n)), 1)",
        param_map={"n": "n"},
        notes="F_n = 2^(2^n) + 1 - Fermat numbers"
    ),

    "polygonal": Decomposition(
        expression="divide(multiply(n, subtract(multiply(subtract(s, 2), n), subtract(s, 4))), 2)",
        param_map={"n": "n", "s": "s"},
        notes="P(s,n) = n*((s-2)*n - (s-4))/2 - s-gonal number"
    ),

    "centered_polygonal": Decomposition(
        expression="add(divide(multiply(multiply(s, n), subtract(n, 1)), 2), 1)",
        param_map={"n": "n", "s": "s"},
        notes="Centered s-gonal number"
    ),

    "pronic": Decomposition(
        expression="multiply(n, add(n, 1))",
        param_map={"n": "n"},
        notes="n*(n+1) - pronic/oblong numbers"
    ),

    "tetrahedral": Decomposition(
        expression="divide(multiply(multiply(n, add(n, 1)), add(n, 2)), 6)",
        param_map={"n": "n"},
        notes="Te_n = n(n+1)(n+2)/6"
    ),

    # =========================================================================
    # 4. FIGURATE NUMBERS
    # =========================================================================

    "triangular_number": Decomposition(
        expression="divide(multiply(n, add(n, constant_one())), 2)",
        param_map={"n": "n"},
        notes="T_n = n(n+1)/2"
    ),

    "square_number": Decomposition(
        expression="multiply(n, n)",
        param_map={"n": "n"},
        notes="n^2"
    ),

    "pentagonal_number": Decomposition(
        expression="divide(multiply(n, subtract(multiply(3, n), constant_one())), 2)",
        param_map={"n": "n"},
        notes="P_n = n(3n-1)/2"
    ),

    "hexagonal_number": Decomposition(
        expression="multiply(n, subtract(multiply(2, n), constant_one()))",
        param_map={"n": "n"},
        notes="H_n = n(2n-1)"
    ),

    "heptagonal_number": Decomposition(
        expression="divide(multiply(n, subtract(multiply(5, n), 3)), 2)",
        param_map={"n": "n"},
        notes="Hep_n = n(5n-3)/2"
    ),

    "octagonal_number": Decomposition(
        expression="multiply(n, subtract(multiply(3, n), 2))",
        param_map={"n": "n"},
        notes="Oct_n = n(3n-2)"
    ),

    "pyramidal_square": Decomposition(
        expression="divide(multiply(multiply(n, add(n, 1)), add(multiply(2, n), 1)), 6)",
        param_map={"n": "n"},
        notes="Square pyramidal = n(n+1)(2n+1)/6"
    ),

    "pyramidal_triangular": Decomposition(
        expression="divide(multiply(multiply(n, add(n, 1)), add(n, 2)), 6)",
        param_map={"n": "n"},
        notes="Triangular pyramidal = n(n+1)(n+2)/6"
    ),

    "star_number": Decomposition(
        expression="add(multiply(6, multiply(n, subtract(n, 1))), 1)",
        param_map={"n": "n"},
        notes="S_n = 6n(n-1) + 1"
    ),

    "centered_square": Decomposition(
        expression="add(multiply(n, n), multiply(subtract(n, 1), subtract(n, 1)))",
        param_map={"n": "n"},
        notes="n^2 + (n-1)^2"
    ),

    # =========================================================================
    # 5. POLYNOMIAL OPERATIONS
    # =========================================================================

    "polynomial_eval": Decomposition(
        expression="horner_eval(coeffs, x)",
        param_map={"coeffs": "coeffs", "x": "x"},
        notes="Evaluate polynomial at x using Horner's method"
    ),

    "polynomial_degree": Decomposition(
        expression="subtract(length(coeffs), 1)",
        param_map={"coeffs": "coeffs"},
        notes="Degree = length of coefficients - 1"
    ),

    "vieta_sum_roots": Decomposition(
        expression="negate(divide(coeff_1, coeff_0))",
        param_map={"coeff_0": "coeff_0", "coeff_1": "coeff_1"},
        notes="Sum of roots = -a_{n-1}/a_n"
    ),

    "vieta_product_roots": Decomposition(
        expression="divide(multiply(power(negate(1), n), coeff_n), coeff_0)",
        param_map={"coeff_0": "coeff_0", "coeff_n": "coeff_n", "n": "n"},
        notes="Product of roots = (-1)^n * a_0/a_n"
    ),

    "quadratic_discriminant": Decomposition(
        expression="subtract(power(b, 2), multiply(multiply(4, a), c))",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="b^2 - 4ac"
    ),

    "quadratic_root_plus": Decomposition(
        expression="divide(add(negate(b), power(subtract(power(b, 2), multiply(multiply(4, a), c)), 0.5)), multiply(2, a))",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="(-b + sqrt(b^2-4ac)) / 2a"
    ),

    "quadratic_root_minus": Decomposition(
        expression="divide(subtract(negate(b), power(subtract(power(b, 2), multiply(multiply(4, a), c)), 0.5)), multiply(2, a))",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="(-b - sqrt(b^2-4ac)) / 2a"
    ),

    "legendre_poly": Decomposition(
        expression="legendre_polynomial(n, x)",
        param_map={"n": "n", "x": "x"},
        notes="P_n(x) - Legendre polynomial"
    ),

    # =========================================================================
    # 17. ALGEBRA DECOMPOSITIONS (Vieta, Newton, matrices, series)
    # =========================================================================

    "vieta_sum": Decomposition(
        expression="negate(divide(coeff_1, coeff_0))",
        param_map={"coeff_0": "coeff_0", "coeff_1": "coeff_1"},
        notes="Sum of polynomial roots = -a_{n-1}/a_n (Vieta)"
    ),

    "vieta_product": Decomposition(
        expression="divide(multiply(power(negate(1), degree), constant_term), leading_coeff)",
        param_map={"degree": "degree", "constant_term": "constant_term", "leading_coeff": "leading_coeff"},
        notes="Product of roots = (-1)^n * a_0/a_n"
    ),

    "vieta_inverse_coeffs": Decomposition(
        expression="[1, negate(root_sum), root_product]",
        param_map={"root_sum": "root_sum", "root_product": "root_product"},
        notes="Quadratic x^2 - sx + p from sum s and product p"
    ),

    "vieta_quadratic_sum": Decomposition(
        expression="negate(divide(b, a))",
        param_map={"a": "a", "b": "b"},
        notes="For ax^2+bx+c: r1+r2 = -b/a"
    ),

    "vieta_quadratic_product": Decomposition(
        expression="divide(c, a)",
        param_map={"a": "a", "c": "c"},
        notes="For ax^2+bx+c: r1*r2 = c/a"
    ),

    "vieta_cubic_sum": Decomposition(
        expression="negate(divide(b, a))",
        param_map={"a": "a", "b": "b"},
        notes="For ax^3+bx^2+cx+d: r1+r2+r3 = -b/a"
    ),

    "vieta_cubic_product": Decomposition(
        expression="negate(divide(d, a))",
        param_map={"a": "a", "d": "d"},
        notes="For ax^3+bx^2+cx+d: r1*r2*r3 = -d/a"
    ),

    "newton_power_sum": Decomposition(
        expression="sum_powers(roots, k)",
        param_map={"roots": "roots", "k": "k"},
        notes="p_k = sum(r_i^k) power sum of roots"
    ),

    "newton_inverse_elemsym": Decomposition(
        expression="divide(subtract(power(p1, 2), p2), 2)",
        param_map={"p1": "p1", "p2": "p2"},
        notes="e2 = (p1^2 - p2)/2 from Newton's identities"
    ),

    "newton_p2_from_e1_e2": Decomposition(
        expression="subtract(power(e1, 2), multiply(2, e2))",
        param_map={"e1": "e1", "e2": "e2"},
        notes="p_2 = e1^2 - 2*e2"
    ),

    "matrix_trace": Decomposition(
        expression="sum_diagonal(matrix)",
        param_map={"matrix": "matrix"},
        notes="tr(M) = sum of diagonal elements M[i][i]"
    ),

    "matrix_det": Decomposition(
        expression="determinant(matrix)",
        param_map={"matrix": "matrix"},
        notes="det(M) using cofactor expansion or LU"
    ),

    "determinant_2x2": Decomposition(
        expression="subtract(multiply(a, d), multiply(b, c))",
        param_map={"a": "a", "b": "b", "c": "c", "d": "d"},
        notes="det([[a,b],[c,d]]) = ad - bc"
    ),

    "matrix_power": Decomposition(
        expression="matrix_exponentiation(matrix, n)",
        param_map={"matrix": "matrix", "n": "n"},
        notes="M^n via repeated squaring"
    ),

    "simplify_dot_product": Decomposition(
        expression="sum_products(v1, v2)",
        param_map={"v1": "v1", "v2": "v2"},
        notes="v1.v2 = sum(v1[i]*v2[i])"
    ),

    "quadratic_vertex": Decomposition(
        expression="[negate(divide(b, multiply(2, a))), subtract(c, divide(power(b, 2), multiply(4, a)))]",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="Vertex (h,k) = (-b/2a, c-b^2/4a)"
    ),

    "quadratic_axis_of_symmetry": Decomposition(
        expression="negate(divide(b, multiply(2, a)))",
        param_map={"a": "a", "b": "b"},
        notes="x = -b/(2a) axis of symmetry"
    ),

    "arithmetic_sum_formula": Decomposition(
        expression="divide(multiply(n, add(multiply(2, a), multiply(subtract(n, 1), d))), 2)",
        param_map={"a": "a", "d": "d", "n": "n"},
        notes="S_n = n(2a+(n-1)d)/2 arithmetic series"
    ),

    "sum_range_formula": Decomposition(
        expression="divide(multiply(subtract(add(end, 1), start), add(start, end)), 2)",
        param_map={"start": "start", "end": "end"},
        notes="sum(start..end) = (end-start+1)(start+end)/2"
    ),

    "geometric_sum_formula": Decomposition(
        expression="divide(multiply(a, subtract(power(r, n), 1)), subtract(r, 1))",
        param_map={"a": "a", "r": "r", "n": "n"},
        notes="a(r^n-1)/(r-1) geometric series"
    ),

    "linear_recurrence": Decomposition(
        expression="recurrence_iterate(coeffs, initial, n)",
        param_map={"coeffs": "coeffs", "initial": "initial", "n": "n"},
        notes="a_n = c1*a_{n-1} + c2*a_{n-2} + ..."
    ),

    "characteristic_polynomial": Decomposition(
        expression="roots_of_polynomial([1, negate(c1), negate(c2)])",
        param_map={"c1": "c1", "c2": "c2"},
        notes="Roots of x^2 - c1*x - c2 = 0"
    ),

    "binet_formula": Decomposition(
        expression="round(add(multiply(A, power(phi, n)), multiply(B, power(psi, n))))",
        param_map={"A": "A", "B": "B", "phi": "phi", "psi": "psi", "n": "n"},
        notes="F_n via Binet's formula"
    ),

    "telescoping": Decomposition(
        expression="subtract(f_1, f_n_plus_1)",
        param_map={"f_1": "f_1", "f_n_plus_1": "f_n_plus_1"},
        notes="Telescoping sum = f(1) - f(n+1)"
    ),

    "golden_ratio_identity": Decomposition(
        expression="add(multiply(fib_n, phi), fib_n_minus_1)",
        param_map={"fib_n": "fib_n", "fib_n_minus_1": "fib_n_minus_1", "phi": "phi"},
        notes="phi^n = F_n*phi + F_{n-1}"
    ),

    "arithmetic_geometric_mean_min": Decomposition(
        expression="multiply(n, power(product, divide(1, n)))",
        param_map={"product": "product", "n": "n"},
        notes="min(sum) with product P fixed = n*P^(1/n)"
    ),

    "cauchy_schwarz_bound": Decomposition(
        expression="floor(power(multiply(sum_a_sq, sum_b_sq), 0.5))",
        param_map={"sum_a_sq": "sum_a_sq", "sum_b_sq": "sum_b_sq"},
        notes="|a.b| <= sqrt(sum(a_i^2)*sum(b_i^2))"
    ),

    "lagrange_optimize": Decomposition(
        expression="floor(divide(power(sum_val, 2), 4))",
        param_map={"sum_val": "sum_val"},
        notes="max(xy) with x+y=S is S^2/4"
    ),

    "functional_eq_cauchy": Decomposition(
        expression="multiply(c, x)",
        param_map={"c": "c", "x": "x"},
        notes="f(x+y)=f(x)+f(y) => f(x)=cx"
    ),

    "functional_eq_multiplicative": Decomposition(
        expression="valuation_based(n, prime)",
        param_map={"n": "n", "prime": "prime"},
        notes="f(mn)=f(m)+f(n) like log"
    ),

    "functional_eq_power": Decomposition(
        expression="multiply(n, f_base)",
        param_map={"n": "n", "f_base": "f_base"},
        notes="f(x^n)=n*f(x)"
    ),

    "functional_eq_add_mult": Decomposition(
        expression="power(a, x)",
        param_map={"a": "a", "x": "x"},
        notes="f(x+y)=f(x)*f(y) => f(x)=a^x"
    ),

    "polynomial_evaluation": Decomposition(
        expression="horner_eval(coeffs, x)",
        param_map={"coeffs": "coeffs", "x": "x"},
        notes="P(x) via Horner's method"
    ),

    "cyclotomic_eval": Decomposition(
        expression="cyclotomic_polynomial_at(n, x)",
        param_map={"n": "n", "x": "x"},
        notes="Phi_n(x) cyclotomic polynomial"
    ),

    "cyclotomic_factorization": Decomposition(
        expression="divisors(n)",
        param_map={"n": "n"},
        notes="x^n-1 = product Phi_d(x) for d|n"
    ),

    "expand_square_sum": Decomposition(
        expression="add(add(power(a, 2), multiply(2, multiply(a, b))), power(b, 2))",
        param_map={"a": "a", "b": "b"},
        notes="(a+b)^2 = a^2 + 2ab + b^2"
    ),

    "expand_difference": Decomposition(
        expression="add(subtract(power(a, 2), multiply(2, multiply(a, b))), power(b, 2))",
        param_map={"a": "a", "b": "b"},
        notes="(a-b)^2 = a^2 - 2ab + b^2"
    ),

    "polynomial_division": Decomposition(
        expression="poly_div(p, q)",
        param_map={"p": "p", "q": "q"},
        notes="p(x) / q(x) quotient"
    ),

    "polynomial_factorization": Decomposition(
        expression="factor_polynomial(coeffs)",
        param_map={"coeffs": "coeffs"},
        notes="Factor polynomial into irreducibles"
    ),

    "extract_m_n_sum": Decomposition(
        expression="add(m, n)",
        param_map={"m": "m", "n": "n"},
        notes="m+n from linear expression"
    ),

    "algebraic_manipulation": Decomposition(
        expression="subtract(power(n, k), n)",
        param_map={"n": "n", "k": "k"},
        notes="n^k - n"
    ),

    "binomial_coefficient": Decomposition(
        expression="binomial(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="C(n,k) = n!/(k!(n-k)!)"
    ),

    "diophantine_equations": Decomposition(
        expression="count_nonneg_solutions(a, b, c)",
        param_map={"a": "a", "b": "b", "c": "c"},
        notes="Count solutions ax+by=c with x,y>=0"
    ),

    "summation": Decomposition(
        expression="divide(multiply(n, add(n, 1)), 2)",
        param_map={"n": "n"},
        notes="1+2+...+n = n(n+1)/2"
    ),

    "partial_fractions": Decomposition(
        expression="decompose_rational(num, den)",
        param_map={"num": "num", "den": "den"},
        notes="Partial fraction decomposition"
    ),

    "roots_of_unity": Decomposition(
        expression="sum_kth_powers_of_nth_roots(n, k)",
        param_map={"n": "n", "k": "k"},
        notes="Sum of first k of n-th roots of unity"
    ),

    "substitution": Decomposition(
        expression="multiply(subtract(n, 1), add(n, 1))",
        param_map={"n": "n"},
        notes="(n-1)(n+1) = n^2 - 1"
    ),
}
