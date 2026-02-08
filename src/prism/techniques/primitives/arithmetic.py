"""
Arithmetic decompositions: basic operations, floor/ceiling, digits, simple formulas, summations.

Sections covered:
- 8. FLOOR/CEILING/ROUNDING
- 10. DIGIT OPERATIONS
- 11. SIMPLE FORMULAS/IDENTITIES
- 12. SUMMATION FORMULAS
"""

from ..decomposition import Decomposition

ARITHMETIC_DECOMPOSITIONS = {
    # =========================================================================
    # FLOOR/CEILING/ROUNDING
    # =========================================================================

    "ceil": Decomposition(
        expression="negate(floor(negate(n)))",
        param_map={"n": "n"},
        notes="ceil(n) = -floor(-n)"
    ),

    "round_half_up": Decomposition(
        expression="floor(add(n, 0.5))",
        param_map={"n": "n"},
        notes="Round to nearest, ties go up"
    ),

    "truncate": Decomposition(
        expression="multiply(floor(divide(n, 1)), 1)",
        param_map={"n": "n"},
        notes="Truncate towards zero"
    ),

    "frac": Decomposition(
        expression="subtract(n, floor(n))",
        param_map={"n": "n"},
        notes="Fractional part: n - floor(n)"
    ),

    "floor_sqrt": Decomposition(
        expression="floor(power(n, divide(constant_one(), 2)))",
        param_map={"n": "n"},
        notes="floor(sqrt(n))"
    ),

    "ceil_sqrt": Decomposition(
        expression="negate(floor(negate(power(n, 0.5))))",
        param_map={"n": "n"},
        notes="ceil(sqrt(n))"
    ),

    "floor_div": Decomposition(
        expression="floor(divide(a, b))",
        param_map={"a": "a", "b": "b"},
        notes="floor(a/b)"
    ),

    "ceil_div": Decomposition(
        expression="negate(floor(divide(negate(a), b)))",
        param_map={"a": "a", "b": "b"},
        notes="ceil(a/b) = -floor(-a/b)"
    ),

    # =========================================================================
    # DIGIT OPERATIONS
    # =========================================================================

    "last_digit": Decomposition(
        expression="mod(n, 10)",
        param_map={"n": "n"},
        notes="n mod 10"
    ),

    "remove_last_digit": Decomposition(
        expression="floor(divide(n, 10))",
        param_map={"n": "n"},
        notes="floor(n / 10)"
    ),

    "first_digit": Decomposition(
        expression="floor(divide(n, power(10, subtract(digit_count(n), 1))))",
        param_map={"n": "n"},
        notes="First digit of n"
    ),

    "digit_sum": Decomposition(
        expression="sum_digits(n)",
        param_map={"n": "n"},
        notes="Sum of digits of n"
    ),

    "digit_count": Decomposition(
        expression="add(floor(log10(n)), constant_one())",
        param_map={"n": "n"},
        notes="Number of digits = floor(log10(n)) + 1"
    ),

    "reverse_digits": Decomposition(
        expression="reverse_number(n)",
        param_map={"n": "n"},
        notes="Reverse the digits of n"
    ),

    "kth_digit": Decomposition(
        expression="mod(floor(divide(n, power(10, k))), 10)",
        param_map={"n": "n", "k": "k"},
        notes="k-th digit from right (0-indexed)"
    ),

    "digit_product": Decomposition(
        expression="product_digits(n)",
        param_map={"n": "n"},
        notes="Product of digits of n - calls product_digits PRIMITIVE"
    ),

    # =========================================================================
    # SIMPLE FORMULAS/IDENTITIES
    # =========================================================================

    "average_two": Decomposition(
        expression="divide(add(a, b), 2)",
        param_map={"a": "a", "b": "b"},
        notes="(a + b) / 2"
    ),

    "difference_of_squares": Decomposition(
        expression="multiply(add(a, b), subtract(a, b))",
        param_map={"a": "a", "b": "b"},
        notes="(a+b)(a-b) = a^2 - b^2"
    ),

    "sum_of_squares_two": Decomposition(
        expression="add(power(a, 2), power(b, 2))",
        param_map={"a": "a", "b": "b"},
        notes="a^2 + b^2"
    ),

    "sum_of_cubes_two": Decomposition(
        expression="add(power(a, 3), power(b, 3))",
        param_map={"a": "a", "b": "b"},
        notes="a^3 + b^3"
    ),

    "product_sum_diff": Decomposition(
        expression="subtract(power(a, 2), power(b, 2))",
        param_map={"a": "a", "b": "b"},
        notes="a^2 - b^2 = (a+b)(a-b)"
    ),

    "double": Decomposition(
        expression="multiply(2, n)",
        param_map={"n": "n"},
        notes="2n"
    ),

    "triple": Decomposition(
        expression="multiply(3, n)",
        param_map={"n": "n"},
        notes="3n"
    ),

    "halve": Decomposition(
        expression="divide(n, 2)",
        param_map={"n": "n"},
        notes="n/2"
    ),

    "square": Decomposition(
        expression="power(n, 2)",
        param_map={"n": "n"},
        notes="n^2"
    ),

    "cube": Decomposition(
        expression="power(n, 3)",
        param_map={"n": "n"},
        notes="n^3"
    ),

    "fourth_power": Decomposition(
        expression="power(n, 4)",
        param_map={"n": "n"},
        notes="n^4"
    ),

    "sqrt": Decomposition(
        expression="power(n, 0.5)",
        param_map={"n": "n"},
        notes="sqrt(n) = n^0.5"
    ),

    "cbrt": Decomposition(
        expression="power(n, divide(1, 3))",
        param_map={"n": "n"},
        notes="cbrt(n) = n^(1/3)"
    ),

    "increment": Decomposition(
        expression="add(n, constant_one())",
        param_map={"n": "n"},
        notes="n+1"
    ),

    "decrement": Decomposition(
        expression="subtract(n, constant_one())",
        param_map={"n": "n"},
        notes="n-1"
    ),

    "absolute_value": Decomposition(
        expression="max_value(n, negate(n))",
        param_map={"n": "n"},
        notes="|n| = max(n, -n)"
    ),

    "sign": Decomposition(
        expression="divide(n, max_value(absolute_value(n), 1))",
        param_map={"n": "n"},
        notes="sign(n) = n / |n| for n != 0"
    ),

    "reciprocal": Decomposition(
        expression="divide(1, n)",
        param_map={"n": "n"},
        notes="1/n"
    ),

    # =========================================================================
    # SUMMATION FORMULAS
    # =========================================================================

    "sum_first_n": Decomposition(
        expression="divide(multiply(n, add(n, constant_one())), 2)",
        param_map={"n": "n"},
        notes="1+2+...+n = n(n+1)/2"
    ),

    "sum_squares": Decomposition(
        expression="divide(multiply(multiply(n, add(n, constant_one())), add(multiply(2, n), constant_one())), 6)",
        param_map={"n": "n"},
        notes="1^2+2^2+...+n^2 = n(n+1)(2n+1)/6"
    ),

    "sum_cubes": Decomposition(
        expression="power(divide(multiply(n, add(n, constant_one())), 2), 2)",
        param_map={"n": "n"},
        notes="1^3+2^3+...+n^3 = (n(n+1)/2)^2"
    ),

    "sum_fourth_powers": Decomposition(
        expression="divide(multiply(multiply(multiply(n, add(n, 1)), add(multiply(2, n), 1)), subtract(multiply(multiply(3, n), n), subtract(n, 1))), 30)",
        param_map={"n": "n"},
        notes="Sum of k^4 from k=1 to n"
    ),

    "geometric_sum": Decomposition(
        expression="divide(subtract(power(r, add(n, constant_one())), constant_one()), subtract(r, constant_one()))",
        param_map={"r": "r", "n": "n"},
        notes="1+r+r^2+...+r^n = (r^(n+1)-1)/(r-1)"
    ),

    "geometric_sum_finite": Decomposition(
        expression="divide(multiply(a, subtract(1, power(r, n))), subtract(1, r))",
        param_map={"a": "a", "r": "r", "n": "n"},
        notes="a + ar + ar^2 + ... + ar^(n-1)"
    ),

    "arithmetic_progression_sum": Decomposition(
        expression="divide(multiply(n, add(multiply(2, a), multiply(subtract(n, 1), d))), 2)",
        param_map={"a": "a", "d": "d", "n": "n"},
        notes="a + (a+d) + (a+2d) + ... = n(2a + (n-1)d)/2"
    ),

    "sum_odd_numbers": Decomposition(
        expression="power(n, 2)",
        param_map={"n": "n"},
        notes="1+3+5+...+(2n-1) = n^2"
    ),

    "sum_even_numbers": Decomposition(
        expression="multiply(n, add(n, 1))",
        param_map={"n": "n"},
        notes="2+4+6+...+2n = n(n+1)"
    ),

    "sum_powers_of_two": Decomposition(
        expression="subtract(power(2, add(n, 1)), 1)",
        param_map={"n": "n"},
        notes="1+2+4+...+2^n = 2^(n+1) - 1"
    ),
}
