"""Input factory for method auditing.

Maps input_type strings (from MethodBlock.input_type) to functions that
generate appropriate test values.

Key insight: most prism methods use their own generate_parameters() to provide
all data and either ignore input_value or use it as a fallback via
params.get("n", input_value). So None is safe for most structured types.
"""

import random
from typing import Any


def _rand_int(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _rand_float(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 4)


def _rand_point():
    return (_rand_int(1, 20), _rand_int(1, 20))


def _rand_int_list(min_len: int = 3, max_len: int = 8, lo: int = 1, hi: int = 20):
    return [_rand_int(lo, hi) for _ in range(_rand_int(min_len, max_len))]


# ---------------------------------------------------------------------------
# Lookup table: input_type -> factory callable
# ---------------------------------------------------------------------------

_FACTORIES: dict[str, Any] = {}


def _register(*names):
    """Decorator that registers a factory under one or more type names."""
    def decorator(fn):
        for name in names:
            _FACTORIES[name.lower()] = fn
        return fn
    return decorator


# -- Numeric types ----------------------------------------------------------

@_register("integer", "int")
def _integer():
    return _rand_int(10, 500)

@_register("number", "numeric", "float", "real")
def _float():
    return _rand_float(1.0, 100.0)

@_register("count")
def _count():
    return _rand_int(3, 50)

@_register("geometric_value")
def _geometric_value():
    return _rand_int(5, 50)

@_register("valuation")
def _valuation():
    return _rand_int(2, 20)

@_register("modular")
def _modular():
    return _rand_int(5, 50)

@_register("evaluation", "answer")
def _evaluation():
    return _rand_int(10, 100)

@_register("divisor_sum", "gcd", "root_analysis")
def _numeric_alias():
    return _rand_int(10, 100)

@_register("any")
def _any():
    return _rand_int(10, 100)


# -- None / starter types --------------------------------------------------

@_register("none", "seed", "params")
def _none():
    return None


# -- Structured geometric (params carry the data) --------------------------

@_register(
    "triangle", "polygon", "geometric", "geometric_object",
    "circle", "line", "hexagon", "tetrahedron", "sphere_pair",
    "rectangle", "lattice_polygon", "circle_pair",
)
def _geometric_none():
    return None

@_register("point")
def _point():
    return _rand_point()

@_register("point_pair")
def _point_pair():
    return (_rand_point(), _rand_point())

@_register("point_triple")
def _point_triple():
    return (_rand_point(), _rand_point(), _rand_point())

@_register("point_quadruple")
def _point_quadruple():
    return (_rand_point(), _rand_point(), _rand_point(), _rand_point())

@_register("point_set")
def _point_set():
    return [_rand_point() for _ in range(5)]

@_register("angle")
def _angle():
    return _rand_float(10.0, 170.0)

@_register("trig_value")
def _trig_value():
    return _rand_float(0.0, 1.0)


# -- Algebraic types (params carry the data) --------------------------------

@_register(
    "polynomial", "quadratic", "expression", "linear_expression",
    "symbolic_expression", "sqrt_expression", "equation",
)
def _algebraic_none():
    return None


# -- Collection types -------------------------------------------------------

@_register("list", "sequence")
def _list():
    return _rand_int_list()

@_register("sequences")
def _sequences():
    return [_rand_int_list(), _rand_int_list()]

@_register("vectors")
def _vectors():
    return [(_rand_int(1, 10), _rand_int(1, 10)),
            (_rand_int(1, 10), _rand_int(1, 10))]

@_register("set")
def _set():
    return set(_rand_int_list(min_len=5, max_len=5))

@_register("tuple")
def _tuple():
    return (_rand_int(1, 20), _rand_int(1, 20))

@_register("triple")
def _triple():
    return (3, 4, 5)


# -- Structured types -------------------------------------------------------

@_register("matrix")
def _matrix():
    return [[1, 2], [3, 4]]

@_register("dict")
def _dict():
    return {"a": 1, "b": 2}

@_register("graph", "function", "word_problem")
def _structured_none():
    return None


# -- Special types ----------------------------------------------------------

@_register("string")
def _string():
    return "test"

@_register("bool", "boolean")
def _bool():
    return True

@_register("constraint", "constraint_system")
def _constraint():
    return None

@_register("complex")
def _complex():
    return complex(3, 4)

@_register("complex_set")
def _complex_set():
    return {complex(1, 0), complex(0, 1)}

@_register("rational", "fraction")
def _rational():
    return 3

@_register("partition", "permutation", "subset")
def _combinatorial_none():
    return None

@_register("roots", "factorization", "linear_system")
def _algebraic_result_none():
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_test_input(input_type: str) -> Any:
    """Generate an appropriate test input for the given type."""
    factory = _FACTORIES.get(input_type.lower().strip())
    if factory is not None:
        return factory()
    # Fallback: treat unknown types as integer
    return _rand_int(10, 100)
