# Prism Math

A Python library of **718 composable mathematical method blocks** spanning competition mathematics. Each method is a self-contained, executable unit with deterministic `generate_parameters()` and `compute()` functions -- making it ideal for synthetic problem generation, technique identification, and training data pipelines.

Built for the [AIMO3 Kaggle competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3).

## Installation

```bash
pip install prism-math
```

Or from source:

```bash
git clone https://github.com/sethmoudry/prism-math.git
cd prism-math
pip install -e .
```

Requires Python 3.10+ and depends only on `sympy` and `numpy`.

## Quick Start

```python
from prism.techniques import MethodRegistry

# Load all 718 methods
methods = MethodRegistry.get_all()
print(f"{len(methods)} methods loaded")

# Use a specific method
heron = MethodRegistry.get("area_heron")
params = heron.generate_parameters()
result = heron.compute(None, params)
print(result.value, result.description)
# 30 Area = sqrt(15.0*10.0*3.0*2.0) = 30.0000 ~ 30

# Every method follows the same interface
fib = MethodRegistry.get("fibonacci")
params = fib.generate_parameters()
result = fib.compute(None, params)
print(result.value, result.description)
```

Each method implements the `MethodBlock` interface:

```python
class MethodBlock:
    def generate_parameters(self, input_value=None) -> dict:
        """Generate valid random parameters for this method."""
        ...

    def compute(self, input_value, params: dict) -> MethodResult:
        """Execute the method and return a deterministic result."""
        ...
```

## Domains

| Domain | Methods | Examples |
|--------|---------|----------|
| Geometry | 166 | Heron's formula, circumradius, nine-point circle, spiral similarity, Pascal's theorem |
| Algebra | 103 | Vieta's formulas, polynomial roots, geometric series, substitution |
| Number Theory | 100 | Euler's totient, Mobius function, CRT, quadratic residues |
| Combinatorics | 70 | Catalan numbers, Burnside's lemma, Stirling numbers, derangements |
| Analysis | 56 | Generating functions, recurrence relations, asymptotic bounds |
| Basic Primitives | ~40 | Arithmetic, GCD, modular operations, digit manipulation |
| Advanced Techniques | ~30 | Pigeonhole, extremal principle, invariants |

## Audit

Every method is tested for deterministic correctness across multiple random trials:

```bash
python -m tests.audit_methods --trials 10
# 718/718 PASS, 539/539 known values
```

## How It Works

Methods can be **composed** into multi-step solution chains. For example, a problem might use `area_heron` -> `circumradius` -> `euler_og_distance` to compute Euler's OG distance from triangle side lengths. The library includes a decomposition registry mapping 1,054 composite techniques to their primitive building blocks.

## License

MIT
