"""Known-value ground truth for all prism methods.

Each domain module exports a KNOWN_VALUES dict mapping:
    method_name -> list of {input, params, expected} dicts

This __init__.py merges all domain dicts into a single KNOWN_VALUES export.
"""

import importlib
from typing import Any, Dict, List

KNOWN_VALUES: Dict[str, List[Dict[str, Any]]] = {}

_DOMAIN_MODULES = [
    "geometry",
    "number_theory",
    "algebra",
    "combinatorics",
    "analysis_methods",
    "advanced_techniques",
    "basic_primitives",
]

for _mod_name in _DOMAIN_MODULES:
    try:
        _mod = importlib.import_module(f".{_mod_name}", package=__name__)
        _domain_values = getattr(_mod, "KNOWN_VALUES", {})
        KNOWN_VALUES.update(_domain_values)
    except ImportError:
        pass

__all__ = ["KNOWN_VALUES"]
