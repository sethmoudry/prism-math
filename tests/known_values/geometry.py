# fmt: off
"""
Known-value ground truth for geometry methods.

Merges KNOWN_VALUES from geometry_1.py and geometry_2.py.
"""

from .geometry_1 import KNOWN_VALUES as _KV1
from .geometry_2 import KNOWN_VALUES as _KV2

KNOWN_VALUES = {}
KNOWN_VALUES.update(_KV1)
KNOWN_VALUES.update(_KV2)

__all__ = ["KNOWN_VALUES"]
