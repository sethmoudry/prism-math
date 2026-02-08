# fmt: off
"""Known-value tests for all algebra methods.

Split across algebra_1.py and algebra_2.py to respect the 700-line limit.
This module merges both into a single KNOWN_VALUES export.
"""

from .algebra_1 import KNOWN_VALUES as _KV1
from .algebra_2 import KNOWN_VALUES as _KV2

KNOWN_VALUES = {**_KV1, **_KV2}
