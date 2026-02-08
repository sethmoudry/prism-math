"""
Common utilities and imports for geometry technique modules.
"""

import random
import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from ..base import MethodBlock, MethodResult
from ..registry import register_technique


def ensure_tuple(value: Any, name: str, expected_length: int = 2) -> Tuple:
    """
    Ensure a parameter value is a tuple of the expected length.

    Args:
        value: The value to check/convert
        name: Parameter name (for error messages)
        expected_length: Expected tuple length (default 2 for points)

    Returns:
        Tuple of the expected length

    Raises:
        TypeError: If value cannot be converted to a tuple
        ValueError: If tuple has wrong length
    """
    if isinstance(value, (list, tuple)):
        if len(value) != expected_length:
            raise ValueError(
                f"Parameter '{name}' has length {len(value)}, expected {expected_length}. "
                f"Got: {value}"
            )
        return tuple(value)
    elif isinstance(value, (int, float)):
        # Single value - common cases
        if expected_length == 2:
            # For 2D points, assume (value, 0) - x-coordinate with y=0
            return (value, 0)
        else:
            raise TypeError(
                f"Parameter '{name}' is a single value {value}, but expected a "
                f"{expected_length}-tuple. For 2D points, use (x, y) format. "
                f"For example: center=(0, 0), point=(5, 3)"
            )
    else:
        raise TypeError(
            f"Parameter '{name}' has invalid type {type(value).__name__}. "
            f"Expected a {expected_length}-tuple. Got: {value}"
        )


def line_intersection(line1: Any, line2: Any) -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two lines.

    Args:
        line1: Either ((x1, y1), (x2, y2)) for two points on the line,
               or (a, b, c) for line equation ax + by + c = 0
        line2: Either ((x1, y1), (x2, y2)) for two points on the line,
               or (a, b, c) for line equation ax + by + c = 0

    Returns:
        Intersection point (x, y) as a tuple, or None if lines are parallel

    Examples:
        >>> line_intersection(((0, 0), (1, 1)), ((0, 1), (1, 0)))
        (0.5, 0.5)
        >>> line_intersection((1, -1, 0), (1, 1, -1))  # x=y and x+y=1
        (0.5, 0.5)
    """
    # Convert both lines to ax + by + c = 0 form
    def to_standard_form(line):
        if len(line) == 2:
            # Two points: ((x1, y1), (x2, y2))
            p1, p2 = line
            x1, y1 = p1
            x2, y2 = p2

            # Line through two points: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
            # Simplifies to: a = (y2-y1), b = -(x2-x1), c = (x2-x1)y1 - (y2-y1)x1
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            return a, b, c
        elif len(line) == 3:
            # Already in ax + by + c = 0 form
            return line
        else:
            raise ValueError(f"Invalid line format: {line}. Expected ((x1,y1),(x2,y2)) or (a,b,c)")

    a1, b1, c1 = to_standard_form(line1)
    a2, b2, c2 = to_standard_form(line2)

    # Solve system: a1*x + b1*y + c1 = 0
    #               a2*x + b2*y + c2 = 0
    # Using Cramer's rule:
    # determinant = a1*b2 - a2*b1
    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-10:
        # Lines are parallel or coincident
        return None

    # x = (b1*c2 - b2*c1) / det
    # y = (a2*c1 - a1*c2) / det
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return (x, y)


# Re-export for convenience
__all__ = [
    'MethodBlock', 'MethodResult', 'register_technique',
    'ensure_tuple', 'line_intersection',
    'random', 'math', 'np', 'Dict', 'Any', 'Optional', 'Tuple', 'List'
]
