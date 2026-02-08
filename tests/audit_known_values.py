"""Re-export KNOWN_VALUES from the known_values package.

This module exists for backward compatibility with the audit_methods.py import.
"""

from tests.known_values import KNOWN_VALUES

__all__ = ["KNOWN_VALUES"]
