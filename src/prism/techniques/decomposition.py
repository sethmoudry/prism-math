"""
Decomposition system for expressing composite methods as primitive sequences.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import re

@dataclass
class Decomposition:
    """Describes how to decompose a composite method into primitives."""

    # The primitive expression (string format matching DNA program syntax)
    # e.g., "divide(binomial(multiply(2, n), n), add(n, 1))"
    expression: str

    # Parameter mapping from composite to primitives (usually identity)
    param_map: Dict[str, str] = field(default_factory=dict)

    # List of required primitive method names
    required_primitives: List[str] = field(default_factory=list)

    # Whether this decomposition is verified to produce identical results
    verified: bool = False

    # Optional notes about edge cases or limitations
    notes: Optional[str] = None

    def __post_init__(self):
        """Auto-extract required primitives from expression if not provided."""
        if not self.required_primitives:
            # Extract method names from expression (word followed by opening paren)
            self.required_primitives = list(set(re.findall(r'(\w+)\s*\(', self.expression)))


class DecompositionRegistry:
    """Tracks and validates method decompositions."""

    _decompositions: Dict[str, Decomposition] = {}
    _primitives: Set[str] = set()

    @classmethod
    def register_primitive(cls, method_name: str) -> None:
        """Mark a method as a primitive (cannot be decomposed further)."""
        cls._primitives.add(method_name)

    @classmethod
    def register_primitives(cls, method_names: List[str]) -> None:
        """Mark multiple methods as primitives."""
        cls._primitives.update(method_names)

    @classmethod
    def register_decomposition(cls, method_name: str, decomposition: Decomposition) -> None:
        """Register a decomposition for a composite method."""
        cls._decompositions[method_name] = decomposition

    @classmethod
    def get_decomposition(cls, method_name: str) -> Optional[Decomposition]:
        """Get decomposition for a method."""
        return cls._decompositions.get(method_name)

    @classmethod
    def is_primitive(cls, method_name: str) -> bool:
        """Check if a method is a primitive."""
        return method_name in cls._primitives

    @classmethod
    def is_decomposable(cls, method_name: str) -> bool:
        """Check if a method has a registered decomposition."""
        return method_name in cls._decompositions

    @classmethod
    def get_all_primitives(cls) -> Set[str]:
        """Return all registered primitives."""
        return cls._primitives.copy()

    @classmethod
    def get_all_decompositions(cls) -> Dict[str, Decomposition]:
        """Return all registered decompositions."""
        return cls._decompositions.copy()

    @classmethod
    def expand_expression(cls, expression: str, max_depth: int = 10) -> str:
        """
        Recursively expand all composite methods to primitives.

        Args:
            expression: The expression to expand
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Fully expanded expression using only primitives
        """
        if max_depth <= 0:
            return expression

        # Find all method calls in expression
        method_pattern = r'(\w+)\s*\('

        def expand_match(match):
            method_name = match.group(1)
            if cls.is_primitive(method_name):
                return match.group(0)  # Keep as-is
            decomp = cls.get_decomposition(method_name)
            if decomp:
                # This is simplified - full implementation needs proper parsing
                return f"({decomp.expression.split('(')[0]}("
            return match.group(0)

        expanded = re.sub(method_pattern, expand_match, expression)

        # Recurse if we made changes
        if expanded != expression:
            return cls.expand_expression(expanded, max_depth - 1)
        return expanded

    @classmethod
    def get_coverage_stats(cls) -> Dict[str, Any]:
        """Return statistics about decomposition coverage."""
        return {
            "num_primitives": len(cls._primitives),
            "num_decompositions": len(cls._decompositions),
            "primitives": sorted(cls._primitives),
            "decomposed_methods": sorted(cls._decompositions.keys()),
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._decompositions.clear()
        cls._primitives.clear()
