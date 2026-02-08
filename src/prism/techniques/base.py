"""
Base classes and utilities for composable mathematical methods.

This module provides the foundational infrastructure for building composable
mathematical method blocks that can be chained together to generate
complex math competition problems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import random
import re
import uuid as uuid_lib

if TYPE_CHECKING:
    from .decomposition import Decomposition


@dataclass
class MethodResult:
    """Result of applying a method transformation.

    Attributes:
        value: The computed value after applying the method
        description: Human-readable description of this step
        params: Parameters used in the computation
        metadata: Optional additional information about the computation
    """
    value: Any
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MethodResult(value={self.value}, description='{self.description}')"


class MethodBlock(ABC):
    """Base class for composable mathematical methods.

    Each method block represents a single mathematical transformation
    or operation that can be composed with other blocks to create complex
    problems. Blocks define their input/output types to enable automatic
    chaining.

    Attributes:
        name: Unique identifier for this method (e.g., "legendre_valuation")
        input_type: Type of input this method accepts (e.g., "integer", "polynomial")
        output_type: Type of output this method produces (e.g., "valuation", "count")
        difficulty: Difficulty rating on 1-5 scale
        tags: List of topic tags (e.g., ["number_theory", "valuation"])
    """

    name: str = "base_method"
    input_type: str = "any"
    output_type: str = "any"
    difficulty: int = 1
    tags: List[str] = []

    # Decomposition attributes
    decomposition: Optional["Decomposition"] = None
    is_primitive: bool = False

    @abstractmethod
    def generate_parameters(self, input_value: Any = None) -> Dict[str, Any]:
        """Generate random valid parameters for this method.

        Args:
            input_value: Optional input value to ensure parameter compatibility

        Returns:
            Dictionary of parameters needed for the method

        Example:
            For a modular arithmetic method:
            {'modulus': 1000000007, 'offset': 42}
        """
        pass

    @abstractmethod
    def compute(self, input_value: Any, params: Dict[str, Any]) -> MethodResult:
        """Apply the method transformation forward.

        Args:
            input_value: The input value to transform
            params: Parameters controlling the transformation

        Returns:
            MethodResult containing the output value, description, and params

        Example:
            input_value = 100, params = {'prime': 2}
            returns MethodResult(value=2, description="v_2(100) = 2", params=...)
        """
        pass

    @abstractmethod
    def can_invert(self) -> bool:
        """Whether this method has a well-defined inverse.

        Returns:
            True if invert() is implemented and functional
        """
        pass

    def invert(self, output_value: Any, params: Dict[str, Any]) -> Optional[Any]:
        """Compute the inverse transformation (find input given output).

        Args:
            output_value: The desired output value
            params: Parameters that were used in the forward transformation

        Returns:
            The input value that would produce output_value, or None if not invertible

        Note:
            Only implement if can_invert() returns True
        """
        return None

    def validate_input(self, input_value: Any) -> bool:
        """Check if input is valid for this method.

        Args:
            input_value: The value to validate

        Returns:
            True if the input is valid for this method

        Example:
            For a factorization method, check that input is a positive integer
        """
        return True

    def validate_params(self, params: Dict[str, Any], prev_value: Any = None) -> bool:
        """Check if parameters satisfy mathematical preconditions for this method.

        Override this in techniques that have mathematical preconditions.
        Examples:
        - Modular inverse: gcd(a, m) == 1
        - Triangle operations: triangle inequality holds
        - Square root: value is non-negative
        - Binomial: 0 <= k <= n

        Args:
            params: The parameters for this method
            prev_value: The value from the previous step (used as first param)

        Returns:
            True if the params are mathematically valid for this method
        """
        return True

    def describe(self, params: Dict[str, Any]) -> str:
        """Human-readable description of what this method does.

        Args:
            params: The parameters for this method

        Returns:
            A string describing the method with these parameters

        Example:
            "Find the 2-adic valuation" for params={'prime': 2}
        """
        return f"Apply {self.name}"

    def print_with_seed_names(self) -> str:
        """Return method signature with parameter names for LLM prompt.

        Examples:
            "sieve_count(n, primes)" - shows all params from generate_parameters()
            "mod_inverse(n, a, m)" - shows all params from generate_parameters()
            "binomial_coefficient(n, k)" - shows all params from generate_parameters()

        Uses generate_parameters() to discover ALL parameter names and shows them
        in the signature so the LLM knows what order to pass them.

        Returns:
            Function signature string with all parameter names
        """
        # Get parameter names from generate_parameters()
        try:
            sample_params = self.generate_parameters()
            param_names = list(sample_params.keys())
        except Exception:
            param_names = []

        # Build signature with all params
        if not param_names:
            return f"{self.name}(x)"  # Default fallback
        else:
            params_str = ", ".join(param_names)
            return f"{self.name}({params_str})"

    def get_difficulty(self) -> int:
        """Get the difficulty rating of this method.

        Returns:
            Integer from 1 (easiest) to 5 (hardest)
        """
        return self.difficulty

    def has_tag(self, tag: str) -> bool:
        """Check if this method has a specific tag.

        Args:
            tag: The tag to check for

        Returns:
            True if the tag is present
        """
        return tag in self.tags

    def get_decomposed_program(self, params: Dict[str, Any]) -> Optional[str]:
        """Get the decomposed primitive expression with parameters substituted.

        Args:
            params: Dictionary of parameter values to substitute

        Returns:
            The decomposition expression with parameter values substituted,
            or None if no decomposition exists.

        Example:
            If decomposition.expression is "divide(binomial(multiply(2, n), n), add(n, 1))"
            and params is {"n": 5}, returns "divide(binomial(multiply(2, 5), 5), add(5, 1))"
        """
        if self.decomposition is None:
            return None

        expression = self.decomposition.expression

        # Apply parameter mapping if defined
        param_map = self.decomposition.param_map
        mapped_params = {}
        for orig_name, value in params.items():
            # Use mapping if exists, otherwise use original name
            mapped_name = param_map.get(orig_name, orig_name)
            mapped_params[mapped_name] = value

        # Substitute each parameter in the expression
        # We need to be careful to replace whole words only (not substrings)
        for param_name, value in mapped_params.items():
            # Use word boundary matching to avoid replacing substrings
            pattern = r'\b' + re.escape(param_name) + r'\b'
            expression = re.sub(pattern, str(value), expression)

        return expression

    def is_decomposable(self) -> bool:
        """Check if this method has a decomposition defined.

        Returns:
            True if a decomposition exists for this method
        """
        return self.decomposition is not None

    def _find_params_for_answer(self, target_answer: int) -> Optional[Dict[str, Any]]:
        """Find parameters that produce the target answer (backward generation).

        Override this method to enable backward generation for a method.

        Args:
            target_answer: The desired answer value

        Returns:
            Parameters that would produce target_answer, or None if not possible
        """
        return None  # Default: backward generation not supported

    def generate(self, target_answer: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Generate a complete problem, optionally targeting a specific answer.

        This is the main entry point for problem generation. Subclasses can override
        this entirely, or just override _find_params_for_answer() for backward generation.

        Args:
            target_answer: If provided, try to generate a problem with this answer.
                          If None, generate a random problem.

        Returns:
            Problem dict with keys: relationship, answer, methods, uuid, metadata
            Returns None if generation fails (e.g., can't hit target_answer)
        """
        try:
            if target_answer is not None:
                # Backward generation: find params for target
                params = self._find_params_for_answer(target_answer)
                if params is None:
                    return None  # Can't generate this target
            else:
                # Forward generation: random params
                params = self.generate_parameters()

            # Compute the result
            result = self.compute(None, params)

            # Build problem dict
            # Merge result metadata with base metadata
            metadata = {"params": params, "difficulty": self.difficulty}
            if hasattr(result, 'metadata') and result.metadata:
                metadata.update(result.metadata)

            return create_problem_dict(
                relationship=result.description,
                answer=result.value,
                methods=[self.name],
                uuid=generate_uuid(),
                metadata=metadata
            )
        except Exception:
            return None  # Generation failed


# Utility functions for problem generation

def create_problem_dict(
    relationship: str,
    answer: int,
    methods: List[str],
    uuid: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized problem dictionary.

    Args:
        relationship: The problem statement/relationship in LaTeX format
        answer: The final answer (integer)
        methods: List of method names used to generate this problem
        uuid: Unique identifier for this problem
        metadata: Optional additional metadata

    Returns:
        Dictionary containing the problem in standard format

    Example:
        {
            'relationship': 'Find $v_2(100!)$',
            'answer': 97,
            'methods': ['legendre_valuation'],
            'uuid': '123e4567-e89b-12d3-a456-426614174000',
            'metadata': {'difficulty': 3}
        }
    """
    problem = {
        'relationship': relationship,
        'answer': answer,
        'methods': methods,
        'uuid': uuid
    }

    if metadata:
        problem['metadata'] = metadata

    return problem


def generate_uuid() -> str:
    """Generate a unique identifier for a problem.

    Returns:
        UUID string in standard format
    """
    return str(uuid_lib.uuid4())


def safe_int(x: Any, default: int = 0) -> int:
    """Safely convert a value to integer.

    Args:
        x: Value to convert
        default: Default value if conversion fails

    Returns:
        Integer value or default
    """
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm.

    Args:
        a: First integer
        b: Second integer

    Returns:
        GCD of a and b
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Compute least common multiple.

    Args:
        a: First integer
        b: Second integer

    Returns:
        LCM of a and b
    """
    return abs(a * b) // gcd(a, b) if a and b else 0


def is_prime(n: int) -> bool:
    """Check if a number is prime using trial division.

    Args:
        n: Number to check

    Returns:
        True if n is prime

    Note:
        Efficient for small primes, not suitable for cryptographic use
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def prime_factorization(n: int) -> Dict[int, int]:
    """Compute prime factorization of n.

    Args:
        n: Positive integer to factor

    Returns:
        Dictionary mapping prime -> exponent

    Example:
        prime_factorization(100) returns {2: 2, 5: 2}
    """
    if n <= 1:
        return {}

    factors = {}
    d = 2

    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1

    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    return factors


def get_small_primes(limit: int = 1000) -> List[int]:
    """Get list of small primes up to limit using Sieve of Eratosthenes.

    Args:
        limit: Maximum value for primes

    Returns:
        List of primes up to limit
    """
    if limit < 2:
        return []

    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False

    return [i for i in range(limit + 1) if sieve[i]]


# Common prime list for quick access
SMALL_PRIMES = get_small_primes(1000)


def random_prime(min_val: int = 2, max_val: int = 100) -> int:
    """Get a random prime in the given range.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Random prime in [min_val, max_val]
    """
    primes = [p for p in SMALL_PRIMES if min_val <= p <= max_val]
    if not primes:
        # Fallback to checking random numbers
        for _ in range(1000):
            candidate = random.randint(min_val, max_val)
            if is_prime(candidate):
                return candidate
        return 2  # Ultimate fallback
    return random.choice(primes)


def totient(n: int) -> int:
    """Euler's totient function - count of integers <= n coprime to n.

    Args:
        n: Positive integer

    Returns:
        phi(n)
    """
    if n == 1:
        return 1

    result = n
    factors = prime_factorization(n)

    for p in factors:
        result -= result // p

    return result


def divisor_count(n: int) -> int:
    """Count number of divisors of n.

    Args:
        n: Positive integer

    Returns:
        Number of divisors (including 1 and n)
    """
    if n <= 0:
        return 0

    factors = prime_factorization(n)
    count = 1

    for exponent in factors.values():
        count *= (exponent + 1)

    return count


def divisor_sum(n: int) -> int:
    """Sum of all divisors of n.

    Args:
        n: Positive integer

    Returns:
        Sum of divisors (including 1 and n)
    """
    if n <= 0:
        return 0

    factors = prime_factorization(n)
    result = 1

    for p, exp in factors.items():
        # Sum of geometric series: 1 + p + p^2 + ... + p^exp
        result *= (p**(exp + 1) - 1) // (p - 1)

    return result


# Type compatibility mapping for composable methods
# Maps output types to compatible input types
TYPE_COMPATIBILITY = {
    "integer": ["integer", "modular", "polynomial", "sequence", "any"],
    "polynomial": ["polynomial", "algebraic", "any"],
    "sequence": ["sequence", "integer", "any"],
    "set": ["set", "combinatorial", "any"],
    "modular": ["modular", "integer", "any"],
    "algebraic": ["algebraic", "polynomial", "any"],
    "combinatorial": ["combinatorial", "integer", "any"],
    "geometric": ["geometric", "algebraic", "any"],
    "valuation": ["valuation", "integer", "any"],
    "count": ["count", "integer", "any"],
    "sum": ["sum", "integer", "any"],
    "any": ["any"],
}
