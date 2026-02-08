"""
Shared utilities for number theory deep insight techniques.
"""

import random
import math
from typing import Dict, Any, Optional


def generate_uuid() -> str:
    """Generate a unique identifier."""
    import uuid as uuid_lib
    return str(uuid_lib.uuid4())


def create_problem_dict(relationship: str, answer: int, techniques: list,
                       uuid: str, metadata: dict) -> dict:
    """Create a problem dictionary."""
    return {
        "problem": relationship,
        "answer": answer,
        "techniques": techniques,
        "uuid": uuid,
        "metadata": metadata
    }
