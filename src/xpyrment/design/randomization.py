import hashlib
from typing import List, Union


def hash_assign(unit_id: Union[str, int], salt: str, variants: List[str]) -> str:
    """Assigns a unit to a variant deterministically using MD5/SHA-256 hashing.

    Args:
        unit_id: User/unit identifier.
        salt: Salt representing the experiment ID to ensure orthogonal assignments.
        variants: List of variants to split across.

    Returns:
        str: Assigned variant label.
    """
    # Simple hash-based assignment
    hash_input = f"{salt}:{unit_id}".encode("utf-8")
    hash_hex = hashlib.md5(hash_input).hexdigest()
    # Take first 8 chars, convert to int, and modulo by variants length
    hash_val = int(hash_hex[:8], 16)
    idx = hash_val % len(variants)
    return variants[idx]
