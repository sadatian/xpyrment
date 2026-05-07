"""Deterministic, hash-based randomization engines for user and unit assignments.

This module provides the cryptographic hashing engines used to route users or devices into
experimental variants deterministically. It avoids server-side state lookup, ensuring
zero-latency, horizontal scaling, and perfect repeatability across distributed execution environments.
"""

import hashlib
from typing import List, Union


def hash_assign(unit_id: Union[str, int], salt: str, variants: List[str]) -> str:
    r"""Assigns a unit to a variant deterministically using MD5 hashing and modulo arithmetic.

    To distribute units (e.g., user IDs or device hashes) uniformly and orthogonally across multiple
    concurrent experiments without storing state, we concatenate a static experiment-level unique identifier
    (the "salt") with the unit's unique identifier. The resulting string is hashed, and the lower slice is mapped
    into the variant array using modulo arithmetic.

    Mathematical Representation:
        Let $u$ be the unit identifier, $S$ be the unique experiment salt, and $V = (v_1, v_2, \dots, v_k)$
        be the ordered array of $k$ variants.
        The assignment key is formed by concatenation:
        $$K = S \mathbin{\Vert} \text{str}(u)$$
        We compute the MD5 digest of $K$ (yielding a 128-bit hex string) and extract the first 8 characters,
        representing a 32-bit integer $H$:
        $$H = \text{hex\_to\_int}(\text{MD5}(K)[0:8])$$
        The target variant index $i$ is calculated using the modulo operator:
        $$i = H \pmod{k}$$
        The assigned variant is $V_i$.

    Properties of Hash-based Assignment:
        1. **Repeatability**: For a given unit ID $u$ and salt $S$, the returned variant is always identical,
           eliminating the need for distributed database lookups.
        2. **Uniformity**: The MD5 hash exhibits avalanche-effect characteristics, distributing assignment
           probabilities uniformly: $P(\text{Variant} = v) \approx 1/k$.
        3. **Orthogonal Decorrelation**: By using different salts for different experiments ($S_A \neq S_B$),
           user allocations in experiment A are statistically independent of their allocations in experiment B,
           preventing cross-contamination and carrying effects across concurrent runs.

    Args:
        unit_id (Union[str, int]): Unique identifier for the experimental unit (e.g., visitor ID, account code).
        salt (str): A unique string identifying the active experiment. Acting as the cryptographic salt,
            this prevents correlation between different active experiments.
        variants (List[str]): Ordered list of variant labels (e.g., `["control", "treatment_a", "treatment_b"]`).

    Returns:
        str: The selected variant label from the `variants` list.

    Raises:
        ValueError: If the `variants` list is empty.

    Example:
        >>> variants = ["control", "variant_a", "variant_b"]
        >>> hash_assign(unit_id="user_12345", salt="EXP-201_checkout_revamp", variants=variants)
        'variant_a'
        >>> hash_assign(unit_id="user_12345", salt="EXP-202_price_test", variants=variants) # Orthogonal salt
        'control'
    """
    if not variants:
        raise ValueError("variants list cannot be empty.")

    # Simple hash-based assignment
    hash_input = f"{salt}:{unit_id}".encode("utf-8")
    hash_hex = hashlib.md5(hash_input).hexdigest()
    # Take first 8 chars, convert to int, and modulo by variants length
    hash_val = int(hash_hex[:8], 16)
    idx = hash_val % len(variants)
    return variants[idx]

