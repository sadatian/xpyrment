"""Experimental design, user routing, traffic splits, randomization, and Design of Experiments (DoE).

This package provides utilities for configuring study designs, routing experimental traffic, setting up
randomization schemes, and classical Design of Experiments (DoE) matrices:
- `hash_assign`: Cryptographic deterministic hashing engine to route units state-lessly.
- `TrafficSplitter`: Coordinates multi-variant allocations, long-term holdouts, and progressive ramps.
- `stratified_randomization`: Ensures structural balance on pre-experiment covariates.
- `doe`: Comprehensive classical design matrices (factorial, fractional, Taguchi, DSD, CCD, mixture, etc.).
"""

from xpyrment.design.randomization import hash_assign
from xpyrment.design.stratification import stratified_randomization
from xpyrment.design.splits import TrafficSplitter
from xpyrment.design import doe

__all__ = [
    "hash_assign",
    "stratified_randomization",
    "TrafficSplitter",
    "doe",
]
