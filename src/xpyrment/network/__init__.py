"""Network effects and cluster randomized trial systems.

Submodules:
- `cluster`: Graph community detection and cluster allocations.
- `spillover`: Estimating direct vs. indirect neighborhood exposures.
"""

from xpyrment.network.cluster import ClusterRandomizer
from xpyrment.network.spillover import NeighborhoodExposure
from xpyrment.network.identity import IdentityRegistry
from xpyrment.network.federated import PaillierCryptosystem, federated_averaging, federated_secure_covariance_pooling
from xpyrment.network.partition import EntropyBalancedGraphPartitioner

__all__ = [
    "ClusterRandomizer",
    "NeighborhoodExposure",
    "IdentityRegistry",
    "PaillierCryptosystem",
    "federated_averaging",
    "federated_secure_covariance_pooling",
    "EntropyBalancedGraphPartitioner",
]
