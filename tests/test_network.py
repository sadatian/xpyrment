import pytest
import numpy as np
from xpyrment.network.cluster import ClusterRandomizer
from xpyrment.network.spillover import NeighborhoodExposure


def test_cluster_randomizer():
    """Validates LPA graph partitioning correctness and group treatment assignments."""
    # Create an undirected network with two highly dense disconnected communities
    # Community 1: Nodes 0, 1, 2
    # Community 2: Nodes 3, 4, 5
    adjacency = np.zeros((6, 6))
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    adjacency[0, 2] = adjacency[2, 0] = 1

    adjacency[3, 4] = adjacency[4, 3] = 1
    adjacency[4, 5] = adjacency[5, 4] = 1
    adjacency[3, 5] = adjacency[5, 3] = 1

    # Undirected validation check
    with pytest.raises(ValueError):
        # Pass non-symmetric matrix
        ClusterRandomizer(np.array([[0, 1], [0, 0]]))

    randomizer = ClusterRandomizer(adjacency)
    labels = randomizer.detect_communities(seed=42)

    assert len(labels) == 6
    # Asserts nodes in the same community get identical cluster labels
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    # Asserts the two communities get distinct labels
    assert labels[0] != labels[3]

    # Test cluster randomized assignment
    node_treatment = randomizer.assign_treatments(ratio=0.5, seed=42)
    assert len(node_treatment) == 6
    # Entire clusters must be assigned to treatment/control as groups
    assert node_treatment[0] == node_treatment[1] == node_treatment[2]
    assert node_treatment[3] == node_treatment[4] == node_treatment[5]


def test_neighborhood_exposure_model():
    """Validates exposure classification and DTE/ISE calculations over interconnected chain graphs."""
    # Small chain graph: 0 - 1 - 2 - 3
    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1

    # Assignments: 0, 1 are Control; 2, 3 are Treated
    treatment = np.array([0, 0, 1, 1])

    ne = NeighborhoodExposure(adjacency)
    states = ne.classify_exposure(treatment)

    # Classifications:
    # Node 0: T=0, no treated neighbors -> State 0 (Pure Control)
    # Node 1: T=0, neighbor 2 is treated -> State 1 (Control Spillover/Leakage)
    # Node 2: T=1, neighbor 3 is treated -> State 3 (Treated Exposure)
    # Node 3: T=1, neighbor 2 is treated -> State 3 (Treated Exposure)
    assert states[0] == 0
    assert states[1] == 1
    assert states[2] == 3
    assert states[3] == 3

    # Outcomes:
    y = np.array([10.0, 12.0, 15.0, 15.0])
    results = ne.estimate_effects(y, treatment)

    assert results["mean_pure_control"] == 10.0
    assert results["mean_control_spillover"] == 12.0
    assert results["mean_treated_exposure"] == 15.0

    # Direct Treatment Effect = mean_3 - mean_0 = 15.0 - 10.0 = 5.0
    assert results["direct_treatment_effect"] == pytest.approx(5.0)
    # Indirect Spillover Effect = mean_1 - mean_0 = 12.0 - 10.0 = 2.0
    assert results["indirect_spillover_effect"] == pytest.approx(2.0)
