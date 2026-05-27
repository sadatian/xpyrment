import pytest
import numpy as np
from xpyrment.network.cluster import ClusterRandomizer
from xpyrment.network.spillover import NeighborhoodExposure
from xpyrment.network.identity import IdentityRegistry
import pandas as pd


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


def test_identity_resolution():
    """Validates disjoint-set-union (DSU) based cross-device identity resolution and stitching."""
    registry = IdentityRegistry()

    # 1. Assert isolated nodes resolve to themselves
    assert registry.resolve_id("device_1") == "device_1"
    assert registry.resolve_id("user_A") == "user_A"

    # 2. Register links and verify lexicographical ordering
    # Link: device_1 <-> user_B. Since 'device_1' < 'user_B', the representative must be 'device_1'
    registry.register_link("device_1", "user_B")
    assert registry.resolve_id("device_1") == "device_1"
    assert registry.resolve_id("user_B") == "device_1"

    # Link: user_B <-> ad_id_Z. Representatives are 'device_1' and 'ad_id_Z'.
    # Since 'ad_id_Z' < 'device_1' (lexicographically), 'ad_id_Z' should become the root!
    registry.register_link("user_B", "ad_id_Z")
    assert registry.resolve_id("device_1") == "ad_id_Z"
    assert registry.resolve_id("user_B") == "ad_id_Z"
    assert registry.resolve_id("ad_id_Z") == "ad_id_Z"

    # 3. Test get_component
    assert registry.get_component("user_B") == {"device_1", "user_B", "ad_id_Z"}

    # 4. Test dataframe resolution with auto-linking
    data = pd.DataFrame([
        {"cookie": "cookie_x", "login": "user_1", "sales": 100},
        {"cookie": "cookie_y", "login": "user_2", "sales": 200},
        # Row 3 stitches user_1 and cookie_y together!
        # Now component is: cookie_x, user_1, cookie_y, user_2.
        # Lexicographically smallest identifier in this component: "cookie_x"
        {"cookie": "cookie_y", "login": "user_1", "sales": 300},
        # Row 4 is isolated
        {"cookie": "cookie_z", "login": None, "sales": 150},
    ])

    resolved = registry.resolve_dataframe(data, id_cols=["cookie", "login"], target_col="stitched_id", auto_link=True)

    # Asserts cookie_x and cookie_y/user_1/user_2 are all stitched to the smallest identifier "cookie_x"
    assert resolved.loc[0, "stitched_id"] == "cookie_x"
    assert resolved.loc[1, "stitched_id"] == "cookie_x"
    assert resolved.loc[2, "stitched_id"] == "cookie_x"
    # Row 4 resolves to cookie_z since login is Null and no prior link exists
    assert resolved.loc[3, "stitched_id"] == "cookie_z"

    # Verify component contains all stitched IDs
    assert registry.get_component("user_2") == {"cookie_x", "cookie_y", "user_1", "user_2"}


def test_gcd():
    """Validates the Greatest Common Divisor (GCD) calculation for positive, negative, and zero inputs."""
    from xpyrment.network.federated import gcd

    # Standard positive inputs
    assert gcd(48, 18) == 6
    assert gcd(18, 48) == 6

    # Negative inputs should return positive GCD
    assert gcd(-48, 18) == 6
    assert gcd(48, -18) == 6
    assert gcd(-48, -18) == 6

    # Zero handling
    assert gcd(5, 0) == 5
    assert gcd(0, 5) == 5
    assert gcd(0, 0) == 0

    # Prime numbers
    assert gcd(13, 17) == 1

    # Same numbers
    assert gcd(7, 7) == 7
    assert gcd(-7, 7) == 7


def test_federated_pooling():
    """Validates from-scratch Paillier cryptosystem, SMPC covariance pooling, and FedAvg pooling."""
    from xpyrment.network.federated import PaillierCryptosystem, federated_averaging, federated_secure_covariance_pooling
    import numpy as np

    # 1. Test Paillier Homomorphic Cryptosystem
    crypto = PaillierCryptosystem()
    m1 = 15
    m2 = 25
    
    c1 = crypto.encrypt(m1, r=11)
    c2 = crypto.encrypt(m2, r=17)
    
    # Decrypt and assert exact recovery
    assert crypto.decrypt(c1) == m1
    assert crypto.decrypt(c2) == m2

    # Test additive homomorphic summation
    c_sum = crypto.add_encrypted(c1, c2)
    assert crypto.decrypt(c_sum) == m1 + m2

    # 2. Test Secure SMPC Covariance Pooling
    cov_client_1 = np.array([[2.5, 0.8], [0.8, 1.2]])
    cov_client_2 = np.array([[1.5, -0.2], [-0.2, 0.9]])
    
    expected_global_cov = cov_client_1 + cov_client_2

    pooled_cov = federated_secure_covariance_pooling([cov_client_1, cov_client_2], crypto)
    # Asserts homomorphically encrypted sum matches exact mathematical sum
    assert np.allclose(pooled_cov, expected_global_cov, atol=0.02)

    # 3. Test Federated Averaging (FedAvg)
    local_w1 = np.array([1.5, -0.5, 2.0])
    local_w2 = np.array([2.5, 0.5, 1.0])
    
    sample_sizes = [100, 300]
    expected_global_w = 0.25 * local_w1 + 0.75 * local_w2

    global_w = federated_averaging([local_w1, local_w2], sample_sizes)
    assert np.allclose(global_w, expected_global_w)


def test_entropy_balanced_partition():
    """Validates the Entropy-Balanced Graph Partitioner."""
    from xpyrment.network.partition import EntropyBalancedGraphPartitioner

    # Create a graph with a potential giant community and a small community
    # Node 0, 1, 2, 3: large dense community
    # Node 4, 5: small dense community
    adj = {
        0: [1, 2, 3],
        1: [0, 2, 3],
        2: [0, 1, 3],
        3: [0, 1, 2],
        4: [5],
        5: [4]
    }

    # Test with gamma = 0.5 (balanced)
    partitioner = EntropyBalancedGraphPartitioner(adj, gamma=0.5)
    labels = partitioner.fit_predict(seed=42)

    assert len(labels) == 6
    # Node 0, 1, 2, 3 should have same label
    assert labels[0] == labels[1] == labels[2] == labels[3]
    # Node 4, 5 should have same label
    assert labels[4] == labels[5]
    # Small community should be separate from large community
    assert labels[0] != labels[4]

    # Check entropy and normalized cut
    entropy = partitioner.compute_size_entropy(labels)
    assert entropy > 0.1
    
    n_cut = partitioner.compute_normalized_cut(labels)
    assert n_cut == pytest.approx(0.0)  # disconnected components, cut is exactly zero


