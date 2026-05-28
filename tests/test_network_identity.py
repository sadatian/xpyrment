"""Unit tests for graph-based identity resolution and stitch registry (network/identity.py)."""

import pytest
import numpy as np
import pandas as pd
from xpyrment.network.identity import IdentityRegistry


def test_identity_registry_basic_stitching():
    """Asserts that pairwise identity registration stitches and path-compresses correctly."""
    registry = IdentityRegistry()
    
    # Register links: user1 <-> cookieA, user1 <-> deviceX
    # Lexicographically: "cookieA" < "deviceX" < "user1"
    registry.register_link("user1", "cookieA")
    registry.register_link("user1", "deviceX")
    
    # Verify both resolve to "cookieA"
    assert registry.resolve_id("user1") == "cookieA"
    assert registry.resolve_id("cookieA") == "cookieA"
    assert registry.resolve_id("deviceX") == "cookieA"
    
    # Null link check
    registry.register_link("user1", "")
    assert registry.resolve_id("user1") == "cookieA"


def test_identity_registry_register_links():
    """Asserts register_links forms cliques while filtering out nulls and empty values."""
    registry = IdentityRegistry()
    
    # Links with empty, null, and whitespace entries
    nodes = ["deviceB", "deviceA", "", "  ", None, pd.NA]
    registry.register_links(nodes)
    
    # deviceB <-> deviceA should be linked to "deviceA"
    assert registry.resolve_id("deviceB") == "deviceA"
    assert registry.resolve_id("deviceA") == "deviceA"


def test_identity_registry_resolve_id_null_errors():
    """Asserts that trying to resolve null or empty nodes raises a ValueError."""
    registry = IdentityRegistry()
    
    with pytest.raises(ValueError, match="Cannot resolve null or empty"):
        registry.resolve_id("")
        
    with pytest.raises(ValueError, match="Cannot resolve null or empty"):
        registry.resolve_id(None)


def test_identity_registry_get_components():
    """Asserts retrieval of single and all components behaves correctly."""
    registry = IdentityRegistry()
    
    # Component for unregistered node
    assert registry.get_component("missing") == {"missing"}
    
    # Register some links
    registry.register_link("A", "B")
    registry.register_link("C", "D")
    
    comp_ab = registry.get_component("A")
    assert comp_ab == {"A", "B"}
    
    all_comps = registry.get_all_components()
    assert len(all_comps) == 2
    assert {"A", "B"} in all_comps
    assert {"C", "D"} in all_comps


def test_identity_registry_resolve_dataframe():
    """Asserts DataFrame resolution with and without auto-linking works correctly."""
    df = pd.DataFrame({
        "cookie_id": ["cookie1", "cookie1", "cookie2", "cookie3"],
        "login_id": ["userA", "userA", "userB", None],
        "email_hash": [None, "hash1", "hash1", None]
    })
    
    registry = IdentityRegistry()
    
    # First pass: resolve with auto_link = True
    # row 1: cookie1, userA -> linked
    # row 2: cookie1, userA, hash1 -> linked
    # row 3: cookie2, userB, hash1 -> userB, cookie2 are stitched into the cookie1/userA/hash1 component!
    # Lexicographically: "cookie1" < "cookie2" < "cookie3" < "hash1" < "userA" < "userB"
    res_df = registry.resolve_dataframe(df, ["cookie_id", "login_id", "email_hash"], target_col="uid", auto_link=True)
    
    assert res_df["uid"].iloc[0] == "cookie1"
    assert res_df["uid"].iloc[1] == "cookie1"
    assert res_df["uid"].iloc[2] == "cookie1"  # Stitched via hash1!
    assert res_df["uid"].iloc[3] == "cookie3"  # Isolated
    
    # Test resolve with auto_link = False on a new registry
    registry2 = IdentityRegistry()
    registry2.register_link("userA", "cookie1")
    
    res_df2 = registry2.resolve_dataframe(df, ["cookie_id", "login_id", "email_hash"], target_col="uid", auto_link=False)
    # Only resolves rows that match userA/cookie1 link
    assert res_df2["uid"].iloc[0] == "cookie1"
    assert res_df2["uid"].iloc[1] == "cookie1"
    assert res_df2["uid"].iloc[2] == "cookie2"  # cookie2 is lexicographically smaller than userB/hash1, unregistered
    
    # Row with all null IDs
    df_null = pd.DataFrame({
        "cookie_id": [None],
        "login_id": [None]
    })
    res_null = registry2.resolve_dataframe(df_null, ["cookie_id", "login_id"], target_col="uid")
    assert res_null["uid"].iloc[0] is None
