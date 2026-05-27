"""Graph-based identity resolution registry for cross-device tracking and stitching.

Solves user session stitching across distinct device identifiers (e.g., cookies,
mobile advertising IDs, server-side login events) to eliminate cross-arm user leakage.

# TODO: Implement parallelized union-find component graph traversal using multi-threaded batch resolution for large-scale production logs.
"""

from typing import Dict, List, Optional, Set
import pandas as pd


class IdentityRegistry:
    """Graph-based identity resolution registry.

    Connected components of the identity graph represent unified experimental units.
    The resolved 'unified ID' is the lexicographically first node identifier
    within the connected component. This deterministic representative serves as the
    stable variant routing ID, preventing treatment leakage across multiple devices.
    """

    def __init__(self) -> None:
        """Initializes a new graph-based identity resolution registry."""
        # Maps each node to its parent in the disjoint-set-union (DSU) forest
        self.parent: Dict[str, str] = {}

    def _find(self, node: str) -> str:
        """Finds the representative of the connected component containing the node.

        Applies path compression to optimize future lookup queries to O(alpha(N)).
        If the node does not exist in the registry, it is registered as its own
        parent.

        Args:
            node (str): The node identifier to look up.

        Returns:
            str: The lexicographically first node identifier in the component.
        """
        if node not in self.parent:
            self.parent[node] = node
            return node

        # Standard path compression traversal
        path = []
        curr = node
        while self.parent[curr] != curr:
            path.append(curr)
            curr = self.parent[curr]

        for n in path:
            self.parent[n] = curr

        return curr

    def register_link(self, node_a: str, node_b: str) -> None:
        """Registers a bidirectional link (edge) between two identifiers.

        Unions the connected components of node_a and node_b. To ensure that the
        representative of a connected component is always lexicographically first,
        the lexicographically smaller representative is always selected as the parent
        of the lexicographically larger representative.

        Args:
            node_a (str): The first identifier.
            node_b (str): The second identifier.
        """
        # Exclude null/empty string values from links
        if not node_a or not node_b:
            return

        root_a = self._find(node_a)
        root_b = self._find(node_b)

        if root_a != root_b:
            # Enforce lexicographically first representative ordering
            if root_a < root_b:
                self.parent[root_b] = root_a
            else:
                self.parent[root_a] = root_b

    def register_links(self, nodes: List[str]) -> None:
        """Registers links between all identifiers in the provided list.

        This effectively forms a fully connected component (clique) among the non-null
        identifiers in the list.

        Args:
            nodes (List[str]): List of node identifiers to link.
        """
        valid_nodes = [str(n) for n in nodes if pd.notna(n) and str(n).strip() != ""]
        if len(valid_nodes) < 2:
            return

        for i in range(len(valid_nodes) - 1):
            self.register_link(valid_nodes[i], valid_nodes[i + 1])

    def resolve_id(self, node: str) -> str:
        """Resolves an identifier to its unified experimental unit ID.

        If the node does not exist in the registry, it is registered and returned
        as its own representative.

        Args:
            node (str): The node identifier.

        Returns:
            str: The unified representative ID.
        """
        if not node or pd.isna(node):
            raise ValueError("Cannot resolve null or empty node identifier.")
        return self._find(str(node))

    def get_component(self, node: str) -> Set[str]:
        """Gets all node identifiers belonging to the same connected component.

        Args:
            node (str): The node identifier.

        Returns:
            Set[str]: The set of all linked identifiers.
        """
        if node not in self.parent:
            return {node}
        root = self._find(node)
        return {k for k in self.parent if self._find(k) == root}

    def get_all_components(self) -> List[Set[str]]:
        """Returns all connected components in the registry.

        Returns:
            List[Set[str]]: A list of sets, where each set represents a connected component.
        """
        from collections import defaultdict

        groups = defaultdict(set)
        for node in list(self.parent.keys()):
            root = self._find(node)
            groups[root].add(node)
        return list(groups.values())

    def resolve_dataframe(
        self,
        df: pd.DataFrame,
        id_cols: List[str],
        target_col: str = "unified_id",
        auto_link: bool = True,
    ) -> pd.DataFrame:
        """Resolves identifiers in a pandas DataFrame to stable unified unit IDs.

        For each row, extracts non-null values from the specified identifier columns,
        optionally links them, and resolves them to the single unified lexicographically
        first identifier.

        To guarantee consistency when dynamic stitching links are discovered mid-dataset,
        this method operates in two passes:
        1. Link building: If auto_link is True, registers links between all present
           identifiers in each row across the entire dataset.
        2. Resolution: Looks up the unified identifier for each row using the final
           fully constructed identity graph.

        Args:
            df (pd.DataFrame): Input DataFrame containing tracking columns.
            id_cols (List[str]): List of column names containing tracking identifiers
                (e.g., ['cookie_id', 'user_id', 'email_hash']).
            target_col (str): Name of the output column to store resolved unit IDs.
                Defaults to 'unified_id'.
            auto_link (bool): If True, automatically registers links between all
                present (non-null) identifiers in each row. Defaults to True.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with the resolved target column added.
        """
        result_df = df.copy()

        # Pass 1: Build the identity graph from all row links if auto_link is enabled
        if auto_link:
            for row in result_df.itertuples():
                row_ids = []
                for col in id_cols:
                    val = getattr(row, col)
                    if pd.notna(val) and str(val).strip() != "":
                        row_ids.append(str(val))
                if len(row_ids) > 1:
                    self.register_links(row_ids)

        # Pass 2: Resolve identifiers for each row
        resolved_ids = []
        for row in result_df.itertuples():
            row_ids = []
            for col in id_cols:
                val = getattr(row, col)
                if pd.notna(val) and str(val).strip() != "":
                    row_ids.append(str(val))

            if not row_ids:
                resolved_ids.append(None)
            else:
                resolved_ids.append(self._find(row_ids[0]))

        result_df[target_col] = resolved_ids
        return result_df
