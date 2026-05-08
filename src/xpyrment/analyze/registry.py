"""Metric Registry & Directed Acyclic Graph (DAG) Evaluator (Block 49).

Maintains a register of raw and derived metrics, sorting computation trees topologically and
evaluating metrics with optimal execution paths and shared-subexpression caching.
"""

from typing import Dict, List, Callable, Any, Set
import numpy as np


class MetricRegistry:
    """Manages a registry of metrics and evaluates them using topological sorting of a DAG.

    TODO: Implement out-of-core streaming batch evaluations for high-velocity telemetry pipelines.
    TODO: Add symbolic differentiation support to automatically derive Delta Method variance paths for complex algebraic combinations of parents.
    """

    def __init__(self) -> None:
        """Initializes the metric registry."""
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def add_raw(self, name: str) -> "MetricRegistry":
        """Registers a raw, primitive input metric name."""
        self.nodes[name] = {
            "type": "raw",
            "dependencies": [],
            "callable": None,
        }
        return self

    def add_derived(self, name: str, dependencies: List[str], formula: Callable[..., np.ndarray]) -> "MetricRegistry":
        """Registers a derived metric calculated from dependency parents.

        Args:
            name (str): Name of the derived metric.
            dependencies (List[str]): List of dependency metric names.
            formula (Callable): A function returning an np.ndarray given dependency arrays.
        """
        self.nodes[name] = {
            "type": "derived",
            "dependencies": dependencies,
            "callable": formula,
        }
        return self

    def _topological_sort(self) -> List[str]:
        """Performs a standard DFS-based topological sort on the registered metrics DAG.

        Raises:
            ValueError: If a cyclic dependency is detected.
        """
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        order: List[str] = []

        def dfs(node: str) -> None:
            if node in temp_visited:
                raise ValueError(f"Cyclic dependency detected involving node: {node}")
            if node not in visited:
                temp_visited.add(node)
                # If node is missing in registry, treat as an implicit raw node
                if node in self.nodes:
                    for dep in self.nodes[node]["dependencies"]:
                        dfs(dep)
                temp_visited.remove(node)
                visited.add(node)
                order.append(node)

        for name in self.nodes:
            if name not in visited:
                dfs(name)

        return order

    def evaluate(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Evaluates all registered metrics topologically, using caching.

        Args:
            inputs (Dict[str, np.ndarray]): Dictionary of raw primitive arrays.

        Returns:
            Dict[str, np.ndarray]: Complete dictionary containing both raw and derived arrays.
        """
        # Sort node evaluation order
        eval_order = self._topological_sort()
        cache: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in inputs.items()}

        for name in eval_order:
            if name in cache:
                continue

            node = self.nodes.get(name)
            if node is None or node["type"] == "raw":
                # Raw node not provided in inputs
                raise ValueError(f"Raw input data is missing for required metric: {name}")

            # Collect arguments for the formula
            args = []
            for dep in node["dependencies"]:
                if dep not in cache:
                    raise ValueError(f"Dependency metric {dep} was not evaluated for {name}.")
                args.append(cache[dep])

            # Compute and cache
            cache[name] = np.asarray(node["callable"](*args))

        return cache
