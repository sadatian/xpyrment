"""Abstract Base Class for classical Design of Experiments (DoE) matrices.

This module defines `DesignMatrix`, the abstract foundation for all classical DoE modeling
schemes. It enforces standard construction patterns for factors, levels, and generation,
supporting both screen designs (fractional, screening, Plackett-Burman) and response surface methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class DesignMatrix(ABC):
    r"""Abstract base class representing a Design Matrix in Classical Design of Experiments (DoE).

    A Design Matrix is a structured layout of factor combinations designed to evaluate treatment
    effects, screening active factors, or modeling multi-factor non-linear responses with the
    minimal possible experiment run footprint.

    Mathematical Context:
        Let $k$ be the number of independent factors (variables). A design matrix $X$ is an $N \times k$
        matrix where each row represents a specific experiment run (a combination of factor levels) and
        each column represents a factor.
        In standardized designs, factor levels are typically mapped to coded values:
        - Continuous factors: mapped to $[-1, 0, +1]$ representing low, medium, and high levels.
        - Categorical factors: mapped to discrete integer indices.
        
        The primary goal of classical DoE is to construct $X$ such that the parameter estimation variance
        of the linear model:
        $$
        Y = X\beta + \varepsilon
        $$
        is minimized, which is equivalent to maximizing the information matrix $X^T X$.

    Attributes:
        factors (Dict[str, List[float]]): A dictionary mapping factor names (str) to their list
            of possible levels (floats/integers). For example: `{"temperature": [100.0, 150.0], "pressure": [1.0, 2.0]}`.
    """

    def __init__(self, factors: Dict[str, List[float]]) -> None:
        """Initializes a new DesignMatrix.

        Args:
            factors (Dict[str, List[float]]): Mapping of factor labels to their designated levels.
        """
        self.factors = factors

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generates and returns the specific classical DoE design matrix.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the design matrix, where columns represent
                factors and rows represent specific experimental runs.
        """
        pass
