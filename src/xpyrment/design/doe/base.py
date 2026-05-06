from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class DesignMatrix(ABC):
    """Abstract base class representing a Design Matrix in Classical Design of Experiments (DoE)."""

    def __init__(self, factors: Dict[str, List[float]]):
        self.factors = factors

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Generates and returns the DoE design matrix."""
        pass
