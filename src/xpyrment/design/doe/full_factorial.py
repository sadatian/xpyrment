from typing import Dict, List
import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class FullFactorialDesign(DesignMatrix):
    """Generates a full factorial experimental design matrix for all factor combinations."""

    def generate(self) -> pd.DataFrame:
        """Generates full factorial matrix."""
        # Simple placeholder implementation
        import itertools

        keys = list(self.factors.keys())
        values = list(self.factors.values())
        combinations = list(itertools.product(*values))

        df = pd.DataFrame(combinations, columns=keys)
        return df
