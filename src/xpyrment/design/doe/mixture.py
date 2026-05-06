import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class MixtureDesign(DesignMatrix):
    """Generates designs for mixture factor spaces where components must sum to 1.0 (e.g., recipe mixes)."""

    def generate(self) -> pd.DataFrame:
        # TODO: Implement Simplex Lattice or Simplex Centroid mixture designs
        return pd.DataFrame()
