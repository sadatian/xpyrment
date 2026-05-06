import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class CentralCompositeDesign(DesignMatrix):
    """Generates Central Composite Designs (CCD) for response surface modeling."""

    def __init__(self, factors: dict, alpha: str = "orthogonal"):
        super().__init__(factors)
        self.alpha_type = alpha

    def generate(self) -> pd.DataFrame:
        # TODO: Implement CCD factorial, axial, and center points generation
        return pd.DataFrame()
