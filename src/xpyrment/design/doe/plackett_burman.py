import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class PlackettBurmanDesign(DesignMatrix):
    """Generates Plackett-Burman screening design matrices to identify main effects in few runs."""

    def generate(self) -> pd.DataFrame:
        # TODO: Implement Plackett-Burman matrix builder
        return pd.DataFrame()
