import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class BoxBehnkenDesign(DesignMatrix):
    """Generates Box-Behnken Designs to estimate non-linear response surfaces in fewer runs."""

    def generate(self) -> pd.DataFrame:
        # TODO: Implement Box-Behnken matrix builder
        return pd.DataFrame()
