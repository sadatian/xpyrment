import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class FractionalFactorialDesign(DesignMatrix):
    """Generates fractional factorial experimental designs, showing resolution and aliasing."""

    def __init__(self, factors: dict, generator_string: str):
        super().__init__(factors)
        self.generator_string = generator_string

    def generate(self) -> pd.DataFrame:
        """Generates fractional factorial matrix."""
        # TODO: Implement generator parsing and fractional matrix construction
        return pd.DataFrame()
