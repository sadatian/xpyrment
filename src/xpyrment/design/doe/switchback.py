import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class SwitchbackDesign(DesignMatrix):
    """Generates time/geo crossover switchback experimental designs for marketplace tests."""

    def __init__(self, factors: dict, unit_window_hours: int = 2):
        super().__init__(factors)
        self.unit_window_hours = unit_window_hours

    def generate(self) -> pd.DataFrame:
        # TODO: Implement switchback crossover matrix allocations
        return pd.DataFrame()
