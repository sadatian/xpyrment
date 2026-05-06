import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class DefinitiveScreeningDesign(DesignMatrix):
    """Generates a Definitive Screening Design (DSD) to estimate linear, quadratic, and 2-way terms."""

    def generate(self) -> pd.DataFrame:
        # TODO: Implement DSD algorithm (using conference matrices)
        return pd.DataFrame()
