"""Classical Design of Experiments (DoE) generators and optimization engines.

This package houses classical Design of Experiments (DoE) algorithms. It provides a comprehensive set of
standardized design matrices for screening active factors, modeling non-linear response surfaces, optimizing mixtures,
and scheduling marketplace crossover tests.

Available Designs:
- `FullFactorialDesign`: Tests all possible level combinations. Orthogonal, estimates all interaction orders.
- `FractionalFactorialDesign` ($2^{k-p}$): Highly efficient screening fractionals with explicit alias and resolution tracking.
- `PlackettBurmanDesign`: Resolution III screening designs for identifying main effects under extreme trial budgets.
- `TaguchiDesign`: Balanced Orthogonal Arrays mapped with Signal-to-Noise (S/N) ratios to optimize process robustness.
- `DefinitiveScreeningDesign` (DSD): Advanced 3-level designs estimating linear, quadratic, and 2-way terms without follow-up runs.
- `CentralCompositeDesign` (CCD): Standard 5-level Response Surface Methodology (RSM) design (cube + star + center points).
- `BoxBehnkenDesign` (BBD): Efficient 3-level response surface design avoiding physical or operational corner combinations.
- `DOptimalDesign`: Computer-guided coordinate exchange optimization to build designs under irregular constraints or budget sizes.
- `LatinHypercubeDesign` (LHS): Multidimensional space-filling sampling for simulation and black-box computer experiments.
- `MixtureDesign`: Simplex Lattice and Simplex Centroid formulations where component fractions must sum strictly to 1.0.
- `SwitchbackDesign`: Temporal crossover designs for multi-region marketplace tests where network spillovers preclude user-splits.
- `EVOPDesign`: Sequential low-amplitude perturbation loops designed for live, full-scale continuous process optimization.
"""

from xpyrment.design.doe.base import DesignMatrix
from xpyrment.design.doe.box_behnken import BoxBehnkenDesign
from xpyrment.design.doe.ccd import CentralCompositeDesign
from xpyrment.design.doe.d_optimal import DOptimalDesign
from xpyrment.design.doe.dsd import DefinitiveScreeningDesign
from xpyrment.design.doe.evop import EVOPDesign
from xpyrment.design.doe.fractional_factorial import FractionalFactorialDesign
from xpyrment.design.doe.full_factorial import FullFactorialDesign
from xpyrment.design.doe.lhs import LatinHypercubeDesign
from xpyrment.design.doe.mixture import MixtureDesign
from xpyrment.design.doe.plackett_burman import PlackettBurmanDesign
from xpyrment.design.doe.switchback import SwitchbackDesign
from xpyrment.design.doe.taguchi import TaguchiDesign

__all__ = [
    "DesignMatrix",
    "FullFactorialDesign",
    "FractionalFactorialDesign",
    "PlackettBurmanDesign",
    "TaguchiDesign",
    "DefinitiveScreeningDesign",
    "CentralCompositeDesign",
    "BoxBehnkenDesign",
    "DOptimalDesign",
    "LatinHypercubeDesign",
    "MixtureDesign",
    "SwitchbackDesign",
    "EVOPDesign",
]

