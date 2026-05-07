"""Evolutionary Operation (EVOP) continuous live process optimization matrices.

This module provides the `EVOPDesign` class, which constructs Evolutionary Operation (EVOP) designs.
EVOP (Box, 1957) is a continuous optimization methodology engineered specifically for full-scale, live manufacturing
or production environments, where operations must be systematically improved without disrupting output or quality.
"""

import pandas as pd
from xpyrment.design.doe.base import DesignMatrix


class EVOPDesign(DesignMatrix):
    r"""Manages Evolutionary Operation (EVOP) structures for live process improvements.

    EVOP is based on the philosophy that a live process should not only produce a product (revenue),
    but should also generate valuable information on how to improve itself. To avoid producing off-specification
    output or causing system instability, EVOP introduces extremely small, low-amplitude perturbations to factors
    around the current operating center.

    Operational Cycles and Phases:
        1. **Low-Amplitude Perturbations**: Factor levels (typically in a simple $2^2$ or $2^3$ factorial layout
           with a center point) are set extremely close to the current operating standard.
        2. **Sequential Replication (Cycles)**: Because the factor changes are small, the signal-to-noise ratio
           is very low. The same factor combinations are repeated sequentially over multiple "Cycles" ($C_1, C_2, \dots$).
           As cycles accumulate, statistical standard errors decrease ($SE \propto 1/\sqrt{N}$), and the treatment
           effects gradually emerge from the background noise.
        3. **Evolutionary Steps (Phases)**: Once an effect or interaction is shown to be statistically significant
           and beneficial, the operating center is shifted to the new optimal combination. This starts a new "Phase"
           of the study, and the perturbation pattern is repeated around the new center.

    Mathematical Representation:
        For a 2-factor EVOP with current operating settings $(X_{1, \text{opt}}, X_{2, \text{opt}})$:
        - Center Point (0): $(X_{1, \text{opt}}, X_{2, \text{opt}})$
        - Coded run (-1, -1): $(X_{1, \text{opt}} - \delta_1,  X_{2, \text{opt}} - \delta_2)$
        - Coded run (+1, -1): $(X_{1, \text{opt}} + \delta_1,  X_{2, \text{opt}} - \delta_2)$
        - Coded run (-1, +1): $(X_{1, \text{opt}} - \delta_1,  X_{2, \text{opt}} + \delta_2)$
        - Coded run (+1, +1): $(X_{1, \text{opt}} + \delta_1,  X_{2, \text{opt}} + \delta_2)$
        where $\delta_j$ is a very small, non-disruptive increment.

    Pseudocode for the Algorithm:
        ```text
        function generate_evop_matrix(center_settings, delta_increments):
            1. Form standard 2^k factorial + center run in coded space [-1, 0, +1].
            2. For each factor column j:
                 Map 0 -> center_settings[j]
                 Map -1 -> center_settings[j] - delta_increments[j]
                 Map +1 -> center_settings[j] + delta_increments[j]
            3. Structure the output DataFrame with additional columns:
                 "Cycle": tracking the sequence index of the replication.
                 "Phase": tracking the active optimization phase.
            4. Return DataFrame.
        ```
    """

    def generate(
        self,
        center_settings: dict = None,
        deltas: dict = None,
        num_cycles: int = 3,
        phase: int = 1
    ) -> pd.DataFrame:
        """Generates the Evolutionary Operation (EVOP) factor matrix.

        Applies tiny operational increments around current baseline settings, constructs
        the replicated factorial cycle matrix, and returns the result.

        Args:
            center_settings (dict): Target operating center values for each factor.
            deltas (dict): Tiny perturbation step increments for each factor.
            num_cycles (int): Number of cycles (replications) to run. Defaults to 3.
            phase (int): Studying phase identifier. Defaults to 1.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the EVOP design matrix.
        """
        import itertools
        import numpy as np

        k = len(self.factors)
        keys = list(self.factors.keys())

        # Resolve center settings and perturbation step deltas
        if center_settings is None:
            center_settings = {}
            for col in keys:
                levels = self.factors[col]
                center_settings[col] = float(np.mean(levels))

        if deltas is None:
            deltas = {}
            for col in keys:
                deltas[col] = 1.0  # Default small step

        # Standard 2^k factorial corners in coded [-1.0, 1.0] space
        corners = list(itertools.product([-1.0, 1.0], repeat=k))
        # Center point coded as 0.0
        center = [tuple([0.0] * k)]
        cycle_coded_runs = center + corners  # Run size = 2^k + 1

        rows = []
        for cycle in range(1, num_cycles + 1):
            for run in cycle_coded_runs:
                run_dict = {
                    "Cycle": cycle,
                    "Phase": phase
                }
                for idx, col in enumerate(keys):
                    coded_val = run[idx]
                    # Map coded coordinate back to low-amplitude physical space
                    physical_val = center_settings[col] + coded_val * deltas[col]
                    run_dict[col] = physical_val
                rows.append(run_dict)

        # TODO: Add Box-Hunter evolutionary optimization cycle calculation curves to automatically compute main/interaction effects and errors.
        # TODO: Integrate automated stopping rules when treatment boundary shift reaches optimum levels (Simplex Evolutionary Operation optimization).
        return pd.DataFrame(rows)
