import pandas as pd
import numpy as np

def _format_var_reduction(cuped: bool, var_red: float) -> str:
    if not cuped:
        return "-"
    return f"{var_red:.1%}" if pd.notna(var_red) else "-"

df = pd.DataFrame({
    "cuped_applied": [True, False, np.nan, None],
    "variance_reduction": [0.2, 0.2, 0.2, 0.2]
})

old_res = [_format_var_reduction(c, v) for c, v in zip(df["cuped_applied"], df["variance_reduction"])]
print("Old logic:", old_res)

# New vectorized logic to match exactly
cuped_applied = df["cuped_applied"].astype(bool) # Wait, what does astype(bool) do to NaN?
print("astype bool:", df["cuped_applied"].astype(bool).tolist()) # True, False, True, False (None is False?) Let's check

var_red_str = df["variance_reduction"].map("{:.1%}".format, na_action="ignore").fillna("-")

# Let's write equivalent logic using pandas built-ins without apply if possible
print("Equivalent logic:", np.where(df["cuped_applied"].fillna(True).astype(bool), var_red_str, "-").tolist())
