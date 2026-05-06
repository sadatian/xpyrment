from typing import Literal, TypedDict, Union

MetricType = Literal["mean", "proportion", "ratio", "revenue"]


class MetricResult(TypedDict):
    metric_name: str
    metric_type: str
    control_mean: float
    treatment_mean: float
    control_var: float
    treatment_var: float
    control_n: int
    treatment_n: int
    absolute_difference: float
    relative_lift: float
    cuped_applied: bool
    variance_reduction: float
    p_value: float
    ci_lower: float
    ci_upper: float
    rel_ci_lower: float
    rel_ci_upper: float
    power: float
