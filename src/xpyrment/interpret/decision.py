def generate_launch_recommendation(p_value: float, relative_lift: float, cost_threshold: float = 0.0) -> str:
    """Generates automated ship, no-ship, or inconclusive launch recommendations."""
    if p_value < 0.05:
        if relative_lift > cost_threshold:
            return "SHIP: Lifts are statistically significant and exceed deployment costs."
        else:
            return "NO-SHIP: Statistically significant but falls below economic margins."
    return "INCONCLUSIVE: No statistical evidence to assert a positive lift."
