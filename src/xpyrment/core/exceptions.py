class PhaseOrderError(Exception):
    """Raised when an operation is performed in an invalid state/phase."""
    pass


class SRMError(Exception):
    """Raised when Sample Ratio Mismatch (SRM) is detected."""
    pass


class AliasError(Exception):
    """Raised when fractional factorial alias confounding is violated or misconfigured."""
    pass
