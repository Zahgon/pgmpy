import pandas as pd


def collect_state_names(data: pd.DataFrame, variable: str) -> list:
    """Return the sorted observed states for `variable` in `data`."""
    pass


def build_state_names(data: pd.DataFrame, state_names: dict | None = None) -> dict:
    """Build a complete state-name mapping for all variables in `data`."""
    pass


def get_state_counts(
    data: pd.DataFrame,
    state_names: dict,
    variable: str,
    parents=(),
    weighted: bool = False,
    reindex: bool = True,
) -> pd.DataFrame:
    """Return counts for `variable`, optionally conditioned on `parents`."""
    pass
