from __future__ import annotations
import pandas as pd


def filter_liquid_options(df: pd.DataFrame, max_spread: float = 1.00, min_oi: int = 50) -> pd.DataFrame:
    """
    Simple liquidity filters for realism.
    """
    out = df.copy()

    # remove missing mid prices
    out = out.dropna(subset=["mid", "strike"])

    # spread filter (if spread exists)
    if "spread" in out.columns:
        out = out[(out["spread"].isna()) | (out["spread"] <= max_spread)]

    # open interest filter (if exists)
    if "openInterest" in out.columns:
        out = out[(out["openInterest"].isna()) | (out["openInterest"] >= min_oi)]

    return out.reset_index(drop=True)
