from __future__ import annotations
import numpy as np
import pandas as pd

from .payoff import call_payoff, put_payoff


def build_design_matrix(
    S_grid: np.ndarray,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns:
    - X: payoff matrix with shape (n_grid, n_contracts)
    - meta: DataFrame with contract info aligned to columns of X
    """
    call_K = calls["strike"].to_numpy(dtype=float)
    put_K = puts["strike"].to_numpy(dtype=float)

    call_cols = []
    call_meta = []
    for i, K in enumerate(call_K):
        call_cols.append(call_payoff(S_grid, K))
        call_meta.append(
            {"type": "call", "strike": float(K), "mid": float(calls["mid"].iloc[i]), "symbol": calls["contractSymbol"].iloc[i]}
        )

    put_cols = []
    put_meta = []
    for i, K in enumerate(put_K):
        put_cols.append(put_payoff(S_grid, K))
        put_meta.append(
            {"type": "put", "strike": float(K), "mid": float(puts["mid"].iloc[i]), "symbol": puts["contractSymbol"].iloc[i]}
        )

    # X should be (n_grid, n_contracts)
    X = np.column_stack(call_cols + put_cols)
    meta = pd.DataFrame(call_meta + put_meta)
    return X, meta


def cost_vector(meta: pd.DataFrame, contract_multiplier: int = 100) -> np.ndarray:
    """
    Convert option mid prices into dollar cost per contract.
    """
    return meta["mid"].to_numpy(dtype=float) * contract_multiplier
