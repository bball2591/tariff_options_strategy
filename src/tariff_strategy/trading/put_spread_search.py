from __future__ import annotations
import numpy as np
import pandas as pd

from .payoff import put_payoff


def evaluate_put_spread(
    S_grid: np.ndarray,
    K_long: float,
    K_short: float,
    target: np.ndarray,
) -> float:
    """
    Compute squared error between SCALED put spread payoff and target payoff.
    Scaling makes max payoff = 1 so it's comparable to the target curve.
    """
    width = K_long - K_short
    if width <= 0:
        return np.inf

    payoff = put_payoff(S_grid, K_long) - put_payoff(S_grid, K_short)
    payoff_scaled = payoff / width

    error = np.mean((payoff_scaled - target) ** 2)
    return error



def search_best_put_spread(
    S_grid: np.ndarray,
    puts: pd.DataFrame,
    target: np.ndarray,
) -> pd.DataFrame:
    """
    Brute-force search over all valid put spreads (K_long > K_short).
    """
    results = []

    strikes = puts["strike"].to_numpy(dtype=float)
    mids = puts["mid"].to_numpy(dtype=float)

    for i, K_long in enumerate(strikes):
        for j, K_short in enumerate(strikes):
            if K_long <= K_short:
                continue

            cost = mids[i] - mids[j]
            error = evaluate_put_spread(S_grid, K_long, K_short, target)

            results.append({
                "K_long": K_long,
                "K_short": K_short,
                "cost": cost * 100,  # dollar cost
                "error": error,
            })

    return pd.DataFrame(results).sort_values("error").reset_index(drop=True)
