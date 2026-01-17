from __future__ import annotations
import numpy as np
import pandas as pd


def build_price_grid(s0: float, grid_min: float = 0.60, grid_max: float = 1.40, n: int = 250) -> np.ndarray:
    """
    Price grid for terminal prices S_T.
    grid_min/grid_max are multipliers of S0.
    """
    return np.linspace(grid_min * s0, grid_max * s0, n)


def downside_target_payoff(
    S: np.ndarray,
    s0: float,
    floor: float = 0.85,
    cap: float = 1.00,
    scale: float = 1.0,
) -> np.ndarray:
    """
    A simple downside hedge target payoff:

    - payoff = 0 when S >= cap*s0
    - payoff increases linearly as S falls below cap*s0
    - payoff reaches maximum at floor*s0 (then stays flat)

    floor and cap are multipliers of S0.
    """
    floor_price = floor * s0
    cap_price = cap * s0

    # linear ramp from cap_price down to floor_price
    payoff = (cap_price - S) / (cap_price - floor_price)
    payoff = np.clip(payoff, 0.0, 1.0)

    return scale * payoff


def target_dataframe(S: np.ndarray, payoff: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"S_T": S, "target_payoff": payoff})
    return df
