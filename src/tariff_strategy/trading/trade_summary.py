from __future__ import annotations
import numpy as np


def put_spread_metrics(K_long: float, K_short: float, premium_paid: float) -> dict:
    """
    Metrics for a debit put spread.
    premium_paid is in dollars per spread (already *100).
    """
    width = (K_long - K_short) * 100  # dollars
    max_profit = width - premium_paid
    max_loss = premium_paid

    # breakeven for debit put spread:
    # K_long - premium_per_share
    premium_per_share = premium_paid / 100
    breakeven = K_long - premium_per_share

    return {
        "width_$": width,
        "premium_$": premium_paid,
        "max_profit_$": max_profit,
        "max_loss_$": max_loss,
        "breakeven": breakeven,
    }


def put_spread_payoff_dollars(S: np.ndarray, K_long: float, K_short: float, premium_paid: float) -> np.ndarray:
    """
    Terminal P&L in dollars for 1 spread.
    """
    long_put = np.maximum(K_long - S, 0.0) * 100
    short_put = np.maximum(K_short - S, 0.0) * 100
    payoff = long_put - short_put
    pnl = payoff - premium_paid
    return pnl
