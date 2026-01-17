import numpy as np
import pandas as pd

from src.tariff_strategy.data.market_data import (
    fetch_price_history,
    list_option_expiries,
    nearest_expiry,
    fetch_options_chain,
    clean_chain,
)
from src.tariff_strategy.trading.target_payoff import build_price_grid
from src.tariff_strategy.trading.payoff import call_payoff, put_payoff
from src.tariff_strategy.trading.options_universe import filter_liquid_options

TICKER = "SMH"
TARGET_DAYS = 30

if __name__ == "__main__":
    # Get S0 and price grid
    prices = fetch_price_history(TICKER, period="6mo")
    last = prices["Adj Close"].iloc[-1]
    s0 = float(last.iloc[0]) if hasattr(last, "iloc") else float(last)

    S_grid = build_price_grid(s0=s0, grid_min=0.60, grid_max=1.40, n=250)

    # Get expiry and options chain
    expiries = list_option_expiries(TICKER)
    exp = nearest_expiry(expiries, target_days=TARGET_DAYS)
    chain = clean_chain(fetch_options_chain(TICKER, exp))

    calls = filter_liquid_options(chain.calls, max_spread=1.00, min_oi=50)
    puts = filter_liquid_options(chain.puts, max_spread=1.00, min_oi=50)

    print(f"S0: {s0:.2f}")
    print(f"Chosen expiry: {exp}")
    print(f"Calls after filters: {len(calls)}")
    print(f"Puts after filters: {len(puts)}")

    # Build payoff matrices
    # Each row = one option contract payoff across S_grid
    call_payoffs = np.vstack([call_payoff(S_grid, K) for K in calls["strike"].to_numpy()])
    put_payoffs = np.vstack([put_payoff(S_grid, K) for K in puts["strike"].to_numpy()])

    print(f"Call payoff matrix shape: {call_payoffs.shape}")
    print(f"Put payoff matrix shape: {put_payoffs.shape}")

    # Save cleaned options for next step
    calls.to_csv(f"data/step7_calls_clean_{exp}.csv", index=False)
    puts.to_csv(f"data/step7_puts_clean_{exp}.csv", index=False)
    print("Saved cleaned options CSVs in /data")
