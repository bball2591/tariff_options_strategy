import numpy as np
import pandas as pd

from src.tariff_strategy.data.market_data import fetch_price_history
from src.tariff_strategy.trading.target_payoff import build_price_grid
from src.tariff_strategy.trading.optimizer_inputs import build_design_matrix, cost_vector

TICKER = "SMH"
EXPIRY = "2026-02-13"  # use the one you printed in Step 7


if __name__ == "__main__":
    # S_grid
    prices = fetch_price_history(TICKER, period="6mo")
    last = prices["Adj Close"].iloc[-1]
    s0 = float(last.iloc[0]) if hasattr(last, "iloc") else float(last)

    S_grid = build_price_grid(s0=s0, grid_min=0.60, grid_max=1.40, n=250)

    # Load target payoff from Step 6
    target = pd.read_csv("data/step6_target_payoff.csv")
    y = target["target_payoff"].to_numpy(dtype=float)

    # Load cleaned options from Step 7
    calls = pd.read_csv(f"data/step7_calls_clean_{EXPIRY}.csv")
    puts = pd.read_csv(f"data/step7_puts_clean_{EXPIRY}.csv")

    X, meta = build_design_matrix(S_grid, calls, puts)
    c = cost_vector(meta)

    print(f"S_grid length: {len(S_grid)}")
    print(f"Target length: {len(y)}")
    print(f"X shape (n_grid, n_contracts): {X.shape}")
    print(f"Cost vector length: {len(c)}")
    print("\nContracts (first 10):")
    print(meta.head(10))

    # Save for Step 9
    np.save("data/step8_S_grid.npy", S_grid)
    np.save("data/step8_target_y.npy", y)
    np.save("data/step8_payoff_X.npy", X)
    meta.to_csv("data/step8_contract_meta.csv", index=False)
    np.save("data/step8_cost_c.npy", c)

    print("\nSaved Step 8 arrays into /data")
