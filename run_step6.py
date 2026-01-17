import numpy as np
import matplotlib.pyplot as plt

from src.tariff_strategy.data.market_data import fetch_price_history
from src.tariff_strategy.trading.target_payoff import (
    build_price_grid,
    downside_target_payoff,
    target_dataframe,
)

TICKER = "SMH"

if __name__ == "__main__":
    prices = fetch_price_history(TICKER, period="6mo")

    # latest adjusted close (handle multi-index case)
    last = prices["Adj Close"].iloc[-1]
    s0 = float(last.iloc[0]) if hasattr(last, "iloc") else float(last)

    S_grid = build_price_grid(s0=s0, grid_min=0.60, grid_max=1.40, n=250)

    # This is your "desired payoff shape"
    payoff = downside_target_payoff(
        S=S_grid,
        s0=s0,
        floor=0.85,   # max payoff when SMH falls 15%+
        cap=1.00,     # payoff starts once SMH drops below today’s price
        scale=1.0
    )

    df = target_dataframe(S_grid, payoff)
    df.to_csv("data/step6_target_payoff.csv", index=False)

    print(f"S0 used: {s0:.2f}")
    print("Saved: data/step6_target_payoff.csv")

    plt.figure()
    plt.plot(df["S_T"], df["target_payoff"])
    plt.title("Target payoff (downside hedge) to match with options")
    plt.xlabel("SMH terminal price S_T")
    plt.ylabel("Target payoff (scaled units)")
    plt.tight_layout()
    plt.show()

