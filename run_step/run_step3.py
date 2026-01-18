import pandas as pd
import matplotlib.pyplot as plt

from src.tariff_strategy.data.market_data import fetch_price_history
from src.tariff_strategy.modeling.scenarios import tariff_scenarios
from src.tariff_strategy.modeling.distribution import simulate_terminal_prices



TICKER = "SMH"
N_SIMS = 50_000


if __name__ == "__main__":
    prices = fetch_price_history(TICKER, period="6mo")
    s0 = float(prices["Adj Close"].iloc[-1].iloc[0]) if hasattr(prices["Adj Close"].iloc[-1], "iloc") else float(prices["Adj Close"].iloc[-1])
    print(f"Using S0 (latest Adj Close): {s0:.2f}")

    scen = tariff_scenarios()
    print("\nScenarios:")
    print(scen)

    sims = simulate_terminal_prices(s0=s0, scenarios=scen, n_sims=N_SIMS)

    # Summary stats
    q = sims["S_T"].quantile([0.01, 0.05, 0.50, 0.95, 0.99])
    print("\nTerminal price quantiles:")
    print(q)

    # Plot histogram
    plt.figure()
    plt.hist(sims["S_T"], bins=80)
    plt.title("SMH 30-day scenario-weighted terminal price distribution")
    plt.xlabel("S_T")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Save
    sims.to_csv("data/smh_terminal_distribution_step3.csv", index=False)
    print("\nSaved: data/smh_terminal_distribution_step3.csv")
