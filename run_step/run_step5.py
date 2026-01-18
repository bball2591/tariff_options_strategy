import matplotlib.pyplot as plt

from src.tariff_strategy.data.market_data import fetch_price_history
from src.tariff_strategy.modeling.scenarios import tariff_scenarios
from src.tariff_strategy.modeling.distribution import simulate_terminal_prices
from src.tariff_strategy.modeling.calibration import baseline_sigma_30d, baseline_mu_30d


TICKER = "SMH"
N_SIMS = 50_000


def latest_adj_close(prices):
    # handle possible multi-index column case
    last = prices["Adj Close"].iloc[-1]
    return float(last.iloc[0]) if hasattr(last, "iloc") else float(last)


if __name__ == "__main__":
    prices = fetch_price_history(TICKER, period="2y")
    s0 = latest_adj_close(prices)

    mu_base = baseline_mu_30d(prices, window=30, trim=0.10)
    sigma_base = baseline_sigma_30d(prices, window=30)

    print(f"Using S0: {s0:.2f}")
    print(f"Calibrated baseline mu_30 (trimmed mean): {mu_base:.4f}")
    print(f"Calibrated baseline sigma_30 (median rolling): {sigma_base:.4f}")

    scen = tariff_scenarios()
    sims = simulate_terminal_prices(
        s0=s0,
        scenarios=scen,
        mu_base=mu_base,
        sigma_base=sigma_base,
        n_sims=N_SIMS,
    )

    q = sims["S_T"].quantile([0.01, 0.05, 0.50, 0.95, 0.99])
    print("\nTerminal price quantiles (mu + sigma calibrated):")
    print(q)

    plt.figure()
    plt.hist(sims["S_T"], bins=80)
    plt.title("SMH 30-day terminal price distribution (mu and sigma calibrated)")
    plt.xlabel("S_T")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    sims.to_csv("data/smh_terminal_distribution_step5.csv", index=False)
    print("\nSaved: data/smh_terminal_distribution_step5.csv")
