from market_data import (
    fetch_price_history,
    list_option_expiries,
    nearest_expiry,
    fetch_options_chain,
    clean_chain,
)

TICKER = "SMH"
TARGET_DAYS = 30

if __name__ == "__main__":
    prices = fetch_price_history(TICKER, period="2y")
    print("Prices head:")
    print(prices.head(), "\n")

    expiries = list_option_expiries(TICKER)
    exp = nearest_expiry(expiries, target_days=TARGET_DAYS)
    print(f"Chosen expiry near {TARGET_DAYS}d: {exp}\n")

    chain = fetch_options_chain(TICKER, exp)
    chain = clean_chain(chain)

    print("Calls sample:")
    print(chain.calls.head(), "\n")

    print("Puts sample:")
    print(chain.puts.head(), "\n")

    # Save outputs for later steps
    prices.to_csv("data/smh_prices.csv", index=False)
    chain.calls.to_csv(f"data/smh_calls_{exp}.csv", index=False)
    chain.puts.to_csv(f"data/smh_puts_{exp}.csv", index=False)

    print("Saved: data/smh_prices.csv and options chain CSVs in /data")
