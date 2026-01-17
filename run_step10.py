import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tariff_strategy.trading.trade_summary import (
    put_spread_metrics,
    put_spread_payoff_dollars,
)

# Use the result from Step 9
K_LONG = 385.0
K_SHORT = 350.0
PREMIUM_PAID = 577.0  # dollars

if __name__ == "__main__":
    # Load grid and also your Step 5 simulated distribution (for EV)
    S_grid = np.load("data/step8_S_grid.npy")

    # P&L across grid
    pnl = put_spread_payoff_dollars(S_grid, K_LONG, K_SHORT, PREMIUM_PAID)

    # Metrics
    m = put_spread_metrics(K_LONG, K_SHORT, PREMIUM_PAID)
    print("=== Final Trade Summary (1x Put Spread) ===")
    print(f"Long Put Strike:  {K_LONG}")
    print(f"Short Put Strike: {K_SHORT}")
    print(f"Premium Paid:     ${m['premium_$']:.2f}")
    print(f"Max Loss:         ${m['max_loss_$']:.2f}")
    print(f"Max Profit:       ${m['max_profit_$']:.2f}")
    print(f"Breakeven:        {m['breakeven']:.2f}")
    print(f"Spread Width:     ${m['width_$']:.2f}")

    # Plot payoff diagram
    plt.figure()
    plt.plot(S_grid, pnl)
    plt.axhline(0, linewidth=1)
    plt.title("SMH 30D Tariff Hedge: Debit Put Spread P&L at Expiry")
    plt.xlabel("SMH terminal price S_T")
    plt.ylabel("P&L ($)")
    plt.tight_layout()
    plt.savefig("data/step10_payoff_diagram.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved: data/step10_payoff_diagram.png")

    # Expected value under your simulated Step 5 distribution (optional but strong)
    sims5_path = "data/smh_terminal_distribution_step5.csv"
    try:
        sims = pd.read_csv(sims5_path)
        st = sims["S_T"].to_numpy(dtype=float)
        pnl_sims = put_spread_payoff_dollars(st, K_LONG, K_SHORT, PREMIUM_PAID)

        ev = pnl_sims.mean()
        p_profit = (pnl_sims > 0).mean()

        print("\n=== Expected Value under Step 5 Q distribution ===")
        print(f"EV (mean P&L): ${ev:.2f}")
        print(f"P(P&L > 0):    {p_profit:.2%}")

        # Save distribution of pnl
        out = sims.copy()
        out["pnl_$"] = pnl_sims
        out.to_csv("data/step10_pnl_simulated.csv", index=False)
        print("Saved: data/step10_pnl_simulated.csv")

    except FileNotFoundError:
        print("\nStep 5 distribution CSV not found, skipping EV calculation.")
