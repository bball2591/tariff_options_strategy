import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.tariff_strategy.trading.put_spread_search import search_best_put_spread
from src.tariff_strategy.trading.payoff import put_payoff

# Load Step 8 outputs
S_grid = np.load("data/step8_S_grid.npy")
target = np.load("data/step8_target_y.npy")
meta = pd.read_csv("data/step8_contract_meta.csv")

# Restrict to puts only
puts = meta[meta["type"] == "put"].reset_index(drop=True)

results = search_best_put_spread(
    S_grid=S_grid,
    puts=puts,
    target=target,
)

best = results.iloc[0]

print("Best put spread:")
print(best)

# Plot comparison
K_long = best["K_long"]
K_short = best["K_short"]

width = K_long - K_short
spread_payoff = (put_payoff(S_grid, K_long) - put_payoff(S_grid, K_short)) / width


plt.figure()
plt.plot(S_grid, target, label="Target payoff", linewidth=2)
plt.plot(S_grid, spread_payoff, label=f"Put spread {int(K_long)} / {int(K_short)}", linestyle="--")
plt.legend()
plt.xlabel("SMH terminal price")
plt.ylabel("Payoff")
plt.title("Target payoff vs optimal put spread")
plt.tight_layout()
plt.show()
