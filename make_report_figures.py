import pandas as pd
import matplotlib.pyplot as plt

# 1) Q distribution (terminal prices)
sims = pd.read_csv("data/smh_terminal_distribution_step5.csv")

plt.figure()
plt.hist(sims["S_T"], bins=80)
plt.title("SMH 30-day scenario-weighted terminal price distribution (Q)")
plt.xlabel("SMH terminal price $S_T$")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/q_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved: data/q_distribution.png")


# 2) Target payoff curve
target = pd.read_csv("data/step6_target_payoff.csv")

plt.figure()
plt.plot(target["S_T"], target["target_payoff"], linewidth=2)
plt.title("Target payoff curve (downside hedge)")
plt.xlabel("SMH terminal price $S_T$")
plt.ylabel("Target payoff (scaled)")
plt.tight_layout()
plt.savefig("data/target_payoff.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved: data/target_payoff.png")


# 3) Simulated P&L distribution
pnl = pd.read_csv("data/step10_pnl_simulated.csv")

plt.figure()
plt.hist(pnl["pnl_$"], bins=80)
plt.title("Simulated P&L distribution under scenario-weighted Q distribution")
plt.xlabel("P&L ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("data/pnl_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved: data/pnl_distribution.png")
