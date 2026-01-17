# Tariff Shock Options Strategy (SMH) — Scenario-Weighted Hedge + Options Optimisation

This project builds a data-driven derivatives strategy that models how US-imposed tariffs could impact the semiconductor sector and translates that uncertainty into an actionable 30-day options hedge on SMH (VanEck Semiconductor ETF).

The workflow combines:
- Scenario modelling (tariff severity outcomes with probabilities)
- Robust calibration of 30-day drift and volatility from historical SMH returns
- Monte Carlo simulation of a scenario-weighted terminal price distribution (Q distribution)
- Payoff-shape optimisation to select a listed options structure that best matches a desired hedge payoff

---

## Final Trade (Output)

**Instrument:** SMH  
**Horizon:** ~30 days  
**Chosen Expiry:** `2026-02-13`  
**Optimal Hedge Structure:** Debit Put Spread

- Long 385 Put  
- Short 350 Put  
- Premium Paid: **$577**  
- Max Loss: **$577**  
- Max Profit: **$2,923**  
- Breakeven: **379.23**

**Under the scenario-weighted simulated distribution:**
- EV (mean P&L): **+$87.28**
- P(P&L > 0): **25.17%**

This is a convex hedge profile: frequent small losses (premium) with tail protection.

---

## Outputs and Figures

Generated outputs are saved in `data/`.

Key figures:
- `data/q_distribution.png`
- `data/target_payoff.png`
- `data/pnl_distribution.png`
- `data/step10_payoff_diagram.png`

---

## Project Structure

```text
tariff-options-strategy/
  src/
    tariff_strategy/
      data/
      modeling/
      trading/
  notebooks/
  data/
  report/

Method Overview

1) Scenario model (tariff severity)

A discrete set of tariff outcomes is defined (severity 0–3), each with probability p_j. Each scenario implies different short-horizon return parameters.

2) Calibrate baseline 30-day return parameters

Using historical SMH data:
- baseline 30-day drift mu_30
- baseline 30-day volatility sigma_30

3) Scenario-weighted Monte Carlo simulation

Terminal prices are simulated via:

S_T = S_0 * exp(R), where R ~ Normal(mu_j, sigma_j^2)

with scenario j sampled from a categorical distribution.

4) Define a target downside hedge payoff curve

A piecewise linear payoff target is constructed:
- 0 above S_0
- increasing as price falls below S_0
- capped after a chosen downside threshold

5) Optimise a tradable options structure

A brute-force search over put spreads selects strikes (K_long, K_short) minimising mean squared error between:
- scaled put spread payoff
- target payoff curve


How to Run

1) Create and activate a virtual environment

powershell

python -m venv .venv
.\.venv\Scripts\Activate.ps1

2) Install dependencies

powershell

pip install -r requirements.txt

If you use notebooks:

powershell

pip install ipykernel

3) Run the pipeline scripts

powershell

python run_step5.py
python run_step6.py
python run_step7.py
python run_step8.py
python run_step9.py
python run_step10.py

Optional: generate report figures

powershell

python make_report_figures.py


Notes / Limitations
- Scenario probabilities are subjective and could be updated dynamically (news-based or Bayesian updating).
- Mid prices are used as execution cost proxies (bid-ask spread and slippage not fully modelled).
- American exercise is ignored (payoff at maturity is correct, but early exercise is not explicitly handled).
- This is a research/portfolio project and not investment advice.


What I Learned
- Translating macro policy uncertainty into a structured probabilistic model
- Robust calibration of short-horizon return parameters
- Monte Carlo simulation for scenario-weighted terminal distributions
- Payoff engineering and optimisation for listed options structures
- Building a clean, reproducible research pipeline in Python

