from __future__ import annotations
import pandas as pd


def tariff_scenarios() -> pd.DataFrame:
    """
    Simple discrete scenario set for a 30-day horizon.

    severity is an index you can later connect to headlines or a tariff score.
    """
    scenarios = [
        {"scenario": "No new measures", "p": 0.45, "severity": 0},
        {"scenario": "Mild tariffs / tighter language", "p": 0.30, "severity": 1},
        {"scenario": "Aggressive tariffs / restrictions", "p": 0.20, "severity": 2},
        {"scenario": "Aggressive + retaliation", "p": 0.05, "severity": 3},
    ]
    df = pd.DataFrame(scenarios)
    if abs(df["p"].sum() - 1.0) > 1e-9:
        raise ValueError("Scenario probabilities must sum to 1.")
    return df
