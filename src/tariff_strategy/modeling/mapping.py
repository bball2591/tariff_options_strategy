from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioParams:
    mu: float     # expected log-return over 30 days
    sigma: float  # volatility of log-return over 30 days


def params_from_calibration(severity: int, mu_base: float, sigma_base: float) -> ScenarioParams:
    """
    Scenario parameters using calibrated baseline mu and sigma.

    - sigma increases with severity (uncertainty / event risk)
    - mu is baseline drift plus a severity penalty (tariffs are expected to hurt)
    """
    sigma_mult = {0: 1.00, 1: 1.20, 2: 1.60, 3: 2.10}

    # drift penalties in log-return space over 30 days
    mu_penalty = {0: 0.00, 1: -0.01, 2: -0.04, 3: -0.07}

    if severity not in sigma_mult:
        raise ValueError(f"Unknown severity: {severity}")

    mu = mu_base + mu_penalty[severity]
    sigma = sigma_base * sigma_mult[severity]

    return ScenarioParams(mu=mu, sigma=sigma)

