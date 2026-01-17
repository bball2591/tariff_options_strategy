from __future__ import annotations
import numpy as np
import pandas as pd

from .mapping import params_from_calibration


def simulate_terminal_prices(
    s0: float,
    scenarios: pd.DataFrame,
    mu_base: float,
    sigma_base: float,
    n_sims: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate 30-day terminal prices via a mixture of lognormal returns
    using calibrated baseline mu and sigma.
    """
    rng = np.random.default_rng(seed)

    probs = scenarios["p"].to_numpy()
    idx = rng.choice(len(scenarios), size=n_sims, p=probs)

    chosen = scenarios.iloc[idx].reset_index(drop=True)
    severities = chosen["severity"].to_numpy()

    r = np.empty(n_sims, dtype=float)
    for sev in np.unique(severities):
        mask = severities == sev
        params = params_from_calibration(int(sev), mu_base=mu_base, sigma_base=sigma_base)
        r[mask] = rng.normal(loc=params.mu, scale=params.sigma, size=mask.sum())

    st = s0 * np.exp(r)

    out = chosen.copy()
    out["log_return"] = r
    out["S_T"] = st
    return out
