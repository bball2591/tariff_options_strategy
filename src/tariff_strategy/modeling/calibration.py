from __future__ import annotations
import numpy as np
import pandas as pd


def compute_daily_log_returns(prices: pd.DataFrame) -> pd.Series:
    """
    Expects a DataFrame that includes an 'Adj Close' column (from yfinance download).
    Returns daily log returns.
    """
    # yfinance sometimes returns multi-index columns; handle both cases
    if isinstance(prices.columns, pd.MultiIndex):
        adj = prices["Adj Close"].iloc[:, 0].astype(float)
    else:
        adj = prices["Adj Close"].astype(float)

    r = np.log(adj).diff().dropna()
    r.name = "log_return"
    return r


def rolling_sigma_30d(log_returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling volatility of 30 trading days, expressed as sigma over the 30-day horizon:
    sigma_30 = std(daily_returns) * sqrt(30)
    """
    daily_std = log_returns.rolling(window).std()
    sigma_30 = daily_std * np.sqrt(window)
    sigma_30.name = "sigma_30"
    return sigma_30.dropna()


def baseline_sigma_30d(prices: pd.DataFrame, window: int = 30) -> float:
    """
    Robust baseline: median rolling sigma_30 over the sample.
    """
    r = compute_daily_log_returns(prices)
    sig = rolling_sigma_30d(r, window=window)
    return float(sig.median())


def rolling_mu_30d(log_returns: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling 30-trading-day log return (sum of daily log returns).
    """
    mu_30 = log_returns.rolling(window).sum()
    mu_30.name = "mu_30"
    return mu_30.dropna()


def baseline_mu_30d(prices: pd.DataFrame, window: int = 30, trim: float = 0.10) -> float:
    """
    Robust baseline drift estimate:
    - compute rolling 30d log returns
    - take a trimmed mean (drop top/bottom trim proportion)

    trim=0.10 means drop 10% lowest and 10% highest to reduce outlier influence.
    """
    r = compute_daily_log_returns(prices)
    mu_series = rolling_mu_30d(r, window=window)

    mu_sorted = mu_series.sort_values()
    n = len(mu_sorted)
    k = int(n * trim)

    if n < 50:
        # small sample fallback
        return float(mu_series.median())

    trimmed = mu_sorted.iloc[k : n - k]
    return float(trimmed.mean())
