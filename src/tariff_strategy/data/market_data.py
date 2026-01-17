from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionsChain:
    calls: pd.DataFrame
    puts: pd.DataFrame
    expiry: str


def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily OHLCV price history using yfinance.
    """
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}.")
    df = df.reset_index()
    return df


def list_option_expiries(ticker: str) -> List[str]:
    """
    Return available option expiries as ISO date strings.
    """
    t = yf.Ticker(ticker)
    expiries = list(t.options)
    if not expiries:
        raise ValueError(f"No option expiries found for {ticker}.")
    return expiries


def nearest_expiry(expiries: List[str], target_days: int = 30) -> str:
    """
    Choose the expiry closest to target_days from today.
    """
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()  # make tz-naive
    exp_dates = pd.to_datetime(expiries).tz_localize(None)       # make tz-naive

    diffs = (exp_dates - today).days
    diffs = pd.Series(diffs, index=range(len(expiries)))
    diffs = diffs.where(diffs >= 0)  # ignore past

    idx = (diffs - target_days).abs().idxmin()
    return expiries[int(idx)]



def fetch_options_chain(ticker: str, expiry: str) -> OptionsChain:
    """
    Fetch calls and puts for a specific expiry.
    """
    t = yf.Ticker(ticker)
    chain = t.option_chain(expiry)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    if calls.empty or puts.empty:
        raise ValueError(f"Empty options chain for {ticker} at expiry {expiry}.")

    return OptionsChain(calls=calls, puts=puts, expiry=expiry)


def mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Mid price from bid/ask, falls back to lastPrice when bid/ask missing.
    """
    bid = df.get("bid", pd.Series([pd.NA] * len(df)))
    ask = df.get("ask", pd.Series([pd.NA] * len(df)))
    mid = (bid + ask) / 2
    return mid.fillna(df.get("lastPrice"))


def clean_chain(chain: OptionsChain) -> OptionsChain:
    """
    Add mid price, basic spreads, and keep useful columns.
    """
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["mid"] = mid_price(df)
        df["spread"] = (df["ask"] - df["bid"]).fillna(pd.NA)
        keep = [
            "contractSymbol", "strike", "lastPrice", "bid", "ask", "mid", "spread",
            "volume", "openInterest", "impliedVolatility", "inTheMoney"
        ]
        return df[[c for c in keep if c in df.columns]]

    return OptionsChain(calls=_clean(chain.calls), puts=_clean(chain.puts), expiry=chain.expiry)
