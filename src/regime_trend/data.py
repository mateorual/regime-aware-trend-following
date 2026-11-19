"""Market data: OHLC prices and the risk-free rate, cached locally as parquet.

Assets keep their own inception dates (an *unbalanced* panel): an ETF launched
in 2007 simply has NaN before that and joins the portfolio once it has enough
history. Truncating everything to the youngest asset would throw away the
2000-2003 and 2008 bear markets, the periods a regime filter is supposed to help in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

FIELDS = ("Close", "High", "Low")
MAX_FFILL_DAYS = 5
STALE_RUN_DAYS = 5
EXTREME_DAILY_MOVE = 0.40


@dataclass
class MarketData:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    returns: pd.DataFrame  # simple daily returns (NaN before inception)
    risk_free: pd.Series  # daily simple risk-free return

    @property
    def excess_returns(self) -> pd.DataFrame:
        return self.returns.sub(self.risk_free, axis=0)

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log1p(self.returns)


def _download(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      threads=True)
    out = {}
    for f in FIELDS:
        frame = raw[f] if isinstance(raw.columns, pd.MultiIndex) else raw[[f]]
        if isinstance(frame, pd.Series) or frame.shape[1] == 1:
            frame = pd.DataFrame(np.asarray(frame).reshape(-1, 1), index=raw.index,
                                 columns=tickers[:1])
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        missing = [t for t in tickers if t not in frame.columns or frame[t].dropna().empty]
        if missing:
            raise RuntimeError(f"No {f} data returned for {missing}")
        out[f] = frame[tickers]
    return out


def mask_stale_prices(prices: pd.DataFrame, min_run: int = STALE_RUN_DAYS) -> pd.DataFrame:
    """Runs of >= ``min_run`` identical closes are vendor back-fills, not trading."""
    out = prices.copy()
    for col in out:
        valid = out[col].dropna()
        run_len = valid.groupby((valid.diff() != 0).cumsum()).transform("size")
        stale = run_len[run_len >= min_run].index
        if len(stale):
            log.warning("%s: %d stale prices masked (%s -> %s)", col, len(stale),
                        stale[0].date(), stale[-1].date())
            out.loc[stale, col] = np.nan
    return out


def _fill_after_inception(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short gaps, but never before an asset's first valid price."""
    filled = frame.ffill(limit=MAX_FFILL_DAYS)
    started = frame.notna().cummax()
    return filled.where(started)


def load_market_data(cfg: Config, data_root: str | Path = "data", refresh: bool = False
                     ) -> MarketData:
    cache = Path(data_root) / cfg.name
    cache.mkdir(parents=True, exist_ok=True)
    paths = {f: cache / f"{f.lower()}.parquet" for f in FIELDS}
    if refresh or not all(p.exists() for p in paths.values()):
        log.info("Downloading OHLC for %d tickers", len(cfg.tickers))
        for f, frame in _download(cfg.tickers, cfg.data.start, cfg.data.end).items():
            frame.to_parquet(paths[f])
    frames = {f: pd.read_parquet(p)[cfg.tickers] for f, p in paths.items()}

    close = mask_stale_prices(frames["Close"].sort_index()).dropna(how="all")
    close = _fill_after_inception(close)
    high = _fill_after_inception(frames["High"].reindex(close.index)).where(close.notna())
    low = _fill_after_inception(frames["Low"].reindex(close.index)).where(close.notna())
    returns = close.pct_change(fill_method=None)

    extreme = returns.abs().stack()
    for (date, ticker), move in extreme[extreme > EXTREME_DAILY_MOVE].items():
        log.warning("Large daily move %s %s: %.1f%%", ticker, date.date(), 100 * move)

    rf = _risk_free(cfg, close.index, cache, refresh)
    log.info("Data: %d assets, %s -> %s", close.shape[1], close.index[0].date(),
             close.index[-1].date())
    return MarketData(close=close, high=high, low=low, returns=returns, risk_free=rf)


def _risk_free(cfg: Config, index: pd.DatetimeIndex, cache: Path, refresh: bool) -> pd.Series:
    spec = cfg.data.risk_free
    if "constant" in spec:
        annual = pd.Series(float(spec["constant"]), index=index)
    else:
        path = cache / "risk_free.parquet"
        if refresh or not path.exists():
            raw = _download([spec["ticker"]], cfg.data.start, cfg.data.end)["Close"].iloc[:, 0]
            raw.to_frame("yield_pct").to_parquet(path)
        raw = pd.read_parquet(path).iloc[:, 0]
        annual = raw.reindex(index).ffill().bfill() / 100.0
    return ((1.0 + annual) ** (1 / 252) - 1.0).rename("risk_free")
