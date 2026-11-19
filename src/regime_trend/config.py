"""Experiment configuration: one YAML file fully describes one study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    group: str


@dataclass(frozen=True)
class DataConfig:
    start: str
    end: str
    risk_free: dict[str, Any]  # {"ticker": "^IRX"} or {"constant": 0.02}


@dataclass(frozen=True)
class SignalConfig:
    tsmom_lookback: int = 252
    donchian_entry: int = 55
    donchian_exit: int = 20
    kalman_trend_period: int = 126  # cut-off period (days) of the trend filter
    kalman_full_position_sharpe: float = 0.5  # trend Sharpe at which the position is +/-1


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 2
    features: tuple[str, ...] = ("return",)
    train_days: int = 2520  # rolling estimation window (~10y)
    min_train_days: int = 756  # first fit after ~3y of history
    refit_every: int = 21  # monthly re-estimation
    n_iter: int = 200
    tol: float = 1e-4
    n_init: int = 5  # random restarts for the first / full-sample fits
    seed: int = 7


@dataclass(frozen=True)
class VolControlConfig:
    window: int = 21  # realised-vol window
    median_window: int = 756  # trailing median that separates calm / turbulent


@dataclass(frozen=True)
class PortfolioConfig:
    asset_vol_halflife: int = 30
    portfolio_vol_target: float = 0.10
    portfolio_vol_halflife: int = 60
    max_gross_leverage: float = 4.0
    transaction_cost_bps: float = 5.0
    rebalance_buffer: float = 0.25  # trade only if |target - position| > 25% of position
    min_history_days: int = 756


@dataclass(frozen=True)
class EvaluationConfig:
    start: str
    crises: dict[str, tuple[str, str]] = field(default_factory=dict)
    bootstrap_samples: int = 2000
    bootstrap_block: int = 21
    seed: int = 7
    cost_sensitivity_bps: tuple[float, ...] = (0, 2, 5, 10, 20)
    anatomy_asset: str = "SPY"
    anatomy_window: tuple[str, str] = ("2019-06-01", "2021-06-30")


@dataclass(frozen=True)
class Config:
    name: str
    title: str
    description: str
    assets: tuple[Asset, ...]
    data: DataConfig
    signals: SignalConfig
    hmm: HMMConfig
    vol_control: VolControlConfig
    portfolio: PortfolioConfig
    evaluation: EvaluationConfig
    source_path: Path | None = None

    @property
    def tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def groups(self) -> dict[str, str]:
        return {a.ticker: a.group for a in self.assets}


def _build(cls, raw: dict[str, Any] | None):
    raw = dict(raw or {})
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    for k, v in raw.items():
        if isinstance(v, list):
            raw[k] = tuple(v)
        elif isinstance(v, dict) and k == "crises":
            raw[k] = {name: tuple(window) for name, window in v.items()}
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    assets = tuple(Asset(t, s["name"], s["group"]) for t, s in raw["universe"].items())
    regime = raw.get("regime", {})
    return Config(
        name=raw["name"],
        title=raw["title"],
        description=raw.get("description", ""),
        assets=assets,
        data=DataConfig(**raw["data"]),
        signals=_build(SignalConfig, raw.get("signals")),
        hmm=_build(HMMConfig, regime.get("hmm")),
        vol_control=_build(VolControlConfig, regime.get("vol_control")),
        portfolio=_build(PortfolioConfig, raw.get("portfolio")),
        evaluation=_build(EvaluationConfig, raw["evaluation"]),
        source_path=path,
    )
