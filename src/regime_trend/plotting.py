"""Figures - one visual system for the whole project.

Colour encodes the *variant* consistently in every chart: base signal (blue),
+HMM gate (orange), +volatility gate (aqua), HMM with look-ahead (magenta and
hatched, because it is not tradeable), long-only benchmark (grey).
Categorical hues are CVD-validated; every multi-series chart has a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GREY = "#898781"

VARIANT_COLORS = {"base": "#2a78d6", "hmm": "#eb6834", "vol": "#1baf7a",
                  "lookahead": "#e87ba4", "benchmark": GREY}
VARIANT_LABELS = {"base": "Base signal", "hmm": "+ HMM gate (real time)",
                  "vol": "+ volatility gate (control)", "lookahead": "+ HMM gate (look-ahead)",
                  "benchmark": "Long-only, equal risk"}
FAMILY_LABELS = {"tsmom": "Time-series momentum", "donchian": "Donchian breakout",
                 "kalman": "Kalman trend"}
SHADE = "#eb6834"
# signal families get their own (validated) hues in the anatomy figure, so they are
# never confused with the variant colours used everywhere else
FAMILY_COLORS = {"tsmom": "#eda100", "donchian": "#008300", "kalman": "#6250d6"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.8, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_2,
        "hatch.color": SURFACE, "hatch.linewidth": 1.2,
        "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })


def _finish(fig, path: Path, title: str, subtitle: str | None = None,
            legend: tuple[list, list[str]] | None = None) -> Path:
    H = fig.get_figheight()
    title_h, sub_h, leg_h = 0.34, 0.26, 0.34
    header = 0.12 + title_h + (sub_h if subtitle else 0) + (leg_h if legend else 0)
    fig.tight_layout(rect=(0, 0, 1, 1 - header / H))
    y = 1 - 0.08 / H
    fig.text(0.01, y, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    y -= title_h / H
    if subtitle:
        fig.text(0.01, y, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
        y -= sub_h / H
    if legend:
        handles, labels = legend
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, y + 0.02 / H),
                   ncol=len(labels), handlelength=1.4, columnspacing=1.6, borderaxespad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _pct(ax, axis="y", decimals=0):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(
        mticker.PercentFormatter(1.0, decimals=decimals))


def _line_handle(color, lw=2.0):
    return plt.Line2D([0], [0], color=color, lw=lw)


def _log_ticks(ax, lo, hi):
    nice = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]
    ax.yaxis.set_major_locator(mticker.FixedLocator([t for t in nice if lo * 0.9 <= t <= hi * 1.1]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}x"))


# --------------------------------------------------------------------------- #
def wealth_by_family(total: pd.DataFrame, families: list[str], path: Path, title: str) -> Path:
    wealth = (1.0 + total).cumprod()
    fig, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families, strict=True):
        ax.plot(wealth.index, wealth["benchmark"], color=GREY, lw=1.4, zorder=1)
        for v in ("vol", "hmm", "base"):
            ax.plot(wealth.index, wealth[f"{fam}_{v}"], color=VARIANT_COLORS[v], zorder=2)
        ax.set_yscale("log")
        ax.set_title(FAMILY_LABELS[fam])
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _log_ticks(axes[0], float(wealth.min().min()), float(wealth.max().max()))
    keys = ["base", "hmm", "vol", "benchmark"]
    return _finish(fig, path, title, "Growth of 1, net of costs, 10% ex-ante volatility target "
                   "set by the base signal; log scale",
                   legend=([_line_handle(VARIANT_COLORS[k]) for k in keys],
                           [VARIANT_LABELS[k] for k in keys]))


def delta_sharpe(paired: pd.DataFrame, families: list[str], path: Path, title: str) -> Path:
    """Forest plot of paired Sharpe differences of each overlay vs its base signal."""
    rows = []
    for fam in families:
        for v in ("hmm", "vol", "lookahead"):
            key = (f"{fam}_{v}", f"{fam}_base")
            if key in paired.index:
                rows.append((fam, v, paired.loc[key]))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.0))
    ylabels, y = [], 0
    for i, (fam, v, r) in enumerate(rows):
        y = len(rows) - 1 - i
        c = VARIANT_COLORS[v]
        ax.plot([r["CI low"], r["CI high"]], [y, y], color=c, lw=3, alpha=0.4,
                solid_capstyle="round")
        ax.scatter(r["dSharpe"], y, s=60, color=c, edgecolor=SURFACE, linewidth=1.8, zorder=3,
                   marker="D" if v == "lookahead" else "o")
        ax.annotate(f"{r['dSharpe']:+.2f}", (r["CI high"], y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
        ylabels.append((y, f"{FAMILY_LABELS[fam]} {VARIANT_LABELS[v]}"))
    ax.set_yticks([p for p, _ in ylabels], [s for _, s in ylabels], fontsize=8.5)
    ax.axvline(0, color=INK_2, lw=1)
    ax.grid(axis="y", visible=False)
    for k in range(1, len(families)):
        ax.axhline(len(rows) - 3 * k - 0.5, color=GRID, lw=1)
    ax.set_xlabel("Sharpe ratio difference vs. the same signal without a gate")
    return _finish(fig, path, title, "Paired circular block bootstrap, 95% intervals. Diamonds "
                   "use future data (not tradeable): the gap to the orange dot is the look-ahead "
                   "bias.")


def lookahead_bars(sharpe: pd.Series, families: list[str], path: Path, title: str) -> Path:
    variants = ["base", "hmm", "lookahead"]
    x = np.arange(len(families))
    w = 0.24
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for k, v in enumerate(variants):
        vals = [sharpe[f"{f}_{v}"] for f in families]
        bars = ax.bar(x + (k - 1) * w, vals, width=w * 0.92, color=VARIANT_COLORS[v],
                      hatch="///" if v == "lookahead" else None, label=VARIANT_LABELS[v])
        for b, val in zip(bars, vals, strict=True):
            ax.annotate(f"{val:.2f}", (b.get_x() + b.get_width() / 2, max(val, 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK_2)
    ax.axhline(sharpe["benchmark"], color=GREY, lw=1.2, zorder=0,
               label=f"{VARIANT_LABELS['benchmark']} ({sharpe['benchmark']:.2f})")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(x, [FAMILY_LABELS[f] for f in families])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Sharpe ratio (net)")
    handles, labels = ax.get_legend_handles_labels()
    return _finish(fig, path, title, "The same HMM, fitted once on the full sample and smoothed "
                   "with future data, versus re-estimated monthly and filtered in real time",
                   legend=(handles, labels))


def regime_map(close: pd.Series, p_filtered: pd.Series, p_smoothed: pd.Series,
               p_vol: pd.Series, path: Path, title: str, asset: str,
               switches: dict[str, float]) -> Path:
    """Price on top; below, one strip per regime classifier (filled = turbulent)."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    idx = p_filtered.dropna().index
    ax.plot(idx, close.reindex(idx), color=INK_2, lw=1.2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_title(f"{asset} price (log scale)")
    strips = [
        ("Real-time HMM (filtered)", p_filtered, VARIANT_COLORS["hmm"], switches["hmm"]),
        ("Volatility control", p_vol, VARIANT_COLORS["vol"], switches["vol"]),
        ("Hindsight HMM (smoothed)", p_smoothed, VARIANT_COLORS["lookahead"],
         switches["lookahead"]),
    ]
    for row, (_, p, color, _sw) in enumerate(strips):
        y0 = len(strips) - 1 - row
        on = (p.reindex(idx) > 0.5).values
        ax2.fill_between(idx, y0 + 0.12, y0 + 0.88, where=on, color=color, lw=0, step="mid")
    ax2.set_ylim(0, len(strips))
    ax2.set_yticks([len(strips) - 1 - r + 0.5 for r in range(len(strips))],
                   [f"{name}  ({sw:.0f} switches/yr)" for name, _, _, sw in strips], fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    ax2.set_title("Turbulent regime (filled) according to each classifier")
    share = float((p_filtered.reindex(idx) > 0.5).mean())
    return _finish(fig, path, title, f"In real time the HMM flags {share:.0%} of days as "
                   "turbulent and flips regime many times more often than the hindsight "
                   "version that a full-sample backtest silently uses")


def signal_anatomy(close: pd.Series, high: pd.Series, low: pd.Series, kalman_level: pd.Series,
                   signals: dict[str, pd.Series], p_turb: pd.Series, entry: int, window,
                   path: Path, title: str, asset: str) -> Path:
    s, e = window
    sl = slice(s, e)
    c = close.loc[sl]
    hi = high.rolling(entry).max().shift(1).loc[sl]
    lo = low.rolling(entry).min().shift(1).loc[sl]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.3]})
    turb = (p_turb.reindex(c.index) > 0.5).values
    ax.fill_between(c.index, 0, 1, where=turb, transform=ax.get_xaxis_transform(), color=SHADE,
                    alpha=0.12, lw=0, step="mid")
    ax.fill_between(c.index, lo, hi, color=GRID, alpha=0.7, lw=0)
    ax.plot(c.index, c, color=INK_2, lw=1.3)
    ax.plot(c.index, kalman_level.loc[sl], color=FAMILY_COLORS["kalman"], lw=1.8)
    ax.set_title("Price, 55-day channel (grey band) and Kalman trend level; shaded = real-time "
                 "turbulent regime")
    for fam, sig in signals.items():
        ax2.step(sig.loc[sl].index, sig.loc[sl], where="post", color=FAMILY_COLORS[fam], lw=1.6)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_yticks([-1, 0, 1], ["short", "flat", "long"])
    ax2.set_title("Signal before volatility scaling")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [_line_handle(INK_2), plt.Rectangle((0, 0), 1, 1, color=GRID)] + \
        [_line_handle(FAMILY_COLORS[f]) for f in signals]
    labels = [f"{asset} close", f"{entry}-day channel"] + \
        [FAMILY_LABELS[f] + (" (level & signal)" if f == "kalman" else "") for f in signals]
    return _finish(fig, path, title, "Three ways to read the same trend: the momentum sign, a "
                   "stateful breakout and a Kalman slope with graded conviction",
                   legend=(handles, labels))


def crisis_bars(table: pd.DataFrame, families: list[str], path: Path, title: str) -> Path:
    """Compounded return in each crisis window; one panel per crisis."""
    order = ["benchmark"] + [f"{f}_{v}" for f in families for v in ("base", "hmm")]
    labels = ["Long-only"] + [f"{FAMILY_LABELS[f].split()[0] if f != 'tsmom' else 'TSMOM'}"
                              f"{'' if v == 'base' else ' + HMM'}"
                              for f in families for v in ("base", "hmm")]
    colors = [GREY] + [VARIANT_COLORS[v] for _ in families for v in ("base", "hmm")]
    crises = list(table.index)
    ncols = 3
    nrows = int(np.ceil(len(crises) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.55 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y = np.arange(len(order))[::-1]
    for ax, crisis in zip(axes, crises, strict=False):
        vals = table.loc[crisis, order].values
        ax.barh(y, vals, color=colors, height=0.7)
        ax.axvline(0, color=INK_2, lw=1)
        lim = max(0.05, np.nanmax(np.abs(vals)) * 1.35)
        ax.set_xlim(-lim, lim)
        for yy, v in zip(y, vals, strict=True):
            ax.annotate(f"{v:+.0%}", (v, yy), xytext=(3 if v >= 0 else -3, 0),
                        textcoords="offset points", ha="left" if v >= 0 else "right",
                        va="center", fontsize=7.5, color=INK_2)
        ax.set_title(crisis, fontsize=10)
        ax.grid(axis="y", visible=False)
        _pct(ax, "x")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    for ax in axes[len(crises):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_yticks(y, labels, fontsize=8.5)
    keys = ["base", "hmm", "benchmark"]
    return _finish(fig, path, title, "Compounded total return over each stress window, net of "
                   "costs", legend=([plt.Rectangle((0, 0), 1, 1, color=VARIANT_COLORS[k])
                                     for k in keys], [VARIANT_LABELS[k] for k in keys]))


def cost_sensitivity(table: pd.DataFrame, families: list[str], path: Path, title: str,
                     base_bps: float) -> Path:
    fig, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families, strict=True):
        ax.plot(table.index, table["benchmark"], color=GREY, lw=1.4)
        for v in ("base", "hmm", "vol"):
            ax.plot(table.index, table[f"{fam}_{v}"], color=VARIANT_COLORS[v], marker="o", ms=4.5,
                    mec=SURFACE, mew=1.2)
        ax.axvline(base_bps, color=AXIS, lw=1)
        ax.set_title(FAMILY_LABELS[fam])
        ax.set_xlabel("Cost (bps per unit traded)")
    axes[0].set_ylabel("Sharpe ratio")
    keys = ["base", "hmm", "vol", "benchmark"]
    return _finish(fig, path, title, "Soft HMM gating trades continuously as probabilities move, "
                   "so its edge (if any) decays fastest with costs",
                   legend=([_line_handle(VARIANT_COLORS[k]) for k in keys],
                           [VARIANT_LABELS[k] for k in keys]))
