"""
ShockArb Score Visualizer.

Two-panel figure:
  Top:    Confidence-delta bubble chart (circle = positive, triangle-down = negative),
          sized by |confidence_delta|, coloured by R².
  Bottom: Factor-loading heatmap (z-scored per factor), stocks on Y-axis,
          factors on X-axis.

Usage
-----
    python utils/score_viz.py
    python utils/score_viz.py --data-dir data --top 30 --out score_viz.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _find_latest_model(data_dir: str, regime: str | None) -> str | None:
    pattern = (
        os.path.join(data_dir, f"{regime}_us_*.json")
        if regime else
        os.path.join(data_dir, "*us_*.json")
    )
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_scores(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "live_alpha_us.csv")
    if not os.path.exists(path):
        logger.error(f"Score file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def load_loadings(data_dir: str, regime: str | None) -> pd.DataFrame:
    """Return a DataFrame of factor loadings indexed by ticker."""
    model_path = _find_latest_model(data_dir, regime)
    if not model_path:
        logger.error("No model .json found in data dir.")
        sys.exit(1)
    logger.info(f"Model: {model_path}")

    with open(model_path) as f:
        d = json.load(f)

    df = pd.DataFrame(d["loadings"], index=d["stock_columns"])
    df.columns = [f"Factor_{i+1}" for i in range(df.shape[1])]
    return df


# ---------------------------------------------------------------------------
# Bubble chart (top panel)
# ---------------------------------------------------------------------------

def _bubble_panel(ax: plt.Axes, scores: pd.DataFrame, top_n: int) -> None:
    """Confidence-delta bubble chart."""
    # Sort by |confidence_delta| descending, take top_n
    df = scores.copy()
    df["abs_cd"] = df["confidence_delta"].abs()
    df = df.nlargest(top_n, "abs_cd").sort_values("confidence_delta")

    pos = df[df["confidence_delta"] >= 0]
    neg = df[df["confidence_delta"] < 0]

    # Size: map |confidence_delta| to marker area 40–600
    all_abs = df["abs_cd"]
    size_min, size_max = 40, 600
    def _scale(series: pd.Series) -> pd.Series:
        span = all_abs.max() - all_abs.min()
        if span == 0:
            return pd.Series([200] * len(series), index=series.index)
        return size_min + (series / span) * (size_max - size_min)

    cmap = plt.cm.RdYlGn  # red (low R²) → green (high R²)
    vmin, vmax = 0.0, 1.0

    def _scatter(subset, marker, label):
        if subset.empty:
            return
        ax.scatter(
            subset["confidence_delta"],
            subset.index,
            s=_scale(subset["abs_cd"]),
            c=subset["r_squared"],
            cmap=cmap, vmin=vmin, vmax=vmax,
            marker=marker,
            edgecolors="white", linewidths=0.5,
            zorder=3,
            label=label,
        )

    _scatter(pos, "o", "Positive Δ")
    _scatter(neg, "v", "Negative Δ")

    ax.axvline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Confidence Δ  (delta_rel × R²)", color="white")
    ax.set_title("Signal Strength by Ticker", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = ax.get_figure().colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("R²", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Legend for marker shapes
    circle = mpatches.Patch(facecolor="grey", label="○ Positive Δ")
    triangle = mpatches.Patch(facecolor="grey", label="▼ Negative Δ")
    ax.legend(handles=[circle, triangle], loc="lower right",
              facecolor="#222", edgecolor="#555", labelcolor="white", fontsize=8)


# ---------------------------------------------------------------------------
# Factor heatmap (bottom panel)
# ---------------------------------------------------------------------------

def _heatmap_panel(ax: plt.Axes, scores: pd.DataFrame,
                   loadings: pd.DataFrame, top_n: int) -> None:
    """Z-scored factor loading heatmap for tickers in the score output."""
    # Restrict to tickers visible in bubble chart
    df_scores = scores.copy()
    df_scores["abs_cd"] = df_scores["confidence_delta"].abs()
    visible = df_scores.nlargest(top_n, "abs_cd").index

    # Intersect with tickers that have loadings
    tickers = [t for t in visible if t in loadings.index]
    if not tickers:
        ax.set_visible(False)
        return

    ldf = loadings.loc[tickers].copy()

    # Z-score each factor column independently
    z = (ldf - ldf.mean()) / ldf.std().replace(0, 1)

    # Order rows to match bubble chart (same sort: ascending confidence_delta)
    row_order = df_scores.loc[tickers].sort_values("confidence_delta").index
    z = z.loc[row_order]

    im = ax.imshow(z.values, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)

    ax.set_xticks(range(z.shape[1]))
    ax.set_xticklabels(z.columns, color="white", fontsize=9)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, color="white", fontsize=8)
    ax.set_title("Factor Loadings (z-scored per factor)", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    cbar = ax.get_figure().colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("z-score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_score_viz(data_dir: str, regime: str | None, top_n: int,
                   save_path: str | None) -> None:
    scores = load_scores(data_dir)
    loadings = load_loadings(data_dir, regime)

    n_tickers = min(top_n, len(scores))
    # Height ratio: bubble chart taller if many tickers
    bubble_h = max(4, n_tickers * 0.35)
    heat_h   = max(3, n_tickers * 0.28)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(11, bubble_h + heat_h),
        gridspec_kw={"height_ratios": [bubble_h, heat_h]},
        facecolor="#111",
    )
    ax_top.set_facecolor("#1a1a1a")
    ax_bot.set_facecolor("#1a1a1a")

    _bubble_panel(ax_top, scores, top_n)
    _heatmap_panel(ax_bot, scores, loadings, top_n)

    plt.tight_layout(pad=1.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111")
        logger.success(f"Saved: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ShockArb scores — bubble chart + factor heatmap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", "-d", default="data",
                        help="Directory containing live_alpha_us.csv and model .json")
    parser.add_argument("--regime",   "-r", default=None,
                        help="Regime prefix for model file lookup (default: sticky file)")
    parser.add_argument("--top",      "-n", type=int, default=25,
                        help="Number of tickers to display (default: 25)")
    parser.add_argument("--out",      "-o", default=None, metavar="PATH",
                        help="Save figure to PATH instead of showing interactively")
    args = parser.parse_args()

    # Honour sticky regime if --regime not supplied
    if args.regime is None:
        sticky = os.path.join(args.data_dir, ".shockarb_regime")
        if os.path.exists(sticky):
            args.regime = open(sticky).read().strip() or None

    plot_score_viz(args.data_dir, args.regime, args.top, args.out)


if __name__ == "__main__":
    main()
