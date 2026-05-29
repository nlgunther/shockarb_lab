"""
ShockArb x Value Frontier Visualizer.

Three figures + one CSV:
  1. Scatter: Conviction Score (X) vs Discount (Y).
  2. Heatmap: Factor loadings (z-scored).
  3. Heatmap: ETF beta loadings (z-scored).
  4. CSV: all value screener + ShockArb attributes, signed frontier distance.

Usage
-----
    python utils/value_score_viz.py
    python utils/value_score_viz.py --regime ukraine_shock --out data/viz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

SIZE_SCALE  = 300
SIZE_MIN    = 20
CMAP_SIGNAL = "plasma"
CMAP_HEAT   = "coolwarm"

VALUE_TICKER_MAP: dict[str, str] = {
    "Alphabet":                  "GOOGL",
    "Amazon.com":                "AMZN",
    "Meta Platforms":            "META",
    "Apple":                     "AAPL",
    "Microsoft":                 "MSFT",
    "Booking Holdings":          "BKNG",
    "MercadoLibre":              "MELI",
    "Visa":                      "V",
    "Mastercard":                "MA",
    "American Express":          "AXP",
    "Salesforce":                "CRM",
    "Adobe":                     "ADBE",
    "Autodesk":                  "ADSK",
    "PTC":                       "PTC",
    "Bentley Systems":           "BSY",
    "Guidewire":                 "GWRE",
    "Jack Henry":                "JKHY",
    "MarketAxess":               "MKTX",
    "Broadridge":                "BR",
    "Broadcom":                  "AVGO",
    "Applied Materials":         "AMAT",
    "KLA":                       "KLAC",
    "Synopsys":                  "SNPS",
    "Amphenol":                  "APH",
    "Motorola Solutions":        "MSI",
    "Fair Isaac":                "FICO",
    "Equifax":                   "EFX",
    "LPL Financial":             "LPLA",
    "Nike":                      "NKE",
    "McDonald":                  "MCD",
    "Domino":                    "DPZ",
    "McCormick":                 "MKC",
    "Molson Coors":              "TAP",
    "Brown-Forman":              "BF-B",
    "Mondelez":                  "MDLZ",
    "Bristol-Myers":             "BMY",
    "Zoetis":                    "ZTS",
    "Edwards Lifesciences":      "EW",
    "Boston Scientific":         "BSX",
    "Intuitive Surgical":        "ISRG",
    "Ecolab":                    "ECL",
    "Masco":                     "MAS",
    "Otis Worldwide":            "OTIS",
    "Allegion":                  "ALLE",
    "Eaton":                     "ETN",
    "Agilent":                   "A",
    "Arista Networks":           "ANET",
    "Airbnb":                    "ABNB",
    "BlackRock":                 "BLK",
    "Bank of America":           "BAC",
    "S&P Global":                "SPGI",
    "Moody":                     "MCO",
    "Intercontinental Exchange": "ICE",
    "General Dynamics":          "GD",
    "Lockheed Martin":           "LMT",
    "Raytheon":                  "RTX",
    "TransDigm":                 "TDG",
}

ETF_NAMES: dict[str, str] = {
    "VOO": "Vanguard S&P 500",
    "VYM": "Vanguard High Dividend Yield",
    "VEU": "Vanguard FTSE All-World ex-US",
    "VDE": "Vanguard Energy",
    "VNQ": "Vanguard Real Estate",
    "TLT": "iShares 20+ Year Treasury",
    "GLD": "SPDR Gold",
    "USO": "US Oil Fund",
    "ITA": "iShares US Aerospace & Defense",
    "HYG": "iShares High Yield Corp Bond",
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _find_latest_model(data_dir: str, regime: str | None) -> str:
    pattern = (
        os.path.join(data_dir, f"{regime}_us_*.json") if regime
        else os.path.join(data_dir, "*us_*.json")
    )
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No model .json found: {pattern}")
        sys.exit(1)
    return files[-1]


def load_value_data(txt_path: str) -> pd.DataFrame:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from value_analyzer import parse_value_report, calculate_conviction

    df = calculate_conviction(parse_value_report(txt_path)).copy()
    df["is_usd"] = df["Cur1"] == "USD"
    df["Ticker"] = df["Company"].map(
        lambda name: next(
            (v for k, v in VALUE_TICKER_MAP.items() if k.lower() in name.lower()),
            None,
        )
    )
    df.loc[~df["is_usd"], "Ticker"] = None
    logger.info(
        f"Value screener total:{len(df)}  USD:{df['is_usd'].sum()}  "
        f"mapped:{df['Ticker'].notna().sum()}"
    )
    return df.reset_index(drop=True)


def load_scores(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "live_alpha_us.csv")
    if not os.path.exists(path):
        logger.error(f"Score file not found: {path}")
        sys.exit(1)
    return pd.read_csv(path, index_col=0)


def load_model(data_dir: str, regime: str | None) -> dict:
    path = _find_latest_model(data_dir, regime)
    logger.info(f"Model: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def build_frontier(usd: pd.DataFrame,
                   x_col: str = "Conviction_Score",
                   y_col: str = "Discount") -> np.ndarray:
    """Upper convex hull trimmed to non-dominated region (starts at Y-maximum)."""
    points = sorted(zip(usd[x_col].tolist(), usd[y_col].tolist()))
    upper: list = []
    for p in points:
        while len(upper) >= 2:
            p1, p2 = upper[-2], upper[-1]
            cross = (p2[0]-p1[0])*(p[1]-p2[1]) - (p2[1]-p1[1])*(p[0]-p2[0])
            if cross >= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    hull = np.array(upper)
    return hull[int(np.argmax(hull[:, 1])):]


def _dist_to_segment(px, py, x1, y1, x2, y2) -> float:
    mag = np.hypot(x2-x1, y2-y1)
    if mag == 0:
        return np.hypot(px-x1, py-y1)
    u = max(0.0, min(1.0, ((px-x1)*(x2-x1)+(py-y1)*(y2-y1)) / mag**2))
    return np.hypot(px-(x1+u*(x2-x1)), py-(y1+u*(y2-y1)))


def _is_outside(px: float, py: float, frontier: np.ndarray) -> bool:
    """
    Outside = on the non-origin side of ANY frontier segment.

    For segment (A,B): line f(x,y)=0 oriented so f(0,0)<0.
    If f(P) > 0 for any segment, P is outside.
    """
    for i in range(len(frontier) - 1):
        x1, y1 = frontier[i]
        x2, y2 = frontier[i+1]
        f  = (y2-y1)*px - (x2-x1)*py + (x2-x1)*y1 - (y2-y1)*x1
        f0 = (y2-y1)*0  - (x2-x1)*0  + (x2-x1)*y1 - (y2-y1)*x1
        if f0 == 0:
            continue
        if (f / f0) < 0:
            return True
    return False


def _signed_dist(px: float, py: float, frontier: np.ndarray) -> float:
    """Positive = outside frontier, negative = inside."""
    unsigned = min(
        _dist_to_segment(px, py,
                         frontier[i,0], frontier[i,1],
                         frontier[i+1,0], frontier[i+1,1])
        for i in range(len(frontier)-1)
    )
    return unsigned if _is_outside(px, py, frontier) else -unsigned


def add_frontier_distances(df: pd.DataFrame, frontier: np.ndarray,
                            x_col: str = "Conviction_Score",
                            y_col: str = "Discount") -> pd.DataFrame:
    """Add _dist (signed) and _size columns to df in-place (returns copy)."""
    df = df.copy()
    df["_dist"] = df.apply(
        lambda r: _signed_dist(r[x_col], r[y_col], frontier), axis=1
    )
    abs_d = df["_dist"].abs()
    span  = abs_d.max() - abs_d.min()
    df["_size"] = (
        SIZE_MIN + SIZE_SCALE/2 if span == 0
        else SIZE_MIN + SIZE_SCALE * (1 - (abs_d - abs_d.min()) / span)
    )
    return df


# ---------------------------------------------------------------------------
# Figure 1: scatter
# ---------------------------------------------------------------------------

def fig_scatter(value: pd.DataFrame, scores: pd.DataFrame,
                frontier: np.ndarray, save_path: str | None) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 9), facecolor="#111")
    ax.set_facecolor("#111")

    x_col, y_col = "Conviction_Score", "Discount"

    value   = add_frontier_distances(value, frontier, x_col, y_col)
    usd     = value[value["is_usd"]].copy()
    foreign = value[~value["is_usd"]].copy()

    usd_sc = usd.join(
        scores[["confidence_delta"]].rename(columns={"confidence_delta": "conf_delta"}),
        on="Ticker", how="left",
    )
    in_shock = usd_sc[usd_sc["conf_delta"].notna()].copy()
    unscored = usd_sc[usd_sc["conf_delta"].isna()].copy()

    # Layer 1: foreign stocks
    ax.scatter(foreign[x_col], foreign[y_col],
               s=foreign["_size"], marker="s",
               facecolors="grey", edgecolors="grey",
               linewidths=0.5, alpha=0.5, zorder=2,
               label="Foreign (non-USD)")

    # Layer 2: USD not scored
    ax.scatter(unscored[x_col], unscored[y_col],
               s=unscored["_size"], marker="o",
               facecolors="#111", edgecolors="steelblue",
               linewidths=1.0, alpha=0.8, zorder=3,
               label="USD (not in today's score)")

    # Layer 3: ShockArb scored
    abs_cd = in_shock["conf_delta"].abs()
    vmin, vmax = abs_cd.min(), abs_cd.max()

    pos = in_shock[in_shock["conf_delta"] >= 0]
    neg = in_shock[in_shock["conf_delta"] <  0]

    sc_pos = ax.scatter(pos[x_col], pos[y_col],
                        s=pos["_size"], marker="o",
                        c=pos["conf_delta"].abs(),
                        cmap=CMAP_SIGNAL, vmin=vmin, vmax=vmax,
                        edgecolors="white", linewidths=0.4,
                        alpha=0.95, zorder=4,
                        label="Positive Conf.Delta (circle)")
    ax.scatter(neg[x_col], neg[y_col],
               s=neg["_size"], marker="v",
               c=neg["conf_delta"].abs(),
               cmap=CMAP_SIGNAL, vmin=vmin, vmax=vmax,
               edgecolors="white", linewidths=0.4,
               alpha=0.95, zorder=4,
               label="Negative Conf.Delta (triangle)")

    cbar = fig.colorbar(sc_pos, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("|Conf. Delta|", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.ax.yaxis.set_tick_params(color="white")

    # Efficient frontier line
    ax.plot(frontier[:, 0], frontier[:, 1],
            color="lime", linestyle="--", linewidth=1.8,
            marker="s", markersize=5, alpha=0.9, zorder=5,
            label="Efficient Frontier")

    # Labels
    labelled: set[str] = set()

    def _label(row, color="white"):
        name = str(row.get("Ticker") or row.get("Company", "?"))
        if name in labelled:
            return
        labelled.add(name)
        ax.annotate(name, (row[x_col], row[y_col]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=color, zorder=6)

    # All stocks sorted by signed distance descending (outside first, then nearest inside).
    # Label outside points in yellow; top-20 nearest-or-outside from full universe.
    all_sorted = value.copy()
    all_sorted["_abs"] = all_sorted["_dist"].abs()
    all_sorted = all_sorted.sort_values("_dist", ascending=False)

    outside = all_sorted[all_sorted["_dist"] > 0]
    for _, row in outside.iterrows():
        _label(row, color="yellow")

    # Top-20 by proximity (signed distance, highest = closest to or beyond frontier)
    # already sorted descending, so take first 20 not yet labelled
    count = 0
    for _, row in all_sorted.iterrows():
        if count >= 20:
            break
        name = str(row.get("Ticker") or row.get("Company", "?"))
        if name in labelled:
            count += 1
            continue
        _label(row)
        count += 1

    ax.set_xlabel(
        "Conviction Score  [(1-P/FV) x log10(MarketCap) / Uncertainty]",
        color="white",
    )
    ax.set_ylabel("Discount to Fair Value  (1 - P/FV)", color="white")
    ax.set_title("ShockArb x Wide-Moat Value Frontier",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, alpha=0.15)
    ax.legend(facecolor="#222", edgecolor="#555", labelcolor="white",
              fontsize=8, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111")
        logger.success(f"Saved scatter: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: factor heatmap
# ---------------------------------------------------------------------------

def fig_factor_heatmap(value: pd.DataFrame, model: dict,
                        save_path: str | None) -> None:
    stock_cols = model["stock_columns"]
    ldf = pd.DataFrame(model["loadings"], index=stock_cols)
    ldf.columns = [f"Factor_{i+1}" for i in range(ldf.shape[1])]

    present = [t for t in value["Ticker"].dropna() if t in ldf.index]
    if not present:
        logger.warning("No value screener tickers in model -- skipping factor heatmap.")
        return

    sub = ldf.loc[present].sort_index()
    z   = (sub - sub.mean()) / sub.std().replace(0, 1)

    fig, ax = plt.subplots(figsize=(6, max(6, len(present)*0.38)), facecolor="#111")
    sns.heatmap(z, ax=ax, cmap=CMAP_HEAT, center=0, vmin=-2.5, vmax=2.5,
                linewidths=0.4, linecolor="#333",
                annot=True, fmt=".2f", annot_kws={"size": 8, "color": "white"},
                cbar_kws={"label": "z-score", "shrink": 0.6})
    ax.set_title("Factor Loadings (z-scored)", color="white", fontsize=12, pad=8)
    ax.set_xlabel("Latent Factor", color="white")
    ax.set_ylabel("Ticker", color="white")
    ax.tick_params(colors="white")
    ax.figure.axes[-1].yaxis.label.set_color("white")
    plt.setp(ax.figure.axes[-1].yaxis.get_ticklabels(), color="white")
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111")
        logger.success(f"Saved factor heatmap: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: ETF beta heatmap
# ---------------------------------------------------------------------------

def fig_etf_heatmap(value: pd.DataFrame, model: dict,
                     save_path: str | None) -> None:
    stock_cols = model["stock_columns"]
    etf_cols   = model["etf_columns"]
    Vt         = np.array(model["Vt"])
    loadings   = np.array(model["loadings"])

    beta_df = pd.DataFrame(loadings @ Vt, index=stock_cols, columns=etf_cols)
    present = [t for t in value["Ticker"].dropna() if t in beta_df.index]
    if not present:
        logger.warning("No value screener tickers in model -- skipping ETF heatmap.")
        return

    sub = beta_df.loc[present].sort_index()
    z   = (sub - sub.mean()) / sub.std().replace(0, 1)

    fig, ax = plt.subplots(
        figsize=(max(8, len(etf_cols)*0.9), max(6, len(present)*0.38)),
        facecolor="#111",
    )
    sns.heatmap(z, ax=ax, cmap=CMAP_HEAT, center=0, vmin=-2.5, vmax=2.5,
                linewidths=0.4, linecolor="#333",
                annot=True, fmt=".2f", annot_kws={"size": 8, "color": "white"},
                cbar_kws={"label": "z-score", "shrink": 0.6})
    ax.set_title("ETF Beta Loadings (z-scored)", color="white", fontsize=12, pad=8)
    ax.set_xlabel("ETF", color="white")
    ax.set_ylabel("Ticker", color="white")
    ax.tick_params(colors="white")
    ax.figure.axes[-1].yaxis.label.set_color("white")
    plt.setp(ax.figure.axes[-1].yaxis.get_ticklabels(), color="white")
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")
    legend = "   ".join(f"{t}: {ETF_NAMES.get(t, t)}" for t in etf_cols)
    fig.text(0.5, -0.01, legend, ha="center", va="top", fontsize=7.5, color="#aaa")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#111")
        logger.success(f"Saved ETF heatmap: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _load_name_lookup(data_dir: str) -> dict[str, str]:
    """Build Ticker -> Company Name from NYSE and NASDAQ listing files."""
    lookup: dict[str, str] = {}
    for fname in ("nyse_1668526574444.csv", "nasdaq_1668526380140.csv"):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        ref = pd.read_csv(path, usecols=["Symbol", "Name"])
        for _, row in ref.iterrows():
            sym = str(row["Symbol"]).strip()
            if sym and sym not in lookup:
                lookup[sym] = str(row["Name"]).strip()
    return lookup


def export_combined_csv(value: pd.DataFrame, scores: pd.DataFrame,
                         frontier: np.ndarray, save_path: str,
                         data_dir: str = "data") -> None:
    """
    Full outer join: every value screener stock + every ShockArb stock.

    Column order
    ------------
    Name (col A) : company name from NYSE/NASDAQ listing, or value screener Company field
    Ticker       : exchange ticker
    in_value, in_shockarb : membership flags ('Yes' / '')
    ... all other value screener + ShockArb columns ...
    frontier_distance : signed distance (+ = outside, - = inside, NaN = no value data)
    """
    x_col, y_col = "Conviction_Score", "Discount"

    name_lookup = _load_name_lookup(data_dir)

    df = value.copy()
    df["frontier_distance"] = df.apply(
        lambda r: _signed_dist(r[x_col], r[y_col], frontier), axis=1
    )
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df["in_value"] = "Yes"

    scores_reset = scores.reset_index().rename(columns={"index": "Ticker"})
    scores_reset["in_shockarb"] = "Yes"

    merged = df.merge(scores_reset, on="Ticker", how="outer")
    merged["in_value"]    = merged["in_value"].fillna("")
    merged["in_shockarb"] = merged["in_shockarb"].fillna("")

    # Build Name column: prefer exchange listing, fall back to value screener Company field
    def _name(row):
        ticker = row.get("Ticker")
        if ticker and ticker in name_lookup:
            return name_lookup[ticker]
        return row.get("Company", "")

    merged.insert(0, "Name", merged.apply(_name, axis=1))

    # Sort: outside frontier first, then frontier_distance desc, then ticker alpha
    merged = merged.sort_values(
        ["frontier_distance", "Ticker"],
        ascending=[False, True],
        na_position="last",
    )

    merged.to_csv(save_path, index=False)
    n_both = (merged["in_value"].eq("Yes") & merged["in_shockarb"].eq("Yes")).sum()
    logger.success(
        f"Saved CSV: {save_path}  ({len(merged)} rows, {len(merged.columns)} cols)  "
        f"Value:{merged['in_value'].eq('Yes').sum()}  "
        f"ShockArb:{merged['in_shockarb'].eq('Yes').sum()}  overlap:{n_both}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ShockArb x Wide-Moat Value visualization suite.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--value',     '-v', default='value051826.txt',
                        help='Path to value screener text file')
    parser.add_argument('--data-dir',  '-d', default='data')
    parser.add_argument('--regime',    '-r', default=None)
    parser.add_argument('--out',       '-o', default=None, metavar='DIR')
    args = parser.parse_args()

    if args.regime is None:
        sticky = os.path.join(args.data_dir, '.shockarb_regime')
        if os.path.exists(sticky):
            args.regime = open(sticky).read().strip() or None

    def _out(name):
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            return os.path.join(args.out, name)
        return None

    value    = load_value_data(args.value)
    scores   = load_scores(args.data_dir)
    model    = load_model(args.data_dir, args.regime)
    frontier = build_frontier(value[value["is_usd"]])

    fig_scatter(value, scores, frontier, _out("value_scatter.png"))
    fig_factor_heatmap(value, model, _out("value_factor_heatmap.png"))
    fig_etf_heatmap(value, model, _out("value_etf_heatmap.png"))

    csv_path = _out("value_combined.csv") or "value_combined.csv"
    export_combined_csv(value, scores, frontier, csv_path, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
