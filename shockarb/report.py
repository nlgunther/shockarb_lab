"""
Presentation layer — all terminal formatting and display logic.

Keeps display concerns out of the engine and pipeline, maintaining a
clean separation of computation from presentation.

Functions
---------
  print_scores       Pretty-print a score DataFrame from model.score().
  print_model_state  Render a saved JSON model file as a structural report.
  print_live_alpha   Actionable alpha sheet for intraday execution scanning.
"""

from __future__ import annotations

import json

import pandas as pd
from loguru import logger


# =============================================================================
# Score display (used by CLI and score_live.py)
# =============================================================================

def print_scores(
    scores: pd.DataFrame,
    title: str,
    top_n: int = 20,
    min_confidence: float = 0.001,
    min_r_squared: float = 0.3,
) -> None:
    """
    Pretty-print a score DataFrame from FactorModel.score().

    Parameters
    ----------
    scores : DataFrame
        Output of model.score() — must contain confidence_delta, r_squared,
        actual_return, expected_rel, delta_rel columns.
    title : str
        Header label, e.g. "US | 2022-03-01" or "GLOBAL | LIVE".
    top_n : int
        Maximum number of actionable signals to display.
    min_confidence : float
        Minimum confidence_delta to be considered actionable. Default 0.1%.
    min_r_squared : float
        Minimum R² to be considered actionable. Default 0.30.
    """
    print(f"\n{'='*90}")
    print(f"  ⚡ SHOCKARB SCORES: {title}")
    print(f"{'='*90}")

    actionable = scores[
        (scores["confidence_delta"] > min_confidence) &
        (scores["r_squared"] > min_r_squared)
    ].head(top_n)

    if actionable.empty:
        print(
            f"\n  No actionable signals "
            f"(confidence_delta > {min_confidence:.1%} & R² > {min_r_squared:.2f})"
        )
    else:
        print(f"\n  Top {len(actionable)} actionable signals:\n")
        print(
            f"  {'Ticker':<8} {'Actual':>10} {'Exp.Rel':>10} "
            f"{'Delta':>10} {'R²':>8} {'Conf.Δ':>10}"
        )
        print("  " + "─" * 62)
        for ticker, row in actionable.iterrows():
            print(
                f"  {ticker:<8} "
                f"{row['actual_return']:>+9.2%} "
                f"{row['expected_rel']:>+9.2%} "
                f"{row['delta_rel']:>+9.2%} "
                f"{row['r_squared']:>7.2f} "
                f"{row['confidence_delta']:>+9.2%}"
            )

    # Bottom signals (outperformed factors → potential shorts or avoids)
    worst = scores.nsmallest(5, "confidence_delta")
    if not worst.empty and worst["confidence_delta"].iloc[0] < -min_confidence:
        print(f"\n  ⚠️  Bottom 5 (outperformed factors — avoid or consider short):\n")
        for ticker, row in worst.iterrows():
            print(
                f"  {ticker:<8} "
                f"{row['actual_return']:>+9.2%} "
                f"{row['expected_rel']:>+9.2%} "
                f"{row['delta_rel']:>+9.2%}"
            )

    print(f"\n{'='*90}\n")


# =============================================================================
# Model state display (used by CLI 'show' command)
# =============================================================================

def print_model_state(json_path: str) -> None:
    """
    Parse a saved JSON model file and render a human-readable structural report.

    Parameters
    ----------
    json_path : str
        Path to a .json file produced by pipeline.save_model().
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load model at {json_path}: {exc}")
        return

    meta = data["metadata"]
    name = meta.get("name", "UNKNOWN")
    n_factors = meta["n_factors"]

    print(f"\n{'='*80}")
    print(f"SHOCKARB MODEL REPORT: {name.upper()}")
    print(f"{'='*80}")
    print(f"  Created:             {meta.get('created_at', 'unknown')}")
    print(f"  Calibration window:  {meta['n_observations']} trading days")
    print(f"  Factors extracted:   {n_factors}")
    print(f"  Cumulative variance: {meta['cumulative_variance']:.2%}")

    print("\n  Variance by factor:")
    for i, var in enumerate(data["explained_variance_ratio"]):
        print(f"    Factor {i+1}: {var:.2%}")

    factor_cols = [f"Factor_{i+1}" for i in range(n_factors)]

    print(f"\n{'─'*80}")
    print("  ETF MACRO BASIS VECTORS (alphabetical)")
    print(f"{'─'*80}")
    basis = pd.DataFrame(
        pd.DataFrame(data["Vt"]).values.T,
        index=data["etf_columns"],
        columns=factor_cols,
    ).sort_index()
    print(basis.map(lambda x: f"{x:.4f}").to_string())

    print(f"\n{'─'*80}")
    print("  INDIVIDUAL STOCK FACTOR LOADINGS (sorted by Factor 3 if present)")
    print(f"{'─'*80}")
    loadings = pd.DataFrame(
        data["loadings"],
        index=data["stock_columns"],
        columns=factor_cols,
    )
    if "stock_r_squared" in data:
        loadings["R_Squared"] = pd.Series(data["stock_r_squared"]).values
    if "Factor_3" in loadings.columns:
        loadings = loadings.sort_values("Factor_3", ascending=False)
    print(loadings.map(lambda x: f"{x:.4f}").to_string())
    print()


# =============================================================================
# Live alpha display (used by score_live.py)
# =============================================================================

def print_live_alpha(
    score_df: pd.DataFrame,
    model_name: str = "UNKNOWN",
    min_delta: float = 0.005,
) -> None:
    """
    Render the mathematical output of FactorModel.score() as an actionable
    intraday trading sheet.

    Parameters
    ----------
    score_df : DataFrame
        Output of model.score().
    model_name : str
        Label for the header line.
    min_delta : float
        Minimum delta_rel threshold to display a signal. Default 0.5%.
    """
    print(f"\n{'='*110}")
    print(f"⚡ LIVE SHOCKARB DELTA ({model_name.upper()} TARGETS) ⚡")
    print(f"{'='*110}")

    actionable = score_df[score_df["delta_rel"] > min_delta].copy()

    if actionable.empty:
        print(
            f"  Tape is efficient. "
            f"No targets currently exceed the {min_delta:.1%} relative alpha threshold."
        )
    else:
        # Format numeric columns for display
        fmt = {
            "actual_return": "{:.2%}",
            "expected_rel":  "{:.2%}",
            "expected_abs":  "{:.2%}",
            "delta_rel":     "{:.2%}",
            "delta_abs":     "{:.2%}",
            "r_squared":     "{:.3f}",
            "residual_vol":  "{:.2%}",
            "confidence_delta": "{:.2%}",
        }
        display = actionable.copy()
        for col, f in fmt.items():
            if col in display.columns:
                display[col] = display[col].map(lambda x, _f=f: _f.format(x))
        print(display.to_string())

    print(f"{'='*110}\n")
