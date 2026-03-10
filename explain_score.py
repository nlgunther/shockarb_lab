"""
explain_score.py  —  decompose the Exp. Rel. score for a given stock.

Usage (from repo root):
    python explain_score.py NOW
    python explain_score.py NOW CRM INTU
    python explain_score.py NOW --top-etfs 8

No redundant downloads — fetch_live_returns now goes through the datamgr
coordinator, so any data already cached is never re-downloaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from shockarb.config import ExecutionConfig, US_UNIVERSE
from shockarb.pipeline import fetch_live_returns, find_latest_model, load_model


def explain(model, today_etf_returns: pd.Series, tickers: List[str], top_etfs: int = 5) -> None:
    r_e       = today_etf_returns.reindex(model._etf_mean.index).fillna(0) - model._etf_mean
    f_today   = model._Vt @ r_e.values
    k         = len(f_today)
    etf_names = model._etf_mean.index.tolist()

    print("\n" + "=" * 72)
    print("FACTOR SCORES TODAY")
    print("=" * 72)
    print(f"  {'Factor':<12}  {'Score':>8}  Direction")
    print("  " + "-" * 40)
    for i, fs in enumerate(f_today):
        print(f"  Factor {i+1:<4}  {fs:>+8.4f}  {'risk-on  ▲' if fs > 0 else 'risk-off ▼'}")

    print(f"\n{'─'*72}")
    print(f"ETF CONTRIBUTIONS TO EACH FACTOR  (top {top_etfs} by |loading × demeaned return|)")
    print(f"{'─'*72}")

    for i in range(k):
        vt_row    = model._Vt[i]
        contrib_s = pd.Series(vt_row * r_e.values, index=etf_names)
        top       = contrib_s.abs().nlargest(top_etfs).index
        print(f"\n  Factor {i+1}  (score = {f_today[i]:+.4f})")
        print(f"  {'ETF':<8}  {'Today Ret':>10}  {'Loading':>10}  {'Contribution':>12}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}")
        for etf in top:
            j = etf_names.index(etf)
            print(
                f"  {etf:<8}  "
                f"{today_etf_returns.get(etf, float('nan')):>+10.4%}  "
                f"{vt_row[j]:>+10.4f}  "
                f"{contrib_s[etf]:>+12.4f}"
            )
        print(f"  {'':8}  {'':10}  {'Sum':>10}  {contrib_s.sum():>+12.4f}")

    for ticker in tickers:
        if ticker not in model.loadings.index:
            print(f"\n  ⚠️  {ticker} not in model — not part of the fitted universe.")
            continue
        betas           = model.loadings.loc[ticker].values
        factor_contribs = betas * f_today
        expected_rel    = factor_contribs.sum()
        r2              = model.diagnostics.stock_r_squared.loc[ticker]

        print(f"\n{'='*72}")
        print(f"DECOMPOSITION FOR {ticker}  (R² = {r2:.2f})")
        print(f"{'='*72}")
        print(f"  {'Factor':<12}  {'Beta':>12}  {'Factor Score':>13}  {'Contribution':>13}")
        print(f"  {'─'*12}  {'─'*12}  {'─'*13}  {'─'*13}")
        for i in range(k):
            print(
                f"  Factor {i+1:<4}  "
                f"{betas[i]:>+12.4f}  "
                f"{f_today[i]:>+13.4f}  "
                f"{factor_contribs[i]:>+13.4%}"
            )
        print(f"  {'─'*12}  {'─'*12}  {'─'*13}  {'─'*13}")
        print(f"  {'Exp. Rel.':<12}  {'':>12}  {'':>13}  {expected_rel:>+13.4%}")
        print(f"\n  {ticker}'s Exp. Rel. = sum of {k} beta-weighted factor scores.")
        print(f"  Each factor is a principal component of your {len(etf_names)}-ETF basket.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose Exp. Rel. for scored stocks.")
    parser.add_argument("tickers", nargs="+", help="Ticker(s) to explain, e.g. NOW CRM")
    parser.add_argument("--top-etfs", type=int, default=5, help="ETF contributors per factor")
    args    = parser.parse_args()
    tickers = [t.strip("'\"").upper() for t in args.tickers]

    exec_cfg   = ExecutionConfig()
    model_path = find_latest_model(US_UNIVERSE.name, exec_config=exec_cfg)
    if model_path is None:
        print("No saved model found. Run 'shockarb score' first.")
        sys.exit(1)

    print(f"Loading model from {Path(model_path).name} ...")
    model = load_model(model_path)

    etf_tickers = list(model._etf_mean.index)
    print(f"Fetching live returns for {len(etf_tickers)} ETFs ...")
    try:
        today_etf = fetch_live_returns(etf_tickers, exec_config=exec_cfg)
    except Exception as exc:
        print(f"Could not fetch live ETF returns: {exc}")
        sys.exit(1)

    explain(model, today_etf, tickers, top_etfs=args.top_etfs)


if __name__ == "__main__":
    main()
