"""
ShockArb Portfolio Sizing Utility.

Reads one or more ShockArb score CSVs, selects the top-N positive signals
by conviction (confidence_delta), and prints a dollar-denominated trade
ticket with allocation weights and take-profit limit prices.

Current prices are fetched via the DataCoordinator (parquet cache + tail-fetch),
so prices already cached from today's score run cost nothing to retrieve.

Output is saved to data/portfolio_sizer.csv by default. Suppress with --no-out.

Usage examples
--------------
    # Size $100k across the top 5 US signals
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000

    # Merge US + Global into a single ticket
    python utils/portfolio_sizer.py \\
        --csv data/live_alpha_us.csv data/live_alpha_global.csv \\
        --capital 50000 --top 8

    # Exclude specific tickers (output still saved to data/portfolio_sizer.csv by default)
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 \\
        --exclude SNPS BSX

    # Size only specific tickers (bypasses CSV ranking entirely)
    python utils/portfolio_sizer.py --tickers AMAT ADI ETN --capital 10000

    # Suppress file output entirely
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 --no-out

    # Save to a custom path
    python utils/portfolio_sizer.py --csv data/live_alpha_us.csv --capital 100000 \\
        --out data/ticket.csv

    # Mark your actual holdings against today's ShockArb fair value (not an
    # analyst target) — reads a brokerage positions export, intersects it
    # with today's --csv, and reports price vs price*(1+delta_rel) using
    # your real shares/cost basis instead of a capital-weighted allocation
    python utils/portfolio_sizer.py \\
        --positions data/Individual-Positions-2026-07-01-100821.csv

    # Same, but also append this run's read to the durable position log
    # (data/shockarb_position_log.csv) so it survives future runs overwriting
    # data/portfolio_sizer.csv / data/shockarb_position_mark.csv
    python utils/portfolio_sizer.py \\
        --positions data/Individual-Positions-2026-07-01-100821.csv --execute

    # --execute also works on a normal sizing ticket, to log the entry
    # context (price, ShockArb fair value, signal quality) at the moment
    # you actually place the trade
    python utils/portfolio_sizer.py --tickers AMAT ADI ETN --capital 10000 --execute
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from paths import POSITION_LOG, POSITION_MARK_OUT

_DEFAULT_OUT = "./data/portfolio_sizer.csv"
_DEFAULT_MARK_OUT = str(POSITION_MARK_OUT)
_LOG_PATH = POSITION_LOG

# Fixed column order for the durable log — both event types ("ticket" and
# "mark") always write exactly this set, so appends never produce a ragged
# CSV even though the two flows populate different subsets of columns.
_LOG_COLUMNS = [
    "timestamp", "ticker", "event", "price", "delta_rel", "fair_price",
    "confidence_delta", "r_squared", "shares", "cost_basis", "csv_source",
]


def _check_cwd() -> None:
    """Exit with a clear error if not run from the project root."""
    if not Path("data").is_dir():
        print(
            "\n❌  portfolio_sizer.py must be run from the project root.\n"
            "\n"
            "    Correct usage:\n"
            "        cd <project_root>\n"
            "        python utils\\portfolio_sizer.py --tickers MSFT IDXX --capital 10000\n"
            "\n"
            f"    Current directory: {Path.cwd()}\n"
        )
        sys.exit(1)


def _fetch_current_prices(tickers: list[str]) -> pd.Series:
    """
    Return the most recent adj_close for each ticker via the DataCoordinator.

    Uses a 7-day window so the result is correct across weekends and holidays.
    Data already in the parquet cache from today's score run is served without
    a network call.

    Imports are lazy so the module remains importable in test environments that
    don't have the project root on sys.path (the function is patched in tests).

    Example
    -------
        _fetch_current_prices(["MSFT", "BLK"])
        # → pd.Series({"MSFT": 362.80, "BLK": 988.50})
    """
    from datetime import date, timedelta

    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

    from shockarb.store import DataStore as _InnerStore
    from datamgr.coordinator import DataCoordinator
    from datamgr.stores.parquet import ParquetStore
    from datamgr.providers.yfinance import YFinanceProvider
    from datamgr.requests import DataRequest, Frequency

    data  = _root / "data"
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=7)).isoformat()

    inner = _InnerStore(data)
    coord = DataCoordinator(ParquetStore(inner), provider=YFinanceProvider())
    coord.register(DataRequest(
        tickers   = tuple(tickers),
        start     = start,
        end       = end,
        frequency = Frequency.DAILY,
        retention = "permanent",
        requester = "portfolio_sizer",
    ))
    results = coord.fulfill()
    prices  = results.get("portfolio_sizer", pd.DataFrame())
    return prices.iloc[-1] if not prices.empty else pd.Series(dtype=float)


def _load_master(csv_paths: list[str]) -> pd.DataFrame | None:
    """
    Load and merge one or more ShockArb score CSVs.

    Shared by generate_orders() and mark_positions() — the same "which
    tickers did ShockArb score today, with what delta_rel/confidence_delta"
    question underlies both a new trade ticket and a mark of existing
    holdings. Returns None (after logging why) if nothing usable loads.

    Example:
        _load_master(["data/live_alpha_us.csv"])
        # → DataFrame with columns Ticker, confidence_delta, delta_rel, r_squared, ...
    """
    dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            logger.warning(f"Alpha report not found: {path}")
            continue
        try:
            df = pd.read_csv(path)
            if "Ticker" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Ticker"})
            dfs.append(df)
        except Exception as exc:
            logger.error(f"Failed to read {path}: {exc}")

    if not dfs:
        logger.error("No valid CSVs loaded.")
        return None

    master = pd.concat(dfs, ignore_index=True)
    required = {"confidence_delta", "delta_rel"}
    missing = required - set(master.columns)
    if missing:
        logger.error(f"CSV is missing required columns: {missing}")
        logger.error(f"  Available columns: {list(master.columns)}")
        return None
    return master


def _load_held_tickers(positions_path: str, known_tickers: set[str]) -> dict[str, dict]:
    """
    Parse a brokerage positions export and return ShockArb-scored holdings.

    The export has two header lines (an account title, then a blank line)
    before the real column row, plus dollar/comma-formatted numeric strings.
    Only "Equity" rows whose symbol is in `known_tickers` (the tickers
    ShockArb actually scored today, from _load_master) are kept — this is
    what keeps a name like BRK/B or a bond ETF out of the mark table without
    hardcoding a ticker universe here.

    Example:
        _load_held_tickers("data/Individual-Positions-....csv", {"ADI", "CRM"})
        # → {"ADI": {"shares": 9.0, "cost_basis": 386.91}}
    """
    df = pd.read_csv(positions_path, skiprows=2)
    df = df[df.get("Asset Type") == "Equity"].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df = df[df["Symbol"].isin(known_tickers)]

    held = {}
    for _, row in df.iterrows():
        qty  = pd.to_numeric(str(row["Qty (Quantity)"]).replace(",", ""), errors="coerce")
        cost = pd.to_numeric(str(row["Cost Basis"]).replace(",", "").replace("$", ""), errors="coerce")
        if pd.isna(qty) or qty == 0 or pd.isna(cost):
            continue
        held[row["Symbol"]] = {"shares": float(qty), "cost_basis": float(cost) / float(qty)}
    return held


def _append_position_log(rows: list[dict], log_path: Path | str | None = None) -> None:
    """
    Append rows to the durable, never-overwritten ShockArb position log.

    Unlike data/portfolio_sizer.csv or data/shockarb_position_mark.csv (both
    overwritten every run), this file only grows. Every row is reindexed to
    _LOG_COLUMNS so "ticket" rows (from generate_orders) and "mark" rows
    (from mark_positions) share one rectangular schema even though each
    populates a different subset of columns.

    `log_path` defaults to None (resolved to the module-level _LOG_PATH
    inside the function body) rather than being bound as a default argument
    value — that keeps it patchable via `patch("portfolio_sizer._LOG_PATH", ...)`
    in tests, since default-argument values are captured once at import time.

    Example:
        _append_position_log([{"timestamp": "...", "ticker": "ADI", "event": "mark", ...}])
    """
    log_path = Path(log_path) if log_path is not None else Path(_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).reindex(columns=_LOG_COLUMNS)
    df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    logger.success(f"Logged {len(rows)} row(s) to {log_path}")


def generate_orders(
    csv_paths: list[str],
    capital: float,
    top_n: int = 5,
    exclude: list[str] | None = None,
    out: str | None = _DEFAULT_OUT,
    tickers: list[str] | None = None,
    execute: bool = False,
) -> None:
    """
    Print a trade ticket for the top-N conviction signals.

    Parameters
    ----------
    csv_paths : list of str
        Paths to ShockArb score CSVs.  Multiple files are merged before ranking.
    capital : float
        Total dollar capital to allocate.
    top_n : int
        Number of positions to take.  Ignored when tickers is set.
    exclude : list of str, optional
        Tickers to exclude before ranking (e.g. catalyst-driven traps).
        Ignored when tickers is set.
    out : str, optional
        Path to save the ticket CSV.  Defaults to data/portfolio_sizer.csv.
        Pass None to suppress file output.
    tickers : list of str, optional
        If supplied, only these tickers are sized (CSV ranking is bypassed).
        Overrides top_n and exclude.
    execute : bool
        Append this ticket's rows to the durable data/shockarb_position_log.csv
        (event="ticket"). Use this at the moment you actually place the trade,
        so the entry price and ShockArb fair-value read aren't lost the next
        time this CSV or data/portfolio_sizer.csv gets overwritten.
    """
    exclude = [t.upper() for t in (exclude or [])]
    tickers = [t.upper() for t in (tickers or [])]
    master = _load_master(csv_paths)
    if master is None:
        return

    if tickers:
        master = master[master["Ticker"].str.upper().isin(tickers)]
    elif exclude:
        master = master[~master["Ticker"].str.upper().isin(exclude)]

    buys = (
        master[master["confidence_delta"] > 0]
        .sort_values("confidence_delta", ascending=False)
        .head(top_n if not tickers else len(master))
    )

    if buys.empty:
        logger.warning("No positive alpha signals found.")
        return

    # Fetch current prices via shared parquet cache
    ticker_list = buys["Ticker"].tolist()
    logger.info(f"Fetching current prices for: {ticker_list}")
    current = _fetch_current_prices(ticker_list)

    # Conviction-weighted allocation
    total_conviction = buys["confidence_delta"].sum()
    buys = buys.copy()
    buys["Weight"]       = buys["confidence_delta"] / total_conviction
    buys["Dollar_Alloc"] = buys["Weight"] * capital

    # Print ticket
    print("\n" + "=" * 122)
    print(f"  SHOCKARB TRADE TICKET  |  Capital: ${capital:,.2f}  |  Positions: {len(buys)}")
    print("=" * 122)
    print(f"  {'TICKER':<8}  {'WEIGHT':>8}  {'ALLOC':>14}  {'COST':>14}  {'CURRENT':>10}  {'TARGET':>10}  {'SHARES':>6}")
    print("-" * 122)

    rows = []
    for _, row in buys.iterrows():
        ticker = row["Ticker"]
        if ticker not in current.index or pd.isna(current[ticker]):
            logger.warning(f"No live price for {ticker} — skipping row.")
            continue

        price    = float(current[ticker])
        target   = price * (1 + row["delta_rel"])
        shares   = int(row["Dollar_Alloc"] / price)
        cost     = shares * price                        # actual dollars deployed (shares × price)

        print(
            f"  {ticker:<8}  {row['Weight']:>7.1%}  ${row['Dollar_Alloc']:>13,.2f}"
            f"  ${cost:>13,.2f}  ${price:>9.2f}  ${target:>9.2f}  {shares:>6}"
        )
        rows.append({
            "Ticker":           ticker,
            "Weight":           round(row["Weight"], 4),
            "Alloc":            round(row["Dollar_Alloc"], 2),
            "Cost":             round(cost, 2),
            "Current":          round(price, 2),
            "Target":           round(target, 2),
            "Shares":           shares,
            "confidence_delta": round(row["confidence_delta"], 6),
            "r_squared":        round(row.get("r_squared", float("nan")), 4),
            "delta_rel":        round(row["delta_rel"], 6),
        })

    total_alloc = sum(r["Alloc"] for r in rows)
    total_cost  = sum(r["Cost"]  for r in rows)
    print("-" * 122)
    print(f"  {'TOTAL':<8}  {'':>8}  ${total_alloc:>13,.2f}  ${total_cost:>13,.2f}")
    print("=" * 122)
    print("  ALLOC = conviction-weighted target  |  COST = shares × price (actual spend)")
    print("  EXIT: Place GTC sell-limit orders at the Target price.")
    print()

    if out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        logger.success(f"Ticket saved: {out}")

    if execute and rows:
        timestamp = datetime.now().isoformat(timespec="seconds")
        log_rows = [{
            "timestamp": timestamp, "ticker": r["Ticker"], "event": "ticket",
            "price": r["Current"], "delta_rel": r["delta_rel"], "fair_price": r["Target"],
            "confidence_delta": r["confidence_delta"], "r_squared": r["r_squared"],
            "shares": r["Shares"], "cost_basis": None,
            "csv_source": ",".join(csv_paths),
        } for r in rows]
        _append_position_log(log_rows)


def mark_positions(
    positions_path: str,
    csv_paths: list[str],
    out: str | None = _DEFAULT_MARK_OUT,
    execute: bool = False,
) -> None:
    """
    Mark currently-held ShockArb tickers against today's factor-model fair
    value — price * (1 + delta_rel) — independent of any analyst target.

    Read-only counterpart to generate_orders(): instead of sizing a *new*
    order, this reports where ShockArb's own model thinks each existing
    holding is fairly priced today, using the real shares/cost basis from a
    brokerage positions export rather than a capital-weighted allocation.

    Parameters
    ----------
    positions_path : str
        Path to a brokerage positions export, e.g.
        data/Individual-Positions-2026-07-01-100821.csv.
    csv_paths : list of str
        ShockArb score CSV(s) — same role as generate_orders' --csv.
    out : str, optional
        Save the mark table to CSV. Defaults to data/shockarb_position_mark.csv.
        Pass None to suppress.
    execute : bool
        Append this run's rows to the durable data/shockarb_position_log.csv
        (event="mark"). Run this right after your daily `shockarb score` to
        build a continuous fair-value history for whatever you currently
        hold, with no need to remember a ticker list.

    Example
    -------
        mark_positions("data/Individual-Positions-2026-07-01-100821.csv",
                        ["data/live_alpha_us.csv"], execute=True)
    """
    master = _load_master(csv_paths)
    if master is None:
        return

    known = set(master["Ticker"].astype(str).str.upper())
    held = _load_held_tickers(positions_path, known)
    if not held:
        logger.warning("No ShockArb-scored tickers found in the positions file.")
        return

    prices = _fetch_current_prices(list(held.keys()))
    master_idx = master.set_index(master["Ticker"].astype(str).str.upper())

    print("\n" + "=" * 122)
    print(f"  SHOCKARB POSITION MARK  |  Source: {positions_path}")
    print("=" * 122)
    print(f"  {'TICKER':<8}  {'SHARES':>8}  {'COST/SH':>10}  {'CURRENT':>10}  "
          f"{'FAIR (ShockArb)':>16}  {'GAP':>7}  {'GAIN %':>8}  {'r2':>6}")
    print("-" * 122)

    rows, timestamp = [], datetime.now().isoformat(timespec="seconds")
    for ticker, info in held.items():
        if ticker not in prices.index or pd.isna(prices[ticker]):
            logger.warning(f"No live price for {ticker} — skipping row.")
            continue
        price     = float(prices[ticker])
        delta_rel = float(master_idx.loc[ticker, "delta_rel"])
        fair      = price * (1 + delta_rel)
        gain_pct  = (price - info["cost_basis"]) / info["cost_basis"]
        r_squared = float(master_idx.loc[ticker, "r_squared"])

        print(f"  {ticker:<8}  {info['shares']:>8.2f}  ${info['cost_basis']:>9.2f}  "
              f"${price:>9.2f}  ${fair:>15.2f}  {delta_rel:>6.1%}  {gain_pct:>7.1%}  {r_squared:>6.3f}")

        rows.append({
            "timestamp": timestamp, "ticker": ticker, "event": "mark",
            "price": round(price, 2), "delta_rel": round(delta_rel, 6),
            "fair_price": round(fair, 2),
            "confidence_delta": round(float(master_idx.loc[ticker, "confidence_delta"]), 6),
            "r_squared": round(r_squared, 4),
            "shares": info["shares"], "cost_basis": round(info["cost_basis"], 2),
            "csv_source": ",".join(csv_paths),
        })

    print("=" * 122)
    print("  FAIR (ShockArb) = price × (1 + delta_rel) — the factor model's own read, not an analyst target.")
    print()

    if out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        pd.DataFrame(rows).drop(columns=["timestamp", "event", "csv_source"]).to_csv(out, index=False)
        logger.success(f"Mark saved: {out}")

    if execute and rows:
        _append_position_log(rows)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    _check_cwd()
    parser = argparse.ArgumentParser(
        description="Generate a conviction-weighted ShockArb trade ticket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv", nargs="+", default=["./data/live_alpha_us.csv"],
        help="Path(s) to ShockArb score CSV files",
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Total capital to allocate in dollars (default: 100000)",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top positions (default: 5); ignored when --tickers is set",
    )
    parser.add_argument(
        "--exclude", "-e", nargs="+", default=[],
        help="Tickers to exclude before ranking (e.g. --exclude SNPS BSX); ignored when --tickers is set",
    )
    parser.add_argument(
        "--tickers", "-t", nargs="+", default=[],
        help="Size only these tickers; bypasses CSV ranking, --top, and --exclude (e.g. --tickers AMAT ADI ETN)",
    )
    parser.add_argument(
        "--positions", default=None,
        help=(
            "Path to a brokerage positions export (e.g. data/Individual-Positions-*.csv). "
            "When set, marks your currently-held ShockArb-scored tickers against today's "
            "factor-model fair value (price * (1 + delta_rel)) using real shares/cost basis, "
            "instead of sizing a new ticket. Bypasses --capital, --top, --exclude, and --tickers."
        ),
    )
    parser.add_argument(
        "--execute", action="store_true",
        help=(
            "Append this run's rows to the durable data/shockarb_position_log.csv "
            "(never overwritten, unlike --out). Use at the moment you place a trade, "
            "or any time after `shockarb score` to log a fair-value mark of --positions."
        ),
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help=(
            f"Save output to CSV (default: {_DEFAULT_OUT} for a ticket, "
            f"{_DEFAULT_MARK_OUT} for --positions)"
        ),
    )
    parser.add_argument(
        "--no-out", "-sout", action="store_true",
        help="Suppress CSV output (do not write a file)",
    )
    args = parser.parse_args()

    if args.positions:
        out = None if args.no_out else (args.out or _DEFAULT_MARK_OUT)
        mark_positions(args.positions, args.csv, out=out, execute=args.execute)
    else:
        out = None if args.no_out else (args.out or _DEFAULT_OUT)
        generate_orders(
            args.csv, args.capital, args.top, args.exclude, out, args.tickers,
            execute=args.execute,
        )
