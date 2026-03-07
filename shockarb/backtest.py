"""
Walk-forward backtester for the ShockArb factor model.

Architecture
------------
The backtest steps through a historical price series one trading day at a
time.  On each step it:

  1. Fits a FactorModel on a fixed-length calibration window ending at T.
  2. Scores the tape on T+1 (the day *after* the calibration window closes)
     to generate simulated entry signals — no look-ahead.
  3. Marks those positions to market at T+2, T+3, T+4, T+5 to measure
     mean-reversion decay.  This answers "am I a 1-day trade or a 3-day
     trade?" before sizing real positions.
  4. Slides the calibration window forward by one day and repeats.

The output is a ``BacktestResults`` object containing:
  - ``ledger``     : every trade with entry/exit returns at each horizon
  - ``summary``    : win rate, mean return, Sharpe, hit rate by holding period
  - ``daily_pnl``  : aggregate P&L series for equity-curve plotting

Design principles
-----------------
* No data leakage: the model fitted on [T-window+1 … T] never sees T+1.
* Same FactorModel / score() path as live execution — no special backtest
  codepath in the engine.
* Holding-period returns are *gross* (no transaction cost).  Subtract your
  own slippage / commission assumptions in post-processing.
* The calibration window is the same length throughout; no expanding window.
  This keeps the factor extraction stationary.

Typical usage
-------------
    from shockarb.backtest import Backtest, BacktestConfig
    from shockarb.config import US_UNIVERSE

    cfg = BacktestConfig(
        universe=US_UNIVERSE,
        # Override the calibration window length (default 35 days)
        calib_window=35,
        # Score each T+1 and mark-to-market through T+5
        holding_periods=[1, 2, 3, 5],
        # Only trade signals above these thresholds
        min_confidence=0.005,
        min_r_squared=0.50,
        # Walk-forward evaluation window (separate from the calib data)
        eval_start="2023-01-01",
        eval_end="2024-12-31",
    )

    bt = Backtest(cfg)
    results = bt.run()
    results.print_summary()
    results.ledger.to_csv("backtest_trades.csv")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

import shockarb.pipeline as pipeline
from shockarb.config import ExecutionConfig, UniverseConfig
from shockarb.engine import FactorModel


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BacktestConfig:
    """
    Full specification for a walk-forward backtest run.

    Parameters
    ----------
    universe : UniverseConfig
        Defines ETF and stock tickers, n_components, and (crucially) the
        *calibration window dates*.  The backtest uses those dates as the
        first calibration window, then slides forward.
    calib_window : int
        Number of trading days in each calibration window.  Default 35
        (matches the Ukraine shock duration).  Shorter windows adapt faster
        but noisier; longer windows are more stable but lag regime shifts.
    holding_periods : list of int
        Days-forward at which to measure mean-reversion.  Default [1,2,3,5].
        P&L is measured as the *stock's* return over the holding period
        (positive = stock rallied after our buy signal, i.e., we were right).
    min_confidence : float
        Minimum confidence_delta to enter a long signal.  Default 0.005.
    min_r_squared : float
        Minimum R² to enter a long signal.  Default 0.50.
    eval_start : str
        YYYY-MM-DD — first date of the walk-forward evaluation window.
        Must be *after* the universe's end_date (the calibration anchor).
    eval_end : str
        YYYY-MM-DD — last date of the evaluation window (inclusive).
    top_n : int
        Maximum signals to enter per day.  Default 5 (mimics portfolio_sizer).
    n_components : int, optional
        Override the universe's n_components.  Useful for sensitivity tests.
    """
    universe: UniverseConfig
    calib_window: int = 35
    holding_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    min_confidence: float = 0.005
    min_r_squared: float = 0.50
    eval_start: str = ""
    eval_end: str = ""
    top_n: int = 5
    n_components: Optional[int] = None

    def __post_init__(self):
        if not self.eval_start or not self.eval_end:
            raise ValueError("BacktestConfig requires eval_start and eval_end.")
        if pd.Timestamp(self.eval_start) >= pd.Timestamp(self.eval_end):
            raise ValueError("eval_start must be before eval_end.")
        if self.calib_window < 10:
            raise ValueError("calib_window must be >= 10 trading days.")
        if not self.holding_periods:
            raise ValueError("holding_periods cannot be empty.")

    @property
    def effective_n_components(self) -> int:
        return self.n_components or self.universe.n_components


# =============================================================================
# Results container
# =============================================================================

@dataclass
class BacktestResults:
    """
    Output of a completed backtest run.

    Attributes
    ----------
    ledger : DataFrame
        One row per (entry_date, ticker) signal.  Columns:
          entry_date      — date the signal was generated (T+1 after calib end)
          ticker          — stock symbol
          confidence_delta — signal strength at entry
          r_squared       — model fit quality at entry
          delta_rel       — raw mispricing (expected − actual) at entry
          actual_return   — actual return on the entry day
          ret_T+N         — forward return over N days for each holding period
          profitable_T+N  — bool, True if ret_T+N > 0
    summary : DataFrame
        Per-holding-period statistics: win_rate, mean_return, median_return,
        sharpe, n_trades.
    config : BacktestConfig
        The config that produced these results (for reproducibility).
    equity_curve : Series
        Cumulative equal-weight P&L of all T+1 signals (daily, indexed by date).
    """
    ledger: pd.DataFrame
    summary: pd.DataFrame
    config: BacktestConfig
    equity_curve: pd.Series

    def print_summary(self) -> None:
        """Print a formatted summary to the terminal."""
        cfg = self.config
        print(f"\n{'='*75}")
        print(f"  ⚡ SHOCKARB WALK-FORWARD BACKTEST RESULTS")
        print(f"{'='*75}")
        print(f"  Universe   : {cfg.universe.name.upper()}")
        print(f"  Eval window: {cfg.eval_start}  →  {cfg.eval_end}")
        print(f"  Calib days : {cfg.calib_window}  |  "
              f"Factors: {cfg.effective_n_components}  |  "
              f"Top-N/day: {cfg.top_n}")
        print(f"  Min conf.Δ : {cfg.min_confidence:.3f}  |  "
              f"Min R²: {cfg.min_r_squared:.2f}")
        print(f"  Total trades: {len(self.ledger)}")

        if self.ledger.empty:
            print("\n  No trades generated. Loosen thresholds or expand eval window.")
            print(f"{'='*75}\n")
            return

        print(f"\n  {'Holding':>10}  {'N Trades':>9}  {'Win Rate':>9}  "
              f"{'Mean Ret':>9}  {'Median Ret':>11}  {'Sharpe':>8}")
        print(f"  {'─'*73}")

        for _, row in self.summary.iterrows():
            print(
                f"  {row.name:>10}  "
                f"{int(row['n_trades']):>9}  "
                f"{row['win_rate']:>8.1%}  "
                f"{row['mean_return']:>+8.2%}  "
                f"{row['median_return']:>+10.2%}  "
                f"{row['sharpe']:>8.2f}"
            )

        print(f"\n  Equity curve (T+1, equal-weight): "
              f"total return {self.equity_curve.iloc[-1] - 1:+.2%}" if len(self.equity_curve) > 0
              else "")
        print(f"{'='*75}\n")


# =============================================================================
# Backtest engine
# =============================================================================

class Backtest:
    """
    Walk-forward backtester.

    The outer loop iterates over every evaluation date.  On each date T:
      - The calibration window is [T - calib_window … T-1]  (T exclusive)
      - The model is fitted on those calib_window days
      - Signals are generated for T (the *scoring day*)
      - Forward returns are looked up at T+1, T+2, … T+max(holding_periods)

    Parameters
    ----------
    config : BacktestConfig
    exec_config : ExecutionConfig, optional
        Controls caching and data directory.  Uses defaults if omitted.
    """

    def __init__(
        self,
        config: BacktestConfig,
        exec_config: Optional[ExecutionConfig] = None,
    ):
        self.cfg = config
        self.exec_cfg = exec_config or ExecutionConfig(log_to_file=False)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def run(self) -> BacktestResults:
        """
        Execute the full walk-forward backtest.

        Returns
        -------
        BacktestResults
        """
        logger.info(
            f"Starting walk-forward backtest: {self.cfg.eval_start} → {self.cfg.eval_end}"
        )

        # 1. Download all the price history we need in one shot
        etf_prices, stock_prices = self._fetch_all_prices()

        # 2. Convert to returns (fills, drops low-coverage tickers)
        etf_rets = pipeline.prices_to_returns(etf_prices)
        stock_rets = pipeline.prices_to_returns(stock_prices)

        # Align both return series to a common index
        common_dates = etf_rets.index.intersection(stock_rets.index)
        etf_rets = etf_rets.loc[common_dates]
        stock_rets = stock_rets.loc[common_dates]

        if len(common_dates) == 0:
            logger.error("No common trading days found in downloaded data.")
            return self._empty_results()

        logger.info(
            f"Price history: {len(common_dates)} trading days "
            f"({common_dates[0].date()} \u2192 {common_dates[-1].date()})"
        )

        # 3. Identify the evaluation dates — days within [eval_start, eval_end]
        eval_start = pd.Timestamp(self.cfg.eval_start)
        eval_end = pd.Timestamp(self.cfg.eval_end)
        eval_dates = common_dates[
            (common_dates >= eval_start) & (common_dates <= eval_end)
        ]

        if len(eval_dates) == 0:
            logger.error("No trading days found in the evaluation window.")
            return self._empty_results()

        logger.info(f"Evaluation dates: {len(eval_dates)} trading days")

        max_horizon = max(self.cfg.holding_periods)

        # 4. Walk-forward loop
        trade_rows = []

        for i, score_date in enumerate(eval_dates):
            # Find the calibration window ending just before score_date
            score_pos = common_dates.get_loc(score_date)
            calib_end_pos = score_pos            # exclusive
            calib_start_pos = calib_end_pos - self.cfg.calib_window

            if calib_start_pos < 0:
                logger.debug(
                    f"Skipping {score_date.date()}: insufficient history "
                    f"({calib_end_pos} days available, need {self.cfg.calib_window})"
                )
                continue

            calib_etf = etf_rets.iloc[calib_start_pos:calib_end_pos]
            calib_stock = stock_rets.iloc[calib_start_pos:calib_end_pos]

            # Fit the model on the calibration window
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = FactorModel(calib_etf, calib_stock)
                    model.fit(n_components=self.cfg.effective_n_components)
            except Exception as exc:
                logger.warning(f"{score_date.date()}: fit failed — {exc}")
                continue

            # Score the tape on score_date
            today_etf = etf_rets.loc[score_date]
            today_stock = stock_rets.loc[score_date]
            scores = model.score(today_etf, today_stock)

            # Filter to actionable signals
            signals = scores[
                (scores["confidence_delta"] >= self.cfg.min_confidence) &
                (scores["r_squared"] >= self.cfg.min_r_squared)
            ].head(self.cfg.top_n)

            if signals.empty:
                continue

            # Look up forward returns for each holding period
            for ticker, row in signals.iterrows():
                trade = {
                    "entry_date":       score_date,
                    "ticker":           ticker,
                    "confidence_delta": row["confidence_delta"],
                    "r_squared":        row["r_squared"],
                    "delta_rel":        row["delta_rel"],
                    "actual_return":    row["actual_return"],
                }

                for h in self.cfg.holding_periods:
                    fwd_pos = score_pos + h
                    if fwd_pos < len(common_dates) and ticker in stock_rets.columns:
                        # Forward return = compounded return over h days
                        fwd_dates = common_dates[score_pos + 1 : fwd_pos + 1]
                        if len(fwd_dates) > 0:
                            fwd_rets = stock_rets.loc[fwd_dates, ticker]
                            compound = (1 + fwd_rets).prod() - 1
                        else:
                            compound = np.nan
                    else:
                        compound = np.nan

                    trade[f"ret_T+{h}"] = compound
                    trade[f"profitable_T+{h}"] = (
                        bool(compound > 0) if not np.isnan(compound) else None
                    )

                trade_rows.append(trade)

            if (i + 1) % 20 == 0:
                logger.info(
                    f"  {i+1}/{len(eval_dates)} days  |  "
                    f"{len(trade_rows)} trades so far"
                )

        logger.info(f"Walk-forward complete. {len(trade_rows)} total trades.")

        if not trade_rows:
            logger.warning(
                "No trades generated. Consider loosening min_confidence "
                f"(currently {self.cfg.min_confidence:.3f}) or min_r_squared "
                f"(currently {self.cfg.min_r_squared:.2f})."
            )
            return self._empty_results()

        ledger = pd.DataFrame(trade_rows).set_index(["entry_date", "ticker"])
        summary = self._build_summary(ledger)
        equity_curve = self._build_equity_curve(ledger, stock_rets)

        return BacktestResults(
            ledger=ledger,
            summary=summary,
            config=self.cfg,
            equity_curve=equity_curve,
        )

    # -------------------------------------------------------------------------
    # Data fetching
    # -------------------------------------------------------------------------

    def _fetch_all_prices(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download the full price history needed for both the calibration
        anchor and the walk-forward evaluation window in one pass.

        The download window starts calib_window trading days before eval_start
        (so the first evaluation day has a full calibration window) and ends
        max(holding_periods) days after eval_end (so the last signal has
        forward returns available).

        We pad generously — 2× calib_window before, and 30 calendar days
        after — then let prices_to_returns trim to actual trading days.
        """
        # Pad start: go back far enough to have a full calib window
        # before the first evaluation date
        raw_start = (
            pd.Timestamp(self.cfg.eval_start)
            - pd.tseries.offsets.BDay(self.cfg.calib_window + 10)
        ).strftime("%Y-%m-%d")

        # Also make sure we include the universe's original calibration data
        # so the anchor window is available if eval_start is close to end_date
        fetch_start = min(raw_start, self.cfg.universe.start_date)

        # Pad end: max holding period + buffer
        fetch_end = (
            pd.Timestamp(self.cfg.eval_end)
            + pd.tseries.offsets.BDay(max(self.cfg.holding_periods) + 5)
        ).strftime("%Y-%m-%d")

        logger.info(
            f"Downloading prices: {fetch_start} → {fetch_end} "
            f"({len(self.cfg.universe.market_etfs)} ETFs + "
            f"{len(self.cfg.universe.individual_stocks)} stocks)"
        )

        etf_prices = pipeline.fetch_prices(
            self.cfg.universe.market_etfs,
            start=fetch_start,
            end=fetch_end,
            cache_name=f"{self.cfg.universe.name}_etf_bt",
            exec_config=self.exec_cfg,
        )
        stock_prices = pipeline.fetch_prices(
            self.cfg.universe.individual_stocks,
            start=fetch_start,
            end=fetch_end,
            cache_name=f"{self.cfg.universe.name}_stock_bt",
            exec_config=self.exec_cfg,
        )
        return etf_prices, stock_prices

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _build_summary(self, ledger: pd.DataFrame) -> pd.DataFrame:
        """Compute per-holding-period statistics from the trade ledger."""
        rows = []
        for h in self.cfg.holding_periods:
            col = f"ret_T+{h}"
            if col not in ledger.columns:
                continue
            rets = ledger[col].dropna()
            if len(rets) == 0:
                continue

            # Sharpe: mean / std of trade returns, annualised by sqrt(252/h)
            std = rets.std()
            sharpe = (
                (rets.mean() / std) * np.sqrt(252 / h)
                if std > 1e-10 else 0.0
            )

            rows.append({
                "holding_period": f"T+{h}",
                "n_trades":       len(rets),
                "win_rate":       (rets > 0).mean(),
                "mean_return":    rets.mean(),
                "median_return":  rets.median(),
                "sharpe":         sharpe,
            })

        summary = pd.DataFrame(rows).set_index("holding_period")
        return summary

    def _build_equity_curve(
        self,
        ledger: pd.DataFrame,
        stock_rets: pd.DataFrame,
    ) -> pd.Series:
        """
        Build a daily equal-weight equity curve from T+1 signals.

        Each day's P&L = mean of the T+1 returns for all signals entered
        that day.  The curve is cumulative product starting at 1.0.
        """
        col = "ret_T+1"
        if col not in ledger.columns:
            return pd.Series(dtype=float)

        # ledger index is (entry_date, ticker); group by entry_date
        daily = (
            ledger[col]
            .dropna()
            .reset_index(level="ticker", drop=True)
            .groupby(level="entry_date")
            .mean()
        )

        if daily.empty:
            return pd.Series(dtype=float)

        # Fill calendar gaps with 0 (days where no signals were active)
        full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
        daily = daily.reindex(full_index, fill_value=0.0)
        curve = (1 + daily).cumprod()
        curve.name = "equity_curve"
        return curve

    def _empty_results(self) -> BacktestResults:
        """Return a valid but empty BacktestResults."""
        return BacktestResults(
            ledger=pd.DataFrame(),
            summary=pd.DataFrame(),
            config=self.cfg,
            equity_curve=pd.Series(dtype=float),
        )
