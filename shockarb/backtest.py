"""
shockarb.backtest — Walk-forward backtester for the factor model.

Architecture
------------
Simulates live daily execution by stepping through history, fitting a model
on T-window, scoring T+1, and tracking target returns through T+n.

Key Design: "Cohort Tracking" Pattern
-------------------------------------
To avoid O(N^2) recalculations when measuring residual (factor-hedged) returns,
this module uses a cohort tracking pattern. Instead of re-scoring each open position
against the day's ETF returns every single day (which scales as N positions × D days),
we hold the fitted FactorModel for each entry date in memory and vectorize the
mark-to-market calculations for all active cohorts in a single pass per day.

This keeps complexity at O(D) instead of O(N × D), enabling backtests over thousands
of stocks without prohibitive computation cost.

Why Two Return Types?
---------------------
- raw_return: Gross P&L (stock return from entry to exit). Used for raw profitability.
- residual_return: Factor-hedged return (stock return minus what factors predict).
  This isolates the trade's "alpha" from broader market moves, showing whether the
  factor model's mispricing call actually worked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd

from shockarb.config import UniverseConfig, ExecutionConfig
from shockarb.engine import FactorModel

# =============================================================================
# Configuration & Results
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """Immutable configuration for a walk-forward backtest run."""
    universe: UniverseConfig
    calibration_window: int = 35
    holding_periods: tuple[int, ...] = (1, 2, 3, 5)  # tuple for hashability/immutability
    top_n: int = 10
    min_confidence: float = 0.005
    min_r_squared: float = 0.50
    return_type: str = "both"  # "raw", "residual", or "both"

    def __post_init__(self):
        if self.return_type not in {"raw", "residual", "both"}:
            raise ValueError("return_type must be 'raw', 'residual', or 'both'")

@dataclass(frozen=True)
class BacktestResults:
    """Container for backtest outputs. Pass data, not objects."""
    ledger: pd.DataFrame
    summary: pd.DataFrame
    daily_pnl: pd.Series

# =============================================================================
# Core Execution Engine
# =============================================================================

class Backtest:
    """
    Stateful runner for the walk-forward simulation.

    Stateful runner for the walk-forward simulation.
    Supports both Rolling Calibration (default) and Static Model evaluation.
    
    NOTE: I/O (fetching historical prices) is assumed to be handled before 
    calling run() and passed into the instance, adhering to the standard 
    of pushing I/O to the edges.
    """
    def __init__(
        self, 
        config: BacktestConfig, 
        historical_etf_returns: pd.DataFrame, 
        historical_stock_returns: pd.DataFrame,
        static_model: FactorModel | None = None  # <--- ADD THIS
    ):
        self.config = config
        self._etfs = historical_etf_returns
        self._stocks = historical_stock_returns
        self.static_model = static_model         # <--- ADD THIS
        
        # If static_model is provided, we don't need a calibration warmup.
        if self.static_model is not None:
            self._walk_forward_dates = self._stocks.index
        else:
            self._walk_forward_dates = self._stocks.index[self.config.calibration_window:]

    def run(self) -> BacktestResults:
        """
        Execute the walk-forward loop and return compiled results.

        The main loop:
          1. Each day, update P&L for all active cohorts (positions still held)
          2. Fit a new factor model on the prior N days of calibration data
          3. Score today's stocks to find new mispricing candidates
          4. Open new trades on candidates passing confidence and R² thresholds
          5. Record trades when they hit designated holding period milestones
          6. Garbage-collect cohorts that exceed max holding period
        """

        # Dictionary of active trades keyed by entry_date (the date the trade was opened).
        # Minimal state: holds only what is needed to compute ongoing returns.
        # Structure for each cohort:
        #   {
        #       "model": FactorModel,              # Fitted on entry_date's calibration window
        #       "targets": List[str],              # Tickers in this trade
        #       "raw_pnl": pd.Series,              # Cumulative raw return per ticker
        #       "residual_pnl": pd.Series,         # Cumulative residual return per ticker
        #       "days_held": int,                  # Number of days since entry
        #       "entry_metrics": pd.DataFrame      # Entry-day signal metrics (delta_rel, r_squared, confidence_delta)
        #   }
        active_cohorts: Dict[pd.Timestamp, Dict[str, Any]] = {}
        completed_trades: List[Dict[str, Any]] = []

        for current_date in self._walk_forward_dates:
            today_etfs = self._etfs.loc[current_date]
            today_stocks = self._stocks.loc[current_date]

            # ---------------------------------------------------------
            # 1. Mark active cohorts to market (Vectorized)
            # ---------------------------------------------------------
            # Iterate over a list of keys so we can delete expired cohorts safely
            for entry_date in list(active_cohorts.keys()):
                cohort = active_cohorts[entry_date]
                model: FactorModel = cohort["model"]
                targets: List[str] = cohort["targets"]

                # Extract today's return specifically for this cohort's active targets.
                # If a ticker is missing (delisted, bad data, or never had a price), treat as 0% return.
                # This is a conservative assumption: we assume the position was closed at yesterday's price,
                # avoiding artificially pessimistic returns from data gaps. In production, you'd want to
                # either: (a) flag missing tickers and remove them from the cohort, or (b) fetch from a
                # backup data source. For backtesting purposes, 0% is a reasonable approximation.
                target_returns = today_stocks.reindex(targets).fillna(0)

                # Raw (Gross) Returns: simply compound today's actual return
                if self.config.return_type in {"raw", "both"}:
                    cohort["raw_pnl"] = (1 + cohort["raw_pnl"]) * (1 + target_returns) - 1

                # Residual (Hedged) Returns: compute the idiosyncratic surprise.
                # Residual return = (actual return) - (factor-implied return).
                # This isolates what the stock did above/below what the factors predict,
                # measuring the "alpha" from the trade signal (independent of beta).
                #
                # Example: Stock returns +5%, factors predict +3% → residual = +2%
                #          Stock returns +3%, factors predict +5% → residual = -2% (underperformed)
                if self.config.return_type in {"residual", "both"}:
                    scores = model.score(today_etfs, target_returns)
                    # WARNING: model.score() defines delta_rel = (expected_rel - actual).
                    # This is backward from what we want for realized P&L.
                    # A positive delta means the stock fell more than expected (a buy signal).
                    # But when measuring actual P&L, we want: actual - expected (so negative delta becomes positive residual).
                    # We invert the sign: residual_return = -delta_rel = actual - expected.
                    daily_residual = -1 * scores["delta_rel"]
                    cohort["residual_pnl"] = (1 + cohort["residual_pnl"]) * (1 + daily_residual) - 1

                cohort["days_held"] += 1

                # Record returns if we hit a designated holding period milestone
                if cohort["days_held"] in self.config.holding_periods:
                    self._record_ledger(completed_trades, cohort, current_date, entry_date)

                # Garbage collection: drop cohort once it passes max holding period
                if cohort["days_held"] >= max(self.config.holding_periods):
                    del active_cohorts[entry_date]

           # ---------------------------------------------------------
            # 2. Calibrate new model & generate today's signals
            # ---------------------------------------------------------
            # Use the frozen static model if provided; otherwise roll the calibration window
            if self.static_model is not None:
                new_model = self.static_model
            else:
                calib_start = self._stocks.index.get_loc(current_date) - self.config.calibration_window
                calib_end = self._stocks.index.get_loc(current_date)
                
                calib_etfs = self._etfs.iloc[calib_start:calib_end]
                calib_stocks = self._stocks.iloc[calib_start:calib_end]

                new_model = FactorModel(calib_etfs, calib_stocks).fit(n_components=self.config.universe.n_components)
            
            new_signals = new_model.score(today_etfs, today_stocks)

            # Filter mathematically
            actionable = new_signals[
                (new_signals["confidence_delta"] >= self.config.min_confidence) &
                (new_signals["r_squared"] >= self.config.min_r_squared)
            ].head(self.config.top_n)

            # If signals exist, initialize a new cohort
            if not actionable.empty:
                active_cohorts[current_date] = {
                    "model": new_model, 
                    "targets": actionable.index.tolist(),
                    "raw_pnl": pd.Series(0.0, index=actionable.index),
                    "residual_pnl": pd.Series(0.0, index=actionable.index),
                    "days_held": 0,
                    "entry_metrics": actionable
                }

        # Compile and return results
        ledger_df = pd.DataFrame(completed_trades)

        # Guard: ensure columns always exist for downstream consumers,
        # even if no trades were generated. This prevents KeyError when accessing columns.
        if ledger_df.empty:
            ledger_df = pd.DataFrame(columns=[
                "entry_date", "exit_date", "holding_period", "ticker",
                "confidence_delta", "r_squared", "raw_return", "residual_return"
            ])

        summary_df = self._compute_summary(ledger_df)
        daily_pnl = self._compute_daily_pnl(ledger_df)

        return BacktestResults(ledger=ledger_df, summary=summary_df, daily_pnl=daily_pnl)

    def _record_ledger(self, completed_trades: List[Dict[str, Any]], cohort: Dict[str, Any], current_date: pd.Timestamp, entry_date: pd.Timestamp) -> None:
        """
        Extracts the required metrics from the cohort and flattens them 
        into individual trade records for the ledger.
        """
        for ticker in cohort["targets"]:
            record = {
                "entry_date": entry_date,
                "exit_date": current_date,
                "holding_period": cohort["days_held"],
                "ticker": ticker,
                "confidence_delta": cohort["entry_metrics"].at[ticker, "confidence_delta"],
                "r_squared": cohort["entry_metrics"].at[ticker, "r_squared"],
            }
            
            # Conditionally append returns based on the config
            if self.config.return_type in {"raw", "both"}:
                record["raw_return"] = cohort["raw_pnl"][ticker]
                
            if self.config.return_type in {"residual", "both"}:
                record["residual_return"] = cohort["residual_pnl"][ticker]
                
            completed_trades.append(record)
        
    def _compute_summary(self, ledger_df: pd.DataFrame) -> pd.DataFrame:
        """Groups ledger by holding period and computes win rate and mean return."""
        if ledger_df.empty:
            return pd.DataFrame()
            
        summary_records = []
        for hp, group in ledger_df.groupby("holding_period"):
            record = {
                "holding_period": hp, 
                "trade_count": len(group)
            }
            
            if "raw_return" in group.columns:
                record["raw_win_rate"] = (group["raw_return"] > 0).mean()
                record["mean_raw_return"] = group["raw_return"].mean()
                
            if "residual_return" in group.columns:
                record["residual_win_rate"] = (group["residual_return"] > 0).mean()
                record["mean_residual_return"] = group["residual_return"].mean()
                
            summary_records.append(record)
            
        return pd.DataFrame(summary_records).set_index("holding_period")

    def _compute_daily_pnl(self, ledger_df: pd.DataFrame) -> pd.Series:
        """
        Aggregates overlapping trades into a simplified daily proxy curve 
        by averaging the returns of positions closed on a given date.
        """
        if ledger_df.empty:
            return pd.Series(dtype=float)
            
        target_col = "residual_return" if "residual_return" in ledger_df.columns else "raw_return"
        return ledger_df.groupby("exit_date")[target_col].mean()