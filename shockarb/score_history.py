"""
score_history.py — rolling daily score archive for regime health and alpha validation.

Stores one parquet file per trading day under data_dir/recent_scores/.
Each file holds one row per scored stock with the fields needed for SNR
and Spearman rank-correlation tests (Steps 2–4 in KT_REGIME_EFFECTIVENESS.md).

Design constraint -- this module never fetches prices.
    All price data enters through the `scores_df` argument of `save_row()`,
    which must be the output of `pipeline.score_universe()`. That function
    uses the datamgr DataCoordinator, so all data is cached and deduplicated
    there. Nothing here calls yfinance, the coordinator, or any data provider.
    --save-recent carries zero marginal download cost.

Usage:
    archive = ScoreArchive("data")
    archive.save_row(date.today(), scores_df, regime_name="ukraine_shock", model_file="ukraine_shock_us_...json")
    df = archive.load_window(days=30)
    archive.purge_stale(retention_days=90)
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import datetime as _dt
import pandas as pd

# Auto-purge threshold — files older than this are deleted on each save_row() call.
RECENT_SCORES_RETENTION_DAYS = 90

# Minimum days of archive data before regime health output is shown.
MIN_WINDOW_DAYS = 5

# SNR display thresholds (hardcoded display values; move to config if needed).
SNR_GOOD_THRESHOLD = 2.0   # R² > 0.67 — regime highly explanatory
SNR_WARN_THRESHOLD = 1.0   # R² 0.50–0.67 — regime drifting

# Archive column names (compact; avoids collisions with engine column names).

_COLS = ["date", "ticker", "actual", "expected", "delta", "r2", "conf_delta",
         "regime", "model_file"]


class ScoreArchive:
    """
    Manages the rolling parquet archive under data_dir/recent_scores/.

    One file per trading day: YYYY-MM-DD.parquet.
    Purge is time-based (unlink files older than retention_days).

    Example:
        archive = ScoreArchive("data")
        archive.save_row(date.today(), scores_df, "ukraine_shock", "ukraine_shock_us_20260528.json")
        df = archive.load_window(days=30)   # DataFrame with columns in _COLS
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir) / "recent_scores"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_row(
        self,
        score_date: date,
        scores_df: pd.DataFrame,
        regime_name: str,
        model_file: str,
    ) -> Path:
        """
        Persist today's scores to a daily parquet file.

        scores_df is the raw engine output: index = ticker, columns include
        actual_return, expected_rel, delta_rel, r_squared, confidence_delta.
        Only the columns we need are extracted; extras are silently ignored.

        Returns the path of the file written.
        """
        rows = pd.DataFrame(
            {
                "date":       score_date.isoformat(),
                "ticker":     scores_df.index,
                "actual":     scores_df["actual_return"].values,
                "expected":   scores_df["expected_rel"].values,
                "delta":      scores_df["delta_rel"].values,
                "r2":         scores_df["r_squared"].values,
                "conf_delta": scores_df["confidence_delta"].values,
                "regime":     regime_name,
                "model_file": os.path.basename(model_file),
                # next_day_actual is backfilled by the *next* save_row() call
                "next_day_actual": float("nan"),
            }
        )

        out = self._path_for(score_date)
        rows.to_parquet(out, index=False)

        # Backfill yesterday's next_day_actual with today's actual returns
        self._backfill_yesterday(score_date, scores_df)

        return out

    def purge_stale(self, retention_days: int = RECENT_SCORES_RETENTION_DAYS) -> int:
        """
        Delete files older than retention_days. Returns number of files removed.
        """
        cutoff = date.today() - timedelta(days=retention_days)
        removed = 0
        for f in self._dir.glob("*.parquet"):
            file_date = _date_from_stem(f.stem)
            if file_date is None:
                continue
            if file_date < cutoff:
                f.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_window(self, days: int = 30) -> pd.DataFrame:
        """
        Return a DataFrame of the last `days` distinct data-days in the archive.

        `days` is a count of trading days present in the archive, not a calendar
        span. If fewer than `days` data-days exist, all available data is returned.
        When multiple files share the same date (multiple runs), the latest
        timestamp wins and earlier files for that date are ignored.

        Returns an empty DataFrame (with correct columns) if no data is present.
        """
        # latest file per date
        by_date: dict[date, Path] = {}
        for f in self._dir.glob("*.parquet"):
            d = _date_from_stem(f.stem)
            if d is not None:
                if d not in by_date or f.stem > by_date[d].stem:
                    by_date[d] = f

        sorted_dates = sorted(by_date)
        recent = sorted_dates[-days:] if len(sorted_dates) > days else sorted_dates

        frames = [pd.read_parquet(by_date[d]) for d in recent]
        if not frames:
            return pd.DataFrame(columns=_COLS + ["next_day_actual"])
        return pd.concat(frames, ignore_index=True)

    def available_days(self) -> int:
        """Count of distinct trading days currently in the archive."""
        return len({
            _date_from_stem(f.stem)
            for f in self._dir.glob("*.parquet")
            if _date_from_stem(f.stem) is not None
        })

    # ------------------------------------------------------------------
    # Regime health
    # ------------------------------------------------------------------

    def compute_snr(
        self,
        regime_name: str,
        days: int = 30,
    ) -> dict:
        """
        Compute out-of-sample R² and SNR for a regime over a rolling window.

        Uses the delta (residual) and actual columns from the archive.
        Only rows where regime == regime_name contribute.

        Formula:
            r2  = 1 - Var(delta) / Var(actual)
            snr = r2 / (1 - r2)

        Returns dict with keys: regime, r2, snr, n_days, n_stocks, status.
        Returns status='NO DATA' when the regime has no archive rows.

        Example:
            result = archive.compute_snr('ukraine_shock', days=30)
            # {'regime': 'ukraine_shock', 'r2': 0.41, 'snr': 0.70,
            #  'n_days': 22, 'n_stocks': 66, 'status': 'DEGRADED'}
        """
        df = self.load_window(days)
        rdf = df[df["regime"] == regime_name]

        n_days = int(rdf["date"].nunique())
        n_stocks = int(rdf["ticker"].nunique()) if n_days > 0 else 0

        if n_days == 0:
            return {"regime": regime_name, "r2": None, "snr": None,
                    "n_days": 0, "n_stocks": 0, "status": "NO DATA"}

        var_actual = float(rdf["actual"].var())
        var_delta  = float(rdf["delta"].var())

        if var_actual < 1e-10:
            r2 = 0.0
        else:
            r2 = max(-1.0, min(1.0, 1.0 - var_delta / var_actual))

        if r2 >= 1.0:
            snr = float("inf")
        elif r2 <= 0.0:
            snr = 0.0
        else:
            snr = r2 / (1.0 - r2)

        if snr >= SNR_GOOD_THRESHOLD:
            status = "ACTIVE"
        elif snr >= SNR_WARN_THRESHOLD:
            status = "DEGRADED"
        else:
            status = "POOR"

        return {
            "regime":   regime_name,
            "r2":       round(r2, 3),
            "snr":      round(snr, 2),
            "n_days":   n_days,
            "n_stocks": n_stocks,
            "status":   status,
        }

    def regime_competition(self, days: int = 30) -> list[dict]:
        """
        Compute SNR for every regime in REGIME_REGISTRY.

        Regimes with archive data are ranked by R² descending.
        Regimes with no archive data appear last with status='NO DATA'.
        The top-ranked regime with actual data is labelled 'BEST FIT'.

        Example:
            for r in archive.regime_competition(days=30):
                print(r['regime'], r['r2'], r['status'])
        """
        from shockarb.regimes import REGIME_REGISTRY
        results = [self.compute_snr(name, days) for name in REGIME_REGISTRY]
        ranked = sorted(
            results,
            key=lambda x: (x["r2"] is not None, x["r2"] or 0.0),
            reverse=True,
        )
        # Mark the top result that has actual data as BEST FIT
        for r in ranked:
            if r["r2"] is not None:
                r["status"] = "BEST FIT"
                break
        return ranked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


    def _path_for(self, d: date) -> Path:
        ts = _dt.datetime.now().strftime("%H%M%S")
        return self._dir / f"{d.isoformat()}_{ts}.parquet"

    @staticmethod
    def _date_from_path(p: Path) -> date | None:
        """Extract date from YYYY-MM-DD or YYYY-MM-DD_HHMMSS stem. Returns None if invalid."""
        return _date_from_stem(p.stem)

    def _backfill_yesterday(self, today: date, scores_df: pd.DataFrame) -> None:
        """
        Write today's actual returns into the most recent prior archive file
        as next_day_actual. Uses most-recent-prior rather than today-1 so that
        a Monday run correctly backfills Friday across weekends and holidays.
        The only write-twice pattern in the archive; t+1 returns are unavailable
        until the next scoring run. All values from scores_df (datamgr) -- no fetch.
        """
        # most-recent-prior rather than today-1: handles weekends and missed days
        prior = sorted(
            f for f in self._dir.glob("*.parquet")
            if _date_from_stem(f.stem) is not None and _date_from_stem(f.stem) < today
        )
        if not prior:
            return
        ypath = prior[-1]
        ydf = pd.read_parquet(ypath)
        # Values already in memory from datamgr -- no new fetch
        actual_map = scores_df["actual_return"].rename("next_day_actual")
        ydf = ydf.set_index("ticker")
        ydf["next_day_actual"] = ydf.index.map(actual_map)
        ydf = ydf.reset_index()
        ydf.to_parquet(ypath, index=False)


def _is_valid_date_stem(stem: str) -> bool:
    return _date_from_stem(stem) is not None


def _date_from_stem(stem: str) -> date | None:
    """Parse date from YYYY-MM-DD or YYYY-MM-DD_HHMMSS stem. Returns None if invalid."""
    try:
        return date.fromisoformat(stem[:10])
    except ValueError:
        return None
