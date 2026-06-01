"""
BackfillArchive — ScoreArchive adapter for the weekly backfill CSVs.

Presents data from data/backfill_us/ as if it were the daily parquet archive
in data/recent_scores/, giving full access to ScoreArchive's regime health
API (compute_snr, regime_competition, load_window) without writing anything
to the live archive.

The backfill CSVs are weekly (one per week), so the "days" parameter in
load_window/compute_snr counts distinct weekly data-points, not calendar days.

Usage
-----
    from utils.backfill_analysis import BackfillArchive

    archive = BackfillArchive("data/backfill_us")

    # Same API as ScoreArchive
    print(archive.available_days())
    df = archive.load_window(days=52)       # last 52 weekly data-points
    snr = archive.compute_snr("ukraine_shock", days=78)
    for r in archive.regime_competition(days=78):
        print(r["regime"], r["r2"], r["status"])

    # Convenience: print a formatted health table (like the CLI command)
    archive.print_health(days=78)

CLI
---
    python utils/backfill_analysis.py
    python utils/backfill_analysis.py --days 26   # last 6 months only
    python utils/backfill_analysis.py --dir data/backfill_us
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ScoreArchive lives in shockarb package — add project root to path if needed
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shockarb.score_history import ScoreArchive, _COLS


# Column mapping: backfill CSV → archive format
_COL_MAP = {
    "actual_return":    "actual",
    "expected_rel":     "expected",
    "delta_rel":        "delta",
    "r_squared":        "r2",
    "confidence_delta": "conf_delta",
}


class BackfillArchive(ScoreArchive):
    """
    ScoreArchive subclass that reads from weekly backfill CSVs.

    Overrides load_window() and available_days() so all inherited methods
    (compute_snr, regime_competition) work transparently.  Nothing is written
    to data/recent_scores/.

    Parameters
    ----------
    backfill_dir : str or Path
        Directory containing alpha_YYYY-MM-DD.csv files.
    regime : str
        Regime name to tag all rows with. Default: "ukraine_shock".
    data_dir : str or Path, optional
        Passed to ScoreArchive.__init__ but the live archive is never read.
        Defaults to a non-existent temp path so no real archive data leaks in.
    """

    def __init__(
        self,
        backfill_dir: str | Path = "data/backfill_us",
        regime: str = "ukraine_shock",
        data_dir: str | Path | None = None,
    ) -> None:
        # Point parent at a harmless path — we override the read methods anyway
        super().__init__(data_dir or Path(backfill_dir) / "_unused_archive")
        self._backfill_dir = Path(backfill_dir)
        self._regime = regime

    # ------------------------------------------------------------------
    # Override: read from CSV backfill, not parquet archive
    # ------------------------------------------------------------------

    def load_window(self, days: int = 30) -> pd.DataFrame:
        """
        Return a DataFrame of the last `days` weekly data-points from the
        backfill directory.  Column layout matches the ScoreArchive format.
        """
        files = self._sorted_backfill_files()
        recent = files[-days:] if len(files) > days else files

        frames = []
        for f in recent:
            try:
                df = self._load_csv_as_archive_row(f)
                frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame(columns=_COLS + ["next_day_actual"])
        return pd.concat(frames, ignore_index=True)

    def available_days(self) -> int:
        """Count of weekly data-points in the backfill directory."""
        return len(self._sorted_backfill_files())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_health(self, days: int | None = None) -> None:
        """Print a formatted regime health table (mirrors the CLI output)."""
        n = days or self.available_days()
        results = self.regime_competition(days=n)
        print(f"\n  REGIME HEALTH — backfill data ({n}-week window)")
        print(f"  {'─'*56}")
        for r in results:
            if r["r2"] is None:
                print(f"  {r['regime']:<30}  ——  NO DATA")
            else:
                icon = "✅" if r["status"] in ("ACTIVE", "BEST FIT") else ("⚠️" if r["status"] == "DEGRADED" else "❌")
                print(f"  {r['regime']:<30}  R²={r['r2']:.2f}  SNR={r['snr']:.2f}  {icon}  {r['status']}")
        best = next((r for r in results if r["r2"] is not None), None)
        if best:
            print(f"\n  → Best fit over {n} weeks: {best['regime']}")
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sorted_backfill_files(self) -> list[Path]:
        """Return backfill CSV files sorted chronologically."""
        files = sorted(
            f for f in self._backfill_dir.glob("alpha_*.csv")
        )
        return files

    def _load_csv_as_archive_row(self, csv_path: Path) -> pd.DataFrame:
        """Load one backfill CSV and return it in archive column format."""
        date_str = csv_path.stem.replace("alpha_", "")
        df = pd.read_csv(csv_path, index_col=0)

        # Rename to archive column names
        df = df.rename(columns=_COL_MAP)

        # Keep only archive columns that exist in the CSV
        keep = [c for c in _COL_MAP.values() if c in df.columns]
        df = df[keep].copy()

        df.index.name = "ticker"
        df = df.reset_index()

        df["date"]           = date_str
        df["regime"]         = self._regime
        df["model_file"]     = "backfill"
        df["next_day_actual"] = float("nan")

        # Ensure all archive columns are present
        for col in _COLS + ["next_day_actual"]:
            if col not in df.columns:
                df[col] = float("nan")

        return df[_COLS + ["next_day_actual"]]


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute regime health from backfill CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dir",  default="data/backfill_us",
                        help="Backfill CSV directory (default: data/backfill_us)")
    parser.add_argument("--days", type=int, default=None,
                        help="Number of most-recent weekly data-points to use (default: all)")
    parser.add_argument("--regime", default="ukraine_shock",
                        help="Regime tag for SNR filter (default: ukraine_shock)")
    args = parser.parse_args()

    archive = BackfillArchive(backfill_dir=args.dir, regime=args.regime)
    n = archive.available_days()
    print(f"Backfill directory: {args.dir}")
    print(f"Data-points available: {n} weeks")

    days = args.days or n
    archive.print_health(days=days)
