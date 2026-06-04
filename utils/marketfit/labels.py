"""
marketfit.labels — Build labeled training rows from score archive + forward prices.

NOT YET IMPLEMENTED — stub only. The rule engine (rules.py) is the active
primary path. This module is a placeholder for the ML training pipeline.

Label design (agreed 2026-06-03)
----------------------------------
For each archived trading day T:
  1. Take top-N positive-confidence_delta picks (default N=5, r²>0.50).
  2. entry  = close[T]
     forward = close[T+h]  (default h=5 trading days, via yf.download Adj Close)
  3. For each pick:
       realized_return = (forward - entry) / entry
       gap_recovery    = realized_return / delta_rel
       (measures how much of the factor-implied gap closed, normalized to signal size)
  4. outcome = mean gap_recovery across picks for day T
  5. y = 1 if outcome >= TAU (default 0.5), else 0
       (TAU=0.5 means picks closed ≥50% of their implied gap on average)

This two-part label captures:
  (a) Gap closure: did stocks revert toward their model-implied price?
  (b) Model drift: did the market environment sustain rising factor-implied prices?

Constants to tune (document changes here):
  N     = 5       top picks per day
  H     = 5       forward horizon in trading days
  TAU   = 0.5     gap-recovery threshold for y=1
"""

from __future__ import annotations


N   = 5
H   = 5
TAU = 0.5


def build_training_rows(
    archive_dir: str,
    snapshot_dir: str,
    out_path: str,
) -> None:
    """
    Build and cache labeled training rows to out_path (parquet).

    NOT YET IMPLEMENTED.

    Parameters
    ----------
    archive_dir  : path to data/recent_scores/ (ScoreArchive parquet files)
    snapshot_dir : path to directory of historical market_snapshot.json files
    out_path     : destination parquet path (append-only, dedupe by date)
    """
    raise NotImplementedError(
        "labels.build_training_rows is not yet implemented. "
        "The rule engine (rules.evaluate) is the active primary path. "
        "Implement this once ScoreArchive has ≥30 days of persisted signals."
    )
