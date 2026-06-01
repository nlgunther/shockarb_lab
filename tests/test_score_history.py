"""
Unit tests for shockarb.score_history — ScoreArchive.

All tests are fully offline and use temporary directories.

Coverage areas
--------------
  TestScoreArchiveInit       — auto-creates recent_scores dir
  TestSaveRow                — file created, correct schema, correct row count
  TestLoadWindow             — returns rows within window, excludes older files
  TestPurgeStale             — deletes old files, keeps recent, returns count
  TestBackfill               — save_row backfills next_day_actual into most-recent-prior (handles weekends)
  TestAvailableDays          — counts distinct days, ignores non-date files
  TestEmptyWindow            — load_window returns empty DF with correct columns
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shockarb.score_history import (
    ScoreArchive,
    MIN_WINDOW_DAYS,
    RECENT_SCORES_RETENTION_DAYS,
    _COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TICKERS = ["AAPL", "MSFT", "GOOGL", "V", "UNH"]


def _make_scores(tickers=_TICKERS, seed=0) -> pd.DataFrame:
    """
    Minimal engine-output DataFrame: index = ticker, columns match what
    ScoreArchive.save_row() extracts (actual_return, expected_rel, delta_rel,
    r_squared, confidence_delta).
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    actual = rng.normal(0, 0.01, n)
    expected = rng.normal(0, 0.008, n)
    delta = expected - actual
    r2 = rng.uniform(0.3, 0.9, n)
    return pd.DataFrame(
        {
            "actual_return":    actual,
            "expected_rel":     expected,
            "expected_abs":     expected + rng.normal(0, 0.001, n),
            "delta_rel":        delta,
            "delta_abs":        delta + rng.normal(0, 0.001, n),
            "r_squared":        r2,
            "residual_vol":     rng.uniform(0.1, 0.4, n),
            "confidence_delta": delta * r2,
        },
        index=pd.Index(tickers, name="ticker"),
    )


def _find_file_for_date(archive, d):
    """Return the (latest-timestamp) parquet file for date d, or None."""
    candidates = sorted(
        f for f in archive._dir.glob("*.parquet")
        if f.stem.startswith(d.isoformat())
    )
    return candidates[-1] if candidates else None



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def archive(tmp_path) -> ScoreArchive:
    return ScoreArchive(tmp_path)


@pytest.fixture
def today() -> date:
    return date(2026, 5, 30)


@pytest.fixture
def scores() -> pd.DataFrame:
    return _make_scores()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScoreArchiveInit:
    def test_creates_recent_scores_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        assert not (data_dir / "recent_scores").exists()
        ScoreArchive(data_dir)
        assert (data_dir / "recent_scores").is_dir()

    def test_idempotent_on_existing_dir(self, tmp_path):
        """Constructing twice must not raise."""
        ScoreArchive(tmp_path)
        ScoreArchive(tmp_path)  # should not raise


class TestSaveRow:
    def test_creates_parquet_file(self, archive, today, scores):
        path = archive.save_row(today, scores, "ukraine_shock", "ukraine_shock_us_20260528.json")
        assert path.exists()
        assert path.suffix == ".parquet"
        assert path.stem.startswith(today.isoformat())

    def test_row_count_matches_tickers(self, archive, today, scores):
        path = archive.save_row(today, scores, "ukraine_shock", "ukraine_shock_us_20260528.json")
        df = pd.read_parquet(path)
        assert len(df) == len(_TICKERS)

    def test_schema(self, archive, today, scores):
        path = archive.save_row(today, scores, "ukraine_shock", "ukraine_shock_us_20260528.json")
        df = pd.read_parquet(path)
        required = {"date", "ticker", "actual", "expected", "delta", "r2",
                    "conf_delta", "regime", "model_file", "next_day_actual"}
        assert required.issubset(set(df.columns))

    def test_model_file_basename_only(self, archive, today, scores):
        # Full path should be stripped to basename
        path = archive.save_row(
            today, scores, "ukraine_shock",
            "/some/long/path/ukraine_shock_us_20260528.json"
        )
        df = pd.read_parquet(path)
        assert df["model_file"].iloc[0] == "ukraine_shock_us_20260528.json"

    def test_regime_name_stored(self, archive, today, scores):
        archive.save_row(today, scores, "liberation_day_recovery", "model.json")
        f = _find_file_for_date(archive, today)
        df = pd.read_parquet(f)
        assert (df["regime"] == "liberation_day_recovery").all()

    def test_values_match_scores_df(self, archive, today, scores):
        archive.save_row(today, scores, "ukraine_shock", "model.json")
        df = pd.read_parquet(_find_file_for_date(archive, today))
        df_indexed = df.set_index("ticker")
        for ticker in _TICKERS:
            assert math.isclose(
                df_indexed.loc[ticker, "actual"],
                scores.loc[ticker, "actual_return"],
                rel_tol=1e-6,
            )
            assert math.isclose(
                df_indexed.loc[ticker, "conf_delta"],
                scores.loc[ticker, "confidence_delta"],
                rel_tol=1e-6,
            )


class TestLoadWindow:
    def test_returns_all_rows_within_window(self, archive):
        base = date(2026, 5, 25)
        for offset in range(5):
            d = base + timedelta(days=offset)
            archive.save_row(d, _make_scores(seed=offset), "ukraine_shock", "m.json")
        df = archive.load_window(days=10)
        assert len(df) == 5 * len(_TICKERS)

    def test_load_window_returns_last_n_data_days(self, archive):
        """load_window(days=N) returns the N most recent data-days by count, not calendar span."""
        dates = [date(2026, 1, 1), date(2026, 2, 15), date(2026, 3, 20),
                 date(2026, 4, 10), date(2026, 5, 30)]
        for i, d in enumerate(dates):
            archive.save_row(d, _make_scores(seed=i), "ukraine_shock", "m.json")
        # Ask for last 3 data-days — should get the 3 most recent regardless of calendar gap
        df = archive.load_window(days=3)
        result_dates = set(df["date"].unique())
        assert date(2026, 3, 20).isoformat() in result_dates
        assert date(2026, 4, 10).isoformat() in result_dates
        assert date(2026, 5, 30).isoformat() in result_dates
        assert date(2026, 2, 15).isoformat() not in result_dates
        assert date(2026, 1, 1).isoformat() not in result_dates

    def test_returns_dataframe_with_correct_columns(self, archive):
        archive.save_row(date(2026, 5, 30), _make_scores(), "ukraine_shock", "m.json")
        df = archive.load_window(days=30)
        required = {"date", "ticker", "actual", "expected", "delta", "r2",
                    "conf_delta", "regime", "model_file"}
        assert required.issubset(set(df.columns))


class TestEmptyWindow:
    def test_empty_archive_returns_empty_df(self, archive):
        df = archive.load_window(days=30)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_df_has_correct_columns(self, archive):
        df = archive.load_window(days=30)
        required = set(_COLS) | {"next_day_actual"}
        assert required.issubset(set(df.columns))


class TestPurgeStale:
    def test_removes_old_files(self, archive):
        old = date(2020, 1, 1)
        recent = date.today()
        archive.save_row(old, _make_scores(seed=0), "ukraine_shock", "m.json")
        archive.save_row(recent, _make_scores(seed=1), "ukraine_shock", "m.json")
        removed = archive.purge_stale(retention_days=90)
        assert removed == 1
        assert _find_file_for_date(archive, old) is None
        assert _find_file_for_date(archive, recent) is not None

    def test_keeps_files_within_retention(self, archive):
        recent = date.today() - timedelta(days=10)
        archive.save_row(recent, _make_scores(), "ukraine_shock", "m.json")
        removed = archive.purge_stale(retention_days=90)
        assert removed == 0
        assert _find_file_for_date(archive, recent) is not None

    def test_returns_count_of_removed_files(self, archive):
        for year in [2019, 2020, 2021]:
            archive.save_row(date(year, 1, 1), _make_scores(), "ukraine_shock", "m.json")
        removed = archive.purge_stale(retention_days=90)
        assert removed == 3

    def test_ignores_non_date_files(self, archive):
        # A stray file with a non-date stem should not cause an error
        (archive._dir / "README.txt").write_text("ignore me")
        removed = archive.purge_stale(retention_days=0)
        assert (archive._dir / "README.txt").exists()  # untouched


class TestBackfill:
    def test_backfills_next_day_actual_into_yesterday(self, archive):
        yesterday = date(2026, 5, 29)
        today = date(2026, 5, 30)

        scores_yesterday = _make_scores(seed=0)
        scores_today = _make_scores(seed=1)

        archive.save_row(yesterday, scores_yesterday, "ukraine_shock", "m.json")
        archive.save_row(today, scores_today, "ukraine_shock", "m.json")

        # Yesterday's file should now have next_day_actual filled for tickers
        # present in both days.
        ydf = pd.read_parquet(_find_file_for_date(archive, yesterday))
        ydf = ydf.set_index("ticker")

        for ticker in _TICKERS:
            expected_val = scores_today.loc[ticker, "actual_return"]
            assert math.isclose(
                ydf.loc[ticker, "next_day_actual"],
                expected_val,
                rel_tol=1e-6,
            ), f"Backfill mismatch for {ticker}"

    def test_backfill_noop_when_no_yesterday_file(self, archive):
        """save_row on the first day should not raise even without a prior file."""
        today = date(2026, 5, 30)
        archive.save_row(today, _make_scores(), "ukraine_shock", "m.json")
        # No exception = pass

    def test_backfills_across_weekend(self, archive):
        """Monday's save_row must backfill Friday, not Sunday (no file exists)."""
        friday = date(2026, 5, 29)   # Friday
        monday = date(2026, 6, 1)    # Monday (gap: Sat + Sun)

        scores_friday = _make_scores(seed=10)
        scores_monday = _make_scores(seed=11)

        archive.save_row(friday, scores_friday, "ukraine_shock", "m.json")
        archive.save_row(monday, scores_monday, "ukraine_shock", "m.json")

        friday_df = pd.read_parquet(_find_file_for_date(archive, friday))
        friday_df = friday_df.set_index("ticker")

        for ticker in _TICKERS:
            assert math.isclose(
                friday_df.loc[ticker, "next_day_actual"],
                scores_monday.loc[ticker, "actual_return"],
                rel_tol=1e-6,
            ), f"Weekend backfill mismatch for {ticker}"

    def test_backfill_skips_saturday_file(self, archive):
        """Saturday files (if they somehow exist) are handled without error."""
        friday = date(2026, 5, 29)
        saturday = date(2026, 5, 30)
        monday = date(2026, 6, 1)

        archive.save_row(friday, _make_scores(seed=20), "ukraine_shock", "m.json")
        archive.save_row(saturday, _make_scores(seed=21), "ukraine_shock", "m.json")
        archive.save_row(monday, _make_scores(seed=22), "ukraine_shock", "m.json")

        # Saturday's file should be backfilled by Monday (most recent prior)
        sat_df = pd.read_parquet(_find_file_for_date(archive, saturday))
        assert not sat_df["next_day_actual"].isna().all()

    def test_today_next_day_actual_is_nan(self, archive):
        today = date(2026, 5, 30)
        path = archive.save_row(today, _make_scores(), "ukraine_shock", "m.json")
        df = pd.read_parquet(path)
        # next_day_actual should be NaN until tomorrow's run backfills it
        assert df["next_day_actual"].isna().all()


class TestAvailableDays:
    def test_counts_distinct_days(self, archive):
        for offset in range(3):
            archive.save_row(
                date(2026, 5, 28) + timedelta(days=offset),
                _make_scores(),
                "ukraine_shock", "m.json",
            )
        assert archive.available_days() == 3

    def test_zero_when_empty(self, archive):
        assert archive.available_days() == 0

    def test_ignores_non_date_parquet_files(self, archive):
        (archive._dir / "not_a_date.parquet").write_bytes(b"")
        assert archive.available_days() == 0



class TestComputeSNR:
    def test_perfect_model_gives_r2_one(self, archive):
        """When delta is zero for all rows, R² should be 1.0."""
        d = date(2026, 5, 28)
        scores = _make_scores(seed=5)
        # Force delta=0 by making expected == actual
        scores["delta_rel"] = 0.0
        scores["confidence_delta"] = 0.0
        archive.save_row(d, scores, "ukraine_shock", "m.json")
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["r2"] == 1.0
        assert result["status"] == "ACTIVE"  # BEST FIT is only set by regime_competition()

    def test_random_model_gives_low_r2(self, archive):
        """When delta is as large as actual, R² should be near zero or negative."""
        rng = np.random.default_rng(99)
        n = len(_TICKERS)
        scores = _make_scores(seed=7)
        # Make delta independent noise — same variance as actual → R²≈0
        scores["delta_rel"] = rng.normal(0, scores["actual_return"].std(), n)
        scores["confidence_delta"] = scores["delta_rel"] * scores["r_squared"]
        archive.save_row(date(2026, 5, 28), scores, "ukraine_shock", "m.json")
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["r2"] is not None
        assert result["r2"] <= 0.5

    def test_no_data_returns_no_data_status(self, archive):
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["r2"] is None
        assert result["snr"] is None
        assert result["n_days"] == 0
        assert result["status"] == "NO DATA"

    def test_filters_by_regime_name(self, archive):
        """Rows from a different regime must not pollute the SNR calculation."""
        scores_a = _make_scores(seed=1)
        scores_b = _make_scores(seed=2)
        archive.save_row(date(2026, 5, 27), scores_a, "ukraine_shock", "m.json")
        archive.save_row(date(2026, 5, 28), scores_b, "gulf_war_recovery", "m2.json")
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["n_days"] == 1
        result2 = archive.compute_snr("gulf_war_recovery", days=30)
        assert result2["n_days"] == 1

    def test_n_days_and_n_stocks_populated(self, archive):
        for i in range(3):
            archive.save_row(
                date(2026, 5, 26) + timedelta(days=i),
                _make_scores(seed=i), "ukraine_shock", "m.json",
            )
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["n_days"] == 3
        assert result["n_stocks"] == len(_TICKERS)

    def test_snr_thresholds_active(self, archive):
        """SNR >= SNR_GOOD_THRESHOLD → ACTIVE (or BEST FIT after competition)."""
        from shockarb.score_history import SNR_GOOD_THRESHOLD
        scores = _make_scores(seed=3)
        # Zero delta → R²=1, SNR=inf → should be ACTIVE
        scores["delta_rel"] = 0.0
        scores["confidence_delta"] = 0.0
        archive.save_row(date(2026, 5, 28), scores, "ukraine_shock", "m.json")
        result = archive.compute_snr("ukraine_shock", days=30)
        assert result["status"] in ("ACTIVE", "BEST FIT")

    def test_multi_day_window(self, archive):
        for i in range(5):
            archive.save_row(
                date(2026, 5, 25) + timedelta(days=i),
                _make_scores(seed=i), "ukraine_shock", "m.json",
            )
        result30 = archive.compute_snr("ukraine_shock", days=30)
        result2  = archive.compute_snr("ukraine_shock", days=2)
        assert result30["n_days"] >= result2["n_days"]


class TestRegimeCompetition:
    def _populate(self, archive, regime_name: str, n_days: int = 3, seed_offset: int = 0):
        for i in range(n_days):
            archive.save_row(
                date(2026, 5, 26) + timedelta(days=i),
                _make_scores(seed=i + seed_offset),
                regime_name, "m.json",
            )

    def test_returns_list_of_dicts(self, archive):
        self._populate(archive, "ukraine_shock")
        results = archive.regime_competition(days=30)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_all_registered_regimes_present(self, archive):
        from shockarb.regimes import REGIME_REGISTRY
        self._populate(archive, "ukraine_shock")
        results = archive.regime_competition(days=30)
        names = {r["regime"] for r in results}
        assert names == set(REGIME_REGISTRY.keys())

    def test_ranked_by_r2_descending(self, archive):
        """Regimes with data should appear before NO DATA regimes, sorted by R²."""
        self._populate(archive, "ukraine_shock")
        self._populate(archive, "gulf_war_recovery", seed_offset=10)
        results = archive.regime_competition(days=30)
        r2_values = [r["r2"] for r in results if r["r2"] is not None]
        assert r2_values == sorted(r2_values, reverse=True)

    def test_no_data_regimes_appear_last(self, archive):
        self._populate(archive, "ukraine_shock")
        results = archive.regime_competition(days=30)
        has_data = [r for r in results if r["r2"] is not None]
        no_data  = [r for r in results if r["r2"] is None]
        # All has_data entries come before no_data entries
        if has_data and no_data:
            last_data_idx  = max(results.index(r) for r in has_data)
            first_nodata_idx = min(results.index(r) for r in no_data)
            assert last_data_idx < first_nodata_idx

    def test_best_fit_label_on_top_result(self, archive):
        self._populate(archive, "ukraine_shock")
        results = archive.regime_competition(days=30)
        top = next((r for r in results if r["r2"] is not None), None)
        assert top is not None
        assert top["status"] == "BEST FIT"

    def test_empty_archive_all_no_data(self, archive):
        results = archive.regime_competition(days=30)
        assert all(r["status"] == "NO DATA" for r in results)

class TestConstants:
    def test_retention_days_is_positive(self):
        assert RECENT_SCORES_RETENTION_DAYS > 0

    def test_min_window_days_is_positive(self):
        assert MIN_WINDOW_DAYS > 0
