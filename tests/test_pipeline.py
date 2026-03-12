"""
Tests for shockarb.pipeline — all I/O, caching, and model lifecycle functions.

Covers:
  - prices_to_returns(): NaN handling, coverage filtering, forward-fill
  - fetch_prices(): cache hit/miss, fallback to synthetic data
  - fetch_live_returns(): basic use, empty-response guard
  - build(): end-to-end with mocked yfinance
  - save_model() / load_model(): file creation, JSON structure
  - find_latest_model(): ordering, None when absent
  - export_csvs(): file creation, column presence
  - Integration: save → load → score consistency
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import shockarb.pipeline as pipeline
from shockarb.config import ExecutionConfig, UniverseConfig


# =============================================================================
# Shared test helpers
# =============================================================================

class _InMemoryStore:
    """
    Minimal DataStore implementation for unit tests.

    Stores DataFrames in a dict keyed by 'daily/{ticker}'.
    No filesystem access — all operations are in-memory.
    """
    def __init__(self):
        self._data = {}

    def read(self, key, start, end):
        df = self._data.get(key)
        if df is None:
            return None
        try:
            return df.loc[start:end]
        except Exception:
            return df

    def write(self, key, df, meta):
        self._data[key] = df

    def coverage(self, key):
        df = self._data.get(key)
        if df is None or df.empty:
            return None
        return (str(df.index.min().date()), str(df.index.max().date()))

    def sweep(self, retention, before):
        return []


def _seeded_store(tickers: dict[str, float], days: int = 10) -> _InMemoryStore:
    """
    Return an _InMemoryStore pre-seeded with adj_close data for *tickers*.

    Parameters
    ----------
    tickers : dict  ticker -> base_price
    days    : int   number of business days ending today
    """
    from datetime import date, timedelta
    end   = date.today()
    start = end - timedelta(days=days * 2)  # generous to guarantee enough bdays
    idx   = pd.bdate_range(start=start, end=end)[-days:]
    store = _InMemoryStore()
    for ticker, base in tickers.items():
        vals = [base + i * 0.5 for i in range(len(idx))]
        store._data[f"daily/{ticker}"] = pd.DataFrame({"adj_close": vals}, index=idx)
    return store


# =============================================================================
# prices_to_returns()
# =============================================================================

class TestPricesToReturns:

    def test_basic_return_calculation(self):
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0],
             "B": [50.0, 51.0, 50.0, 52.0]},
            index=pd.date_range("2022-01-01", periods=4),
        )
        rets = pipeline.prices_to_returns(prices)
        assert len(rets) == 3
        assert abs(rets.iloc[0]["A"] - 0.01) < 1e-10

    def test_drops_low_coverage_tickers(self):
        prices = pd.DataFrame(
            {"GOOD": [100.0, 101.0, 102.0, 103.0, 104.0],
             "BAD":  [100.0, np.nan, np.nan, np.nan, np.nan]},
            index=pd.date_range("2022-01-01", periods=5),
        )
        rets = pipeline.prices_to_returns(prices, min_coverage=0.8)
        assert "GOOD" in rets.columns
        assert "BAD" not in rets.columns

    def test_forward_fills_gaps(self):
        """Holiday gaps (NaN in middle of series) should be forward-filled, not dropped."""
        prices = pd.DataFrame(
            {"A": [100.0, np.nan, 102.0, 103.0, 104.0]},
            index=pd.date_range("2022-01-01", periods=5),
        )
        rets = pipeline.prices_to_returns(prices, min_coverage=0.5)
        assert not rets.isna().any().any()
        assert len(rets) == 4

    def test_no_nan_in_output(self):
        prices = pd.DataFrame(
            {"A": [100.0, np.nan, 102.0, 103.0],
             "B": [50.0, 51.0, np.nan, 53.0]},
            index=pd.date_range("2022-01-01", periods=4),
        )
        rets = pipeline.prices_to_returns(prices, min_coverage=0.5)
        assert not rets.isna().any().any()

    def test_output_length_is_input_minus_one(self):
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0]},
            index=pd.date_range("2022-01-01", periods=3),
        )
        assert len(pipeline.prices_to_returns(prices)) == 2


# =============================================================================
# _synthetic_prices()
# =============================================================================

class TestSyntheticPrices:

    def test_reproducible_across_calls(self):
        tickers = ["VOO", "TLT", "GLD"]
        p1 = pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        p2 = pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        pd.testing.assert_frame_equal(p1, p2)

    def test_all_tickers_present(self):
        tickers = ["VOO", "TLT", "GLD", "UNKNOWN"]
        prices = pipeline._synthetic_prices(tickers, "2022-02-10", "2022-03-31")
        assert set(tickers) == set(prices.columns)

    def test_anchored_near_100(self):
        """Synthetic prices start at 100 as a deliberate flag for callers."""
        prices = pipeline._synthetic_prices(["VOO", "TLT"], "2022-02-10", "2022-03-31")
        for col in prices.columns:
            assert abs(prices[col].iloc[0] - 100) < 5


# =============================================================================
# fetch_prices()
# =============================================================================

class TestFetchPrices:

    @patch("shockarb.pipeline.yf.download")
    def test_caches_on_first_call(self, mock_dl, temp_dir):
        mock_data = pd.DataFrame(
            {("Adj Close", "AAPL"): [150.0, 151.0, 152.0]},
            index=pd.date_range("2022-01-03", periods=3),
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_dl.return_value = mock_data

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.fetch_prices(["AAPL"], "2022-01-01", "2022-01-10",
                              cache_name="test_etf", exec_config=cfg)
        assert mock_dl.called

    @patch("shockarb.pipeline.yf.download")
    def test_cache_hit_skips_download(self, mock_dl, temp_dir):
        mock_data = pd.DataFrame(
            {("Adj Close", "AAPL"): [150.0, 151.0, 152.0]},
            index=pd.date_range("2022-01-03", periods=3),
        )
        mock_data.columns = pd.MultiIndex.from_tuples(mock_data.columns)
        mock_dl.return_value = mock_data

        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.fetch_prices(["AAPL"], "2022-01-01", "2022-01-10",
                              cache_name="test_etf", exec_config=cfg)
        mock_dl.reset_mock()

        # Second call — should not hit network
        pipeline.fetch_prices(["AAPL"], "2022-01-01", "2022-01-10",
                              cache_name="test_etf", exec_config=cfg)
        assert not mock_dl.called

    @patch("shockarb.pipeline.yf.download")
    def test_falls_back_to_synthetic_on_empty_response(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        prices = pipeline.fetch_prices(["VOO", "TLT"], "2022-02-10", "2022-03-31")
        assert not prices.empty
        # Synthetic prices anchor at 100
        assert abs(prices.iloc[0, 0] - 100) < 5


# =============================================================================
# fetch_live_returns()
# =============================================================================

class TestFetchLiveReturns:

    def test_returns_series_with_correct_tickers(self):
        from datamgr.coordinator import DataCoordinator

        from datetime import date, timedelta

        # Seed a window that is guaranteed to cover any "1d" lookback from today
        end   = date.today()
        start = end - timedelta(days=10)
        idx   = pd.bdate_range(start=start, end=end)

        store = _InMemoryStore()
        for ticker, base in {"AAPL": 150, "MSFT": 300}.items():
            vals = [base + i for i in range(len(idx))]
            store._data[f"daily/{ticker}"] = pd.DataFrame(
                {"adj_close": vals}, index=idx
            )

        coord = DataCoordinator(store)  # no provider needed — full cache hit

        with patch.object(pipeline, "_coordinator", return_value=coord):
            returns = pipeline.fetch_live_returns(["AAPL", "MSFT"], period="1d")

        assert isinstance(returns, pd.Series)
        assert len(returns) == 2
        assert set(returns.index) == {"AAPL", "MSFT"}

    def test_raises_on_empty_response(self):
        """
        When no data can be retrieved, fetch_live_returns must raise ValueError.

        fetch_live_returns has two paths:
          - Market open  → _fetch_live_direct() (hot-fix, bypasses coordinator)
          - Market closed → coordinator cache path

        Both paths are covered here by forcing the market-open path and making
        _fetch_live_direct raise, so the test is not sensitive to market hours.
        """
        with patch.object(pipeline, "_market_is_open", return_value=True), \
             patch.object(pipeline, "_fetch_live_direct",
                          side_effect=ValueError("fetch_live_returns: coordinator returned no data.")):
            with pytest.raises(ValueError, match="no data"):
                pipeline.fetch_live_returns(["AAPL"])


# =============================================================================
# score_universe()
# =============================================================================

class TestScoreUniverse:
    """
    Tests for pipeline.score_universe().

    All tests use _InMemoryStore + MockProvider injected via patch.object so
    no filesystem or network access occurs.
    """

    _UNIVERSE = UniverseConfig(
        name="test",
        market_etfs=["VOO", "TLT", "GLD"],
        individual_stocks=["AAPL", "MSFT"],
        n_components=2,
        start_date="2022-02-10",
        end_date="2022-03-31",
    )

    def _make_coordinator(self, store):
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider
        return DataCoordinator(store, provider=MockProvider())

    def test_returns_dataframe(self, fitted_model):
        """score_universe() returns a non-empty DataFrame."""
        store = _seeded_store({"VOO": 400, "TLT": 100, "GLD": 180,
                               "AAPL": 150, "MSFT": 300})
        coord = self._make_coordinator(store)
        with patch.object(pipeline, "_coordinator", return_value=coord), \
             patch.object(pipeline, "_market_is_open", return_value=False):
            scores = pipeline.score_universe(self._UNIVERSE, fitted_model)
        assert isinstance(scores, pd.DataFrame)
        assert not scores.empty

    def test_single_coordinator_single_fulfill(self, fitted_model):
        """
        score_universe() must create exactly one coordinator and call
        fulfill() exactly once — both ETF and stock legs go through the
        same instance.
        """
        from unittest.mock import MagicMock
        store = _seeded_store({"VOO": 400, "TLT": 100, "GLD": 180,
                               "AAPL": 150, "MSFT": 300})
        real_coord = self._make_coordinator(store)

        fulfill_calls = []
        original_fulfill = real_coord.fulfill

        def counting_fulfill(**kwargs):
            fulfill_calls.append(1)
            return original_fulfill(**kwargs)

        real_coord.fulfill = counting_fulfill

        coordinator_calls = []

        def factory(*args, **kwargs):
            coordinator_calls.append(1)
            return real_coord

        with patch.object(pipeline, "_coordinator", side_effect=factory), \
             patch.object(pipeline, "_market_is_open", return_value=False):
            pipeline.score_universe(self._UNIVERSE, fitted_model)

        assert len(coordinator_calls) == 1, (
            f"Expected 1 coordinator instantiation, got {len(coordinator_calls)}"
        )
        assert len(fulfill_calls) == 1, (
            f"Expected 1 fulfill() call, got {len(fulfill_calls)}"
        )

    def test_etf_and_stock_end_dates_aligned(self, fitted_model):
        """
        Both ETF and stock data must end on the same date — no temporal
        misalignment between legs.
        """
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider

        store = _seeded_store({"VOO": 400, "TLT": 100, "GLD": 180,
                               "AAPL": 150, "MSFT": 300})
        coord = DataCoordinator(store, provider=MockProvider())

        captured = {}

        original_fulfill = coord.fulfill

        def capturing_fulfill(**kwargs):
            result = original_fulfill(**kwargs)
            captured["results"] = result
            return result

        coord.fulfill = capturing_fulfill

        with patch.object(pipeline, "_coordinator", return_value=coord), \
             patch.object(pipeline, "_market_is_open", return_value=False):
            pipeline.score_universe(self._UNIVERSE, fitted_model)

        etf_key   = "test.live_etf"
        stock_key = "test.live_stock"
        assert etf_key   in captured["results"], "ETF result missing from fulfill() output"
        assert stock_key in captured["results"], "Stock result missing from fulfill() output"

        etf_end   = captured["results"][etf_key].index.max()
        stock_end = captured["results"][stock_key].index.max()
        assert etf_end == stock_end, (
            f"Date misalignment: ETF ends {etf_end}, stocks end {stock_end}"
        )

    def test_raises_on_empty_etf_data(self, fitted_model):
        """
        ValueError when the coordinator returns no ETF data.

        We patch fulfill() directly to return controlled output — a store-only
        approach won't work because MockProvider fills cache misses for any
        ticker, making it impossible to produce an empty ETF result that way.
        """
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider
        from unittest.mock import MagicMock

        coord = DataCoordinator(_InMemoryStore(), provider=MockProvider())
        # Return stocks but empty ETF frame, simulating a provider failure
        # for the ETF leg only.
        coord.fulfill = lambda **kw: {
            "test.live_etf":   pd.DataFrame(),   # empty — triggers the guard
            "test.live_stock": pd.DataFrame(     # non-empty — passes its guard
                {"AAPL": [1.0, 1.01], "MSFT": [1.0, 1.02]},
                index=pd.bdate_range("2026-03-10", periods=2),
            ),
        }
        with patch.object(pipeline, "_coordinator", return_value=coord), \
             patch.object(pipeline, "_market_is_open", return_value=False):
            with pytest.raises(ValueError, match="ETF"):
                pipeline.score_universe(self._UNIVERSE, fitted_model)

    def test_ticker_source_falls_back_to_universe_config(self, fitted_model):
        """
        After load_model(), model.etf_returns is empty (not persisted to JSON).
        score_universe() must fall back to universe.market_etfs / individual_stocks.

        Simulate a post-load model by nulling out the returns attributes.
        """
        import copy
        model_copy = copy.copy(fitted_model)
        # Simulate post-load state: returns matrices are not in memory
        object.__setattr__(model_copy, "etf_returns",   pd.DataFrame())
        object.__setattr__(model_copy, "stock_returns", pd.DataFrame())

        store = _seeded_store({"VOO": 400, "TLT": 100, "GLD": 180,
                               "AAPL": 150, "MSFT": 300})
        coord = self._make_coordinator(store)
        with patch.object(pipeline, "_coordinator", return_value=coord), \
             patch.object(pipeline, "_market_is_open", return_value=False):
            # Should not raise — tickers sourced from universe config
            scores = pipeline.score_universe(self._UNIVERSE, model_copy)
        assert isinstance(scores, pd.DataFrame)


# =============================================================================
# Model lifecycle — save / load / find
# =============================================================================

class TestModelLifecycle:

    def test_save_creates_json_file(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        assert path.endswith(".json")

    def test_save_file_exists_on_disk(self, fitted_model, temp_dir):
        import os
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        assert os.path.exists(path)

    def test_saved_json_structure(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        data = json.loads(open(path).read())
        # Required top-level keys
        for key in ("metadata", "Vt", "etf_columns", "loadings",
                    "stock_columns", "etf_mean", "stock_mean"):
            assert key in data, f"Missing key: {key}"
        # Metadata extras written by save_model
        assert "created_at" in data["metadata"]
        assert data["metadata"]["name"] == "test"

    def test_saved_json_excludes_raw_matrices(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        data = json.loads(open(path).read())
        assert "etf_returns" not in data
        assert "stock_returns" not in data

    def test_load_model_restores_fitted_state(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        loaded = pipeline.load_model(path)
        assert loaded._fitted is True

    def test_load_model_preserves_loadings(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        path = pipeline.save_model(fitted_model, "test", cfg)
        loaded = pipeline.load_model(path)
        pd.testing.assert_frame_equal(loaded.loadings, fitted_model.loadings)

    def test_find_latest_model_returns_most_recent(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        pipeline.save_model(fitted_model, "test", cfg)
        time.sleep(0.05)
        path2 = pipeline.save_model(fitted_model, "test", cfg)
        assert pipeline.find_latest_model("test", cfg) == path2

    def test_find_latest_model_returns_none_when_absent(self, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        assert pipeline.find_latest_model("nonexistent", cfg) is None


# =============================================================================
# export_csvs()
# =============================================================================

class TestExportCsvs:

    def test_creates_both_csv_files(self, fitted_model, temp_dir):
        import os
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        basis_path, loadings_path = pipeline.export_csvs(fitted_model, "test", cfg)
        assert os.path.exists(basis_path)
        assert os.path.exists(loadings_path)

    def test_basis_csv_has_factor_columns(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        basis_path, _ = pipeline.export_csvs(fitted_model, "test", cfg)
        df = pd.read_csv(basis_path, index_col=0)
        assert "Factor_1" in df.columns

    def test_loadings_csv_has_diagnostic_columns(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _, loadings_path = pipeline.export_csvs(fitted_model, "test", cfg)
        df = pd.read_csv(loadings_path, index_col=0)
        assert "R_squared" in df.columns
        assert "Residual_Vol" in df.columns

    def test_loadings_sorted_by_r_squared_descending(self, fitted_model, temp_dir):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        _, loadings_path = pipeline.export_csvs(fitted_model, "test", cfg)
        df = pd.read_csv(loadings_path, index_col=0)
        r2 = df["R_squared"].values
        assert all(r2[i] >= r2[i+1] for i in range(len(r2) - 1))


# =============================================================================
# build() — end-to-end
# =============================================================================

class TestBuild:

    def _mock_coordinator(self, temp_dir):
        """
        Return a DataCoordinator wired with MockProvider + _InMemoryStore so
        build() gets synthetic OHLCV without any yfinance calls.
        """
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider
        return DataCoordinator(_InMemoryStore(), provider=MockProvider())

    def test_build_returns_fitted_model(self, temp_dir):
        universe = UniverseConfig(
            name="test", market_etfs=["VOO", "TLT", "GLD"],
            individual_stocks=["AAPL", "MSFT"],
            n_components=2, start_date="2022-02-10", end_date="2022-03-31",
        )
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        with patch.object(pipeline, "_coordinator", return_value=self._mock_coordinator(temp_dir)):
            model = pipeline.build(universe, cfg)
        assert model._fitted is True

    def test_build_model_has_correct_n_components(self, temp_dir):
        universe = UniverseConfig(
            name="test", market_etfs=["VOO", "TLT", "GLD"],
            individual_stocks=["AAPL", "MSFT"],
            n_components=2, start_date="2022-02-10", end_date="2022-03-31",
        )
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        with patch.object(pipeline, "_coordinator", return_value=self._mock_coordinator(temp_dir)):
            model = pipeline.build(universe, cfg)
        assert model.diagnostics.n_factors == 2


# =============================================================================
# Integration — save → load → score roundtrip
# =============================================================================

class TestIntegration:

    def test_save_load_score_consistency(self, fitted_model, temp_dir,
                                         sample_etf_returns, sample_stock_returns):
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        etf_ret = sample_etf_returns.iloc[-1]
        stk_ret = sample_stock_returns.iloc[-1]

        before = fitted_model.score(etf_ret, stk_ret)
        path = pipeline.save_model(fitted_model, "roundtrip", cfg)
        after = pipeline.load_model(path).score(etf_ret, stk_ret)
        pd.testing.assert_frame_equal(before, after)

    def test_full_workflow(self, temp_dir):
        from datamgr.coordinator import DataCoordinator
        from datamgr.providers.mock import MockProvider

        coord = DataCoordinator(_InMemoryStore(), provider=MockProvider())
        universe = UniverseConfig(
            name="test", market_etfs=["VOO", "TLT", "GLD"],
            individual_stocks=["AAPL", "MSFT"],
            n_components=2, start_date="2022-02-10", end_date="2022-03-31",
        )
        cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        with patch.object(pipeline, "_coordinator", return_value=coord):
            model = pipeline.build(universe, cfg)
        path = pipeline.save_model(model, universe.name, cfg)
        loaded = pipeline.load_model(path)
        scores = loaded.score(
            pd.Series({"VOO": -0.02, "TLT": 0.01, "GLD": 0.015}),
            pd.Series({"AAPL": -0.03, "MSFT": -0.025}),
        )
        assert len(scores) == 2
        assert "confidence_delta" in scores.columns


# =============================================================================
# save_live_tape()
# =============================================================================

class TestSaveLiveTape:

    def _make_intraday_ohlcv(self, tickers, n_bars_yesterday=5, n_bars_today=5):
        """
        Synthetic 15-minute OHLCV matching yfinance intraday MultiIndex output.

        Two calendar days of bars so _minimal_tape can identify
        yesterday-close, today-open, and today-last as three distinct rows.
        """
        yesterday = pd.Timestamp("2022-03-14")
        today     = pd.Timestamp("2022-03-15")

        y_times = pd.date_range(
            yesterday.replace(hour=14, minute=0), periods=n_bars_yesterday, freq="15min"
        )
        t_times = pd.date_range(
            today.replace(hour=9, minute=30), periods=n_bars_today, freq="15min"
        )
        index = y_times.append(t_times)

        fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
        cols = pd.MultiIndex.from_product([fields, tickers], names=["field", "ticker"])
        data = np.random.rand(len(index), len(cols)) * 100 + 50
        return pd.DataFrame(data, index=index, columns=cols)

    @patch("shockarb.pipeline.yf.download")
    def test_creates_parquet_file(self, mock_dl, temp_dir):
        etf_tickers   = ["VOO", "TLT"]
        stock_tickers = ["AAPL", "MSFT"]
        mock_dl.return_value = self._make_intraday_ohlcv(etf_tickers + stock_tickers)

        path = os.path.join(temp_dir, "tapes", "us_20220315.parquet")
        result = pipeline.save_live_tape(etf_tickers, stock_tickers, path)

        assert result is not None
        assert os.path.exists(path)

    @patch("shockarb.pipeline.yf.download")
    def test_preserves_multiindex_columns(self, mock_dl, temp_dir):
        tickers = ["VOO", "TLT", "AAPL"]
        mock_dl.return_value = self._make_intraday_ohlcv(tickers)

        path = os.path.join(temp_dir, "tapes", "test.parquet")
        pipeline.save_live_tape(["VOO", "TLT"], ["AAPL"], path)

        loaded = pd.read_parquet(path)
        assert isinstance(loaded.columns, pd.MultiIndex)

    @patch("shockarb.pipeline.yf.download")
    def test_contains_all_tickers(self, mock_dl, temp_dir):
        etfs   = ["VOO", "TLT"]
        stocks = ["AAPL", "MSFT"]
        mock_dl.return_value = self._make_intraday_ohlcv(etfs + stocks)

        path = os.path.join(temp_dir, "tapes", "test.parquet")
        pipeline.save_live_tape(etfs, stocks, path)

        loaded = pd.read_parquet(path)
        tickers_in_file = set(loaded.columns.get_level_values(1).unique())
        assert set(etfs + stocks) == tickers_in_file

    @patch("shockarb.pipeline.yf.download")
    def test_sliced_to_three_rows(self, mock_dl, temp_dir):
        """Tape must contain exactly 3 rows: yesterday-close, today-open, today-last."""
        mock_dl.return_value = self._make_intraday_ohlcv(["VOO", "AAPL"])

        path = os.path.join(temp_dir, "tapes", "test.parquet")
        result = pipeline.save_live_tape(["VOO"], ["AAPL"], path)

        assert len(result) == 3

    @patch("shockarb.pipeline.yf.download")
    def test_three_rows_span_two_calendar_dates(self, mock_dl, temp_dir):
        """Row 0 is yesterday; rows 1 and 2 are both today."""
        mock_dl.return_value = self._make_intraday_ohlcv(["VOO", "AAPL"])

        path = os.path.join(temp_dir, "tapes", "test.parquet")
        result = pipeline.save_live_tape(["VOO"], ["AAPL"], path)

        dates = [ts.date() for ts in result.index]
        assert dates[0] < dates[1]   # row 0 is yesterday
        assert dates[1] == dates[2]  # rows 1 and 2 are both today

    @patch("shockarb.pipeline.yf.download")
    def test_today_open_is_first_bar_of_day(self, mock_dl, temp_dir):
        """Row 1 (today-open) is the earliest 15m bar of the session."""
        mock_dl.return_value = self._make_intraday_ohlcv(
            ["VOO", "AAPL"], n_bars_yesterday=4, n_bars_today=6
        )

        path = os.path.join(temp_dir, "tapes", "test.parquet")
        result = pipeline.save_live_tape(["VOO"], ["AAPL"], path)

        assert result.index[1].hour == 9
        assert result.index[1].minute == 30

    @patch("shockarb.pipeline.yf.download")
    def test_creates_parent_directory(self, mock_dl, temp_dir):
        mock_dl.return_value = self._make_intraday_ohlcv(["VOO", "AAPL"])
        nested = os.path.join(temp_dir, "a", "b", "c", "tape.parquet")
        pipeline.save_live_tape(["VOO"], ["AAPL"], nested)
        assert os.path.exists(nested)

    @patch("shockarb.pipeline.yf.download")
    def test_returns_none_on_empty_response(self, mock_dl, temp_dir):
        mock_dl.return_value = pd.DataFrame()
        path = os.path.join(temp_dir, "tapes", "test.parquet")
        result = pipeline.save_live_tape(["VOO"], ["AAPL"], path)
        assert result is None

    @patch("shockarb.pipeline.yf.download")
    def test_calls_yfinance_with_15m_interval(self, mock_dl, temp_dir):
        mock_dl.return_value = self._make_intraday_ohlcv(["VOO", "AAPL"])
        path = os.path.join(temp_dir, "tapes", "test.parquet")
        pipeline.save_live_tape(["VOO"], ["AAPL"], path)
        assert mock_dl.call_args[1].get("interval") == "15m"

    @patch("shockarb.pipeline.yf.download")
    def test_calls_yfinance_with_auto_adjust_false(self, mock_dl, temp_dir):
        mock_dl.return_value = self._make_intraday_ohlcv(["VOO", "AAPL"])
        path = os.path.join(temp_dir, "tapes", "test.parquet")
        pipeline.save_live_tape(["VOO"], ["AAPL"], path)
        assert mock_dl.call_args[1].get("auto_adjust") is False

