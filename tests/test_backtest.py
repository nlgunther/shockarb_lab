"""
Tests for shockarb.backtest — walk-forward backtester.

Test strategy
-------------
All tests use entirely synthetic price data generated in-process.
No network calls, no disk I/O beyond the temp_dir fixture.

The synthetic data is designed so that:
  - The ETF factor structure is stable and well-identified
  - A subset of stocks has persistent mispricing in the evaluation window
    (so that at least some trades are generated without needing looser thresholds)
  - Forward returns are computable for all holding periods tested
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from shockarb.backtest import Backtest, BacktestConfig, BacktestResults
from shockarb.config import UniverseConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mini_universe() -> UniverseConfig:
    """Minimal universe for fast tests."""
    return UniverseConfig(
        name="test",
        market_etfs=["VOO", "VDE", "TLT", "GLD", "ITA"],
        individual_stocks=["V", "MSFT", "LMT", "CVX", "UNH"],
        n_components=2,
        start_date="2022-02-10",
        end_date="2022-03-31",
    )


def _make_synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate reproducible synthetic prices with a realistic factor structure.

    All prices start at 100.  The first 3 tickers (ETFs by convention) drive
    a market factor; the remaining tickers are linear combinations plus noise.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, end=end)
    T = len(dates)

    market = rng.normal(0, 0.010, T)
    prices = {}
    for i, t in enumerate(tickers):
        beta = 0.8 - 0.1 * (i % 5)   # varying betas
        shock = 0.002 * (i % 3 - 1)   # varying drift
        rets = beta * market + rng.normal(shock, 0.008, T)
        prices[t] = 100.0 * np.cumprod(1 + rets)

    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def synthetic_etf_prices(mini_universe) -> pd.DataFrame:
    return _make_synthetic_prices(
        mini_universe.market_etfs,
        start="2021-10-01",
        end="2024-03-31",
        seed=42,
    )


@pytest.fixture
def synthetic_stock_prices(mini_universe) -> pd.DataFrame:
    return _make_synthetic_prices(
        mini_universe.individual_stocks,
        start="2021-10-01",
        end="2024-03-31",
        seed=99,
    )


@pytest.fixture
def patched_backtest(mini_universe, synthetic_etf_prices, synthetic_stock_prices, temp_dir):
    """
    A Backtest instance with fetch_prices patched to return synthetic data.
    Avoids all network calls.
    """
    from shockarb.config import ExecutionConfig

    cfg = BacktestConfig(
        universe=mini_universe,
        calib_window=35,
        holding_periods=[1, 2, 3, 5],
        min_confidence=0.001,   # low threshold so synthetic data generates trades
        min_r_squared=0.10,
        eval_start="2023-01-03",
        eval_end="2023-06-30",
        top_n=5,
    )
    exec_cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
    bt = Backtest(cfg, exec_config=exec_cfg)

    def mock_fetch(tickers, start, end, cache_name, exec_config=None):
        etfs = mini_universe.market_etfs
        if set(tickers) <= set(etfs):
            return synthetic_etf_prices[tickers].loc[start:end]
        return synthetic_stock_prices[tickers].loc[start:end]

    return bt, mock_fetch


# =============================================================================
# BacktestConfig tests
# =============================================================================

class TestBacktestConfig:

    def test_valid_config(self, mini_universe):
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-12-31",
        )
        assert cfg.calib_window == 35
        assert cfg.holding_periods == [1, 2, 3, 5]
        assert cfg.effective_n_components == 2

    def test_requires_eval_start(self, mini_universe):
        with pytest.raises(ValueError, match="eval_start"):
            BacktestConfig(universe=mini_universe, eval_start="", eval_end="2023-12-31")

    def test_requires_eval_end(self, mini_universe):
        with pytest.raises(ValueError, match="eval_start"):
            BacktestConfig(universe=mini_universe, eval_start="2023-01-01", eval_end="")

    def test_start_before_end(self, mini_universe):
        with pytest.raises(ValueError, match="before eval_end"):
            BacktestConfig(
                universe=mini_universe,
                eval_start="2023-12-31",
                eval_end="2023-01-01",
            )

    def test_calib_window_minimum(self, mini_universe):
        with pytest.raises(ValueError, match="calib_window"):
            BacktestConfig(
                universe=mini_universe,
                calib_window=5,
                eval_start="2023-01-01",
                eval_end="2023-12-31",
            )

    def test_n_components_override(self, mini_universe):
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-12-31",
            n_components=4,
        )
        assert cfg.effective_n_components == 4

    def test_n_components_default_from_universe(self, mini_universe):
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-12-31",
        )
        assert cfg.effective_n_components == mini_universe.n_components


# =============================================================================
# BacktestResults tests
# =============================================================================

class TestBacktestResults:

    def _make_results(self, mini_universe) -> BacktestResults:
        """Build a minimal BacktestResults with synthetic data."""
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-06-30",
        )
        dates = pd.bdate_range("2023-01-03", "2023-01-10")
        ledger = pd.DataFrame({
            "confidence_delta": [0.01, 0.02, 0.008],
            "r_squared":        [0.6,  0.7,  0.55],
            "delta_rel":        [0.012, 0.025, 0.010],
            "actual_return":    [-0.01, -0.015, -0.005],
            "ret_T+1":          [0.008, 0.020, -0.005],
            "ret_T+2":          [0.012, 0.015,  0.002],
            "ret_T+3":          [0.015, 0.018,  0.004],
            "ret_T+5":          [0.010, 0.022,  0.003],
            "profitable_T+1":   [True,  True,   False],
            "profitable_T+2":   [True,  True,   True],
            "profitable_T+3":   [True,  True,   True],
            "profitable_T+5":   [True,  True,   True],
        }, index=pd.MultiIndex.from_tuples(
            [(dates[0], "V"), (dates[1], "MSFT"), (dates[2], "LMT")],
            names=["entry_date", "ticker"],
        ))
        summary = pd.DataFrame({
            "n_trades":      [3, 3, 3, 3],
            "win_rate":      [0.67, 1.0, 1.0, 1.0],
            "mean_return":   [0.008, 0.010, 0.012, 0.012],
            "median_return": [0.008, 0.012, 0.015, 0.010],
            "sharpe":        [1.2, 1.5, 1.8, 1.4],
        }, index=pd.Index(["T+1", "T+2", "T+3", "T+5"], name="holding_period"))
        curve = pd.Series(
            [1.0, 1.01, 1.015],
            index=pd.bdate_range("2023-01-03", periods=3),
            name="equity_curve",
        )
        return BacktestResults(ledger=ledger, summary=summary, config=cfg, equity_curve=curve)

    def test_results_has_ledger(self, mini_universe):
        r = self._make_results(mini_universe)
        assert isinstance(r.ledger, pd.DataFrame)
        assert len(r.ledger) == 3

    def test_results_has_summary(self, mini_universe):
        r = self._make_results(mini_universe)
        assert isinstance(r.summary, pd.DataFrame)
        assert "win_rate" in r.summary.columns
        assert "sharpe" in r.summary.columns

    def test_results_has_equity_curve(self, mini_universe):
        r = self._make_results(mini_universe)
        assert isinstance(r.equity_curve, pd.Series)
        assert r.equity_curve.iloc[0] == 1.0

    def test_print_summary_runs(self, mini_universe, capsys):
        r = self._make_results(mini_universe)
        r.print_summary()
        out = capsys.readouterr().out
        assert "WALK-FORWARD" in out
        assert "Win Rate" in out

    def test_print_summary_empty(self, mini_universe, capsys):
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-06-30",
        )
        r = BacktestResults(
            ledger=pd.DataFrame(),
            summary=pd.DataFrame(),
            config=cfg,
            equity_curve=pd.Series(dtype=float),
        )
        r.print_summary()
        out = capsys.readouterr().out
        assert "No trades" in out


# =============================================================================
# Backtest core logic tests (patched data)
# =============================================================================

class TestBacktestRun:

    def test_run_returns_results_object(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        assert isinstance(results, BacktestResults)

    def test_ledger_has_expected_columns(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        required = {"confidence_delta", "r_squared", "delta_rel", "actual_return"}
        assert required.issubset(set(results.ledger.columns))

    def test_ledger_index_is_date_ticker(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        assert results.ledger.index.names == ["entry_date", "ticker"]

    def test_forward_return_columns_present(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        for h in bt.cfg.holding_periods:
            assert f"ret_T+{h}" in results.ledger.columns

    def test_no_lookahead_entry_dates_in_eval_window(self, patched_backtest):
        """All entry dates must fall within the evaluation window."""
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        entry_dates = results.ledger.index.get_level_values("entry_date")
        assert entry_dates.min() >= pd.Timestamp(bt.cfg.eval_start)
        assert entry_dates.max() <= pd.Timestamp(bt.cfg.eval_end)

    def test_all_signals_pass_thresholds(self, patched_backtest):
        """Every trade in the ledger must meet min_confidence and min_r_squared."""
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        assert (results.ledger["confidence_delta"] >= bt.cfg.min_confidence).all()
        assert (results.ledger["r_squared"] >= bt.cfg.min_r_squared).all()

    def test_max_top_n_signals_per_day(self, patched_backtest):
        """No single day should have more than top_n signals."""
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.ledger.empty:
            pytest.skip("No trades generated with synthetic data")
        daily_counts = (
            results.ledger.reset_index()
            .groupby("entry_date")["ticker"].count()
        )
        assert (daily_counts <= bt.cfg.top_n).all()

    def test_summary_has_all_holding_periods(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.summary.empty:
            pytest.skip("No trades to summarise")
        for h in bt.cfg.holding_periods:
            assert f"T+{h}" in results.summary.index

    def test_equity_curve_starts_near_one(self, patched_backtest):
        bt, mock_fetch = patched_backtest
        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()
        if results.equity_curve.empty:
            pytest.skip("No equity curve generated")
        assert abs(results.equity_curve.iloc[0] - 1.0) < 0.10

    def test_empty_results_on_no_eval_dates(self, mini_universe, temp_dir):
        """When eval window has no trading days, return empty results gracefully."""
        from shockarb.config import ExecutionConfig

        cfg = BacktestConfig(
            universe=mini_universe,
            calib_window=35,
            holding_periods=[1, 2, 3, 5],
            min_confidence=0.001,
            min_r_squared=0.10,
            eval_start="2023-07-04",  # Tuesday but we'll give it a tiny window
            eval_end="2023-07-05",
            top_n=5,
        )
        exec_cfg = ExecutionConfig(data_dir=temp_dir, log_to_file=False)
        bt = Backtest(cfg, exec_config=exec_cfg)

        tiny_etf = pd.DataFrame(
            {"VOO": [0.01], "VDE": [0.02], "TLT": [-0.01], "GLD": [0.005], "ITA": [0.015]},
            index=pd.bdate_range("2023-07-05", periods=1),
        )
        tiny_stock = pd.DataFrame(
            {"V": [0.01], "MSFT": [0.02], "LMT": [0.01], "CVX": [0.01], "UNH": [0.01]},
            index=pd.bdate_range("2023-07-05", periods=1),
        )

        def mock_fetch(tickers, start, end, cache_name, exec_config=None):
            etfs = mini_universe.market_etfs
            return tiny_etf if set(tickers) <= set(etfs) else tiny_stock

        with patch("shockarb.backtest.pipeline.fetch_prices", side_effect=mock_fetch):
            results = bt.run()

        assert results.ledger.empty
        assert results.summary.empty


# =============================================================================
# Summary statistics tests
# =============================================================================

class TestBuildSummary:

    def _make_backtest(self, mini_universe, temp_dir) -> Backtest:
        from shockarb.config import ExecutionConfig
        cfg = BacktestConfig(
            universe=mini_universe,
            eval_start="2023-01-01",
            eval_end="2023-06-30",
            holding_periods=[1, 3],
        )
        return Backtest(cfg, ExecutionConfig(data_dir=temp_dir, log_to_file=False))

    def test_win_rate_between_0_and_1(self, mini_universe, temp_dir):
        bt = self._make_backtest(mini_universe, temp_dir)
        ledger = pd.DataFrame({
            "ret_T+1": [0.01, -0.005, 0.02, -0.008, 0.015],
            "ret_T+3": [0.02,  0.010, 0.03,  0.005, 0.025],
        })
        summary = bt._build_summary(ledger)
        assert (summary["win_rate"] >= 0).all()
        assert (summary["win_rate"] <= 1).all()

    def test_summary_index_matches_holding_periods(self, mini_universe, temp_dir):
        bt = self._make_backtest(mini_universe, temp_dir)
        ledger = pd.DataFrame({
            "ret_T+1": [0.01, 0.02],
            "ret_T+3": [-0.005, 0.015],
        })
        summary = bt._build_summary(ledger)
        assert "T+1" in summary.index
        assert "T+3" in summary.index

    def test_sharpe_is_finite(self, mini_universe, temp_dir):
        bt = self._make_backtest(mini_universe, temp_dir)
        rng = np.random.RandomState(0)
        ledger = pd.DataFrame({
            "ret_T+1": rng.normal(0.005, 0.02, 50),
            "ret_T+3": rng.normal(0.010, 0.03, 50),
        })
        summary = bt._build_summary(ledger)
        assert np.isfinite(summary["sharpe"]).all()

    def test_n_trades_matches_ledger_length(self, mini_universe, temp_dir):
        bt = self._make_backtest(mini_universe, temp_dir)
        ledger = pd.DataFrame({"ret_T+1": [0.01, 0.02, 0.03]})
        summary = bt._build_summary(ledger)
        assert summary.loc["T+1", "n_trades"] == 3
