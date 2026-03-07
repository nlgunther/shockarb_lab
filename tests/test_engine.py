"""
Unit tests for shockarb.engine — FactorDiagnostics and FactorModel.

Pure unit tests: no network, no filesystem access. All assertions
operate on in-memory DataFrames constructed from deterministic
synthetic data via conftest.py fixtures.

Coverage areas
--------------
  TestFactorDiagnostics        — summary() output format and type
  TestFactorModelConstruction  — __init__ guards, pre-fit property guards
  TestFactorModelFit           — shapes, stored means, diagnostic values,
                                 mathematical properties (orthogonality)
  TestFactorModelScore         — output columns, sort order, formula,
                                 partial overlap, extreme inputs
  TestProjectSecurity          — basic use, overlap guard
  TestSerialisation            — to_dict / from_dict round-trip, no bloat
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shockarb.engine import FactorModel


class TestFactorDiagnostics:

    def test_summary_returns_string(self, fitted_model):
        assert isinstance(fitted_model.diagnostics.summary(), str)

    def test_summary_contains_key_fields(self, fitted_model):
        s = fitted_model.diagnostics.summary()
        for word in ("FactorModel Diagnostics", "days", "ETFs", "Stocks", "Factors"):
            assert word in s, f"Expected '{word}' in summary"
        assert "variance" in s.lower()


class TestFactorModelConstruction:

    def test_stores_input_dataframes(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        assert m.etf_returns is sample_etf_returns
        assert m.stock_returns is sample_stock_returns

    def test_mismatched_index_raises(self, sample_etf_returns):
        bad = pd.DataFrame({"A": [0.01]}, index=pd.date_range("2020-01-01", periods=1))
        with pytest.raises(ValueError, match="Index mismatch"):
            FactorModel(sample_etf_returns, bad)

    def test_etf_basis_before_fit_raises(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.etf_basis

    def test_factor_returns_before_fit_raises(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.factor_returns

    def test_score_before_fit_raises(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.score(pd.Series(dtype=float), pd.Series(dtype=float))


class TestFactorModelFit:

    def test_returns_self_for_chaining(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        assert m.fit(2) is m

    def test_too_many_components_raises(self, sample_etf_returns, sample_stock_returns):
        with pytest.raises(ValueError, match="n_components"):
            FactorModel(sample_etf_returns, sample_stock_returns).fit(100)

    def test_sets_fitted_flag(self, fitted_model):
        assert fitted_model._fitted is True

    def test_all_internal_attributes_set(self, fitted_model):
        for attr in ("_Vt", "_F", "_etf_mean", "_stock_mean", "loadings", "diagnostics"):
            assert getattr(fitted_model, attr) is not None, f"Expected {attr} to be set"

    def test_etf_mean_matches_input(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns).fit(2)
        pd.testing.assert_series_equal(m._etf_mean, sample_etf_returns.mean(), check_names=False)

    def test_stock_mean_matches_input(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns).fit(2)
        pd.testing.assert_series_equal(m._stock_mean, sample_stock_returns.mean(), check_names=False)

    def test_diagnostics_factor_count(self, fitted_model):
        assert fitted_model.diagnostics.n_factors == 2

    def test_diagnostics_observation_count(self, fitted_model, sample_etf_returns):
        assert fitted_model.diagnostics.n_observations == len(sample_etf_returns)

    def test_diagnostics_etf_count(self, fitted_model, sample_etf_returns):
        assert fitted_model.diagnostics.n_etfs == sample_etf_returns.shape[1]

    def test_diagnostics_stock_count(self, fitted_model, sample_stock_returns):
        assert fitted_model.diagnostics.n_stocks == sample_stock_returns.shape[1]

    def test_cumulative_variance_equals_sum_of_ratios(self, fitted_model):
        d = fitted_model.diagnostics
        assert abs(d.explained_variance_ratio.sum() - d.cumulative_variance) < 1e-10

    def test_variance_ratios_non_negative(self, fitted_model):
        assert all(v >= 0 for v in fitted_model.diagnostics.explained_variance_ratio)

    def test_r_squared_in_unit_interval(self, fitted_model):
        assert all(0 <= v <= 1 for v in fitted_model.diagnostics.stock_r_squared)

    def test_r_squared_indexed_by_stock_tickers(self, fitted_model, sample_stock_returns):
        assert list(fitted_model.diagnostics.stock_r_squared.index) == list(sample_stock_returns.columns)

    def test_residual_vol_non_negative(self, fitted_model):
        assert all(v >= 0 for v in fitted_model.diagnostics.residual_vol)

    def test_etf_basis_shape_and_columns(self, fitted_model, sample_etf_returns):
        b = fitted_model.etf_basis
        assert b.shape == (sample_etf_returns.shape[1], 2)
        assert list(b.columns) == ["Factor_1", "Factor_2"]
        assert list(b.index) == list(sample_etf_returns.columns)

    def test_factor_returns_shape_and_columns(self, fitted_model, sample_etf_returns):
        f = fitted_model.factor_returns
        assert f.shape == (len(sample_etf_returns), 2)
        assert list(f.columns) == ["Factor_1", "Factor_2"]

    def test_factors_are_orthogonal(self, fitted_model):
        """SVD guarantees orthogonal factors: |corr(F1, F2)| must be near zero."""
        corr = fitted_model.factor_returns.corr()
        assert abs(corr.loc["Factor_1", "Factor_2"]) < 0.05

    def test_single_factor_model(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns).fit(1)
        assert m.diagnostics.n_factors == 1
        assert m.loadings.shape[1] == 1

    def test_minimum_data_points(self):
        dates = pd.bdate_range("2022-01-01", periods=5)
        etf = pd.DataFrame({"A": np.random.randn(5) * 0.01,
                            "B": np.random.randn(5) * 0.01}, index=dates)
        stk = pd.DataFrame({"X": np.random.randn(5) * 0.01}, index=dates)
        assert FactorModel(etf, stk).fit(1)._fitted is True

    def test_zero_variance_etf_no_nan_loadings(self):
        """Constant-return ETF (zero variance) must not produce NaN loadings."""
        dates = pd.bdate_range("2022-01-01", periods=10)
        etf = pd.DataFrame({"A": np.zeros(10), "B": np.random.randn(10) * 0.01}, index=dates)
        stk = pd.DataFrame({"X": np.random.randn(10) * 0.01}, index=dates)
        m = FactorModel(etf, stk).fit(1)
        assert not m.loadings.isna().any().any()


class TestFactorModelScore:

    _ETF = pd.Series({"VOO": -0.01, "VDE": 0.02, "TLT": 0.005, "GLD": 0.01, "ITA": 0.015})

    def test_output_has_required_columns(self, fitted_model, sample_stock_returns):
        scores = fitted_model.score(self._ETF, sample_stock_returns.iloc[-1])
        for col in ("actual_return", "expected_rel", "expected_abs",
                    "delta_rel", "delta_abs", "r_squared",
                    "residual_vol", "confidence_delta"):
            assert col in scores.columns, f"Missing column: {col}"

    def test_sorted_descending_by_confidence_delta(self, fitted_model, sample_etf_returns, sample_stock_returns):
        scores = fitted_model.score(sample_etf_returns.iloc[-1], sample_stock_returns.iloc[-1])
        cd = scores["confidence_delta"].values
        assert all(cd[i] >= cd[i + 1] for i in range(len(cd) - 1))

    def test_confidence_delta_formula(self, fitted_model, sample_stock_returns):
        """confidence_delta must equal delta_rel × r_squared exactly."""
        scores = fitted_model.score(self._ETF, sample_stock_returns.iloc[-1])
        for _, row in scores.iterrows():
            assert pytest.approx(row["confidence_delta"], rel=1e-5) == (
                row["delta_rel"] * row["r_squared"]
            )

    def test_partial_ticker_overlap(self, fitted_model):
        scores = fitted_model.score(
            pd.Series({"VOO": -0.01, "TLT": 0.005}),
            pd.Series({"V": -0.02, "MSFT": -0.015}),
        )
        assert len(scores) == 2

    def test_no_overlap_returns_empty(self, fitted_model):
        scores = fitted_model.score(
            pd.Series({"UNKNOWN_ETF": 0.01}),
            pd.Series({"UNKNOWN_STOCK": 0.01}),
        )
        assert len(scores) == 0

    def test_uses_stored_means_not_live_data(self, sample_etf_returns, sample_stock_returns):
        """Corrupting raw return frames after fit() must not change score output."""
        m = FactorModel(sample_etf_returns.copy(), sample_stock_returns.copy()).fit(2)
        before = m.score(sample_etf_returns.iloc[-1], sample_stock_returns.iloc[-1])
        m.etf_returns.iloc[0] *= 1000
        m.stock_returns.iloc[0] *= 1000
        after = m.score(sample_etf_returns.iloc[-1], sample_stock_returns.iloc[-1])
        pd.testing.assert_frame_equal(before, after)

    def test_extreme_returns_produce_no_nan(self, fitted_model):
        etf = pd.Series({"VOO": -0.20, "VDE": 0.50, "TLT": 0.10, "GLD": 0.15, "ITA": 0.30})
        stk = pd.Series({"V": -0.30, "MSFT": -0.25, "LMT": 0.40, "CVX": 0.60, "UNH": -0.10})
        assert not fitted_model.score(etf, stk).isna().any().any()

    def test_missing_etfs_filled_with_zero(self, fitted_model, sample_stock_returns):
        """Partial ETF input should not raise — missing tickers filled with 0."""
        scores = fitted_model.score(
            pd.Series({"VOO": -0.01}),
            sample_stock_returns.iloc[-1],
        )
        assert len(scores) == len(sample_stock_returns.columns)
        assert not scores.isna().any().any()


class TestProjectSecurity:

    def test_returns_named_series(self, fitted_model, sample_stock_returns):
        result = fitted_model.project_security("V_NEW", sample_stock_returns["V"])
        assert isinstance(result, pd.Series)
        assert result.name == "V_NEW"
        assert len(result) == 2

    def test_factor_index_labels(self, fitted_model, sample_stock_returns):
        result = fitted_model.project_security("TEST", sample_stock_returns["V"])
        assert list(result.index) == ["Factor_1", "Factor_2"]

    def test_insufficient_overlap_raises(self, fitted_model):
        bad = pd.Series([0.01, 0.02], index=pd.date_range("2020-01-01", periods=2))
        with pytest.raises(ValueError, match="overlap"):
            fitted_model.project_security("BAD", bad)


class TestSerialisation:

    def test_to_dict_required_keys_present(self, fitted_model):
        d = fitted_model.to_dict()
        for key in ("metadata", "Vt", "etf_columns", "etf_mean",
                    "loadings", "stock_columns", "stock_mean",
                    "explained_variance_ratio", "stock_r_squared", "residual_vol"):
            assert key in d, f"Missing serialisation key: {key}"

    def test_to_dict_excludes_raw_return_matrices(self, fitted_model):
        d = fitted_model.to_dict()
        assert "etf_returns" not in d
        assert "stock_returns" not in d

    def test_round_trip_preserves_loadings(self, fitted_model):
        pd.testing.assert_frame_equal(
            FactorModel.from_dict(fitted_model.to_dict()).loadings,
            fitted_model.loadings,
        )

    def test_round_trip_preserves_etf_mean(self, fitted_model):
        pd.testing.assert_series_equal(
            FactorModel.from_dict(fitted_model.to_dict())._etf_mean,
            fitted_model._etf_mean,
        )

    def test_round_trip_preserves_diagnostics(self, fitted_model):
        restored = FactorModel.from_dict(fitted_model.to_dict())
        assert restored.diagnostics.n_factors == fitted_model.diagnostics.n_factors
        assert restored.diagnostics.cumulative_variance == pytest.approx(
            fitted_model.diagnostics.cumulative_variance
        )

    def test_score_unchanged_after_round_trip(self, fitted_model, sample_etf_returns, sample_stock_returns):
        etf_ret = sample_etf_returns.iloc[-1]
        stk_ret = sample_stock_returns.iloc[-1]
        before = fitted_model.score(etf_ret, stk_ret)
        after = FactorModel.from_dict(fitted_model.to_dict()).score(etf_ret, stk_ret)
        pd.testing.assert_frame_equal(before, after)

    def test_loaded_model_has_empty_return_stubs(self, fitted_model):
        restored = FactorModel.from_dict(fitted_model.to_dict())
        assert restored.etf_returns.empty
        assert restored.stock_returns.empty

    def test_unfitted_model_to_dict_raises(self, sample_etf_returns, sample_stock_returns):
        m = FactorModel(sample_etf_returns, sample_stock_returns)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.to_dict()
