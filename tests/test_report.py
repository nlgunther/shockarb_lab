"""
Unit tests for shockarb.report — terminal display functions.

Covers:
  - print_scores(): columns, actionable threshold, r_squared filter, top_n, bottom signals
  - print_model_state(): JSON parsing, structural output, bad-path safety
  - print_live_alpha(): threshold filtering, model name in header
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from shockarb.report import print_live_alpha, print_model_state, print_scores


# =============================================================================
# Helpers
# =============================================================================

def _make_scores(n: int = 5, conf: float = 0.02, delta_rel: float = 0.03) -> pd.DataFrame:
    """Synthetic score DataFrame matching the exact columns from model.score()."""
    return pd.DataFrame(
        {
            "actual_return":    [-0.02] * n,
            "expected_rel":     [0.01] * n,
            "expected_abs":     [0.01] * n,
            "delta_rel":        [delta_rel] * n,
            "delta_abs":        [delta_rel] * n,
            "r_squared":        [0.60] * n,
            "residual_vol":     [0.15] * n,
            "confidence_delta": [conf] * n,
        },
        index=[f"T{i}" for i in range(n)],
    )


def _write_model_json(path: str) -> None:
    """Write a minimal but structurally valid model JSON to disk."""
    data = {
        "metadata": {
            "name": "test",
            "n_factors": 2,
            "n_observations": 36,
            "cumulative_variance": 0.82,
            "created_at": "2025-01-01T00:00:00",
        },
        "Vt": [[0.5, 0.5, 0.3, -0.4, 0.2],
               [0.3, -0.4, 0.5, 0.2, -0.3]],
        "etf_columns": ["VOO", "VDE", "TLT", "GLD", "ITA"],
        "etf_mean": [0.001, 0.002, 0.0005, 0.001, 0.0015],
        "loadings": [[0.8, 0.3], [1.1, -0.1]],
        "stock_columns": ["V", "MSFT"],
        "stock_mean": [-0.001, -0.001],
        "explained_variance_ratio": [0.60, 0.22],
        "stock_r_squared": {"V": 0.72, "MSFT": 0.65},
        "residual_vol": {"V": 0.12, "MSFT": 0.18},
    }
    with open(path, "w") as f:
        json.dump(data, f)


# =============================================================================
# print_scores()
# =============================================================================

class TestPrintScores:

    def test_header_contains_title(self, capsys):
        print_scores(_make_scores(), "MY TITLE")
        assert "MY TITLE" in capsys.readouterr().out

    def test_shockarb_scores_label_present(self, capsys):
        print_scores(_make_scores(), "TEST")
        assert "SHOCKARB SCORES" in capsys.readouterr().out

    def test_actionable_tickers_displayed(self, capsys):
        print_scores(_make_scores(), "TEST")
        assert "T0" in capsys.readouterr().out

    def test_no_actionable_message_when_below_threshold(self, capsys):
        print_scores(_make_scores(conf=0.000001), "TEST", min_confidence=0.001)
        assert "No actionable signals" in capsys.readouterr().out

    def test_top_n_limits_output(self, capsys):
        scores = _make_scores(n=10, conf=0.05)
        print_scores(scores, "TEST", top_n=3, min_confidence=0.001)
        out = capsys.readouterr().out
        # T3 and beyond should not appear (top_n=3 → T0,T1,T2 only)
        assert "T3" not in out

    def test_bottom_signals_shown_when_negative(self, capsys):
        scores = _make_scores(conf=-0.02)
        print_scores(scores, "TEST", min_confidence=0.001)
        out = capsys.readouterr().out
        assert "Bottom" in out or "avoid" in out.lower()

    def test_r_squared_filter_hides_low_fit_stocks(self, capsys):
        scores = _make_scores(conf=0.05)
        scores["r_squared"] = 0.1  # all below default min_r_squared=0.30
        print_scores(scores, "TEST")
        assert "No actionable signals" in capsys.readouterr().out

    def test_custom_min_confidence(self, capsys):
        scores = _make_scores(conf=0.003)
        # With default min_confidence=0.001 this is above threshold
        print_scores(scores, "TEST", min_confidence=0.001)
        assert "T0" in capsys.readouterr().out


# =============================================================================
# print_model_state()
# =============================================================================

class TestPrintModelState:

    def test_prints_model_name_in_header(self, temp_dir, capsys):
        path = os.path.join(temp_dir, "model.json")
        _write_model_json(path)
        print_model_state(path)
        out = capsys.readouterr().out
        assert "SHOCKARB MODEL REPORT" in out
        assert "TEST" in out

    def test_prints_cumulative_variance(self, temp_dir, capsys):
        path = os.path.join(temp_dir, "model.json")
        _write_model_json(path)
        print_model_state(path)
        out = capsys.readouterr().out
        assert "82" in out   # 82% cumulative variance

    def test_prints_etf_basis_section(self, temp_dir, capsys):
        path = os.path.join(temp_dir, "model.json")
        _write_model_json(path)
        print_model_state(path)
        out = capsys.readouterr().out
        assert "ETF" in out or "BASIS" in out

    def test_prints_etf_tickers(self, temp_dir, capsys):
        path = os.path.join(temp_dir, "model.json")
        _write_model_json(path)
        print_model_state(path)
        out = capsys.readouterr().out
        assert "VOO" in out

    def test_bad_path_does_not_raise(self, capsys):
        """A missing file should log an error but never raise an exception."""
        print_model_state("/nonexistent/path/model.json")
        # No assertion on output — just confirming no exception propagates


# =============================================================================
# print_live_alpha()
# =============================================================================

class TestPrintLiveAlpha:

    def test_header_contains_model_name(self, capsys):
        print_live_alpha(_make_scores(), model_name="US")
        assert "US" in capsys.readouterr().out

    def test_live_shockarb_delta_label_present(self, capsys):
        print_live_alpha(_make_scores(), model_name="US")
        assert "LIVE SHOCKARB DELTA" in capsys.readouterr().out

    def test_efficient_tape_message_when_no_signals(self, capsys):
        scores = _make_scores(delta_rel=0.0001)
        print_live_alpha(scores, model_name="US", min_delta=0.005)
        out = capsys.readouterr().out
        assert "efficient" in out.lower() or "No targets" in out

    def test_actionable_signals_displayed_above_threshold(self, capsys):
        scores = _make_scores(delta_rel=0.02)
        print_live_alpha(scores, model_name="US", min_delta=0.005)
        assert "T0" in capsys.readouterr().out
