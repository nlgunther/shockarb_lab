"""
Tests for the global_eod workflow.

Covers:
  TestGlobalWorkflowBat       -- bat file has global_eod wired correctly
  TestGlobalUniverseIsolation -- global stocks are disjoint from US stocks
  TestGlobalScoreRouting      -- cmd_score with global regime uses global model + correct out path
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from shockarb.regimes import (
    GLOBAL_UKRAINE_SHOCK,
    UKRAINE_SHOCK,
    _GLOBAL_STOCKS,
    _US_STOCKS,
)


# =============================================================================
# TestGlobalWorkflowBat -- bat file structure
# =============================================================================

class TestGlobalWorkflowBat:
    """Sanity-check that shockarb_workflows.bat has global_eod correctly wired."""

    @pytest.fixture
    def bat_text(self):
        bat = Path(__file__).parent.parent / "scripts" / "shockarb_workflows.bat"
        return bat.read_text(encoding="utf-8")

    def test_global_eod_target_exists(self, bat_text):
        """:global_eod label is present."""
        assert ":global_eod" in bat_text

    def test_global_eod_uses_global_regime(self, bat_text):
        """global_eod scores against global_ukraine_shock, not sticky."""
        assert "--regime global_ukraine_shock" in bat_text

    def test_global_eod_routes_scores_to_global_csv(self, bat_text):
        """global_eod writes scores to live_alpha_global.csv, not live_alpha_us.csv."""
        assert "--out data\\live_alpha_global.csv" in bat_text

    def test_global_eod_routes_reports_to_global_subdir(self, bat_text):
        """global_eod routes stock report to reports\\global, not reports\\."""
        assert "--reports-dir reports\\global" in bat_text

    def test_global_eod_in_help(self, bat_text):
        """global_eod appears in the help output."""
        assert "global_eod" in bat_text

    def test_global_eod_does_not_clobber_us_csv(self, bat_text):
        """Within :global_eod block, live_alpha_us.csv is never written."""
        in_block = False
        block_lines = []
        for line in bat_text.splitlines():
            if line.strip() == ":global_eod":
                in_block = True
                continue
            if in_block:
                if line.startswith(":") and not line.startswith("::"):
                    break
                block_lines.append(line)
        assert block_lines, "global_eod block should have content"
        block = "\n".join(block_lines)
        assert "live_alpha_us.csv" not in block


# =============================================================================
# TestGlobalUniverseIsolation -- universe-level correctness
# =============================================================================

class TestGlobalUniverseIsolation:
    """Global and US stock universes are non-overlapping."""

    def test_global_stocks_disjoint_from_us_stocks(self):
        """Only the documented overlap (ASML) appears in both universes.

        ASML Holding (Netherlands) is in _US_STOCKS as a heavily US-traded semi,
        and in _GLOBAL_STOCKS as a European ADR. This known duplication means
        global_eod and eod will both score ASML under different regimes.
        If additional tickers are added to both lists, this test will fail to
        surface the new overlap for review.
        """
        overlap = set(_GLOBAL_STOCKS) & set(_US_STOCKS)
        known_overlap = {"ASML"}
        new_overlap = overlap - known_overlap
        assert new_overlap == set(), (
            f"New tickers added to both universes (beyond known ASML): {new_overlap}"
        )

    def test_global_etfs_include_international(self):
        """Global ETF basket includes at least one ex-US ETF."""
        global_etfs = set(GLOBAL_UKRAINE_SHOCK.universe.market_etfs)
        ex_us = {"VEU", "VGK", "VPL", "VWO", "EWJ", "EWG", "EWU", "FXI"}
        assert global_etfs & ex_us, "Global ETF basket should include ex-US ETFs"

    def test_us_etfs_not_dominant_in_global(self):
        """Global basket uses different ETFs than the US basket."""
        global_etfs = set(GLOBAL_UKRAINE_SHOCK.universe.market_etfs)
        us_etfs = set(UKRAINE_SHOCK.universe.market_etfs)
        assert global_etfs != us_etfs, "Global and US ETF baskets should differ"

    def test_global_stocks_are_adrs(self):
        """All global stocks are expected ADRs (non-US domicile)."""
        expected = {
            "TTE", "SAN", "ASML", "SAP", "NVO", "SHEL", "HSBC",
            "TSM", "SONY", "TM", "BHP", "RIO", "HDB", "RY", "VALE",
        }
        assert set(_GLOBAL_STOCKS) == expected


# =============================================================================
# TestGlobalScoreRouting -- CLI score routes to global model and output path
# =============================================================================

class TestGlobalScoreRouting:
    """cmd_score with --regime global_ukraine_shock uses the right model and output path."""

    @patch("shockarb.cli.pipeline.score_universe")
    @patch("shockarb.cli.pipeline.find_latest_model")
    @patch("shockarb.cli.pipeline.load_model")
    def test_global_regime_searches_global_model(
        self, mock_load, mock_find, mock_score
    ):
        """find_latest_model is called with regime='global_ukraine_shock'."""
        from shockarb.cli import cmd_score

        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.regime = "global_ukraine_shock"
            args.universe = None
            args.data_dir = tmpdir
            args.no_log = True
            args.date = None
            args.model = None
            args.output = None
            args.top = 20
            args.min_confidence = 0.0
            args.min_r_squared = 0.0
            args.use_prior_close = False
            args.from_open = False
            args.save_tape = False
            args.save_recent = False
            args.no_out = True

            mock_find.return_value = "/fake/global_ukraine_shock_global_20260522_170000.json"
            mock_model = MagicMock()
            mock_model.etf_returns.columns = ["VOO"]
            mock_model.stock_returns.columns = ["TSM"]
            mock_load.return_value = mock_model

            mock_scores = pd.DataFrame({"ticker": ["TSM"], "delta": [-0.04]})
            mock_prov = MagicMock()
            mock_score.return_value = (mock_scores, mock_prov)

            with patch("shockarb.cli.print_scores"):
                cmd_score(args)

        assert mock_find.called
        call_kwargs = mock_find.call_args.kwargs
        assert call_kwargs.get("regime") == "global_ukraine_shock", (
            f"Expected find_latest_model called with regime='global_ukraine_shock', "
            f"got: {call_kwargs}"
        )

    @patch("shockarb.cli.pipeline.score_universe")
    @patch("shockarb.cli.pipeline.find_latest_model")
    @patch("shockarb.cli.pipeline.load_model")
    def test_global_out_path_saved_to_global_csv(
        self, mock_load, mock_find, mock_score, tmp_path
    ):
        """When --out points to live_alpha_global.csv that file is written, not live_alpha_us.csv."""
        from shockarb.cli import cmd_score

        out_path = str(tmp_path / "live_alpha_global.csv")

        args = MagicMock()
        args.regime = "global_ukraine_shock"
        args.universe = None
        args.data_dir = str(tmp_path)
        args.no_log = True
        args.date = None
        args.model = None
        args.output = None
        args.top = 20
        args.min_confidence = 0.0
        args.min_r_squared = 0.0
        args.use_prior_close = False
        args.from_open = False
        args.save_tape = False
        args.save_recent = False
        args.no_out = False
        args.out = out_path

        mock_find.return_value = "/fake/global_ukraine_shock_global_20260522_170000.json"
        mock_model = MagicMock()
        mock_model.etf_returns.columns = ["VOO"]
        mock_model.stock_returns.columns = ["TSM"]
        mock_load.return_value = mock_model

        mock_scores = pd.DataFrame(
            {"confidence_delta": [0.03], "r_squared": [0.72]},
            index=pd.Index(["TSM"], name="Ticker"),
        )
        mock_prov = MagicMock()
        mock_score.return_value = (mock_scores, mock_prov)

        with patch("shockarb.cli.print_scores"):
            cmd_score(args)

        assert Path(out_path).exists(), "live_alpha_global.csv should have been written"
        assert not (tmp_path / "live_alpha_us.csv").exists(), (
            "live_alpha_us.csv should not be touched by a global_ukraine_shock score"
        )
