"""
Uniform --out / -o flag contract tests.

Two abstract base classes define the contract that every utility's --out flag
must satisfy.  Concrete subclasses provide only the invoke() method and any
overrides — all contract tests are inherited automatically.

    OutFileContract  — --out points to a single output file (CSV, Markdown, …)
    OutDirContract   — --out points to a directory; multiple files written inside

Concrete subclasses
-------------------
    TestPortfolioSizerOutFlag   (OutFileContract)
    TestCsvToMdOutFlag          (OutFileContract)  — overrides test_no_out*
    TestDailyScannerOutFlag     (OutDirContract)

Adding a new utility
--------------------
Subclass the appropriate contract, implement invoke(), done.
"""

from __future__ import annotations

import os
import sys
from abc import abstractmethod
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))


# =============================================================================
# Shared fixtures
# =============================================================================

_ALPHA_ROWS = [
    {"Ticker": "BLK",  "confidence_delta": 0.050, "delta_rel": 0.04, "r_squared": 0.80},
    {"Ticker": "TXN",  "confidence_delta": 0.040, "delta_rel": 0.03, "r_squared": 0.75},
    {"Ticker": "SNPS", "confidence_delta": 0.035, "delta_rel": 0.03, "r_squared": 0.70},
]


def _write_alpha_csv(path: str, rows: list[dict] | None = None) -> str:
    pd.DataFrame(rows or _ALPHA_ROWS).to_csv(path, index=False)
    return path


def _mock_yf_prices(tickers: list[str], price: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2026-05-29"])
    return pd.DataFrame({t: [price] for t in tickers}, index=idx)


# =============================================================================
# OutFileContract — --out writes a single file
# =============================================================================

class OutFileContract:
    """
    Contract for utilities where --out <path> writes a single output file.

    Subclasses must implement:
        invoke(out_path: str | None, tmp_path) -> None
        extension: str   — expected file extension, e.g. ".csv" or ".md"
    """

    extension: str = ""

    @abstractmethod
    def invoke(self, out_path: str | None, tmp_path) -> None:
        raise NotImplementedError

    def test_out_creates_file(self, tmp_path):
        """--out path exists after invocation."""
        out = tmp_path / f"output{self.extension}"
        self.invoke(str(out), tmp_path)
        assert out.exists(), f"Expected {out} to be created by --out flag"

    def test_out_creates_intermediate_dirs(self, tmp_path):
        """--out path with nonexistent parent directories is created."""
        out = tmp_path / "nested" / "deep" / f"output{self.extension}"
        self.invoke(str(out), tmp_path)
        assert out.exists()

    def test_no_out_does_not_write_to_tmp(self, tmp_path):
        """When out=None no output file is written into tmp_path."""
        self.invoke(None, tmp_path)
        written = [
            f for f in tmp_path.iterdir()
            if f.suffix == self.extension and f.name != "alpha.csv"
        ]
        assert written == [], f"out=None should not create any {self.extension} file; found {written}"

    def test_out_file_is_nonempty(self, tmp_path):
        """The output file must contain content."""
        out = tmp_path / f"output{self.extension}"
        self.invoke(str(out), tmp_path)
        assert out.stat().st_size > 0


# =============================================================================
# OutDirContract — --out writes multiple files into a directory
# =============================================================================

class OutDirContract:
    """
    Contract for utilities where --out <dir> writes one or more files into
    the named directory.

    Subclasses must implement:
        invoke(out_dir: str | None, tmp_path) -> None
        expected_filenames: list[str]
    """

    expected_filenames: list[str] = []

    @abstractmethod
    def invoke(self, out_dir: str | None, tmp_path) -> None:
        raise NotImplementedError

    def test_out_creates_expected_files(self, tmp_path):
        """--out dir contains the expected output files after invocation."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        self.invoke(str(out_dir), tmp_path)
        for fname in self.expected_filenames:
            assert (out_dir / fname).exists(), f"Expected {fname} in output dir"

    def test_out_creates_dir_if_missing(self, tmp_path):
        """--out dir is created if it does not exist."""
        out_dir = tmp_path / "nested" / "out"
        self.invoke(str(out_dir), tmp_path)
        assert out_dir.exists()

    def test_no_out_does_not_write_to_tmp(self, tmp_path):
        """
        When out=None the utility falls back to its configured default dir.
        Base implementation is a no-op pass; subclasses may override.
        """
        pass


# =============================================================================
# Concrete: portfolio_sizer  (OutFileContract)
# =============================================================================

from portfolio_sizer import generate_orders


class TestPortfolioSizerOutFlag(OutFileContract):
    extension = ".csv"

    def invoke(self, out_path: str | None, tmp_path) -> None:
        csv = tmp_path / "alpha.csv"
        _write_alpha_csv(str(csv))
        tickers = [r["Ticker"] for r in _ALPHA_ROWS]
        with patch("portfolio_sizer.yf.download",
                   return_value=_mock_yf_prices(tickers)):
            generate_orders([str(csv)], capital=10_000, top_n=3, out=out_path)

    def test_csv_has_expected_columns(self, tmp_path):
        out = tmp_path / "ticket.csv"
        self.invoke(str(out), tmp_path)
        df = pd.read_csv(str(out))
        for col in ("Ticker", "Weight", "Dollar_Alloc", "Current", "Target", "Shares"):
            assert col in df.columns

    def test_csv_row_count_matches_top_n(self, tmp_path):
        out = tmp_path / "ticket.csv"
        self.invoke(str(out), tmp_path)
        df = pd.read_csv(str(out))
        assert len(df) == 3


# =============================================================================
# Concrete: csv_to_md  (OutFileContract)
# =============================================================================

from csv_to_md import generate_markdown_report


class TestCsvToMdOutFlag(OutFileContract):
    extension = ".md"

    def invoke(self, out_path: str | None, tmp_path) -> None:
        csv = tmp_path / "alpha.csv"
        pd.DataFrame(_ALPHA_ROWS).set_index("Ticker").to_csv(str(csv))
        generate_markdown_report(
            str(csv),
            output_path=out_path,
            enrich_names=False,
            data_dir=str(tmp_path),
        )

    def test_no_out_does_not_write_to_tmp(self, tmp_path):
        """
        Override: csv_to_md intentionally writes <csv_stem>.md when out=None.
        Verify it lands at the expected default path rather than asserting no file.
        """
        csv = tmp_path / "alpha.csv"
        pd.DataFrame(_ALPHA_ROWS).set_index("Ticker").to_csv(str(csv))
        generate_markdown_report(
            str(csv), output_path=None, enrich_names=False, data_dir=str(tmp_path)
        )
        assert (tmp_path / "alpha.md").exists(), \
            "csv_to_md should default to <csv_stem>.md when out=None"

    def test_md_contains_ticker(self, tmp_path):
        out = tmp_path / "report.md"
        self.invoke(str(out), tmp_path)
        assert "BLK" in out.read_text(encoding="utf-8")


# =============================================================================
# Concrete: daily_scanner  (OutDirContract)
# =============================================================================

import shockarb.pipeline as _pipeline
from shockarb.config import ExecutionConfig
from daily_scanner import run_scanner


def _fake_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {"confidence_delta": [0.05, 0.03], "delta_rel": [0.04, 0.02], "r_squared": [0.8, 0.7]},
        index=pd.Index(["BLK", "TXN"], name="Ticker"),
    )


class _FakeProvenance:
    """Minimal stub — supports attribute assignment and to_dict()."""
    model_file: str = ""

    def summary(self) -> str:
        return "mock provenance"

    def to_dict(self) -> dict:
        return {"model_file": self.model_file, "universe": "us"}


class TestDailyScannerOutFlag(OutDirContract):
    expected_filenames = ["live_alpha_us.csv"]

    def invoke(self, out_dir: str | None, tmp_path) -> None:
        exec_cfg = ExecutionConfig(data_dir=str(tmp_path))
        with patch.object(_pipeline, "find_latest_model", return_value="fake.json"), \
             patch.object(_pipeline, "load_model", return_value=MagicMock()), \
             patch.object(_pipeline, "score_universe",
                          return_value=(_fake_scores(), _FakeProvenance())), \
             patch("daily_scanner._get_sticky_regime", return_value="ukraine_shock"):
            run_scanner(["us"], exec_cfg, out_dir=out_dir)

    def test_csv_contains_scored_tickers(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        self.invoke(str(out_dir), tmp_path)
        df = pd.read_csv(str(out_dir / "live_alpha_us.csv"), index_col=0)
        assert "BLK" in df.index

    def test_provenance_sidecar_written(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        self.invoke(str(out_dir), tmp_path)
        assert (out_dir / "live_alpha_us_provenance.json").exists()
