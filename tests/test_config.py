"""
Unit tests for shockarb.config — UniverseConfig and ExecutionConfig.

Pure unit tests: no I/O, no network. All assertions operate on
in-memory objects or temporary directories (via temp_dir fixture).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from shockarb.config import (
    GLOBAL_UNIVERSE,
    US_UNIVERSE,
    ExecutionConfig,
    UniverseConfig,
)


class TestUniverseConfig:
    """UniverseConfig: creation, validation, immutability, pre-built constants."""

    def test_basic_creation(self):
        cfg = UniverseConfig(
            name="test",
            market_etfs=["SPY", "TLT"],
            individual_stocks=["AAPL"],
            n_components=2,
            start_date="2022-01-01",
            end_date="2022-03-31",
        )
        assert cfg.name == "test"
        assert cfg.n_components == 2
        assert "SPY" in cfg.market_etfs

    def test_frozen_immutability(self):
        cfg = UniverseConfig(
            name="x", market_etfs=["A"], individual_stocks=["B"],
            n_components=1, start_date="2022-01-01", end_date="2022-02-01",
        )
        with pytest.raises(AttributeError):
            cfg.name = "mutated"

    def test_validation_empty_etfs(self):
        with pytest.raises(ValueError, match="market_etfs cannot be empty"):
            UniverseConfig(
                name="x", market_etfs=[], individual_stocks=["A"],
                n_components=1, start_date="2022-01-01", end_date="2022-02-01",
            )

    def test_validation_empty_stocks(self):
        with pytest.raises(ValueError, match="individual_stocks cannot be empty"):
            UniverseConfig(
                name="x", market_etfs=["A"], individual_stocks=[],
                n_components=1, start_date="2022-01-01", end_date="2022-02-01",
            )

    def test_validation_zero_components(self):
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            UniverseConfig(
                name="x", market_etfs=["A"], individual_stocks=["B"],
                n_components=0, start_date="2022-01-01", end_date="2022-02-01",
            )

    def test_us_universe_structure(self):
        assert US_UNIVERSE.name == "us"
        assert len(US_UNIVERSE.market_etfs) > 10
        assert len(US_UNIVERSE.individual_stocks) > 30
        assert US_UNIVERSE.n_components == 3
        assert "VOO" in US_UNIVERSE.market_etfs
        assert "V" in US_UNIVERSE.individual_stocks

    def test_global_universe_has_international_etfs(self):
        assert "VGK" in GLOBAL_UNIVERSE.market_etfs   # European equities
        assert "BNO" in GLOBAL_UNIVERSE.market_etfs   # Brent crude
        assert "UUP" in GLOBAL_UNIVERSE.market_etfs   # USD index


class TestExecutionConfig:
    """ExecutionConfig: defaults, env-var resolution, path helpers, logging."""

    def test_default_data_dir_contains_data(self):
        old = os.environ.pop("SHOCK_ARB_DATA_DIR", None)
        try:
            assert "data" in ExecutionConfig().data_dir
        finally:
            if old:
                os.environ["SHOCK_ARB_DATA_DIR"] = old

    def test_explicit_data_dir(self):
        cfg = ExecutionConfig(data_dir="/custom/path")
        assert cfg.data_dir == "/custom/path"

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"SHOCK_ARB_DATA_DIR": "/env/path"}):
            assert ExecutionConfig().data_dir == "/env/path"

    def test_none_data_dir_falls_back_to_default(self):
        old = os.environ.pop("SHOCK_ARB_DATA_DIR", None)
        try:
            cfg = ExecutionConfig(data_dir=None)
            assert cfg.data_dir is not None
            assert "data" in cfg.data_dir
        finally:
            if old:
                os.environ["SHOCK_ARB_DATA_DIR"] = old

    def test_resolve_path_creates_directory(self, temp_dir):
        sub = os.path.join(temp_dir, "sub")
        path = ExecutionConfig(data_dir=sub).resolve_path("model.json")
        assert os.path.isdir(sub)
        assert path == os.path.join(sub, "model.json")

    def test_configure_logger_idempotent(self):
        cfg = ExecutionConfig(log_to_file=False)
        cfg.configure_logger()
        cfg.configure_logger()
        assert cfg._logger_configured is True

    def test_configure_logger_sets_flag(self):
        cfg = ExecutionConfig(log_to_file=False)
        assert not cfg._logger_configured
        cfg.configure_logger()
        assert cfg._logger_configured
