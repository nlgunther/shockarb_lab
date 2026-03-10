"""
Configuration for the ShockArb factor model.

Two classes:

  UniverseConfig  — immutable definition of *what* to analyze:
                    which ETFs, which stocks, date window, and
                    how many SVD components to extract.

  ExecutionConfig — mutable settings for *how* to run:
                    caching, logging, and file paths.

Pre-built universe constants (US_UNIVERSE, GLOBAL_UNIVERSE) are defined
at module level so any script can import them without constructing anything.

Example
-------
    from shockarb.config import US_UNIVERSE, ExecutionConfig

    cfg = ExecutionConfig(data_dir="/data/shockarb", log_level="DEBUG")
    cfg.configure_logger()
    print(US_UNIVERSE.market_etfs)
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


# =============================================================================
# UniverseConfig — what to analyze
# =============================================================================

@dataclass(frozen=True)
class UniverseConfig:
    """
    Immutable specification of a trading universe.

    Frozen so that pre-built constants (US_UNIVERSE, GLOBAL_UNIVERSE) can
    be shared freely without risk of accidental mutation.

    Attributes
    ----------
    name : str
        Short identifier used in filenames, log output, and as the requester
        prefix in DataRequest labels (e.g. "us.etf", "us.stock").
    market_etfs : list of str
        ETF tickers used to build the macro factor basis via SVD.
        Rule of thumb: include at least one per major asset class present
        in the target stock universe.  The US_UNIVERSE uses 19 ETFs covering
        broad equity, sectors, fixed income, commodities, and defense.
    individual_stocks : list of str
        Stocks to project onto the factor basis and score.  Tickers that are
        delisted or changed will silently produce NaN columns — check logs
        for warnings from prices_to_returns() after each build.
    n_components : int
        Number of SVD factors to retain.  Typical values:
          2 — market + one sector axis
          3 — market + sector + shock (recommended default)
          4+ — risks overfitting on short (~35-day) windows
    start_date, end_date : str
        Calibration window in YYYY-MM-DD format.
        end_date is exclusive (yfinance convention).
        The default window (2022-02-10 to 2022-03-31) captures the
        Russia-Ukraine invasion shock — the founding event for ShockArb.
    """
    market_etfs: List[str]
    individual_stocks: List[str]
    n_components: int
    start_date: str
    end_date: str
    name: str = "custom"

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1")
        if not self.market_etfs:
            raise ValueError("market_etfs cannot be empty")
        if not self.individual_stocks:
            raise ValueError("individual_stocks cannot be empty")


# =============================================================================
# ExecutionConfig — how to run
# =============================================================================

@dataclass
class ExecutionConfig:
    """
    Mutable runtime settings: paths, caching, and logging.

    The data_dir default resolves from the SHOCK_ARB_DATA_DIR environment
    variable, falling back to ./data relative to the working directory.
    Pass data_dir explicitly in tests to avoid cross-test pollution.

    Attributes
    ----------
    data_dir : str
        Root directory for all cached data and saved models.
        Sub-directories used at runtime:
          data/prices/daily/   — per-ticker parquet files (ParquetStore)
          data/manifest.json   — coverage registry (ParquetStore)
          data/cache/          — legacy CacheManager files (unused after
                                 datamgr migration; safe to delete)
          data/backups/        — timestamped parquet backups (CacheManager)
          data/shockarb.log    — rotating log file (if log_to_file=True)
    log_to_file : bool
        Whether to write a rotating log file inside data_dir.
        Disabled in tests to avoid creating files in the working directory.
    log_level : str
        Loguru level string: "DEBUG", "INFO", "WARNING", "ERROR".
        Set via the --log-level CLI flag (global, before the subcommand)::

            python -m shockarb --log-level DEBUG build --universe us
            python -m shockarb --log-level DEBUG score --universe us

        DEBUG enables DataStore read/write traces showing exact parquet
        paths and row counts — useful for diagnosing cache misses.

    Notes
    -----
    configure_logger() is idempotent: safe to call multiple times.
    The internal _logger_configured guard prevents duplicate sink
    registration across multiple ExecutionConfig instances in one process.
    """
    data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "SHOCK_ARB_DATA_DIR",
            os.path.join(os.getcwd(), "data"),
        )
    )
    log_to_file: bool = True
    log_level: str = "INFO"

    # Internal guard — not part of public interface
    _logger_configured: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        # Guard against callers that pass data_dir=None explicitly
        if self.data_dir is None:
            self.data_dir = os.environ.get(
                "SHOCK_ARB_DATA_DIR",
                os.path.join(os.getcwd(), "data"),
            )

    def resolve_path(self, filename: str) -> str:
        """
        Return the full path for *filename* inside data_dir.

        Creates data_dir if it does not exist.  Used by save_model(),
        find_latest_model(), and export_csvs() to avoid repeating
        os.makedirs() calls.

        Parameters
        ----------
        filename : str
            Relative filename, e.g. "us_20260310_120000.json".

        Returns
        -------
        str
            Absolute path: {data_dir}/{filename}.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        return os.path.join(self.data_dir, filename)

    def configure_logger(self) -> None:
        """
        Configure loguru for stdout and optional rotating file output.

        Idempotent — does nothing if already configured.  The first call
        removes the default loguru sink, installs a colourised stdout sink
        at self.log_level, and (if log_to_file=True) adds a rotating file
        sink at INFO level inside data_dir.

        The stdout format::

            12:01:22 | INFO     | Building model: us
            12:01:22 | DEBUG    | [ParquetStore] read  OK    data\\prices\\daily\\VOO.parquet (34 rows)
        """
        if self._logger_configured:
            return

        logger.remove()
        logger.add(
            sys.stdout,
            colorize=True,
            level=self.log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<level>{message}</level>"
            ),
        )

        if self.log_to_file:
            logger.add(
                self.resolve_path("shockarb.log"),
                rotation="10 MB",
                retention="10 days",
                level="INFO",
            )

        self._logger_configured = True


# =============================================================================
# Pre-built universes
# =============================================================================

US_UNIVERSE = UniverseConfig(
    name="us",
    market_etfs=[
        # Broad equity
        "VOO", "VYM", "VEU",
        # Sectors
        "VDE", "VFH", "VHT", "VIS", "VGT", "VAW", "VNQ",
        "VOX", "VPU", "VCR", "VDC",
        # Cross-asset
        "TLT", "GLD", "USO",
        # Defense / semis
        "ITA", "XAR", "XSD",
    ],
    individual_stocks=[
        # Financials / payments
        "V", "MA", "AXP", "SPGI", "MCO", "CME", "ICE", "BLK", "FIS",
        # FI (Fiserv) removed — delisted/ticker changed; causes 404 at runtime
        # Software
        "MSFT", "NOW", "ADBE", "CRM", "INTU", "ORCL", "SNPS", "CDNS",
        # Cybersecurity
        "PANW", "CRWD", "FTNT",
        # Healthcare
        "UNH", "SYK", "TMO", "DHR", "ZTS", "IDXX", "EW", "BSX", "ISRG",
        "LLY", "JNJ", "ABBV", "ROP",
        # Defense
        "LMT", "RTX", "NOC", "GD", "LHX", "HII",
        # Industrials
        "HON", "ETN", "PH", "GWW", "ROK", "CPRT", "FAST",
        # Semiconductors
        "ASML", "KLAC", "TXN", "LRCX", "AMAT", "ADI", "QCOM",
        # Consumer staples
        "COST", "WMT", "PG", "PEP", "KO", "HSY",
        # Energy
        "CVX",
    ],
    n_components=3,
    start_date="2022-02-10",
    end_date="2022-03-31",
)

GLOBAL_UNIVERSE = UniverseConfig(
    name="global",
    market_etfs=[
        # US equity + sectors
        "VOO", "VYM", "VEU", "VDE", "VFH", "VHT", "VIS", "VGT", "VAW",
        "VNQ", "VOX", "VPU", "VCR", "VDC", "TLT", "GLD", "ITA",
        # International / FX / commodities
        "VGK",   # European equities
        "BNO",   # Brent crude
        "UUP",   # USD index
    ],
    individual_stocks=[
        # US blue chips
        "V", "MA", "MSFT", "NOW", "SYK", "UNH", "ROP", "LMT", "CVX",
        # European / international ADRs
        "BP", "TTE", "SLB", "SAP", "NVO", "SHEL", "AZN", "HSBC",
        # Canadian banks
        "RY", "TD",
        # Miners / autos / tech ADRs
        "BHP", "RIO", "TM", "SONY",
    ],
    n_components=3,
    start_date="2022-02-10",
    end_date="2022-03-31",
)
