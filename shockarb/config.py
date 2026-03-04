"""
Configuration for the ShockArb factor model.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

# =============================================================================
# Core Configuration Classes
# =============================================================================

@dataclass(frozen=True)
class UniverseConfig:
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

@dataclass
class ExecutionConfig:
    """
    Controls I/O, caching, and logging behavior.
    """
    data_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "SHOCK_ARB_DATA_DIR", 
            os.path.join(os.getcwd(), "data")
        )
    )
    use_cache: bool = True
    log_to_file: bool = True
    log_level: str = "INFO"
    
    _logger_configured: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        # FIX: Catch explicit None values passed by the CLI and apply the default fallback
        if self.data_dir is None:
            self.data_dir = os.environ.get(
                "SHOCK_ARB_DATA_DIR", 
                os.path.join(os.getcwd(), "data")
            )

    def resolve_path(self, filename: str) -> str:
        """Return full path for a file inside data_dir, creating dir if needed."""
        os.makedirs(self.data_dir, exist_ok=True)
        return os.path.join(self.data_dir, filename)
    
    def configure_logger(self) -> None:
        """Configure loguru for terminal and optional file output. Idempotent."""
        if self._logger_configured:
            return
        
        logger.remove()
        logger.add(
            sys.stdout,
            colorize=True,
            level=self.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>"
        )
        
        if self.log_to_file:
            log_path = self.resolve_path("shockarb.log")
            logger.add(
                log_path, 
                rotation="10 MB", 
                retention="10 days", 
                level="INFO"
            )
        
        object.__setattr__(self, '_logger_configured', True)

# =============================================================================
# Pre-built Universe Definitions
# =============================================================================

US_UNIVERSE = UniverseConfig(
    name="us",
    market_etfs=[
        "VOO", "VYM", "VEU", "VDE", "VFH", "VHT", "VIS", "VGT", "VAW", "VNQ", 
        "VOX", "VPU", "VCR", "VDC", "TLT", "GLD", "USO", "ITA", "XAR", "XSD",
    ],
    individual_stocks=[
        "V", "MA", "AXP", "SPGI", "MCO", "CME", "ICE", "BLK", "FI", "FIS",
        "MSFT", "NOW", "ADBE", "CRM", "INTU", "ORCL", "SNPS", "CDNS",
        "PANW", "CRWD", "FTNT",
        "UNH", "SYK", "TMO", "DHR", "ZTS", "IDXX", "EW", "BSX", "ISRG", "LLY", "JNJ", "ABBV", "ROP",
        "LMT", "RTX", "NOC", "GD", "LHX", "HII",
        "HON", "ETN", "PH", "GWW", "ROK", "CPRT", "FAST",
        "ASML", "KLAC", "TXN", "LRCX", "AMAT", "ADI", "QCOM",
        "COST", "WMT", "PG", "PEP", "KO", "HSY",
        "CVX",
    ],
    n_components=3,
    start_date="2022-02-10",
    end_date="2022-03-31",
)

GLOBAL_UNIVERSE = UniverseConfig(
    name="global",
    market_etfs=[
        "VOO", "VYM", "VEU", "VDE", "VFH", "VHT", "VIS", "VGT", "VAW",
        "VNQ", "VOX", "VPU", "VCR", "VDC", "TLT", "GLD", "ITA",
        "VGK", "BNO", "UUP",
    ],
    individual_stocks=[
        "V", "MA", "MSFT", "NOW", "SYK", "UNH", "ROP", "LMT", "CVX",
        "BP", "TTE", "SLB", "SAP", "NVO", "SHEL", "AZN", "HSBC",
        "RY", "TD", "BHP", "RIO", "TM", "SONY",
    ],
    n_components=3,
    start_date="2022-02-10",
    end_date="2022-03-31",
)