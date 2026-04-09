"""
Regime catalogue — decoupled historical period definitions.

Each regime is a HistoricFactorModel: a UniverseConfig (tickers + dates)
bundled with narrative metadata that explains *why* this period matters
and *what sector dynamics* to expect.

Adding a new regime:
  1. Define a new HistoricFactorModel instance at module level.
  2. Add it to REGIME_REGISTRY.
  3. That's it — pipeline.build() and the CLI pick it up automatically.

Example
-------
    from shockarb.regimes import get_regime, list_regimes

    regime = get_regime("ukraine_shock")
    model = pipeline.build(regime.universe)

    # See all available regimes
    for name in list_regimes():
        regime = get_regime(name)
        print(f"{regime.name}: {regime.description}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from shockarb.config import UniverseConfig


# =============================================================================
# HistoricFactorModel
# =============================================================================

@dataclass(frozen=True)
class HistoricFactorModel:
    """
    A named historical regime with its universe spec and interpretive metadata.

    Attributes
    ----------
    name : str
        Short CLI-friendly identifier, e.g. "ukraine_shock", "gulf_war_recovery".
    description : str
        One-line human-readable label for reports and logs.
    narrative : str
        Paragraph explaining the thesis: what happened, what the factor structure
        captures, and what sector dynamics to expect from residuals.
    universe : UniverseConfig
        Tickers, date window, and n_components for this regime.
    expected_dynamics : Dict[str, str]
        Sector → expected behavior mapping, for operator reference.
        e.g. {"energy": "give back shock premium", "tech": "lead recovery"}
    tags : Tuple[str, ...]
        Searchable tags: ("geopolitical", "energy_shock", "recovery", "ceasefire")
        Uses tuple instead of list for hashability.
    supersedes : Optional[str]
        If this regime is intended to replace another in a live rotation,
        name the predecessor here. Purely informational.
    """
    name:               str
    description:        str
    narrative:          str
    universe:           UniverseConfig
    expected_dynamics:  Dict[str, str] = field(default_factory=dict, hash=False, compare=False)
    tags:               Tuple[str, ...] = field(default_factory=tuple)
    supersedes:         Optional[str]  = None


# =============================================================================
# ETF and stock lists — shared across regimes where appropriate
# =============================================================================

# Core US macro ETFs (asset-class coverage for PCA factor extraction)
_US_MACRO_ETFS = [
    "VOO",   # S&P 500
    "VYM",   # High-dividend value
    "VUG",   # Growth
    "VEU",   # International ex-US
    "VDE",   # Energy
    "VNQ",   # REITs
    "TLT",   # Long-term Treasuries
    "GLD",   # Gold
    "USO",   # Crude oil
    "ITA",   # Aerospace & defense
    "XLB",   # Materials
    "XLI",   # Industrials
    "XLK",   # Technology
    "XLP",   # Consumer staples
    "XLY",   # Consumer discretionary
    "XLF",   # Financials
    "XLV",   # Healthcare
    "XLU",   # Utilities
    "HYG",   # High-yield corporate bonds
]

# Standard US stock universe (80 names across sectors)
_US_STOCKS = [
    # Financials / payments
    "V", "MA", "AXP", "SPGI", "MCO", "CME", "ICE", "BLK", "FIS",
    # Software
    "MSFT", "NOW", "ADBE", "CRM", "INTU", "ORCL", "SNPS", "CDNS", "PLTR",
    # Cybersecurity
    "PANW", "CRWD", "FTNT",
    # Healthcare
    "UNH", "SYK", "TMO", "DHR", "ZTS", "IDXX", "EW", "BSX", "ISRG",
    "LLY", "JNJ", "ABBV", "ROP",
    # Defense
    "LMT", "RTX", "NOC", "GD", "LHX", "HII",
    # Industrials
    "HON", "ETN", "PH", "GWW", "ROK", "CPRT", "FAST",
    # Airlines (Negative Beta Shock Anchors)
    "DAL", "UAL",
    # Semiconductors
    "ASML", "KLAC", "TXN", "LRCX", "AMAT", "ADI", "QCOM",
    # Consumer staples
    "COST", "WMT", "PG", "PEP", "KO", "HSY",
    # Energy (Positive Beta Shock Anchors)
    "CVX", "XOM", "OXY", "LNG",
]

# =============================================================================
# Regime definitions
# =============================================================================

UKRAINE_SHOCK = HistoricFactorModel(
    name="ukraine_shock",
    description="Russia-Ukraine invasion shock (Feb–Mar 2022)",
    narrative="""
    The Russia-Ukraine invasion of February 2022 created a textbook geopolitical
    shock: oil spiked 40%, broad equities sold off 15%, while energy and defense
    stocks rallied strongly. The factor model calibrated on this window captures
    three latent axes: (1) market beta, (2) energy/commodity exposure, and
    (3) defensive rotation into bonds/gold. Stocks with high R² that sold off
    harder than factor-implied are candidates for mean-reversion — the core
    ShockArb thesis. This regime assumes the shock persists; use gulf_war_recovery
    when a ceasefire begins normalization.
    """,
    universe=UniverseConfig(
        name="us",
        market_etfs=_US_MACRO_ETFS,
        individual_stocks=_US_STOCKS,
        n_components=3,
        start_date="2022-02-10",
        end_date="2022-03-31",
    ),
    expected_dynamics={
        "energy": "rally (+15-20% vs SPY)",
        "defense": "rally (+10-15% vs SPY)",
        "tech": "sell off (-20-25% vs SPY)",
        "airlines": "severe sell off (-30% vs SPY)",
        "financials": "moderate sell off (-10% vs SPY)",
    },
    tags=("geopolitical", "energy_shock", "invasion", "acute_crisis"),
    supersedes=None,
)

# Gulf War Recovery regime will be populated after running check_1991.py
# Placeholder with tentative tickers from the plan — update after data validation
GULF_WAR_RECOVERY = HistoricFactorModel(
    name="gulf_war_recovery",
    description="Gulf War ceasefire recovery (Mar–Jun 1991)",
    narrative="""
    The Gulf War ceasefire (Feb 28, 1991) marked the beginning of a fragile,
    uncertain recovery from an oil-supply shock. Oil prices fell 31% Jan-Mar 1991
    but stayed ~30% above pre-invasion levels. The ceasefire was followed by months
    of geopolitical uncertainty (Iraqi uprisings, no-fly zones, Kuwaiti oil fires
    burning through Nov 1991). This regime captures the sector rotation from
    energy/defense into cyclicals and tech, but under continued volatility — not
    a clean V-shape. The factor basis uses 1991-era indexes and liquid stocks
    (ETFs didn't exist yet) to extract the same latent structure: market, energy,
    and defensive rotation. Use this regime when a ceasefire is declared but
    fundamental trust deficits remain.
    """,
    universe=UniverseConfig(
        name="us_recovery",  # Different name to avoid collision
        market_etfs=[
            # NOTE: Update these after running check_1991.py
            # 1991-era factor basis uses indexes and stocks, not ETFs
            "^GSPC",  # S&P 500
            # "^DJI",   # Dow Jones
            "^IXIC",  # Nasdaq
            # Treasuries
            "^TYX",
            "^TNX",
            "^IRX",
            # "GC=F",   # Gold futures
            # "CL=F",   # WTI crude futures
        ],
        individual_stocks=[
            # NOTE: Update these after running check_1991.py
            # Confirmed 1991-available tickers
            # "XOM", "GE", "IBM", "BA", "LMT", "JPM", "JNJ", "PG",
            # "KO", "MRK", "PFE", "CAT", "MMM", "INTC", "TXN",
                # Energy
            "XOM",
            "CVX",
            # Industrials
            "GE",
            "BA",
            "MMM",
            "CAT",
            # Defense
            "LMT",
            "RTX",
            "GD",
            "NOC",
            # Technology
            "IBM",
            "MSFT",
            "INTC",
            "TXN",
            # Healthcare
            "JNJ",
            "MRK",
            "PFE",
            "LLY",
            # Consumer
            "PG",
            "KO",
            "PEP",
            "MCD",
            # Financials
            "JPM",
            "BAC",
            "C",
            "WFC",
            # Materials
            "DD",

        ],
        n_components=4,  # 4 factors per the plan (market + energy + rotation + recovery)
        start_date="1991-03-01",
        end_date="1991-06-28",
    ),
    expected_dynamics={
        "energy": "give back shock premium (-10-15% vs SPY)",
        "defense": "moderate give back (-5-10% vs SPY)",
        "tech": "lead recovery (+15-20% vs SPY)",
        "financials": "participate in recovery (+10-15% vs SPY)",
        "materials": "recover with cyclicals (+10% vs SPY)",
    },
    tags=("geopolitical", "recovery", "ceasefire", "uncertain_normalization"),
    supersedes="ukraine_shock",  # Use when transitioning from acute shock to recovery
)

# Liberation Day recovery — tariff-uncertainty to normalization (Apr–Jul 2025)
# Supplementary regime for validating generalization to non-geopolitical uncertainty
LIBERATION_DAY_RECOVERY = HistoricFactorModel(
    name="liberation_day_recovery",
    description="Post-Liberation Day tariff recovery (Apr–Jul 2025)",
    narrative="""
    The Liberation Day tariff announcement (April 2, 2025) created a sharp
    domestic-policy uncertainty shock followed by gradual normalization as
    exemptions and negotiations clarified the actual impact. Unlike geopolitical
    shocks, this was driven by trade policy rather than energy supply disruption,
    but the sector rotation pattern was similar: initial tech weakness, defensive
    rotation into staples/healthcare, then broadening recovery as uncertainty
    resolved. This regime validates that the ShockArb approach generalizes beyond
    geopolitical events to any macro uncertainty shock with eventual resolution.
    """,
    universe=UniverseConfig(
        name="us_lib_day",
        market_etfs=_US_MACRO_ETFS,
        individual_stocks=_US_STOCKS,
        n_components=3,
        start_date="2025-04-01",
        end_date="2025-07-31",
    ),
    expected_dynamics={
        "tech": "initial weakness then recovery",
        "staples": "defensive outperformance early",
        "industrials": "tariff sensitivity then normalization",
        "financials": "rate-sensitive recovery",
    },
    tags=("policy_uncertainty", "recovery", "tariff", "domestic"),
    supersedes=None,
)

# =============================================================================
# Regime registry
# =============================================================================

REGIME_REGISTRY: Dict[str, HistoricFactorModel] = {
    "ukraine_shock":            UKRAINE_SHOCK,
    "gulf_war_recovery":        GULF_WAR_RECOVERY,
    "liberation_day_recovery":  LIBERATION_DAY_RECOVERY,
}


# =============================================================================
# Public API
# =============================================================================

def get_regime(name: str) -> HistoricFactorModel:
    """
    Look up a regime by name (case-insensitive).

    Parameters
    ----------
    name : str
        Regime identifier, e.g. "ukraine_shock" or "gulf_war_recovery".

    Returns
    -------
    HistoricFactorModel

    Raises
    ------
    ValueError
        If regime name is not found in the registry.
    """
    key = name.lower()
    if key not in REGIME_REGISTRY:
        available = ", ".join(sorted(REGIME_REGISTRY.keys()))
        raise ValueError(
            f"Unknown regime: '{name}'. Available regimes: {available}"
        )
    return REGIME_REGISTRY[key]


def list_regimes() -> List[str]:
    """
    Return a sorted list of all registered regime names.

    Returns
    -------
    list of str
        All regime names in alphabetical order.
    """
    return sorted(REGIME_REGISTRY.keys())


def find_by_tag(tag: str) -> List[HistoricFactorModel]:
    """
    Find all regimes matching a given tag.

    Parameters
    ----------
    tag : str
        Tag to search for, e.g. "recovery", "geopolitical".

    Returns
    -------
    list of HistoricFactorModel
        All regimes containing the tag (case-insensitive match).
    """
    tag_lower = tag.lower()
    matches = []
    for regime in REGIME_REGISTRY.values():
        if tag_lower in [t.lower() for t in regime.tags]:
            matches.append(regime)
    return matches
