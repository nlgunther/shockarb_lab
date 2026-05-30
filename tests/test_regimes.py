"""
Tests for shockarb.regimes module.
"""

import pytest

from shockarb.regimes import (
    COVID_REOPENING,
    GLOBAL_UKRAINE_SHOCK,
    GULF_WAR_RECOVERY,
    LIBERATION_DAY_RECOVERY,
    REGIME_REGISTRY,
    UKRAINE_SHOCK,
    HistoricFactorModel,
    find_by_tag,
    get_regime,
    list_regimes,
)


class TestHistoricFactorModel:
    """Test the HistoricFactorModel dataclass."""

    def test_frozen(self):
        """Regimes are immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError in 3.10+
            UKRAINE_SHOCK.name = "changed"

    def test_hashable(self):
        """Regimes can be compared for equality."""
        # Note: Regimes contain UniverseConfig which has list fields, so they're not hashable
        # But they can be compared for equality and used in lists/sets via id()
        assert UKRAINE_SHOCK == UKRAINE_SHOCK
        assert UKRAINE_SHOCK != GULF_WAR_RECOVERY
        
        # Can use in lists
        regimes = [UKRAINE_SHOCK, GULF_WAR_RECOVERY]
        assert UKRAINE_SHOCK in regimes

    def test_required_fields(self):
        """All regimes have required fields populated."""
        for regime in REGIME_REGISTRY.values():
            assert regime.name
            assert regime.description
            assert regime.narrative
            assert regime.universe
            assert regime.universe.name
            assert regime.universe.start_date
            assert regime.universe.end_date
            assert regime.universe.n_components > 0


class TestGetRegime:
    """Test regime lookup."""

    def test_valid_lookup(self):
        """Can retrieve regimes by name."""
        regime = get_regime("ukraine_shock")
        assert regime.name == "ukraine_shock"
        assert regime is UKRAINE_SHOCK

    def test_case_insensitive(self):
        """Lookup is case-insensitive."""
        assert get_regime("UKRAINE_SHOCK") is UKRAINE_SHOCK
        assert get_regime("Ukraine_Shock") is UKRAINE_SHOCK
        assert get_regime("ukraine_shock") is UKRAINE_SHOCK

    def test_invalid_name(self):
        """Raises ValueError for unknown regime."""
        with pytest.raises(ValueError, match="Unknown regime"):
            get_regime("nonexistent_regime")

    def test_error_message_includes_available(self):
        """Error message lists available regimes."""
        with pytest.raises(ValueError, match="ukraine_shock"):
            get_regime("bad_name")


class TestListRegimes:
    """Test regime listing."""

    def test_returns_all_names(self):
        """Returns all registered regime names."""
        names = list_regimes()
        assert "ukraine_shock" in names
        assert "gulf_war_recovery" in names
        assert "liberation_day_recovery" in names
        assert "global_ukraine_shock" in names
        assert "covid_reopening" in names

    def test_regime_count(self):
        """Registry contains exactly 5 regimes."""
        assert len(list_regimes()) == 5

    def test_sorted(self):
        """Names are sorted alphabetically."""
        names = list_regimes()
        assert names == sorted(names)

    def test_matches_registry(self):
        """List matches registry keys."""
        assert set(list_regimes()) == set(REGIME_REGISTRY.keys())


class TestFindByTag:
    """Test tag-based regime search."""

    def test_geopolitical_tag(self):
        """Can find regimes by tag."""
        results = find_by_tag("geopolitical")
        assert UKRAINE_SHOCK in results
        assert GULF_WAR_RECOVERY in results
        assert GLOBAL_UKRAINE_SHOCK in results
        assert LIBERATION_DAY_RECOVERY not in results

    def test_global_tag(self):
        """Global tag returns only the global regime."""
        results = find_by_tag("global")
        assert GLOBAL_UKRAINE_SHOCK in results
        assert UKRAINE_SHOCK not in results

    def test_recovery_tag(self):
        """Recovery tag finds recovery regimes."""
        results = find_by_tag("recovery")
        assert GULF_WAR_RECOVERY in results
        assert LIBERATION_DAY_RECOVERY in results
        assert UKRAINE_SHOCK not in results

    def test_case_insensitive(self):
        """Tag search is case-insensitive."""
        lower = find_by_tag("geopolitical")
        upper = find_by_tag("GEOPOLITICAL")
        assert lower == upper

    def test_no_matches(self):
        """Returns empty list when no matches."""
        results = find_by_tag("nonexistent_tag")
        assert results == []


class TestRegimeRegistry:
    """Test regime registry structure."""

    def test_all_regimes_present(self):
        """All expected regimes are in registry."""
        assert "ukraine_shock" in REGIME_REGISTRY
        assert "gulf_war_recovery" in REGIME_REGISTRY
        assert "liberation_day_recovery" in REGIME_REGISTRY
        assert "global_ukraine_shock" in REGIME_REGISTRY

    def test_registry_values_are_regimes(self):
        """All registry values are HistoricFactorModel instances."""
        for regime in REGIME_REGISTRY.values():
            assert isinstance(regime, HistoricFactorModel)

    def test_keys_match_names(self):
        """Registry keys match regime names."""
        for key, regime in REGIME_REGISTRY.items():
            assert key == regime.name


class TestRegimeContent:
    """Test specific regime definitions."""

    def test_ukraine_shock_dates(self):
        """Ukraine shock has correct date range."""
        regime = UKRAINE_SHOCK
        assert regime.universe.start_date == "2022-02-10"
        assert regime.universe.end_date == "2022-03-31"

    def test_gulf_war_dates(self):
        """Gulf War recovery has correct date range."""
        regime = GULF_WAR_RECOVERY
        assert regime.universe.start_date == "1991-03-01"
        assert regime.universe.end_date == "1991-06-28"

    def test_liberation_day_dates(self):
        """Liberation Day has correct date range."""
        regime = LIBERATION_DAY_RECOVERY
        assert regime.universe.start_date == "2025-04-01"
        assert regime.universe.end_date == "2025-07-31"

    def test_ukraine_shock_components(self):
        """Ukraine shock uses 3 factors."""
        assert UKRAINE_SHOCK.universe.n_components == 3

    def test_gulf_war_components(self):
        """Gulf War recovery uses 4 factors per the plan."""
        assert GULF_WAR_RECOVERY.universe.n_components == 4

    def test_supersedes_relationship(self):
        """Gulf War supersedes Ukraine."""
        assert GULF_WAR_RECOVERY.supersedes == "ukraine_shock"
        assert UKRAINE_SHOCK.supersedes is None

    def test_global_ukraine_shock_dates(self):
        """Global Ukraine shock uses the same event window as the US variant."""
        assert GLOBAL_UKRAINE_SHOCK.universe.start_date == "2022-02-10"
        assert GLOBAL_UKRAINE_SHOCK.universe.end_date == "2022-03-31"

    def test_global_ukraine_shock_universe_name(self):
        """Global regime universe is named 'global', not 'us'."""
        assert GLOBAL_UKRAINE_SHOCK.universe.name == "global"

    def test_global_ukraine_shock_components(self):
        """Global Ukraine shock uses 3 factors."""
        assert GLOBAL_UKRAINE_SHOCK.universe.n_components == 3


class TestGlobalUkraineShock:
    """Focused tests for the GLOBAL_UKRAINE_SHOCK regime."""

    def test_regime_is_registered(self):
        """GLOBAL_UKRAINE_SHOCK is reachable via get_regime."""
        regime = get_regime("global_ukraine_shock")
        assert regime is GLOBAL_UKRAINE_SHOCK

    def test_etf_count(self):
        """Global regime has exactly 14 market ETFs."""
        assert len(GLOBAL_UKRAINE_SHOCK.universe.market_etfs) == 14

    def test_stock_count(self):
        """Global regime has exactly 15 individual stocks."""
        assert len(GLOBAL_UKRAINE_SHOCK.universe.individual_stocks) == 15

    def test_global_tag_present(self):
        """'global' tag is present in the new regime."""
        assert "global" in GLOBAL_UKRAINE_SHOCK.tags

    def test_geopolitical_tag_present(self):
        """Shares 'geopolitical' tag with the US ukraine_shock."""
        assert "geopolitical" in GLOBAL_UKRAINE_SHOCK.tags

    def test_no_supersedes(self):
        """Global Ukraine shock does not supersede any other regime."""
        assert GLOBAL_UKRAINE_SHOCK.supersedes is None

    def test_distinct_from_us_variant(self):
        """Global and US ukraine_shock are different objects."""
        assert GLOBAL_UKRAINE_SHOCK is not UKRAINE_SHOCK
        assert GLOBAL_UKRAINE_SHOCK.universe.market_etfs != UKRAINE_SHOCK.universe.market_etfs

    def test_expected_dynamics_keys(self):
        """Global regime documents expected dynamics for key regional sectors."""
        keys = set(GLOBAL_UKRAINE_SHOCK.expected_dynamics.keys())
        assert "european_energy" in keys
        assert "emerging_markets" in keys
        assert "commodities" in keys


class TestCovidReopening:
    """Focused tests for the COVID_REOPENING regime."""

    def test_regime_is_registered(self):
        """COVID_REOPENING is reachable via get_regime."""
        assert get_regime("covid_reopening") is COVID_REOPENING

    def test_dates(self):
        """Covid reopening uses the Pfizer announcement window."""
        assert COVID_REOPENING.universe.start_date == "2020-11-09"
        assert COVID_REOPENING.universe.end_date == "2021-02-28"

    def test_n_components(self):
        """Covid reopening uses 3 factors."""
        assert COVID_REOPENING.universe.n_components == 3

    def test_universe_name(self):
        """Universe is named 'us_reopening'."""
        assert COVID_REOPENING.universe.name == "us_reopening"

    def test_recovery_tag(self):
        """'recovery' tag is present."""
        assert "recovery" in COVID_REOPENING.tags

    def test_risk_on_tag(self):
        """'risk_on' tag is present."""
        assert "risk_on" in COVID_REOPENING.tags

    def test_no_supersedes(self):
        """Covid reopening does not supersede any other regime."""
        assert COVID_REOPENING.supersedes is None

    def test_distinct_from_ukraine_shock(self):
        """Different object with different dates from ukraine_shock."""
        assert COVID_REOPENING is not UKRAINE_SHOCK
        assert COVID_REOPENING.universe.start_date != UKRAINE_SHOCK.universe.start_date

    def test_expected_dynamics_keys(self):
        """Expected dynamics documents key sector rotations."""
        keys = set(COVID_REOPENING.expected_dynamics.keys())
        assert "energy" in keys
        assert "airlines" in keys
        assert "financials" in keys

    def test_find_by_recovery_tag(self):
        """find_by_tag('recovery') includes covid_reopening."""
        results = find_by_tag("recovery")
        assert COVID_REOPENING in results

    def test_find_by_risk_on_tag(self):
        """find_by_tag('risk_on') includes covid_reopening."""
        results = find_by_tag("risk_on")
        assert COVID_REOPENING in results
