"""
Tests for shockarb.regimes module.
"""

import pytest

from shockarb.regimes import (
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
        assert LIBERATION_DAY_RECOVERY not in results

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
