"""Tests for shockarb.report_compare."""

from __future__ import annotations

import textwrap

import pytest

from shockarb.report_compare import (
    ReportData,
    _clean_number,
    build_comparison,
    parse_report,
    print_comparison,
    write_comparison_md,
)


REPORT_A = textwrap.dedent("""\
    # 📋 ShockArb Stock Opportunity Report

    **2026-06-12 08:00 UTC**
    *Source: live_alpha_us.csv + fundamentals.csv + news.txt*

    > Thresholds applied: r² ≥ 0.65 | conf.Δ ≥ 0.020 | analyst upside ≥ 5%

    ---

    ## ✅ Act on These (1 candidates)

    | Ticker | R²    | Conf.Δ  | Price    | Analyst Tgt | Upside | Fwd P/E |
    | ------ | -----:| -------:| --------:| -----------:| ------:| -------:|
    | ETN    | 0.723 | +0.0260 | \\$393.64 | \\$451.73    | +14.8% | 25.0x   |

    ---

    ## ⚠️ Watch (0 candidates)

    *No candidates in watch tier.*

    ---

    ## ❌ Excluded (2 tickers)

    | Ticker | Reason                                             |
    | ------ | -------------------------------------------------- |
    | MSFT   | confidence_delta=+0.0053 below threshold (0.020)   |
    | CPRT   | r²=0.063 below threshold (0.65) — model fit too weak |
    """)


REPORT_B = textwrap.dedent("""\
    # 📋 ShockArb Stock Opportunity Report

    **2026-06-12 07:29 UTC**
    *Source: live_alpha_us.csv + fundamentals.csv + news.txt*

    > Thresholds applied: r² ≥ 0.65 | conf.Δ ≥ 0.020 | analyst upside ≥ 5%

    ---

    ## ✅ Act on These (1 candidates)

    | Ticker | R²    | Conf.Δ  | Price    | Analyst Tgt | Upside | Fwd P/E |
    | ------ | -----:| -------:| --------:| -----------:| ------:| -------:|
    | CPRT   | 0.812 | +0.0274 | \\$31.06  | \\$41.44     | +33.4% | 18.4x   |

    ---

    ## ⚠️ Watch (1 candidates)

    | Ticker | R²    | Conf.Δ  | Price    | Analyst Tgt | Upside | Fwd P/E |
    | ------ | -----:| -------:| --------:| -----------:| ------:| -------:|
    | ETN    | 0.700 | +0.0210 | \\$390.00 | \\$450.00    | +15.0% | 24.0x   |

    ---

    ## ❌ Excluded (1 tickers)

    | Ticker | Reason                                            |
    | ------ | ------------------------------------------------- |
    | MSFT   | confidence_delta=+0.0053 below threshold (0.020) |
    """)


VERDICTS_CSV = textwrap.dedent("""\
    ticker,tier,reason,r_squared,confidence_delta,analyst_upside,price,analyst_target,fwd_pe,cluster,rvol,rvol_window,intraday_price,intraday_chg_pct,news_headlines,warnings
    ETN,INCLUDE,"r²=0.723, conf.Δ=+0.0260, analyst upside +14.8% — all gates pass",0.723,0.026,0.148,393.64,451.73,25.0,,,,,,,
    MSFT,EXCLUDE,confidence_delta=+0.0053 below threshold (0.020),0.345,0.0053,0.05,300.0,330.0,28.0,Mega-Cap Tech,,,,,,
    CPRT,EXCLUDE,r²=0.063 below threshold (0.65) — model fit too weak,0.063,0.0274,0.334,31.06,41.44,18.4,,,,,,,
    """)


@pytest.fixture
def report_paths(tmp_path):
    a_dir = tmp_path / "iran_shock"
    a_dir.mkdir()
    a_path = a_dir / "stock_report_20260612_0800.md"
    a_path.write_text(REPORT_A, encoding="utf-8")

    b_path = tmp_path / "stock_report_20260612_0729.md"
    b_path.write_text(REPORT_B, encoding="utf-8")

    return str(a_path), str(b_path)


@pytest.fixture
def verdicts_csv_path(tmp_path):
    csv_dir = tmp_path / "iran_shock"
    csv_dir.mkdir(exist_ok=True)
    p = csv_dir / "stock_report_20260612_0945_verdicts.csv"
    p.write_text(VERDICTS_CSV, encoding="utf-8")
    return str(p)


class TestCleanNumber:
    def test_strips_currency_and_backslash(self):
        assert _clean_number("\\$393.64") == 393.64

    def test_strips_percent_sign_and_keeps_sign(self):
        assert _clean_number("+14.8%") == 14.8

    def test_strips_multiplier_suffix_and_parenthetical(self):
        assert _clean_number("1.3x (20d)") == 1.3

    def test_em_dash_returns_none(self):
        assert _clean_number("—") is None


class TestParseReport:
    def test_parses_metadata_and_label(self, report_paths):
        a_path, b_path = report_paths

        a = parse_report(a_path)
        assert a.timestamp == "2026-06-12 08:00"
        assert a.thresholds.startswith("r² ≥ 0.65")
        assert a.label == "iran_shock_0800"

        b = parse_report(b_path)
        assert b.label.endswith("_0729")

    def test_parses_tiers_and_stats(self, report_paths):
        a_path, _ = report_paths
        a = parse_report(a_path)

        assert a.tickers["ETN"]["tier"] == "act_on"
        assert a.tickers["ETN"]["r_squared"] == pytest.approx(0.723)
        assert a.tickers["ETN"]["conf_delta"] == pytest.approx(0.0260)
        assert a.tickers["ETN"]["upside"] == pytest.approx(14.8)

        assert a.tickers["MSFT"]["tier"] == "excluded"
        assert "below threshold" in a.tickers["MSFT"]["reason"]

        assert a.counts == {"act_on": 1, "watch": 0, "excluded": 2}


class TestBuildComparison:
    def test_flags_ticker_with_differing_tier(self, report_paths):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, flagged = build_comparison(reports)

        # ETN is "act_on" in report A but "watch" in report B.
        assert flagged["ETN"]
        # CPRT is "excluded" in A but "act_on" in B.
        assert flagged["CPRT"]
        # MSFT is "excluded" in both.
        assert not flagged["MSFT"]

    def test_union_of_tickers_covers_both_reports(self, report_paths):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, _ = build_comparison(reports)

        assert set(comparison.index) == {"ETN", "MSFT", "CPRT"}


class TestPrintComparison:
    def test_runs_without_error(self, report_paths, capsys):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, flagged = build_comparison(reports)

        print_comparison(reports, comparison, flagged)

        out = capsys.readouterr().out
        assert "REPORT COMPARISON" in out
        assert "ETN" in out
        assert "⚠" in out

    def test_stats_include_flagged_excluded_ticker_with_reason(self, report_paths, capsys):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, flagged = build_comparison(reports)

        print_comparison(reports, comparison, flagged)

        out = capsys.readouterr().out
        # CPRT is "excluded" in report A (flagged, since it's "act_on" in B) —
        # its stats and exclusion reason should appear under report A's Stats.
        stats_a = out.split("STATS — iran_shock_0800")[1].split("STATS —")[0]
        assert "CPRT" in stats_a
        assert "below threshold" in stats_a


class TestParseVerdictsCsv:
    def test_maps_tiers_to_lowercase(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.tickers["ETN"]["tier"] == "act_on"
        assert r.tickers["MSFT"]["tier"] == "excluded"
        assert r.tickers["CPRT"]["tier"] == "excluded"

    def test_excluded_tickers_keep_full_stats(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.tickers["MSFT"]["r_squared"] == pytest.approx(0.345)
        assert r.tickers["MSFT"]["conf_delta"] == pytest.approx(0.0053)

    def test_upside_converted_to_percent(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.tickers["ETN"]["upside"] == pytest.approx(14.8)

    def test_label_strips_verdicts_suffix(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.label == "iran_shock_0945"

    def test_counts_tally_tiers(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.counts == {"act_on": 1, "excluded": 2}

    def test_timestamp_and_thresholds_are_none(self, verdicts_csv_path):
        r = parse_report(verdicts_csv_path)
        assert r.timestamp is None
        assert r.thresholds is None
        assert r.source is None


class TestParseReportDispatch:
    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "notes.txt"
        p.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported report file type"):
            parse_report(str(p))


class TestMixedFormatComparison:
    def test_md_and_csv_reports_can_be_compared(self, report_paths, verdicts_csv_path):
        a_path, _ = report_paths
        reports = [parse_report(a_path), parse_report(verdicts_csv_path)]
        comparison, flagged = build_comparison(reports)

        assert set(comparison.index) == {"ETN", "MSFT", "CPRT"}
        # ETN is "act_on" in both reports — not flagged.
        assert not flagged["ETN"]
        # CPRT is "excluded" in both A and the verdicts CSV — not flagged.
        assert not flagged["CPRT"]
        # MSFT is "excluded" in both.
        assert not flagged["MSFT"]

    def test_union_of_tickers_covers_both_reports(self, report_paths):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, _ = build_comparison(reports)

        assert set(comparison.index) == {"ETN", "MSFT", "CPRT"}


class TestWriteComparisonMd:
    def test_stats_section_includes_flagged_excluded_ticker_with_reason(
        self, tmp_path, report_paths
    ):
        a_path, b_path = report_paths
        reports = [parse_report(a_path), parse_report(b_path)]
        comparison, flagged = build_comparison(reports)

        out_path = str(tmp_path / "comparison.md")
        write_comparison_md(out_path, reports, comparison, flagged)
        text = open(out_path, encoding="utf-8").read()

        # CPRT is flagged (excluded in A, act_on in B) — its
        # Stats — iran_shock_0800 row should show its exclusion reason.
        stats_a = text.split("## Stats — iran_shock_0800")[1].split("## Stats —")[0]
        assert "CPRT" in stats_a
        assert "below threshold" in stats_a

    def test_stats_section_includes_fwd_pe_from_verdicts_csv(
        self, tmp_path, report_paths, verdicts_csv_path
    ):
        a_path, _ = report_paths
        reports = [parse_report(a_path), parse_report(verdicts_csv_path)]
        comparison, flagged = build_comparison(reports)

        out_path = str(tmp_path / "comparison.md")
        write_comparison_md(out_path, reports, comparison, flagged)
        text = open(out_path, encoding="utf-8").read()

        # ETN is "act_on" in both reports, so it's an interesting ticker.
        # Its fwd_pe from the verdicts CSV (25.0) should appear.
        stats_csv = text.split("## Stats — iran_shock_0945")[1]
        assert "25.0x" in stats_csv
