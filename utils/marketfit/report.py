"""
marketfit.report — Assemble the Markdown market report from snapshot, verdict, fundamentals, and news.

Pure module: takes already-loaded data and a Verdict, returns a Markdown string.
No network calls, no file I/O (caller handles saving).
"""
from __future__ import annotations

import os
from typing import Any
import pandas as pd

from trading_calendar import market_open_at_fetch


def _fmt(val: float | None, decimals: int = 2, prefix: str = "") -> str:
    if val is None or pd.isna(val):
        return "—"
    return f"{prefix}{val:,.{decimals}f}"


def _chg(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "—"
    arrow = "▲" if val >= 0 else "▼"
    return f"{arrow} {val:+.2f}%"


def _overseas_status(label: str, fetched_hour_et: int) -> str:
    eu   = {"FTSE", "DAX", "CAC", "Euro Stoxx"}
    asia = {"Nikkei", "Hang Seng", "Shanghai", "BSE Sensex", "ASX"}
    for kw in eu:
        if kw in label:
            return "open" if 2 <= fetched_hour_et <= 11 else "closed"
    for kw in asia:
        if kw in label:
            return "open" if fetched_hour_et <= 4 or fetched_hour_et >= 19 else "closed"
    return "likely open"


def _baseline_note(snapshot: dict[str, Any]) -> str:
    mode          = snapshot.get("mode", "daily")
    baseline_date = snapshot.get("baseline_date")
    baseline_str  = f" vs. close {baseline_date}" if baseline_date else ""

    if mode == "intraday":
        return f"⚡ **Intraday** — live prices{baseline_str}"

    if market_open_at_fetch(snapshot):
        return (
            f"⚠️ Market open — showing current price{baseline_str}. "
            "Run `market_data.py --intraday` for an explicit intraday fetch."
        )

    return f"Daily close{baseline_str}"



def build(
    snapshot: dict[str, Any],
    verdict,        # marketfit.rules.Verdict
    stale: bool = False,
) -> str:
    """
    Build the full Markdown market report string.

    Parameters
    ----------
    snapshot : dict
        Parsed market_snapshot.json.
    verdict : rules.Verdict
        Output of rules.evaluate(features.extract(snapshot)).
    stale : bool
        If True, prepend a stale-data warning.

    Returns
    -------
    str — complete Markdown document, ready to write to market_report.md.

    Example
    -------
        md = build(snapshot, verdict, stale=False)
        Path("data/market_report.md").write_text(md)
    """
    fetched_local = snapshot.get("fetched_at_local", "unknown")
    by_ticker = {t["ticker"]: t for t in snapshot.get("tickers", [])}

    try:
        fetched_hour_et = int(fetched_local.split(" ")[1].split(":")[0])
    except Exception:
        fetched_hour_et = 12

    def row(ticker: str) -> dict:
        return by_ticker.get(ticker, {})

    def close(ticker: str) -> str:
        r = row(ticker)
        if r.get("status") == "error" or r.get("close") is None:
            return "—"
        return f"{r['close']:,.2f}"

    def chg(ticker: str) -> str:
        r = row(ticker)
        if r.get("status") == "error" or r.get("chg_pct") is None:
            return "—"
        suffix = f" (stale {r['last_date']})" if r.get("stale") else ""
        return _chg(r["chg_pct"]) + suffix

    sector_tickers = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]
    sector_labels  = {
        "XLK": "Tech", "XLF": "Financials", "XLE": "Energy",
        "XLV": "Health Care", "XLI": "Industrials", "XLY": "Consumer Disc.",
        "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate",
        "XLB": "Materials", "XLC": "Comm. Services",
    }
    sectors_sorted = sorted(
        sector_tickers,
        key=lambda t: by_ticker.get(t, {}).get("chg_pct") or -999,
        reverse=True,
    )

    def sector_row(ticker: str, rank: int) -> str:
        label   = sector_labels.get(ticker, ticker)
        c       = close(ticker)
        ch      = chg(ticker)
        chg_val = by_ticker.get(ticker, {}).get("chg_pct")
        if rank < 3:
            return f"| **{label} ({ticker})** | **{c}** | **{ch}** |"
        # Only italicise bottom-3 if they are actually negative —
        # on broad up-days the worst sectors may still be positive.
        if rank >= len(sectors_sorted) - 3 and chg_val is not None and chg_val < 0:
            return f"| *{label} ({ticker})* | *{c}* | *{ch}* |"
        return f"| {label} ({ticker}) | {c} | {ch} |"

    eu_tickers   = ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E"]
    asia_tickers = ["^N225", "^HSI", "000001.SS", "^BSESN", "^AXJO"]
    am_tickers   = ["^BVSP"]
    eu_labels    = {
        "^FTSE": "FTSE 100 (London)", "^GDAXI": "DAX (Frankfurt)",
        "^FCHI": "CAC 40 (Paris)", "^STOXX50E": "Euro Stoxx 50",
    }
    asia_labels  = {
        "^N225": "Nikkei 225 (Tokyo)", "^HSI": "Hang Seng (HK)",
        "000001.SS": "Shanghai Composite", "^BSESN": "BSE Sensex (Mumbai)",
        "^AXJO": "ASX 200 (Sydney)",
    }
    am_labels = {"^BVSP": "Bovespa (São Paulo)"}

    def overseas_row(ticker: str, label: str) -> str:
        status = _overseas_status(label, fetched_hour_et)
        return f"| {label} | {close(ticker)} | {chg(ticker)} | ({status} at fetch) |"

    stale_warning = (
        "\n> ⚠️ **Stale data** — snapshot is more than 6 hours old. "
        "Run `python utils/market_data.py` for a fresh fetch.\n"
        if stale else ""
    )

    overall_emoji  = {"GOOD": "🟢", "CAUTION": "🟡", "POOR": "🔴"}.get(verdict.overall, "⚪")
    verdict_source = "" if verdict.source == "rules" else f" *(learned model — {verdict.source})*"
    baseline_note  = _baseline_note(snapshot)

    lines = [
        "# 📊 Market Close Report",
        f"**{fetched_local}** — {baseline_note}",
        stale_warning,
        "---",
        "",
        "## 🇺🇸 Broad Market",
        "",
        "| Index | Close | Change |",
        "|-------|------:|-------:|",
        f"| S&P 500 (SPY) | {close('SPY')} | {chg('SPY')} |",
        f"| Nasdaq 100 (QQQ) | {close('QQQ')} | {chg('QQQ')} |",
        f"| Russell 2000 (IWM) | {close('IWM')} | {chg('IWM')} |",
        f"| Dow Jones (DIA) | {close('DIA')} | {chg('DIA')} |",
        "",
        "---",
        "",
        "## 🏭 Sectors *(best → worst)*",
        "",
        "| Sector | Close | Change |",
        "|--------|------:|-------:|",
    ]
    for i, t in enumerate(sectors_sorted):
        lines.append(sector_row(t, i))

    lines += [
        "",
        "---",
        "",
        "## 💵 Bonds & Rates",
        "",
        "| Instrument | Close | Change |",
        "|------------|------:|-------:|",
        f"| 20yr Treasury (TLT) | {close('TLT')} | {chg('TLT')} |",
        f"| 7-10yr Treasury (IEF) | {close('IEF')} | {chg('IEF')} |",
        f"| High Yield (HYG) | {close('HYG')} | {chg('HYG')} |",
        f"| Inv. Grade Corp. (LQD) | {close('LQD')} | {chg('LQD')} |",
        "",
        "---",
        "",
        "## 🌍 Overseas Markets",
        "",
        f"*Snapshot fetched at {fetched_local} ET.*",
        "",
        "### Europe",
        "",
        "| Market | Close | Change | Status |",
        "|--------|------:|-------:|--------|",
    ]
    for t in eu_tickers:
        lines.append(overseas_row(t, eu_labels[t]))

    lines += [
        "",
        "### Asia-Pacific",
        "",
        "| Market | Close | Change | Status |",
        "|--------|------:|-------:|--------|",
    ]
    for t in asia_tickers:
        lines.append(overseas_row(t, asia_labels[t]))

    lines += [
        "",
        "### Americas (ex-US)",
        "",
        "| Market | Close | Change | Status |",
        "|--------|------:|-------:|--------|",
    ]
    for t in am_tickers:
        lines.append(overseas_row(t, am_labels[t]))

    lines += [
        "",
        "---",
        "",
        "## 📉 Risk Gauges",
        "",
        "| Gauge | Close | Change |",
        "|-------|------:|-------:|",
        f"| VIX | {close('^VIX')} | {chg('^VIX')} |",
        f"| Gold (GLD) | {close('GLD')} | {chg('GLD')} |",
        f"| Oil (USO) | {close('USO')} | {chg('USO')} |",
        "",
        "---",
        "",
        "## 🎯 ShockArb Fit Analysis",
        "",
        "### Condition Checks",
        "",
        verdict.as_markdown_table(),
        "",
        f"### Overall Fit: {overall_emoji} {verdict.overall}{verdict_source}",
        "",
        "### Analysis",
        "",
        _narrative(verdict),
        "",
        "### Recommendation",
        "",
        f"> {verdict.recommendation}",
        "",
        "---",
        f"*Snapshot: {fetched_local} | Mode: {snapshot.get('mode', 'daily')} | Baseline: {snapshot.get('baseline_date', '—')} | Source: {verdict.source}*",
    ]

    return "\n".join(lines)


def _narrative(verdict) -> str:
    """Generate a 3–4 sentence analysis paragraph from the verdict conditions."""
    parts = []

    trend_map = {
        "SHOCK":    "The market is in a shock regime — panic-driven selling is active.",
        "MELT-UP":  "The market is in a sustained melt-up with broad positive breadth and low fear.",
        "RECOVERY": "The market is in recovery mode after a prior selloff.",
        "CHOPPY":   "The market is choppy with no clear directional bias.",
    }
    parts.append(trend_map.get(verdict.trend_status, "Market trend is unclear."))

    if verdict.vix_status == "ELEVATED":
        parts.append(
            f"VIX is elevated ({verdict.notes.get('vix', '')}), signalling fear — "
            "panic-driven dislocations are more likely."
        )
    elif verdict.vix_status == "LOW":
        parts.append(
            f"VIX is low ({verdict.notes.get('vix', '')}), indicating complacency — "
            "stocks are trading close to factor-implied prices with little dislocation."
        )
    else:
        parts.append(f"VIX is moderate ({verdict.notes.get('vix', '')}).")

    if verdict.breadth_status == "NEGATIVE" and verdict.dispersion_status == "WIDE":
        parts.append(
            "Negative breadth and wide sector dispersion suggest genuine idiosyncratic "
            "moves — some stocks are underperforming their macro peers, creating ShockArb candidates."
        )
    elif verdict.breadth_status == "POSITIVE" and verdict.dispersion_status == "NARROW":
        parts.append(
            "Positive breadth and narrow sector dispersion indicate a uniform rally — "
            "rising tides are lifting all boats, erasing the gaps ShockArb needs."
        )
    else:
        parts.append(
            f"Breadth is {verdict.breadth_status.lower()} with "
            f"{verdict.dispersion_status.lower()} sector dispersion "
            f"({verdict.notes.get('dispersion', '')})."
        )

    if verdict.bond_status == "RISK-OFF":
        parts.append(
            "Bonds are bid alongside the equity selloff — a genuine risk-off move "
            "that reinforces the ShockArb thesis."
        )
    elif verdict.bond_status == "RISK-ON":
        parts.append(
            "Bonds are selling off with equities, suggesting the move is driven by "
            "rate/inflation concerns rather than pure panic."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Enhanced report builder — full <!-- LEARN --> template with LLM narratives
# ---------------------------------------------------------------------------

def _learn(section: str, difficulty: str, inputs: str, text: str) -> str:
    """Wrap narrative text in a <!-- LEARN --> block."""
    return (
        f'<!-- LEARN section="{section}" difficulty="{difficulty}" inputs="{inputs}" -->\n\n'
        f"{text.strip()}\n\n"
        f"<!-- /LEARN -->"
    )


def _inputs_str(**kwargs) -> str:
    """Format keyword args as 'k=v,k=v' for the inputs= attribute."""
    return ",".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)


def build_enhanced(
    snapshot:        dict[str, Any],
    verdict,
    narratives:      dict[str, str],
    picks_df:        Any | None = None,   # pandas DataFrame from live_alpha_us.csv
    news_dict:       dict | None = None,  # {ticker: [headline, ...]}
    fundamentals_df: Any | None = None,   # pandas DataFrame from fundamentals.csv
    stale:           bool = False,
) -> str:
    """
    Build the full enhanced market report with <!-- LEARN --> narrative sections.

    Called from cli.py when --llm is active.  Narratives dict is populated by
    MarketfitLLMClient.generate_narratives(); missing sections fall back to
    an empty placeholder rather than erroring.

    Parameters
    ----------
    snapshot        : market_snapshot.json dict
    verdict         : rules.Verdict
    narratives      : {section_name: text} from LLM (may be partial or {})
    picks_df        : DataFrame from live_alpha_us.csv (optional)
    news_dict       : {ticker: [headline, ...]} parsed from news.txt (optional)
    fundamentals_df : DataFrame from fundamentals.csv (optional)
    stale           : True if snapshot > 6h old

    Example
    -------
        md = build_enhanced(snapshot, verdict, narratives, picks_df=df)
        Path("data/market_report_intraday.md").write_text(md)
    """
    fetched_local = snapshot.get("fetched_at_local", "unknown")
    baseline_date = snapshot.get("baseline_date", "")
    by_ticker = {t["ticker"]: t for t in snapshot.get("tickers", [])}

    try:
        fetched_hour_et = int(fetched_local.split(" ")[1].split(":")[0])
    except Exception:
        fetched_hour_et = 12

    def row(ticker):
        return by_ticker.get(ticker, {})

    def close(ticker):
        r = row(ticker)
        if r.get("status") == "error" or r.get("close") is None:
            return "—"
        return f"{r['close']:,.2f}"

    def chg(ticker):
        r = row(ticker)
        if r.get("status") == "error" or r.get("chg_pct") is None:
            return "—"
        return _chg(r["chg_pct"])

    def chg_val(ticker):
        return by_ticker.get(ticker, {}).get("chg_pct")

    sector_tickers = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]
    sector_labels  = {
        "XLK":"Tech","XLF":"Financials","XLE":"Energy","XLV":"Health Care",
        "XLI":"Industrials","XLY":"Consumer Disc.","XLP":"Consumer Staples",
        "XLU":"Utilities","XLRE":"Real Estate","XLB":"Materials","XLC":"Comm. Services",
    }
    sectors_sorted = sorted(
        sector_tickers,
        key=lambda t: by_ticker.get(t, {}).get("chg_pct") or -999,
        reverse=True,
    )

    def sector_row(ticker, rank):
        label   = sector_labels.get(ticker, ticker)
        c, ch   = close(ticker), chg(ticker)
        cv      = chg_val(ticker)
        if rank < 3:
            return f"| **{label} ({ticker})** | **{c}** | **{ch}** |"
        if rank >= len(sectors_sorted) - 3 and cv is not None and cv < 0:
            return f"| *{label} ({ticker})* | *{c}* | *{ch}* |"
        return f"| {label} ({ticker}) | {c} | {ch} |"

    eu_tickers   = ["^FTSE","^GDAXI","^FCHI","^STOXX50E"]
    asia_tickers = ["^N225","^HSI","000001.SS","^BSESN","^AXJO"]
    am_tickers   = ["^BVSP"]
    eu_labels    = {"^FTSE":"FTSE 100 (London)","^GDAXI":"DAX (Frankfurt)",
                    "^FCHI":"CAC 40 (Paris)","^STOXX50E":"Euro Stoxx 50"}
    asia_labels  = {"^N225":"Nikkei 225 (Tokyo)","^HSI":"Hang Seng (HK)",
                    "000001.SS":"Shanghai Composite","^BSESN":"BSE Sensex (Mumbai)",
                    "^AXJO":"ASX 200 (Sydney)"}
    am_labels    = {"^BVSP":"Bovespa (São Paulo)"}

    def overseas_row(ticker, label):
        return f"| {label} | {close(ticker)} | {chg(ticker)} | ({_overseas_status(label, fetched_hour_et)} at fetch) |"

    stale_warning = (
        "\n> ⚠️ **Stale data** — snapshot is more than 6 hours old. "
        "Run `python utils/market_data.py` for a fresh fetch.\n"
        if stale else ""
    )

    overall_emoji  = {"GOOD":"🟢","CAUTION":"🟡","POOR":"🔴"}.get(verdict.overall, "⚪")
    baseline_note  = _baseline_note(snapshot)
    verdict_source = "" if verdict.source == "rules" else f" *(learned model — {verdict.source})*"

    def narr(section):
        """Return narrative text for a section, or empty placeholder."""
        return narratives.get(section, "*(narrative not generated)*")

    # Pre-compute inputs= strings for each LEARN tag
    spy_c  = chg_val("SPY")  or 0
    qqq_c  = chg_val("QQQ")  or 0
    dia_c  = chg_val("DIA")  or 0
    iwm_c  = chg_val("IWM")  or 0
    tlt_c  = chg_val("TLT")  or 0
    hyg_c  = chg_val("HYG")  or 0
    lqd_c  = chg_val("LQD")  or 0
    ief_c  = chg_val("IEF")  or 0
    gld_c  = chg_val("GLD")  or 0
    oil_c  = chg_val("USO")  or 0
    xlk_c  = chg_val("XLK")  or 0
    vix_cl = by_ticker.get("^VIX",{}).get("close") or 0
    vix_c  = chg_val("^VIX") or 0
    xlk_vs_spy = xlk_c - spy_c

    overseas = [t for t in snapshot.get("tickers",[]) if t.get("group")=="overseas" and t.get("chg_pct") is not None]
    ov_vals  = [t["chg_pct"] for t in overseas]
    ov_mean  = round(sum(ov_vals)/len(ov_vals),2) if ov_vals else 0
    hsi_c    = chg_val("^HSI") or 0
    n225_c   = chg_val("^N225") or 0
    cac_c    = chg_val("^FCHI") or 0
    bvsp_c   = chg_val("^BVSP") or 0

    top3 = sectors_sorted[:3]
    disp = round((by_ticker.get(sectors_sorted[0],{}).get("chg_pct",0) or 0) -
                 (by_ticker.get(sectors_sorted[-1],{}).get("chg_pct",0) or 0), 2)

    lines = [
        "# 📊 ShockArb Market Report",
        f"**{fetched_local}** — {baseline_note}",
        stale_warning,
        "",
        _learn(
            "executive_summary", "judgment",
            _inputs_str(spy_chg=f"{spy_c:+.2f}",qqq_chg=f"{qqq_c:+.2f}",
                        dia_chg=f"{dia_c:+.2f}",iwm_chg=f"{iwm_c:+.2f}",
                        vix=f"{vix_cl:.2f}",vix_chg=f"{vix_c:+.2f}",
                        oil_chg=f"{oil_c:+.2f}",tlt_chg=f"{tlt_c:+.2f}",
                        verdict=verdict.overall,trend=verdict.trend_status,
                        baseline_date=baseline_date),
            narr("executive_summary"),
        ),
        "---", "",
        "## 🇺🇸 Broad Market", "",
        "| Index | Close | Change |",
        "|-------|------:|-------:|",
        f"| S&P 500 (SPY) | {close('SPY')} | {chg('SPY')} |",
        f"| Nasdaq 100 (QQQ) | {close('QQQ')} | {chg('QQQ')} |",
        f"| Russell 2000 (IWM) | {close('IWM')} | {chg('IWM')} |",
        f"| Dow Jones (DIA) | {close('DIA')} | {chg('DIA')} |",
        "",
        _learn(
            "broad_market_interpretation", "analytic",
            _inputs_str(spy_chg=f"{spy_c:+.2f}",qqq_chg=f"{qqq_c:+.2f}",
                        dia_chg=f"{dia_c:+.2f}",iwm_chg=f"{iwm_c:+.2f}",
                        qqq_rel=f"{qqq_c-spy_c:+.2f}",iwm_rel=f"{iwm_c-spy_c:+.2f}"),
            narr("broad_market_interpretation"),
        ),
        "---", "",
        "## 🏭 Sectors *(best → worst)*", "",
        "| Sector | Close | Change |",
        "|--------|------:|-------:|",
    ]
    for i, t in enumerate(sectors_sorted):
        lines.append(sector_row(t, i))

    top3_str = ",".join(f"{sector_labels.get(t,t)}_chg={chg_val(t):+.2f}" for t in top3)
    lines += [
        "",
        _learn(
            "sector_rotation_story", "judgment",
            _inputs_str(**{f"{sector_labels.get(t,t).lower().replace(' ','_')}_chg": f"{chg_val(t):+.2f}" for t in top3},
                        xlk_chg=f"{xlk_c:+.2f}",xlk_vs_spy=f"{xlk_vs_spy:+.2f}",
                        dispersion=f"{disp:.2f}pp",breadth=f"{verdict.breadth_status}",
                        oil_chg=f"{oil_c:+.2f}",baseline_date=baseline_date),
            narr("sector_rotation_story"),
        ),
        "---", "",
        "## 💵 Bonds & Rates", "",
        "| Instrument | Close | Change |",
        "|------------|------:|-------:|",
        f"| 20yr Treasury (TLT) | {close('TLT')} | {chg('TLT')} |",
        f"| 7-10yr Treasury (IEF) | {close('IEF')} | {chg('IEF')} |",
        f"| High Yield (HYG) | {close('HYG')} | {chg('HYG')} |",
        f"| Inv. Grade Corp. (LQD) | {close('LQD')} | {chg('LQD')} |",
        "",
        _learn(
            "bond_signal_interpretation", "analytic",
            _inputs_str(tlt_chg=f"{tlt_c:+.2f}",ief_chg=f"{ief_c:+.2f}",
                        hyg_chg=f"{hyg_c:+.2f}",lqd_chg=f"{lqd_c:+.2f}",
                        bond_status=verdict.bond_status,spy_chg=f"{spy_c:+.2f}",
                        vix_chg=f"{vix_c:+.2f}"),
            narr("bond_signal_interpretation"),
        ),
        "---", "",
        "## 🌍 Overseas Markets", "",
        f"*Snapshot fetched at {fetched_local} ET.*", "",
        "### Europe", "",
        "| Market | Close | Change | Status |",
        "|--------|------:|-------:|--------|",
    ]
    for t in eu_tickers:
        lines.append(overseas_row(t, eu_labels[t]))

    lines += ["", "### Asia-Pacific", "",
              "| Market | Close | Change | Status |", "|--------|------:|-------:|--------|"]
    for t in asia_tickers:
        lines.append(overseas_row(t, asia_labels[t]))

    lines += ["", "### Americas (ex-US)", "",
              "| Market | Close | Change | Status |", "|--------|------:|-------:|--------|"]
    for t in am_tickers:
        lines.append(overseas_row(t, am_labels[t]))

    lines += [
        "",
        _learn(
            "overseas_read", "analytic",
            _inputs_str(hsi_chg=f"{hsi_c:+.2f}",n225_chg=f"{n225_c:+.2f}",
                        cac_chg=f"{cac_c:+.2f}",bvsp_chg=f"{bvsp_c:+.2f}",
                        overseas_breadth=f"{ov_mean:+.2f}"),
            narr("overseas_read"),
        ),
        "---", "",
        "## 📉 Risk Gauges", "",
        "| Gauge | Close | Change |",
        "|-------|------:|-------:|",
        f"| VIX | {close('^VIX')} | {chg('^VIX')} |",
        f"| Gold (GLD) | {close('GLD')} | {chg('GLD')} |",
        f"| Oil (USO) | {close('USO')} | {chg('USO')} |",
        "",
        _learn(
            "risk_gauge_read", "analytic",
            _inputs_str(vix=f"{vix_cl:.2f}",vix_chg=f"{vix_c:+.2f}",
                        gold_chg=f"{gld_c:+.2f}",oil_chg=f"{oil_c:+.2f}",
                        vix_status=verdict.vix_status,baseline_date=baseline_date),
            narr("risk_gauge_read"),
        ),
        "---", "",
        "## 🎯 ShockArb Fit Analysis", "",
        "### Condition Checks", "",
        verdict.as_markdown_table(), "",
        f"### Overall Fit: {overall_emoji} {verdict.overall}{verdict_source}", "",
        _learn(
            "shockarb_fit_analysis", "judgment",
            _inputs_str(verdict=verdict.overall,score=verdict.score,
                        trend=verdict.trend_status,breadth=verdict.breadth_status,
                        vix_status=verdict.vix_status,dispersion=verdict.dispersion_status,
                        tech=verdict.tech_status,bond=verdict.bond_status),
            narr("shockarb_fit_analysis"),
        ),
        "",
        "### Recommendation", "",
        _learn(
            "recommendation", "template",
            _inputs_str(verdict=verdict.overall,score=verdict.score),
            f"> {verdict.recommendation}",
        ),
    ]

    # Picks table + commentary
    if picks_df is not None and not picks_df.empty and fundamentals_df is not None:
        lines += ["", "---", "", "## 📋 ShockArb Candidates — Fundamental Cross-Check", ""]
        # Build table from merged picks + fundamentals
        try:
            import pandas as pd
            merged = picks_df.merge(fundamentals_df, left_on="Ticker", right_on="Ticker", how="inner")
            lines += ["| Ticker | Fwd P/E | Analyst Tgt | Implied Upside | Signal (Conf.Δ) |",
                      "|--------|--------:|------------:|---------------:|----------------:|"]
            for _, r in merged.iterrows():
                try:
                    price  = float(str(r.get("Price","")).replace("$","").replace(",",""))
                    tgt    = float(str(r.get("Analyst Tgt","")).replace("$","").replace(",",""))
                    upside = f"{(tgt/price-1)*100:.0f}%" if price > 0 else "—"
                except Exception:
                    upside = "—"
                lines.append(
                    f"| {r['Ticker']} | {r.get('Fwd P/E','—')}x | "
                    f"{r.get('Analyst Tgt','—')} | {upside} | "
                    f"{r.get('confidence_delta', r.get('Conf.Δ','—'))} |"
                )
            lines += [
                "",
                _learn(
                    "picks_commentary", "judgment",
                    _inputs_str(verdict=verdict.overall, n_picks=len(merged)),
                    narr("picks_commentary"),
                ),
            ]
        except Exception:
            pass  # picks table is optional — never crash

    # News catalyst section
    if news_dict:
        lines += ["", "---", "", "## 📡 Catalyst Feed Highlights", ""]
        shown = 0
        for ticker, headlines in list(news_dict.items())[:8]:
            for h in headlines[:2]:
                lines.append(f"- **[{ticker}]** {h}")
                shown += 1
        if shown:
            lines += [
                "",
                _learn(
                    "catalyst_summary", "judgment",
                    _inputs_str(verdict=verdict.overall, n_tickers=len(news_dict)),
                    narr("catalyst_summary") if "catalyst_summary" in narratives
                    else narr("watch_list"),
                ),
            ]

    lines += [
        "", "---", "",
        _learn(
            "watch_list", "judgment",
            _inputs_str(verdict=verdict.overall,trend=verdict.trend_status,
                        oil_chg=f"{oil_c:+.2f}",tlt_chg=f"{tlt_c:+.2f}",
                        vix=f"{vix_cl:.2f}"),
            narr("watch_list"),
        ),
        "---",
        f"*Snapshot: {fetched_local} | Mode: {snapshot.get('mode','daily')} | "
        f"Baseline: {snapshot.get('baseline_date','—')} | Source: {verdict.source} | LLM: enabled*",
    ]

    return "\n".join(lines)
