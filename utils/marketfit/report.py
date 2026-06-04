"""
marketfit.report — Assemble the Markdown market report from snapshot + verdict.

Pure module: takes already-loaded data and a Verdict, returns a Markdown string.
No network calls, no file I/O (caller handles saving).
"""

from __future__ import annotations

from typing import Any


def _fmt(val: float | None, decimals: int = 2, prefix: str = "") -> str:
    if val is None:
        return "—"
    return f"{prefix}{val:,.{decimals}f}"


def _chg(val: float | None) -> str:
    if val is None:
        return "—"
    arrow = "▲" if val >= 0 else "▼"
    return f"{arrow} {val:+.2f}%"


def _market_open_note(group: str, fetched_hour_et: int) -> str:
    """Rough open/closed indicator based on fetch hour (ET)."""
    if group == "overseas":
        return ""   # handled per-region below
    return ""


def _overseas_status(label: str, fetched_hour_et: int) -> str:
    """
    Approximate market-open status at fetch time.
    Europe: ~2:30am–11:30am ET. Asia-Pac: ~7pm–4am ET. Americas ex-US varies.
    """
    eu = {"FTSE", "DAX", "CAC", "Euro Stoxx"}
    asia = {"Nikkei", "Hang Seng", "Shanghai", "BSE Sensex", "ASX"}

    for kw in eu:
        if kw in label:
            return "open" if 2 <= fetched_hour_et <= 11 else "closed"
    for kw in asia:
        if kw in label:
            return "open" if fetched_hour_et <= 4 or fetched_hour_et >= 19 else "closed"
    return "likely open"


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
    """
    fetched_local = snapshot.get("fetched_at_local", "unknown")
    by_ticker = {t["ticker"]: t for t in snapshot.get("tickers", [])}

    # Parse fetch hour for open/closed notes
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
        return _chg(r["chg_pct"])

    # -------------------------------------------------------------------------
    # Sector sort (best → worst)
    # -------------------------------------------------------------------------
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
        label = sector_labels.get(ticker, ticker)
        c = close(ticker)
        ch = chg(ticker)
        cell = f"| {label} ({ticker}) | {c} | {ch} |"
        if rank < 3:
            return f"| **{label} ({ticker})** | **{c}** | **{ch}** |"
        if rank >= len(sectors_sorted) - 3:
            return f"| *{label} ({ticker})* | *{c}* | *{ch}* |"
        return cell

    # -------------------------------------------------------------------------
    # Overseas grouping
    # -------------------------------------------------------------------------
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
    am_labels    = {"^BVSP": "Bovespa (São Paulo)"}

    def overseas_row(ticker: str, label: str) -> str:
        c  = close(ticker)
        ch = chg(ticker)
        status = _overseas_status(label, fetched_hour_et)
        return f"| {label} | {c} | {ch} | ({status} at fetch) |"

    # -------------------------------------------------------------------------
    # Assemble
    # -------------------------------------------------------------------------
    stale_warning = (
        "\n> ⚠️ **Stale data** — snapshot is more than 6 hours old. "
        "Run `python utils/market_data.py` for a fresh fetch.\n"
        if stale else ""
    )

    overall_emoji = {"GOOD": "🟢", "CAUTION": "🟡", "POOR": "🔴"}.get(verdict.overall, "⚪")
    verdict_source = "" if verdict.source == "rules" else f" *(learned model — {verdict.source})*"

    lines = [
        f"# 📊 Market Close Report",
        f"**{fetched_local}**",
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
        f"## 🌍 Overseas Markets",
        f"",
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
        f"*Snapshot: {fetched_local} | Source: {verdict.source}*",
    ]

    return "\n".join(lines)


def _narrative(verdict) -> str:
    """Generate a 3–4 sentence analysis paragraph from the verdict conditions."""
    parts = []

    # Trend sentence
    trend_map = {
        "SHOCK":   "The market is in a shock regime — panic-driven selling is active.",
        "MELT-UP": "The market is in a sustained melt-up with broad positive breadth and low fear.",
        "RECOVERY":"The market is in recovery mode after a prior selloff.",
        "CHOPPY":  "The market is choppy with no clear directional bias.",
    }
    parts.append(trend_map.get(verdict.trend_status, "Market trend is unclear."))

    # VIX sentence
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

    # Breadth + dispersion sentence
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

    # Bond signal sentence
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
