"""
stockfit.report — Assemble the Markdown stock opportunity report.

Pure module: takes a list of StockVerdict objects and metadata, returns a
Markdown string.  No network calls, no file I/O (caller handles saving).

Sections
--------
  Header           — date, data source, threshold summary
  INCLUDE          — ranked table + per-ticker narrative block (with <!-- LEARN --> tags)
  WATCH            — ranked table with brief reason
  EXCLUDE          — compact table with reason
  Data Quality     — flags from target_below_price etc.

LEARN markup
------------
  Each per-ticker narrative block is wrapped in:
    <!-- LEARN section="stock_analysis" ticker="ETN" difficulty="intermediate"
         inputs="r2=0.693 conf_delta=+0.0283 upside=+14.1%" -->
    ... narrative ...
    <!-- /LEARN -->
  This marks the content as a training corpus entry for future LLM fine-tuning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from stockfit.rules import StockVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.{decimals}f}%"


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "—"
    return f"\\${v:,.2f}"


def _learn_open(ticker: str, r2: float, conf_delta: float, upside: float | None) -> str:
    upside_str = f"{upside * 100:+.1f}%" if upside is not None else "N/A"
    return (
        f'<!-- LEARN section="stock_analysis" ticker="{ticker}" difficulty="intermediate" '
        f'inputs="r2={r2:.3f} conf_delta={conf_delta:+.4f} upside={upside_str}" -->'
    )


_LEARN_CLOSE = "<!-- /LEARN -->"


def _table_header() -> str:
    return (
        "| Ticker | R² | Conf.Δ | Price | Analyst Tgt | Upside | Fwd P/E | RVOL | Intraday |\n"
        "| ------ | --:| ------:| -----:| -----------:| ------:| -------:| ----:| --------:|"
    )


# ---------------------------------------------------------------------------
# Basic report (no LLM)
# ---------------------------------------------------------------------------

def build(
    verdicts:  list[StockVerdict],
    date_str:  str | None = None,
    source:    str = "live_alpha_us.csv + fundamentals.csv + news.txt",
    thresholds: dict[str, Any] | None = None,
    stale:     bool = False,
) -> str:
    """
    Build the basic stock opportunity Markdown report (no LLM narratives).

    Parameters
    ----------
    verdicts   : output of rules.evaluate_all()
    date_str   : report date string (e.g. "2026-06-06 15:51"); defaults to now
    source     : data source note for the report header
    thresholds : {min_r2, min_conf_delta, min_upside} overrides for display
    stale      : whether the underlying data is stale

    Returns
    -------
    str — complete Markdown report ready to write to disk.
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    thresh = thresholds or {}
    min_r2   = thresh.get("min_r2",   0.65)
    min_cd   = thresh.get("min_conf_delta", 0.020)
    min_up   = thresh.get("min_upside", 0.05)

    include = [v for v in verdicts if v.tier == "INCLUDE"]
    watch   = [v for v in verdicts if v.tier == "WATCH"]
    exclude = [v for v in verdicts if v.tier == "EXCLUDE"]

    stale_note = "\n> ⚠️ **Data may be stale.** Re-run the pipeline for fresh signals.\n" if stale else ""

    parts = [
        "# 📋 ShockArb Stock Opportunity Report",
        "",
        f"**{date_str}**",
        f"*Source: {source}*",
        stale_note,
        f"> Thresholds applied: r² ≥ {min_r2:.2f} | conf.Δ ≥ {min_cd:.3f} | analyst upside ≥ {min_up * 100:.0f}%",
        "",
        "---",
        "",
    ]

    # --- INCLUDE section ---
    parts += [f"## ✅ Act on These ({len(include)} candidates)", ""]
    if include:
        parts += [_table_header()]
        for v in include:
            parts.append(v.as_markdown_row())
        parts.append("")
        for v in include:
            parts += _ticker_detail_basic(v)
    else:
        parts += ["*No tickers pass all filters for this session.*", ""]

    parts += ["---", ""]

    # --- WATCH section ---
    parts += [f"## ⚠️ Watch ({len(watch)} candidates)", ""]
    if watch:
        parts += [_table_header()]
        for v in watch:
            parts.append(v.as_markdown_row())
        parts.append("")
        for v in watch:
            parts += [
                f"**{v.ticker}** — {v.reason}",
                "",
            ]
            if v.warnings:
                for w in v.warnings:
                    parts.append(f"> ⚠️ {w}")
                parts.append("")
    else:
        parts += ["*No candidates in watch tier.*", ""]

    parts += ["---", ""]

    # --- EXCLUDE section ---
    parts += [f"## ❌ Excluded ({len(exclude)} tickers)", ""]
    if exclude:
        parts += [
            "| Ticker | Reason |",
            "| ------ | ------ |",
        ]
        for v in exclude:
            reason = v.reason.replace("|", "\\|")
            parts.append(f"| {v.ticker} | {reason} |")
        parts.append("")
    else:
        parts += ["*No tickers excluded.*", ""]

    parts += ["---", ""]

    # --- Data quality flags ---
    dq_flags = [v for v in verdicts if v.tier == "EXCLUDE" and v.reason.startswith("Analyst target below")]
    if dq_flags:
        parts += ["## 📌 Data Quality Flags", ""]
        for v in dq_flags:
            upside = _fmt_pct(v.analyst_upside)
            parts.append(
                f"- **{v.ticker}**: analyst target {_fmt_price(v.analyst_target)} "
                f"vs price {_fmt_price(v.price)} — verify via broker before acting."
            )
        parts.append("")

    return "\n".join(parts)


def _ticker_detail_basic(v: StockVerdict) -> list[str]:
    """Generate the basic (non-LLM) per-ticker narrative block with LEARN tags."""
    learn_open = _learn_open(v.ticker, v.r_squared, v.confidence_delta, v.analyst_upside)
    upside_str = _fmt_pct(v.analyst_upside)
    price_str  = _fmt_price(v.price)
    target_str = _fmt_price(v.analyst_target)

    lines = [
        f"### {v.ticker}",
        "",
        learn_open,
        "",
        f"- Price: {price_str} | Analyst target: {target_str} | Upside: **{upside_str}**",
        f"- r²={v.r_squared:.3f} | confidence_delta={v.confidence_delta:+.4f}",
        f"- {v.reason}",
    ]

    if v.warnings:
        lines.append("")
        for w in v.warnings:
            lines.append(f"> ⚠️ {w}")

    if v.news_headlines:
        lines.append("")
        for h in v.news_headlines:
            lines.append(f"  - {h}")

    lines += ["", _LEARN_CLOSE, ""]
    return lines


# ---------------------------------------------------------------------------
# Enhanced report (with LLM narratives)
# ---------------------------------------------------------------------------

def build_enhanced(
    verdicts:   list[StockVerdict],
    narratives: dict[str, str],
    date_str:   str | None = None,
    source:     str = "live_alpha_us.csv + fundamentals.csv + news.txt",
    thresholds: dict[str, Any] | None = None,
    stale:      bool = False,
) -> str:
    """
    Build the enhanced stock opportunity report with LLM-generated narratives.

    narratives keys expected (from stockfit.llm):
        "executive_summary"
        "picks_analysis"    — per-ticker dict {"ETN": "...", "ADI": "..."}
        "watch_list_notes"
        "risk_factors"

    Falls back gracefully to build() if narratives is empty or a key is missing.
    """
    if not narratives:
        return build(verdicts, date_str=date_str, source=source,
                     thresholds=thresholds, stale=stale)

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    thresh  = thresholds or {}
    min_r2  = thresh.get("min_r2", 0.65)
    min_cd  = thresh.get("min_conf_delta", 0.020)
    min_up  = thresh.get("min_upside", 0.05)

    include = [v for v in verdicts if v.tier == "INCLUDE"]
    watch   = [v for v in verdicts if v.tier == "WATCH"]
    exclude = [v for v in verdicts if v.tier == "EXCLUDE"]

    stale_note = "\n> ⚠️ **Data may be stale.** Re-run the pipeline for fresh signals.\n" if stale else ""
    picks_analysis = narratives.get("picks_analysis", {})

    parts = [
        "# 📋 ShockArb Stock Opportunity Report",
        "",
        f"**{date_str}**",
        f"*Source: {source}*",
        stale_note,
        f"> Thresholds applied: r² ≥ {min_r2:.2f} | conf.Δ ≥ {min_cd:.3f} | analyst upside ≥ {min_up * 100:.0f}%",
        "",
        "---",
        "",
    ]

    if "executive_summary" in narratives:
        parts += [
            "## Executive Summary",
            "",
            f'<!-- LEARN section="executive_summary" difficulty="intermediate" inputs="date={date_str}" -->',
            "",
            narratives["executive_summary"],
            "",
            _LEARN_CLOSE,
            "",
            "---",
            "",
        ]

    # --- INCLUDE section ---
    parts += [f"## ✅ Act on These ({len(include)} candidates)", ""]
    if include:
        parts += [_table_header()]
        for v in include:
            parts.append(v.as_markdown_row())
        parts.append("")
        for v in include:
            parts += _ticker_detail_enhanced(v, picks_analysis.get(v.ticker))
    else:
        parts += ["*No tickers pass all filters for this session.*", ""]

    parts += ["---", ""]

    # --- WATCH section ---
    parts += [f"## ⚠️ Watch ({len(watch)} candidates)", ""]
    if watch:
        parts += [_table_header()]
        for v in watch:
            parts.append(v.as_markdown_row())
        parts.append("")
        if "watch_list_notes" in narratives:
            parts += [narratives["watch_list_notes"], ""]
        else:
            for v in watch:
                parts += [f"**{v.ticker}** — {v.reason}", ""]
    else:
        parts += ["*No candidates in watch tier.*", ""]

    parts += ["---", ""]

    # --- EXCLUDE section ---
    parts += [f"## ❌ Excluded ({len(exclude)} tickers)", ""]
    if exclude:
        parts += [
            "| Ticker | Reason |",
            "| ------ | ------ |",
        ]
        for v in exclude:
            reason = v.reason.replace("|", "\\|")
            parts.append(f"| {v.ticker} | {reason} |")
        parts.append("")

    if "risk_factors" in narratives:
        parts += [
            "---",
            "",
            "## ⚡ Risk Factors",
            "",
            narratives["risk_factors"],
            "",
        ]

    # --- Data quality flags ---
    dq_flags = [v for v in verdicts if v.tier == "EXCLUDE" and v.reason.startswith("Analyst target below")]
    if dq_flags:
        parts += ["---", "", "## 📌 Data Quality Flags", ""]
        for v in dq_flags:
            parts.append(
                f"- **{v.ticker}**: analyst target {_fmt_price(v.analyst_target)} "
                f"vs price {_fmt_price(v.price)} — verify via broker before acting."
            )
        parts.append("")

    return "\n".join(parts)


def _ticker_detail_enhanced(v: StockVerdict, narrative: str | None) -> list[str]:
    """Generate the LLM-enhanced per-ticker narrative block with LEARN tags."""
    learn_open = _learn_open(v.ticker, v.r_squared, v.confidence_delta, v.analyst_upside)
    upside_str = _fmt_pct(v.analyst_upside)
    price_str  = _fmt_price(v.price)
    target_str = _fmt_price(v.analyst_target)

    lines = [
        f"### {v.ticker}",
        "",
        f"- Price: {price_str} | Analyst target: {target_str} | Upside: **{upside_str}**",
        "",
        learn_open,
        "",
    ]

    if narrative:
        lines.append(narrative)
    else:
        lines.append(v.reason)

    if v.warnings:
        lines.append("")
        for w in v.warnings:
            lines.append(f"> ⚠️ {w}")

    lines += ["", _LEARN_CLOSE, ""]
    return lines
