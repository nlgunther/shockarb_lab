"""
Compare two or more ShockArb stock_report_*.md or *_verdicts.csv files.

Parses each report's per-ticker tier (Act on / Watch / Excluded) and key
stats, then builds a ticker-indexed comparison highlighting where tiers or
signals diverge across reports — e.g. comparing the same date across regimes,
or the same regime across dates.

Two input formats are supported, dispatched by extension (see _PARSERS):
  .md   stock_report_*.md         — rules.report.build() markdown report.
        EXCLUDE-tier tickers carry only a `reason` string.
  .csv  stock_report_*_verdicts.csv — stockfit `--save-verdicts` output.
        Carries full stats (r², conf.Δ, upside, ...) for EVERY tier,
        including EXCLUDE.

Functions
---------
  parse_report        Parse a .md or .csv report into a ReportData (by extension).
  build_comparison     Build a ticker x report comparison table + flag column.
  print_comparison     Pretty-print the comparison to console.
  write_comparison_md  Write the comparison as a markdown file.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

# =============================================================================
# Parsing
# =============================================================================

_HEADER_RE = re.compile(r"\*\*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\*\*")
_SOURCE_RE = re.compile(r"\*Source:\s*(.+?)\*")
_THRESHOLDS_RE = re.compile(r">\s*Thresholds applied:\s*(.+)")
_SECTION_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

# Matches the leading emoji/words of each tier's "## " heading.
_TIER_TITLES = {
    "Act on": "act_on",
    "Watch": "watch",
    "Excluded": "excluded",
}

# Markdown table column -> ReportData ticker-entry key.
_STAT_COLUMNS = {
    "R²": "r_squared",
    "Conf.Δ": "conf_delta",
    "Price": "price",
    "Analyst Tgt": "analyst_target",
    "Upside": "upside",
    "Fwd P/E": "fwd_pe",
    "RVOL": "rvol",
}

# Fields shown per ticker in the comparison table, in display order.
COMPARISON_FIELDS = ["tier", "r_squared", "conf_delta", "upside", "fwd_pe", "reason"]


@dataclass
class ReportData:
    """Structured contents of one stock_report_*.md file."""

    path: str
    label: str
    timestamp: str | None
    source: str | None
    thresholds: str | None
    counts: dict[str, int] = field(default_factory=dict)
    tickers: dict[str, dict] = field(default_factory=dict)  # ticker -> {tier, ...stats}


def _label_for(path: str, timestamp: str | None) -> str:
    """Build a short report label from its directory and timestamp.

    Example: reports/iran_shock/stock_report_20260612_0800.md -> "iran_shock_0800"
    """
    parent = os.path.basename(os.path.dirname(path)) or "root"
    if parent in ("reports", ""):
        parent = "root"

    if timestamp:
        hhmm = timestamp[-5:].replace(":", "")
    else:
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = stem.removesuffix("_verdicts")  # *_verdicts.csv -> ..._HHMM
        hhmm = stem[-4:]

    return f"{parent}_{hhmm}"


def _clean_number(text: str) -> float | None:
    """
    Strip currency/percent/multiplier markers and return a float, or None
    for "—" placeholders.

    Example:
        _clean_number("+14.8%")    # -> 14.8
        _clean_number("\\$393.64")  # -> 393.64
        _clean_number("1.3x (20d)") # -> 1.3
        _clean_number("—")          # -> None
    """
    text = text.strip()
    if text in ("—", "-", ""):
        return None

    text = text.split("(")[0].strip()  # drop "(20d)" etc.
    for token in ("\\", "$", "x", "%", ","):
        text = text.replace(token, "")

    try:
        return float(text)
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    """Convert a verdicts-CSV cell to a float, or None for empty/missing values.

    Example:
        _to_float("0.0263")  # -> 0.0263
        _to_float("")        # -> None
        _to_float(None)      # -> None
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_table(block: str) -> list[dict]:
    """Parse a markdown pipe table into a list of row dicts keyed by header."""
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:  # skip header row and "|---|---|" separator
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _parse_report_md(path: str) -> ReportData:
    """Parse a stock_report_*.md file into structured per-ticker data."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    ts_match = _HEADER_RE.search(text)
    timestamp = ts_match.group(1) if ts_match else None
    src_match = _SOURCE_RE.search(text)
    thr_match = _THRESHOLDS_RE.search(text)

    tickers: dict[str, dict] = {}
    counts: dict[str, int] = {}

    sections = _SECTION_RE.split(text)  # [pre, title1, body1, title2, body2, ...]
    for i in range(1, len(sections), 2):
        title, body = sections[i], sections[i + 1]
        tier_key = next((v for k, v in _TIER_TITLES.items() if k in title), None)
        if tier_key is None:
            continue

        count_match = re.search(r"\((\d+)\s+(?:candidates|tickers)\)", title)
        counts[tier_key] = int(count_match.group(1)) if count_match else 0

        for row in _parse_table(body):
            ticker = row.get("Ticker")
            if not ticker:
                continue
            entry: dict = {"tier": tier_key}
            if tier_key == "excluded":
                entry["reason"] = row.get("Reason", "")
            else:
                for col, key in _STAT_COLUMNS.items():
                    if col in row:
                        entry[key] = _clean_number(row[col])
            tickers[ticker] = entry

    if not tickers:
        logger.warning(f"No ticker rows parsed from {path} — unexpected report format?")

    return ReportData(
        path=path,
        label=_label_for(path, timestamp),
        timestamp=timestamp,
        source=src_match.group(1) if src_match else None,
        thresholds=thr_match.group(1) if thr_match else None,
        counts=counts,
        tickers=tickers,
    )


# Verdicts-CSV tier -> ReportData tier key (matches _TIER_TITLES values).
_CSV_TIER_MAP = {"INCLUDE": "act_on", "WATCH": "watch", "EXCLUDE": "excluded"}


def _parse_verdicts_csv(path: str) -> ReportData:
    """Parse a stockfit `--save-verdicts` CSV into structured per-ticker data.

    Unlike the markdown report, every tier (including EXCLUDE) carries full
    stats. `analyst_upside` is a fraction in the CSV (e.g. 0.026); it's
    converted to a percent here to match the markdown parser's units.

    No timestamp/source/thresholds header exists in the CSV, so those fields
    are left as None.
    """
    tickers: dict[str, dict] = {}
    counts: dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row.get("ticker")
            if not ticker:
                continue

            tier_key = _CSV_TIER_MAP.get(row.get("tier", ""), "excluded")
            counts[tier_key] = counts.get(tier_key, 0) + 1

            upside = _to_float(row.get("analyst_upside"))
            tickers[ticker] = {
                "tier": tier_key,
                "reason": row.get("reason", ""),
                "r_squared": _to_float(row.get("r_squared")),
                "conf_delta": _to_float(row.get("confidence_delta")),
                "upside": None if upside is None else upside * 100,
                "price": _to_float(row.get("price")),
                "analyst_target": _to_float(row.get("analyst_target")),
                "fwd_pe": _to_float(row.get("fwd_pe")),
                "rvol": _to_float(row.get("rvol")),
            }

    if not tickers:
        logger.warning(f"No ticker rows parsed from {path} — unexpected verdicts CSV format?")

    return ReportData(
        path=path,
        label=_label_for(path, timestamp=None),
        timestamp=None,
        source=None,
        thresholds=None,
        counts=counts,
        tickers=tickers,
    )


# File extension -> parser. Add new formats here.
_PARSERS = {
    ".md": _parse_report_md,
    ".csv": _parse_verdicts_csv,
}


def parse_report(path: str) -> ReportData:
    """Parse a stock_report (.md) or verdicts (.csv) file into a ReportData.

    Dispatches on file extension — see _PARSERS.

    Example:
        parse_report("reports/iran_shock/stock_report_20260612_0800.md")
        parse_report("reports/iran_shock/stock_report_20260612_0945_verdicts.csv")
    """
    ext = os.path.splitext(path)[1].lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"Unsupported report file type {ext!r} for {path!r} "
            f"(supported: {', '.join(sorted(_PARSERS))})"
        )
    return parser(path)


# =============================================================================
# Comparison
# =============================================================================

def build_comparison(reports: list[ReportData]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a ticker x report comparison table.

    Returns
    -------
    comparison : DataFrame
        Index = union of all tickers across reports (sorted). Columns are a
        MultiIndex (report_label, field) for field in COMPARISON_FIELDS.
        Missing ticker/report combinations are NaN.
    flagged : Series
        Boolean per ticker — True if the ticker's tier differs across the
        reports it appears in (ignoring reports where it's absent).
    """
    all_tickers = sorted(set().union(*(r.tickers.keys() for r in reports)))

    columns = pd.MultiIndex.from_product(
        [[r.label for r in reports], COMPARISON_FIELDS], names=["report", "field"]
    )
    comparison = pd.DataFrame(index=all_tickers, columns=columns, dtype=object)

    for r in reports:
        for ticker, entry in r.tickers.items():
            for field_name in COMPARISON_FIELDS:
                comparison.loc[ticker, (r.label, field_name)] = entry.get(field_name)

    tiers_seen = comparison.xs("tier", axis=1, level="field")
    flagged = tiers_seen.nunique(axis=1, dropna=True) > 1

    return comparison, flagged


# =============================================================================
# Formatting helpers
# =============================================================================

_TIER_DISPLAY = {"act_on": "Act on", "watch": "Watch", "excluded": "Excluded", None: "—"}


def _fmt_field(field_name: str, value) -> str:
    """Render one comparison-table cell for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if field_name == "tier":
        return _TIER_DISPLAY.get(value, str(value))
    if field_name == "r_squared":
        return f"{value:.3f}"
    if field_name == "conf_delta":
        return f"{value:+.4f}"
    if field_name == "upside":
        return f"{value:+.1f}%"
    if field_name == "fwd_pe":
        return f"{value:.1f}x"
    return str(value)


def _interesting_tickers(comparison: pd.DataFrame, flagged: pd.Series) -> list[str]:
    """Tickers worth showing in the tier-comparison table: act_on/watch in any
    report, or flagged as having a tier mismatch across reports."""
    tiers_seen = comparison.xs("tier", axis=1, level="field")
    is_actionable = tiers_seen.isin(["act_on", "watch"]).any(axis=1)
    return list(comparison.index[is_actionable | flagged])


# =============================================================================
# Console output
# =============================================================================

def print_comparison(
    reports: list[ReportData],
    comparison: pd.DataFrame,
    flagged: pd.Series,
) -> None:
    """Pretty-print the report comparison to the console."""
    print(f"\n{'='*80}")
    print("  📊 REPORT COMPARISON")
    print(f"{'='*80}")

    print("\n  Reports:")
    for r in reports:
        print(f"    [{r.label}]  {r.timestamp or 'unknown time'} UTC  ({r.path})")
        if r.thresholds:
            print(f"        Thresholds: {r.thresholds}")
        counts_str = " | ".join(f"{_TIER_DISPLAY[k]}: {v}" for k, v in r.counts.items())
        if counts_str:
            print(f"        {counts_str}")

    tickers = _interesting_tickers(comparison, flagged)
    print(f"\n{'─'*80}")
    print("  TIER BY REPORT  (⚠ marks tickers whose tier differs across reports)")
    print(f"{'─'*80}")

    if not tickers:
        print("\n  No Act on / Watch candidates and no tier mismatches.")
    else:
        labels = [r.label for r in reports]
        header = f"  {'Ticker':<8}" + "".join(f"{lbl:>14}" for lbl in labels) + "   "
        print(f"\n{header}")
        for ticker in tickers:
            row = f"  {ticker:<8}"
            for lbl in labels:
                row += f"{_fmt_field('tier', comparison.loc[ticker, (lbl, 'tier')]):>14}"
            row += "  ⚠" if flagged[ticker] else ""
            print(row)

    for r in reports:
        sub = comparison.xs(r.label, axis=1, level="report")
        print(f"\n{'─'*80}")
        print(f"  STATS — {r.label}")
        print(f"{'─'*80}")
        if not tickers:
            print("\n  No Act on / Watch candidates and no tier mismatches.")
            continue
        print(f"\n  {'Ticker':<8}{'Tier':<10}{'R²':>8}{'Conf.Δ':>10}{'Upside':>10}{'Fwd P/E':>10}  Reason")
        for ticker in tickers:
            entry = sub.loc[ticker]
            reason = _fmt_field("reason", entry.get("reason"))
            if len(reason) > 60:
                reason = reason[:57] + "..."
            print(
                f"  {ticker:<8}{_fmt_field('tier', entry.get('tier')):<10}"
                f"{_fmt_field('r_squared', entry.get('r_squared')):>8}"
                f"{_fmt_field('conf_delta', entry.get('conf_delta')):>10}"
                f"{_fmt_field('upside', entry.get('upside')):>10}"
                f"{_fmt_field('fwd_pe', entry.get('fwd_pe')):>10}  {reason}"
            )

    print(f"\n{'='*80}\n")


# =============================================================================
# Markdown output
# =============================================================================

def write_comparison_md(
    out_path: str,
    reports: list[ReportData],
    comparison: pd.DataFrame,
    flagged: pd.Series,
) -> None:
    """Write the report comparison as a markdown file."""
    lines = ["# ShockArb Report Comparison", ""]

    lines.append("## Reports")
    for r in reports:
        lines.append(f"- **{r.label}** — {r.timestamp or 'unknown time'} UTC — `{r.path}`")
        if r.thresholds:
            lines.append(f"  - Thresholds: {r.thresholds}")
        counts_str = " | ".join(f"{_TIER_DISPLAY[k]}: {v}" for k, v in r.counts.items())
        if counts_str:
            lines.append(f"  - {counts_str}")
    lines.append("")

    tickers = _interesting_tickers(comparison, flagged)
    lines.append("## Tier by report")
    lines.append("")
    labels = [r.label for r in reports]
    lines.append("| Ticker | " + " | ".join(labels) + " | Flagged |")
    lines.append("| --- | " + " | ".join("---" for _ in labels) + " | --- |")
    for ticker in tickers:
        cells = [_fmt_field("tier", comparison.loc[ticker, (lbl, "tier")]) for lbl in labels]
        flag = "⚠" if flagged[ticker] else ""
        lines.append(f"| {ticker} | " + " | ".join(cells) + f" | {flag} |")
    lines.append("")

    for r in reports:
        sub = comparison.xs(r.label, axis=1, level="report")
        lines.append(f"## Stats — {r.label}")
        lines.append("")
        if not tickers:
            lines.append("*No Act on / Watch candidates and no tier mismatches.*")
        else:
            lines.append("| Ticker | Tier | R² | Conf.Δ | Upside | Fwd P/E | Reason |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
            for ticker in tickers:
                entry = sub.loc[ticker]
                lines.append(
                    f"| {ticker} | {_fmt_field('tier', entry.get('tier'))} "
                    f"| {_fmt_field('r_squared', entry.get('r_squared'))} "
                    f"| {_fmt_field('conf_delta', entry.get('conf_delta'))} "
                    f"| {_fmt_field('upside', entry.get('upside'))} "
                    f"| {_fmt_field('fwd_pe', entry.get('fwd_pe'))} "
                    f"| {_fmt_field('reason', entry.get('reason'))} |"
                )
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote report comparison to {out_path}")
