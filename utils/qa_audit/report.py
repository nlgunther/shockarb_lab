"""
qa_audit.report — assemble stats_checks + llm_validator output into one
Markdown QA health-check report, matching this project's existing report
style (marketfit/report.py, stockfit/report.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from qa_audit.stats_checks import CheckResult
from qa_audit.llm_validator import ValidationResult

_STATUS_ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
_VERDICT_ICON = {"AGREE": "✅", "DISAGREE": "❌", "UNCERTAIN": "❔", "ERROR": "⚠️"}


def build_qa_report(
    stats_results: list[CheckResult],
    validation_results: list[ValidationResult],
    concordance: Optional[dict] = None,
    universe_size: int = 0,
    n_include: int = 0,
    n_lowconf: int = 0,
    n_watch: int = 0,
) -> str:
    """
    Build the full Markdown report.

    Example
    -------
        from qa_audit.stats_checks import run_all_checks
        from qa_audit.llm_validator import run_validation_batch, summarize_concordance
        from qa_audit.report import build_qa_report

        stats = run_all_checks(features, verdicts)
        validations = run_validation_batch(client, verdicts, ...)
        md = build_qa_report(stats, validations, summarize_concordance(validations),
                              universe_size=len(features), n_include=..., n_lowconf=..., n_watch=...)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# 🩺 ShockArb QA Health Check",
        "",
        f"**{now}**",
        f"*Universe: {universe_size} tickers scored — {n_include} INCLUDE, "
        f"{n_lowconf} LOW_CONFIDENCE, {n_watch} WATCH*",
        "",
        "---",
        "",
    ]

    parts += _stats_section(stats_results)
    parts += ["---", ""]
    parts += _validation_section(validation_results, concordance)
    parts += ["---", ""]
    parts += _bottom_line(stats_results, validation_results, concordance)

    return "\n".join(parts)


def _stats_section(results: list[CheckResult]) -> list[str]:
    parts = ["## 📊 Deterministic Checks (no LLM)", ""]
    parts.append("| Check | Status | Finding |")
    parts.append("| ----- | :----: | ------- |")
    for r in results:
        icon = _STATUS_ICON.get(r.status, r.status)
        parts.append(f"| {r.name} | {icon} {r.status} | {r.message} |")
    parts.append("")
    return parts


def _validation_section(results: list[ValidationResult], concordance: Optional[dict]) -> list[str]:
    parts = ["## 🔍 LLM Independent Validation (gold-standard sample)", ""]

    if not results:
        parts += [
            "*No LLM validation was run this pass — either no key was configured "
            "(`ANTHROPIC_API_KEY`/`GOOGLE_API_KEY`) or `--no-llm` was passed.*",
            "",
        ]
        return parts

    if concordance:
        rate = concordance.get("concordance_rate")
        rate_s = f"{rate:.0%}" if rate is not None else "n/a"
        parts += [
            f"Sampled {concordance['n']} pick(s): "
            f"{concordance['agree']} agree, {concordance['disagree']} disagree, "
            f"{concordance['uncertain']} uncertain, {concordance['error']} errored. "
            f"Concordance rate: **{rate_s}**.",
            "",
        ]
        if concordance.get("disagreements"):
            parts.append(f"⚠️ LLM disagreed with ShockArb on: **{', '.join(concordance['disagreements'])}**")
            parts.append("")

    for r in results:
        icon = _VERDICT_ICON.get(r.llm_verdict, "❔")
        conf_s = f"{r.llm_confidence:.0%}" if r.llm_confidence is not None else "n/a"
        parts += [
            f"### {r.ticker} — {icon} {r.llm_verdict} (confidence {conf_s}, ShockArb tier: {r.shockarb_tier})",
            "",
            r.llm_reasoning or "*(no reasoning returned)*",
            "",
        ]
        if r.red_flags:
            parts.append("**Red flags:**")
            parts += [f"- {f}" for f in r.red_flags]
            parts.append("")
        if r.supporting_points:
            parts.append("**Supporting points:**")
            parts += [f"- {p}" for p in r.supporting_points]
            parts.append("")
        if r.would_need_to_know:
            parts.append("**Before acting on this, verify:**")
            parts += [f"- {w}" for w in r.would_need_to_know]
            parts.append("")

    return parts


def _bottom_line(
    stats_results: list[CheckResult],
    validation_results: list[ValidationResult],
    concordance: Optional[dict],
) -> list[str]:
    parts = ["## Bottom Line", ""]

    failures = [r for r in stats_results if r.status == "FAIL"]
    warnings = [r for r in stats_results if r.status == "WARN"]

    if failures:
        parts.append(
            f"❌ **{len(failures)} data-integrity check(s) failed** "
            f"({', '.join(r.name for r in failures)}) — do not trust today's "
            f"picks until these are resolved."
        )
    elif warnings:
        parts.append(
            f"⚠️ All data-integrity checks passed; {len(warnings)} check(s) "
            f"raised a flag worth a human glance "
            f"({', '.join(r.name for r in warnings)})."
        )
    else:
        parts.append("✅ All deterministic checks passed clean.")

    if concordance:
        rate = concordance.get("concordance_rate")
        if rate is not None and rate < 0.5:
            parts.append(
                f"❌ The LLM validator disagreed with ShockArb on more than half "
                f"of the sampled picks ({concordance['disagree']}/{concordance['n']}) "
                f"— treat today's Act-on tier with real skepticism, not just as a caveat."
            )
        elif rate is not None and rate < 1.0:
            parts.append(
                f"⚠️ The LLM validator disagreed on {concordance['disagree']} of "
                f"{concordance['n']} sampled picks — worth reading those specific "
                f"validations above before acting."
            )

    parts.append("")
    return parts
