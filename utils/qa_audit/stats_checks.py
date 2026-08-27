"""
qa_audit.stats_checks — deterministic, no-LLM sanity checks on a ShockArb run.

These are the "fast, cheap, always-run" half of the QA audit: pure functions
over data already computed by `shockarb score` / `stockfit report`, no
network calls, no LLM calls. They catch the class of bug this project has
hit repeatedly — cache corruption, misaligned dates, stale overrides,
degenerate model fit — before it reaches the expensive/slow LLM validation
layer in `llm_validator.py`.

Each check returns a CheckResult (status PASS/WARN/FAIL + a plain-English
message an operator can act on without reading code). Nothing here decides
whether to trust today's picks — that's a human (or the LLM layer) call;
this module's job is narrower: "does the data behind today's picks look
sane on its face."

Example
-------
    from qa_audit.stats_checks import run_all_checks
    from stockfit.features import extract_all
    from stockfit.rules import evaluate_all

    features = extract_all()
    verdicts = evaluate_all(features)
    results = run_all_checks(features, verdicts)
    for r in results:
        print(r.status, r.name, "-", r.message)
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from loguru import logger

# =============================================================================
# Result type
# =============================================================================

@dataclass
class CheckResult:
    """
    One QA check's outcome.

    status : "PASS" | "WARN" | "FAIL"
        PASS   — nothing unusual found.
        WARN   — worth a human glance, not necessarily wrong (e.g. an
                 unusually high pick count that a real dispersion event
                 would also explain).
        FAIL   — data-integrity problem; the day's picks should not be
                 trusted until this is resolved (e.g. tickers scored off
                 different trading days).
    details : machine-readable payload (offending tickers, numbers) for
        anything that wants to render this beyond the message string.
    """
    name:    str
    status:  str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


_VALID_STATUSES = {"PASS", "WARN", "FAIL"}


def _result(name: str, status: str, message: str, **details) -> CheckResult:
    assert status in _VALID_STATUSES, f"invalid CheckResult status: {status!r}"
    return CheckResult(name=name, status=status, message=message, details=details)


# =============================================================================
# Checks on the raw per-ticker feature list (from stockfit.features.extract_all)
# =============================================================================

def check_cache_date_alignment(tickers: list[str], data_dir: str) -> CheckResult:
    """
    Every ticker scored on the same day must be using the SAME last cached
    close date. If they're not, the factor model is comparing returns from
    different trading sessions as if they were the same day's move —
    silently wrong math, not a "few names looked odd" situation.

    Reads the parquet cache directly (shockarb.store.DataStore._daily_path)
    rather than trusting any in-memory provenance object, so this check is
    independent of whatever the scoring run itself claims happened.

    Root-caused 2026-08-19: NAN-CLOSE-CACHE-CORRUPTION could in principle
    have left tickers at different effective coverage after a partial
    repair — this check exists to catch exactly that class of problem
    before it silently corrupts a score again. See HIL_todo.md.
    """
    from shockarb.store import DataStore
    import pandas as pd

    store = DataStore(data_dir)
    last_dates: dict[str, Optional[str]] = {}
    missing: list[str] = []

    for ticker in tickers:
        path = store._daily_path(ticker)
        if not path.exists():
            missing.append(ticker)
            continue
        try:
            df = pd.read_parquet(path)
            col = "adj_close" if "adj_close" in df.columns else df.columns[0]
            valid = df[col].dropna()
            last_dates[ticker] = str(valid.index.max().date()) if len(valid) else None
        except Exception as exc:
            logger.warning(f"[qa_audit] {ticker}: could not read cache for alignment check: {exc}")
            missing.append(ticker)

    if missing:
        return _result(
            "cache_date_alignment", "FAIL",
            f"{len(missing)} ticker(s) unreadable/missing from cache — "
            f"alignment cannot be verified: {missing}",
            missing=missing,
        )

    dates_seen = set(last_dates.values())
    dates_seen.discard(None)
    if not dates_seen:
        return _result("cache_date_alignment", "FAIL", "No ticker has any valid cached price at all.")

    if len(dates_seen) == 1:
        (only_date,) = dates_seen
        return _result(
            "cache_date_alignment", "PASS",
            f"All {len(tickers)} tickers share the same last cached date ({only_date}).",
            last_date=only_date,
        )

    # More than one distinct "latest date" across the universe — find the
    # majority date and report who's behind it.
    counts: dict[str, int] = {}
    for d in last_dates.values():
        if d is not None:
            counts[d] = counts.get(d, 0) + 1
    majority_date = max(counts, key=counts.get)
    behind = {t: d for t, d in last_dates.items() if d != majority_date}
    return _result(
        "cache_date_alignment", "FAIL",
        f"Tickers are NOT all on the same trading day. {len(behind)} of "
        f"{len(tickers)} lag the majority date ({majority_date}): {behind}",
        majority_date=majority_date, behind=behind,
    )


def check_return_magnitude_outliers(features: list[dict], threshold: float = 0.15) -> CheckResult:
    """
    Flag any single-session |actual_return| above `threshold` (default 15%).

    A real 15%+ one-day move happens, but it's rare enough to be worth a
    human glance — and it's exactly what a multi-day gap silently miscounted
    as "one day" would look like (see NAN-CLOSE-CACHE-CORRUPTION: a name
    that missed several sessions and then catches up all at once shows up
    here as an implausible single-day return).
    """
    outliers = [
        (f["ticker"], f["actual_return"])
        for f in features
        if f.get("actual_return") is not None
        and _is_number(f["actual_return"])
        and abs(f["actual_return"]) > threshold
    ]
    if not outliers:
        return _result(
            "return_magnitude_outliers", "PASS",
            f"No ticker moved more than {threshold:.0%} in the scored session.",
        )
    outliers.sort(key=lambda t: -abs(t[1]))
    formatted = [f"{t}={r:+.1%}" for t, r in outliers]
    return _result(
        "return_magnitude_outliers", "WARN",
        f"{len(outliers)} ticker(s) moved more than {threshold:.0%} in one session: "
        f"{formatted}. Verify these aren't a multi-day gap being read as one day.",
        outliers=dict(outliers),
    )


def check_r2_distribution(features: list[dict]) -> CheckResult:
    """
    Flag a degenerate r² distribution: every name fitting the factor model
    almost identically well is not how real, heterogeneous businesses
    behave — it's what you'd see if the model were regressing against
    itself, or if a data bug flattened everyone's return series the same way.
    """
    r2s = [f["r_squared"] for f in features if _is_number(f.get("r_squared"))]
    if len(r2s) < 5:
        return _result("r2_distribution", "WARN", f"Too few valid r² values ({len(r2s)}) to assess.")

    spread = statistics.pstdev(r2s)
    if spread < 0.02:
        return _result(
            "r2_distribution", "FAIL",
            f"r² is suspiciously uniform across {len(r2s)} tickers "
            f"(stdev={spread:.4f}, mean={statistics.mean(r2s):.3f}) — "
            f"real businesses shouldn't fit a shared factor model this identically.",
            stdev=spread, mean=statistics.mean(r2s),
        )
    return _result(
        "r2_distribution", "PASS",
        f"r² spread looks normal across {len(r2s)} tickers "
        f"(mean={statistics.mean(r2s):.3f}, stdev={spread:.3f}).",
        stdev=spread, mean=statistics.mean(r2s),
    )


def check_pick_count_vs_history(
    n_include: int,
    n_lowconf: int,
    n_watch: int,
    archive=None,          # shockarb.score_history.ScoreArchive, injected for testability
    regime_name: str = "ukraine_shock",
    window_days: int = 30,
    z_threshold: float = 2.0,
) -> CheckResult:
    """
    Compare today's INCLUDE+LOW_CONFIDENCE+WATCH count against the rolling
    history in data/recent_scores/ (via ScoreArchive.load_window()).

    The archive doesn't store tier assignments directly, so "pass" is
    approximated per historical day as (conf_delta >= MIN_CONF_DELTA) &
    (r2 >= MIN_R2_WATCH) — the lower of the two r² floors, since that's the
    band that now covers both action tiers (INCLUDE + LOW_CONFIDENCE); it
    can't reproduce the earnings-window exclusion or the upside gate from
    history alone. This is a deliberate approximation: good enough to catch
    "today is a wild outlier vs. the last month," not precise enough to
    reproduce the exact tier count. Say so in the message so nobody
    over-trusts the exact number.

    Ken's 2026-08-19 question ("lots of recommendations right after a big
    rally — I'd expect few or none") is exactly the scenario this check is
    for: it turns "does this feel like a lot?" into a number against the
    actual recent distribution instead of a gut check.
    """
    from stockfit.rules import MIN_R2_WATCH, MIN_CONF_DELTA

    today_total = n_include + n_lowconf + n_watch

    if archive is None:
        return _result(
            "pick_count_vs_history", "WARN",
            f"No score archive available — can't compare today's {today_total} "
            f"pick(s) ({n_include} INCLUDE + {n_lowconf} LOW_CONFIDENCE + {n_watch} WATCH) "
            f"against recent history. Run `shockarb score --save-recent` to start building one.",
            today_total=today_total,
        )

    hist = archive.load_window(days=window_days)
    if hist is None or hist.empty:
        return _result(
            "pick_count_vs_history", "WARN",
            f"Score archive has no history yet for the last {window_days} days — "
            f"can't compare today's {today_total} pick(s) against a baseline.",
            today_total=today_total,
        )

    regime_hist = hist[hist["regime"] == regime_name] if "regime" in hist.columns else hist
    if regime_hist.empty:
        return _result(
            "pick_count_vs_history", "WARN",
            f"No history found for regime {regime_name!r} — can't compare today's "
            f"{today_total} pick(s) against a baseline.",
            today_total=today_total,
        )

    passing = regime_hist[
        (regime_hist["conf_delta"] >= MIN_CONF_DELTA) & (regime_hist["r2"] >= MIN_R2_WATCH)
    ]
    daily_counts = passing.groupby("date").size()

    n_days = daily_counts.shape[0]
    if n_days < 5:
        return _result(
            "pick_count_vs_history", "WARN",
            f"Only {n_days} historical day(s) in the archive — too few to judge "
            f"whether today's {today_total} pick(s) is unusual.",
            today_total=today_total, n_history_days=n_days,
        )

    mean = daily_counts.mean()
    stdev = daily_counts.std(ddof=0)
    z = (today_total - mean) / stdev if stdev > 0 else float("inf") if today_total != mean else 0.0

    if abs(z) >= z_threshold:
        direction = "more" if today_total > mean else "fewer"
        return _result(
            "pick_count_vs_history", "WARN",
            f"Today's {today_total} pick(s) is a statistical outlier vs. the last "
            f"{n_days} days (mean={mean:.1f}, stdev={stdev:.1f}, z={z:+.1f}) — "
            f"{direction} candidates than the recent norm. Approximate count "
            f"(numeric gates only, not the exact tier logic) — worth a closer look, "
            f"not necessarily wrong.",
            today_total=today_total, mean=mean, stdev=stdev, z=z,
        )
    return _result(
        "pick_count_vs_history", "PASS",
        f"Today's {today_total} pick(s) is within normal range vs. the last "
        f"{n_days} days (mean={mean:.1f}, stdev={stdev:.1f}, z={z:+.1f}).",
        today_total=today_total, mean=mean, stdev=stdev, z=z,
    )


# =============================================================================
# Checks on StockVerdict objects (post-rules.evaluate_all)
# =============================================================================

def check_upside_sanity(verdicts: list, max_reasonable_upside: float = 1.00) -> CheckResult:
    """
    Flag any INCLUDE/LOW_CONFIDENCE/WATCH pick whose analyst upside exceeds
    `max_reasonable_upside` (default 100%). This project's HIL_todo.md has
    a long history of absurd upside numbers tracing back to stale entries
    in analyst_overrides.csv (CRM at $290 when consensus was ~$240; QCOM's
    $220 tracked as stale for six straight checks) — a triple-digit upside
    on an actionable pick is far more often a stale target than a real
    100%+ dislocation.
    """
    flagged = [
        (v.ticker, v.analyst_upside)
        for v in verdicts
        if v.tier in ("INCLUDE", "LOW_CONFIDENCE", "WATCH")
        and v.analyst_upside is not None
        and v.analyst_upside > max_reasonable_upside
    ]
    if not flagged:
        return _result(
            "upside_sanity", "PASS",
            f"No INCLUDE/LOW_CONFIDENCE/WATCH pick claims more than {max_reasonable_upside:.0%} upside.",
        )
    formatted = [f"{t}={u:+.1%}" for t, u in flagged]
    return _result(
        "upside_sanity", "WARN",
        f"{len(flagged)} pick(s) claim more than {max_reasonable_upside:.0%} upside: "
        f"{formatted}. Cross-check the analyst target against live consensus before "
        f"trusting this — see SNPS-NOW-QCOM-OVERRIDE-AUDIT in HIL_todo.md for precedent.",
        flagged=dict(flagged),
    )


def check_cluster_concentration(verdicts: list, max_share: float = 0.5) -> CheckResult:
    """
    Flag if more than `max_share` of INCLUDE+LOW_CONFIDENCE+WATCH picks share the same
    sector cluster. Not automatically wrong — a real sector-wide selloff
    (see the 2026-07-28 semiconductor rout in HIL_todo.md) produces
    exactly this pattern — but concentrated picks mean the "N independent
    opportunities" framing is misleading: they're closer to one bet on one
    factor wearing N tickers.
    """
    actionable = [v for v in verdicts if v.tier in ("INCLUDE", "LOW_CONFIDENCE", "WATCH")]
    if not actionable:
        return _result("cluster_concentration", "PASS", "No INCLUDE/LOW_CONFIDENCE/WATCH picks to assess.")

    counts: dict[str, int] = {}
    for v in actionable:
        if v.cluster:
            counts[v.cluster] = counts.get(v.cluster, 0) + 1

    if not counts:
        return _result(
            "cluster_concentration", "PASS",
            f"None of today's {len(actionable)} pick(s) share a known sector cluster.",
        )

    worst_cluster, worst_count = max(counts.items(), key=lambda kv: kv[1])
    share = worst_count / len(actionable)
    if share > max_share:
        return _result(
            "cluster_concentration", "WARN",
            f"{worst_count} of {len(actionable)} INCLUDE/LOW_CONFIDENCE/WATCH picks ({share:.0%}) "
            f"are in the '{worst_cluster}' cluster — today's picks are concentrated "
            f"in one factor bet, not {len(actionable)} independent ones. Not "
            f"necessarily wrong (could be a real sector-wide dislocation) but treat "
            f"position sizing accordingly.",
            worst_cluster=worst_cluster, worst_count=worst_count, share=share,
        )
    return _result(
        "cluster_concentration", "PASS",
        f"Picks are reasonably spread across clusters (worst: '{worst_cluster}' at {share:.0%}).",
        worst_cluster=worst_cluster, share=share,
    )


def check_tier_count_sanity(verdicts: list) -> CheckResult:
    """
    Baseline sanity check with no history needed: INCLUDE should never
    exceed LOW_CONFIDENCE+WATCH+EXCLUDE combined in a well-calibrated
    universe, and an all-INCLUDE, all-LOW_CONFIDENCE, or all-EXCLUDE result
    on a large universe is itself a signal something upstream is broken
    (thresholds not applied, or a universe-wide data problem making every
    name look identical). A pure-EXCLUDE day is NOT automatically wrong on
    its own — see R2-GATE-NEAR-MISS in HIL_todo.md, 2026-08-21, for a real
    example where zero INCLUDE was correct but the underlying data was fine
    — this check only flags the degenerate "every single ticker, same tier"
    pattern, which is a much stronger signal of a broken gate than "zero
    picks" alone.
    """
    n_include = sum(1 for v in verdicts if v.tier == "INCLUDE")
    n_lowconf = sum(1 for v in verdicts if v.tier == "LOW_CONFIDENCE")
    n_watch   = sum(1 for v in verdicts if v.tier == "WATCH")
    n_exclude = sum(1 for v in verdicts if v.tier == "EXCLUDE")
    total = len(verdicts)

    if total == 0:
        return _result("tier_count_sanity", "FAIL", "No verdicts to assess — empty universe.")

    if n_include == total or n_lowconf == total or n_exclude == total:
        return _result(
            "tier_count_sanity", "FAIL",
            f"Every one of {total} tickers landed in the same tier "
            f"(INCLUDE={n_include}, LOW_CONFIDENCE={n_lowconf}, WATCH={n_watch}, "
            f"EXCLUDE={n_exclude}) — this looks like a broken gate, not a real "
            f"market condition.",
            n_include=n_include, n_lowconf=n_lowconf, n_watch=n_watch, n_exclude=n_exclude,
        )
    return _result(
        "tier_count_sanity", "PASS",
        f"Tier split looks structurally normal: {n_include} INCLUDE, "
        f"{n_lowconf} LOW_CONFIDENCE, {n_watch} WATCH, {n_exclude} EXCLUDE of {total}.",
        n_include=n_include, n_lowconf=n_lowconf, n_watch=n_watch, n_exclude=n_exclude,
    )


# =============================================================================
# Orchestrator
# =============================================================================

def run_all_checks(
    features: list[dict],
    verdicts: list,
    data_dir: str = "./data",
    archive=None,
    return_threshold: float = 0.15,
    max_upside: float = 1.00,
) -> list[CheckResult]:
    """
    Run every deterministic check and return results in a fixed, readable
    order (data-integrity checks first, since a FAIL there makes the
    others moot; then distributional checks; then tier-composition checks).

    Example
    -------
        results = run_all_checks(features, verdicts, data_dir="./data")
        if any(r.status == "FAIL" for r in results):
            print("Do not trust today's picks until these are resolved.")
    """
    tickers   = [f["ticker"] for f in features]
    n_include = sum(1 for v in verdicts if v.tier == "INCLUDE")
    n_lowconf = sum(1 for v in verdicts if v.tier == "LOW_CONFIDENCE")
    n_watch   = sum(1 for v in verdicts if v.tier == "WATCH")

    return [
        check_cache_date_alignment(tickers, data_dir),
        check_return_magnitude_outliers(features, threshold=return_threshold),
        check_r2_distribution(features),
        check_tier_count_sanity(verdicts),
        check_pick_count_vs_history(n_include, n_lowconf, n_watch, archive=archive),
        check_upside_sanity(verdicts, max_reasonable_upside=max_upside),
        check_cluster_concentration(verdicts),
    ]


def _is_number(x: Any) -> bool:
    try:
        return x == x and float(x) == float(x)   # not NaN
    except (TypeError, ValueError):
        return False
