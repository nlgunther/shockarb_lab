"""
qa_audit.llm_validator — the "gold standard" independent check on a sample
of ShockArb's picks.

What this is and isn't
-----------------------
This is NOT a live fact-check. The LLM call here has no web search and no
tool use — it reasons from (a) the evidence this module puts in the prompt
(ShockArb's own numbers, current price/target, cross-attached news
headlines, market-wide context) and (b) whatever general knowledge of the
company and sector it carries from training. It cannot independently
confirm "did X actually happen yesterday" the way a human googling the
ticker could.

What it IS good for: a second, differently-biased reasoning pass over the
same evidence a human analyst would have in front of them, explicitly
instructed to look for the specific failure patterns this project's own
HIL_todo.md has documented over and over — a narrative that ignores a
tagged negative headline, an analyst target that's suspiciously stale, a
"macro panic" framing applied to a name that's actually just tracking the
broad market rather than lagging it. It's a skeptic in the loop, not an
oracle. The PROMPTS constant below is deliberately explicit about this
limitation so the LLM doesn't confabulate specifics it can't know.

Two-step flow
--------------
    1. select_sample()   — pick which tickers to send to the LLM.
    2. validate_pick()    — one LLM call per selected ticker, parsed into
                             a ValidationResult.
    run_validation_batch() does both; summarize_concordance() rolls the
    results up into an agree/disagree/uncertain count against ShockArb's
    own tier assignment.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from qa_audit.llm_client import LLMClient

# =============================================================================
# The prompts (Ken asked to see these explicitly — they are the actual text
# sent to the model, not paraphrased documentation of it)
# =============================================================================

VALIDATION_SYSTEM_PROMPT = """\
You are an independent, skeptical equity research analyst. A quantitative \
trading system called ShockArb has flagged a stock as a "mean-reversion" \
opportunity: it fell (or rose) more than a factor model calibrated on \
broad market/sector ETFs would predict, and the system is betting the gap \
closes.

Your job is to stress-test that specific claim for one stock, using only \
the evidence given to you in this prompt plus your own general knowledge. \
You do NOT have web search or any live data access — you cannot look up \
today's news. Do not invent specific facts, dates, or events you were not \
given. If the evidence provided is not enough to form a confident \
judgment, say so honestly (verdict "UNCERTAIN") rather than guessing.

Do not assume the quantitative signal is correct. Your default posture is \
skepticism: a large statistical residual is consistent with BOTH a real, \
temporary, technical dislocation (bullish for reversion) AND a fundamental \
repricing that the factor model simply doesn't capture (bearish — the \
"dislocation" is actually justified and won't revert). Your job is to \
decide which one this looks like, using the evidence given.

Specifically weigh:
  1. Do the provided headlines describe company-specific news (an earnings \
     miss, a downgrade, a guidance cut, a lawsuit, a leadership change) \
     that would independently justify the price move, regardless of what \
     the broader market did? If so, this is evidence AGAINST mean \
     reversion — the move may be fundamentally justified, not technical.
  2. Does the market-wide context provided (broad index move, sector \
     dispersion) make this stock's move look ordinary (moving with \
     everyone else) or anomalous (diverging from a broad move that should \
     have carried it along)? A stock that lagged a broad, indiscriminate \
     rally with no negative news of its own is a cleaner mean-reversion \
     case than one that fell for a stated, specific reason.
  3. Is the analyst price target/upside figure plausible, or does it look \
     stale or implausibly large (e.g. well above 50-100% upside)? Flag \
     this explicitly if so — you cannot verify the actual current \
     consensus, but you can flag when a number looks suspicious on its \
     face.
  4. Based on your general knowledge of this company and its industry \
     (moat, competitive position, typical volatility), is a move of this \
     magnitude and this quick a reversion plausible at all for this kind \
     of business?

Return ONLY a JSON object, no markdown fences, no preamble, with exactly \
these keys:
  "verdict"              — one of "AGREE", "DISAGREE", "UNCERTAIN"
                            (AGREE = the mean-reversion thesis looks sound;
                             DISAGREE = the move looks fundamentally
                             justified, not a technical dislocation;
                             UNCERTAIN = not enough evidence either way)
  "confidence"            — a number from 0.0 to 1.0
  "reasoning"              — 3-6 sentences explaining the verdict
  "red_flags"              — list of short strings: specific concerns found
                             (empty list if none)
  "supporting_points"      — list of short strings: evidence that DOES
                             support the mean-reversion thesis (empty list
                             if none)
  "would_need_to_know"     — list of short strings: what a human should
                             specifically go check (a live analyst
                             consensus, a specific headline) before acting
                             on this — this is the field that matters most
                             when verdict is UNCERTAIN"""


def build_user_prompt(
    ticker: str,
    company_name: str,
    industry: str,
    actual_return: float,
    expected_return: float,
    r_squared: float,
    confidence_delta: float,
    price: Optional[float],
    analyst_target: Optional[float],
    analyst_upside: Optional[float],
    news_headlines: list[str],
    market_context: dict[str, Any],
    shockarb_tier: str,
    shockarb_reason: str,
) -> str:
    """
    Build the per-ticker user prompt. Kept as a plain function (not a
    template string with .format()) so every value is explicit at the call
    site and a missing field fails loudly instead of silently rendering
    "None" into the prompt text.

    Example
    -------
        prompt = build_user_prompt(
            ticker="KLAC", company_name="KLA Corporation",
            industry="Semiconductor Equipment",
            actual_return=-0.0086, expected_return=0.0086,
            r_squared=0.849, confidence_delta=0.0284,
            price=205.76, analyst_target=305.50, analyst_upside=0.485,
            news_headlines=["[RATING] Some analyst headline..."],
            market_context={"spy_trailing_5d_return": 0.031,
                             "dispersion_status": "MODERATE",
                             "dispersion_note": "..."},
            shockarb_tier="INCLUDE",
            shockarb_reason="Passes r2/conf_delta/upside gates",
        )
    """
    price_s  = f"${price:,.2f}" if price is not None else "not available"
    target_s = f"${analyst_target:,.2f}" if analyst_target is not None else "not available"
    upside_s = f"{analyst_upside:+.1%}" if analyst_upside is not None else "not available"

    headlines_block = (
        "\n".join(f"  - {h}" for h in news_headlines) if news_headlines
        else "  (none attached — the pipeline found no recent headlines for this ticker)"
    )

    spy_ctx = market_context.get("spy_trailing_return")
    spy_days = market_context.get("spy_trailing_days")
    spy_s = (
        f"S&P 500 (SPY) is {spy_ctx:+.1%} over the trailing {spy_days} session(s)."
        if spy_ctx is not None else "Trailing broad-market return not available."
    )
    dispersion_s = market_context.get("dispersion_note") or market_context.get(
        "dispersion_status", "not available"
    )

    return f"""\
TICKER: {ticker} ({company_name}, {industry})

SHOCKARB'S CLAIM:
  ShockArb placed this ticker in tier "{shockarb_tier}" — {shockarb_reason}
  Factor-model-implied return today: {expected_return:+.2%}
  Actual observed return today:      {actual_return:+.2%}
  Residual (implied − actual):        {(expected_return - actual_return):+.2%}
  r² of the factor model fit on this name: {r_squared:.3f}
  confidence_delta (residual × r²):        {confidence_delta:+.4f}
  ShockArb's inference: the gap between implied and actual is a temporary,
  technical dislocation, not a fundamentally justified move, and should
  close (price reverts toward the factor-implied path).

CURRENT PRICE CONTEXT:
  Price: {price_s}
  Analyst target: {target_s}
  Implied upside: {upside_s}

RECENT HEADLINES ATTACHED TO THIS TICKER BY THE PIPELINE:
{headlines_block}

MARKET-WIDE CONTEXT:
  {spy_s}
  Sector dispersion: {dispersion_s}

Evaluate ShockArb's mean-reversion claim for {ticker} using the framework \
in your instructions. Return only the JSON object."""


# =============================================================================
# Sample selection
# =============================================================================

def select_sample(
    verdicts: list,
    n: int = 3,
    mode: str = "stratified",
    seed: Optional[int] = None,
) -> list:
    """
    Choose which INCLUDE/LOW_CONFIDENCE/WATCH verdicts to send to the LLM.

    Parameters
    ----------
    verdicts : list of StockVerdict
    n        : how many to select (capped at the number available)
    mode     : "random"      — uniform random sample from INCLUDE+LOW_CONFIDENCE+WATCH.
               "stratified"  — "thoughtfully selected": always include the
                                single highest-confidence_delta pick (the
                                one most likely to actually get acted on),
                                then fill the rest randomly from what's
                                left. Falls back to plain random if there
                                are 0-1 candidates.
    seed     : optional seed for reproducibility (tests, or re-running the
               same audit twice to compare).

    Example
    -------
        sample = select_sample(verdicts, n=3, mode="stratified", seed=42)
    """
    candidates = [v for v in verdicts if v.tier in ("INCLUDE", "LOW_CONFIDENCE", "WATCH")]
    if not candidates:
        return []

    rng = random.Random(seed)
    n = min(n, len(candidates))

    if mode == "random":
        return rng.sample(candidates, n)

    if mode != "stratified":
        raise ValueError(f"Unknown sample mode: {mode!r} (expected 'random' or 'stratified')")

    ranked = sorted(candidates, key=lambda v: -v.confidence_delta)
    top_pick = ranked[0]
    remaining = [v for v in candidates if v.ticker != top_pick.ticker]
    fill = rng.sample(remaining, min(n - 1, len(remaining)))
    return [top_pick] + fill


# =============================================================================
# Market context
# =============================================================================

def build_market_context(data_dir: str = "./data", trailing_days: int = 5) -> dict[str, Any]:
    """
    Assemble the market-wide backdrop fed into every validation prompt.

    Reads SPY's own cached price history directly (no network) for the
    trailing return, and reuses marketfit's own dispersion verdict from
    the current market_snapshot.json if one is on disk — deliberately
    reusing marketfit.rules.evaluate() rather than recomputing dispersion
    independently, since that's already the codebase's one source of
    truth for "how dispersed is the market today."

    Best-effort throughout: missing files degrade individual context
    fields to None rather than raising, since a validation run shouldn't
    fail outright just because the market snapshot is stale or absent.
    """
    import os
    import pandas as pd

    context: dict[str, Any] = {
        "spy_trailing_return": None,
        "spy_trailing_days": trailing_days,
        "dispersion_status": None,
        "dispersion_note": None,
    }

    spy_path = os.path.join(data_dir, "prices", "daily", "SPY.parquet")
    if os.path.exists(spy_path):
        try:
            df = pd.read_parquet(spy_path)
            col = "adj_close" if "adj_close" in df.columns else df.columns[0]
            closes = df[col].dropna()
            if len(closes) > trailing_days:
                start, end = closes.iloc[-(trailing_days + 1)], closes.iloc[-1]
                context["spy_trailing_return"] = (end - start) / start
        except Exception as exc:
            logger.warning(f"[qa_audit] Could not compute SPY trailing return: {exc}")

    snapshot_path = os.path.join(data_dir, "market_snapshot.json")
    if os.path.exists(snapshot_path):
        try:
            from marketfit import features as mf_features, rules as mf_rules
            import json as _json
            with open(snapshot_path, encoding="utf-8") as f:
                snapshot = _json.load(f)
            feats = mf_features.extract(snapshot)
            verdict = mf_rules.evaluate(feats)
            context["dispersion_status"] = verdict.dispersion_status
            context["dispersion_note"] = verdict.notes.get("dispersion")
        except Exception as exc:
            logger.warning(f"[qa_audit] Could not compute dispersion context: {exc}")

    return context


# =============================================================================
# Validation result + LLM call
# =============================================================================

@dataclass
class ValidationResult:
    ticker:            str
    shockarb_tier:      str
    llm_verdict:         str                    # AGREE / DISAGREE / UNCERTAIN / ERROR
    llm_confidence:      Optional[float]
    llm_reasoning:        str
    red_flags:           list[str] = field(default_factory=list)
    supporting_points:    list[str] = field(default_factory=list)
    would_need_to_know:   list[str] = field(default_factory=list)
    raw_response:         str = ""

    @property
    def concordant(self) -> bool:
        """True if the LLM's verdict agrees with ShockArb acting on this pick."""
        return self.llm_verdict == "AGREE"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_VERDICTS = {"AGREE", "DISAGREE", "UNCERTAIN"}


def _parse_validation_response(text: str) -> dict[str, Any]:
    """
    Parse the LLM's JSON response, tolerating a markdown fence or stray
    preamble text the same way stockfit/marketfit's parsers do (strip
    fences, then fall back to extracting the first {...} span).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    logger.warning(f"[qa_audit] Could not parse LLM validation response as JSON: {text[:200]!r}")
    return {}


def validate_pick(
    client: LLMClient,
    verdict,                    # StockVerdict
    company_name: str,
    industry: str,
    actual_return: float,
    expected_return: float,
    market_context: dict[str, Any],
) -> ValidationResult:
    """
    Run one LLM validation call for a single pick and parse the result.

    Never raises on a malformed/failed LLM response — returns a
    ValidationResult with llm_verdict="ERROR" instead, so one bad response
    in a batch doesn't abort the whole audit run.
    """
    prompt = build_user_prompt(
        ticker=verdict.ticker, company_name=company_name, industry=industry,
        actual_return=actual_return, expected_return=expected_return,
        r_squared=verdict.r_squared, confidence_delta=verdict.confidence_delta,
        price=verdict.price, analyst_target=verdict.analyst_target,
        analyst_upside=verdict.analyst_upside, news_headlines=verdict.news_headlines,
        market_context=market_context, shockarb_tier=verdict.tier,
        shockarb_reason=verdict.reason,
    )

    try:
        raw = client.complete(VALIDATION_SYSTEM_PROMPT, prompt)
    except Exception as exc:
        logger.error(f"[qa_audit] LLM call failed for {verdict.ticker}: {exc}")
        return ValidationResult(
            ticker=verdict.ticker, shockarb_tier=verdict.tier, llm_verdict="ERROR",
            llm_confidence=None, llm_reasoning=f"LLM call failed: {exc}",
        )

    parsed = _parse_validation_response(raw)
    llm_verdict = parsed.get("verdict", "").upper()
    if llm_verdict not in _VALID_VERDICTS:
        llm_verdict = "ERROR" if not parsed else "UNCERTAIN"

    return ValidationResult(
        ticker=verdict.ticker,
        shockarb_tier=verdict.tier,
        llm_verdict=llm_verdict,
        llm_confidence=parsed.get("confidence"),
        llm_reasoning=parsed.get("reasoning", ""),
        red_flags=parsed.get("red_flags", []) or [],
        supporting_points=parsed.get("supporting_points", []) or [],
        would_need_to_know=parsed.get("would_need_to_know", []) or [],
        raw_response=raw,
    )


def run_validation_batch(
    client: LLMClient,
    verdicts: list,
    features_by_ticker: dict[str, dict],
    company_names: dict[str, dict],
    market_context: dict[str, Any],
    n: int = 3,
    mode: str = "stratified",
    seed: Optional[int] = None,
) -> list[ValidationResult]:
    """
    Select a sample and validate each one. The one entry point most
    callers (the CLI) actually use.

    features_by_ticker : {ticker: feature_dict} — from
        stockfit.features.extract_all(), for actual_return/expected_rel
        (not stored on StockVerdict itself).
    company_names : {ticker: {"Name":..., "Industry":...}} — from
        shockarb.names.TickerReferenceResolver.get_reference().
    """
    sample = select_sample(verdicts, n=n, mode=mode, seed=seed)
    results = []
    for v in sample:
        feat = features_by_ticker.get(v.ticker, {})
        name_info = company_names.get(v.ticker, {"Name": v.ticker, "Industry": "Unknown"})
        results.append(validate_pick(
            client, v,
            company_name=name_info.get("Name", v.ticker),
            industry=name_info.get("Industry", "Unknown"),
            actual_return=feat.get("actual_return", float("nan")),
            expected_return=feat.get("expected_rel", float("nan")),
            market_context=market_context,
        ))
    return results


def summarize_concordance(results: list[ValidationResult]) -> dict[str, Any]:
    """
    Roll a batch of ValidationResults up into an agree/disagree/uncertain
    summary against ShockArb's own tier assignment.

    Example
    -------
        summary = summarize_concordance(results)
        # {"n": 3, "agree": 2, "disagree": 1, "uncertain": 0, "error": 0,
        #  "concordance_rate": 0.667, "disagreements": ["QCOM"]}
    """
    n = len(results)
    counts = {"AGREE": 0, "DISAGREE": 0, "UNCERTAIN": 0, "ERROR": 0}
    for r in results:
        counts[r.llm_verdict] = counts.get(r.llm_verdict, 0) + 1

    scored = n - counts["ERROR"]
    concordance_rate = (counts["AGREE"] / scored) if scored else None

    return {
        "n": n,
        "agree": counts["AGREE"],
        "disagree": counts["DISAGREE"],
        "uncertain": counts["UNCERTAIN"],
        "error": counts["ERROR"],
        "concordance_rate": concordance_rate,
        "disagreements": [r.ticker for r in results if r.llm_verdict == "DISAGREE"],
    }
