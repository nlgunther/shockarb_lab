"""
stockfit.llm — Provider-agnostic LLM client for ShockArb per-stock narrative generation.

Parallel to marketfit.llm but focused on per-ticker analysis rather than macro conditions.
Generates the <!-- LEARN --> narrative sections of the enhanced stock report from
structured signal + fundamental + news inputs.

Provider support (auto-selected from environment):
  - Anthropic Claude — set ANTHROPIC_API_KEY (preferred)
  - Google Gemini    — set GOOGLE_API_KEY

Usage
-----
    client = StockfitLLMClient.from_env()
    narratives = client.generate_narratives(verdicts)
    # narratives: {"executive_summary": "...", "picks_analysis": {"ETN": "...", ...}, ...}

Environment variables
---------------------
    ANTHROPIC_API_KEY      Anthropic API key (preferred over Gemini when both set)
    GOOGLE_API_KEY         Gemini API key
    SHOCKARB_LLM_MODEL     Override model
    SHOCKARB_LLM_CALL_LIMIT  Max API calls per day (default 10)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from stockfit.rules import StockVerdict


# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

_RETRYABLE_CODES = {"503", "429"}
_MAX_RETRIES     = 3


def _parse_retry_delay(exc: Exception, default: float = 60.0) -> float:
    match = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else default


# ---------------------------------------------------------------------------
# System prompt — ShockArb stock analyst voice
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a ShockArb stock analyst writing per-ticker commentary for a
daily stock opportunity report. ShockArb is a quantitative strategy that identifies stocks
temporarily mispriced by macro panic. The factor model decomposes each stock's return into
a macro-explained part and a residual; stocks with large positive residuals (fell more than
factors imply) are mean-reversion candidates. The key signal is confidence_delta = delta_rel × r².

Writing style — strictly enforced:
- Direct and concise. No preamble, no LLM padding.
- Open with the key observation: why this stock is dislocated and why it should revert.
- Use specific numbers: r², conf.Δ, price, analyst target, upside %.
- Reference news headlines when relevant to the thesis (positive catalyst or absence of negative).
- 3–5 sentences per stock. Cluster-risk tickers get one additional sentence on the caveat.
- Do not repeat numbers already in the summary table.

Return ONLY a JSON object with these keys:
  "executive_summary"  — 3-4 sentences: today's overall ShockArb opportunity set
  "picks_analysis"     — nested object: {ticker: narrative_string} for each INCLUDE ticker
  "watch_list_notes"   — 2-3 sentences on the WATCH tier tickers collectively
  "risk_factors"       — 2-3 sentences: macro or data risks that could invalidate these signals

No markdown fences, no extra keys, no preamble."""


# ---------------------------------------------------------------------------
# Backends (reuse same pattern as marketfit.llm)
# ---------------------------------------------------------------------------

class _AnthropicBackend:
    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model   = model or self.DEFAULT_MODEL

    def call(self, prompt: str) -> tuple[str, float]:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package required: pip install anthropic")
        client   = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model      = self.model,
            max_tokens = 4096,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        cost = 0.0
        if hasattr(response, "usage"):
            cost = (response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000
        return text, cost


class _GeminiBackend:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model   = model or self.DEFAULT_MODEL

    def call(self, prompt: str) -> tuple[str, float]:
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("google-genai package required: pip install google-genai")
        client      = genai.Client(api_key=self.api_key)
        full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=full_prompt)
                return response.text, 0.0
            except Exception as exc:
                code = re.search(r"^(\d+)", str(exc))
                if code and code.group(1) in _RETRYABLE_CODES:
                    is_daily = "PerDay" in str(exc) or "per_day" in str(exc).lower()
                    if is_daily:
                        raise RuntimeError(
                            f"Daily quota exhausted for {self.model!r}. "
                            "Set SHOCKARB_LLM_MODEL to an alternative or try tomorrow."
                        ) from exc
                    delay = _parse_retry_delay(exc)
                    logger.warning(f"Gemini {code.group(1)} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {delay:.0f}s")
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise
        raise last_exc


# ---------------------------------------------------------------------------
# Daily budget
# ---------------------------------------------------------------------------

@dataclass
class _DailyBudget:
    calls_today: int = 0
    day_key:     str = ""

    def reset_if_new_day(self) -> None:
        from datetime import date
        today = date.today().isoformat()
        if self.day_key != today:
            self.calls_today = 0
            self.day_key     = today

    def can_call(self, limit: int) -> bool:
        self.reset_if_new_day()
        return self.calls_today < limit


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(verdicts: list[StockVerdict]) -> str:
    include = [v for v in verdicts if v.tier == "INCLUDE"]
    watch   = [v for v in verdicts if v.tier == "WATCH"]

    parts = ["SHOCKARB STOCK SIGNALS — INCLUDE TIER:", ""]

    for v in include:
        upside = f"{v.analyst_upside * 100:+.1f}%" if v.analyst_upside is not None else "N/A"
        price  = f"${v.price:,.2f}" if v.price else "N/A"
        target = f"${v.analyst_target:,.2f}" if v.analyst_target else "N/A"
        pe     = f"{v.fwd_pe:.1f}x" if v.fwd_pe else "N/A"
        lines  = [
            f"[{v.ticker}]",
            f"  r²={v.r_squared:.3f}  conf_delta={v.confidence_delta:+.4f}",
            f"  price={price}  analyst_target={target}  upside={upside}  fwd_pe={pe}",
        ]
        if v.cluster:
            lines.append(f"  cluster={v.cluster}")
        if v.warnings:
            for w in v.warnings:
                lines.append(f"  WARNING: {w}")
        if v.news_headlines:
            lines.append("  news:")
            for h in v.news_headlines:
                lines.append(f"    - {h}")
        parts.extend(lines)
        parts.append("")

    if watch:
        parts += ["WATCH TIER:", ""]
        for v in watch:
            upside = f"{v.analyst_upside * 100:+.1f}%" if v.analyst_upside is not None else "N/A"
            parts.append(f"  [{v.ticker}] r²={v.r_squared:.3f} conf_delta={v.confidence_delta:+.4f} upside={upside} — {v.reason}")
        parts.append("")

    parts += [
        "Generate a JSON object with keys:",
        '  "executive_summary"  — 3-4 sentences on the overall opportunity set today',
        '  "picks_analysis"     — {"TICKER": "3-5 sentence narrative"} for each INCLUDE ticker',
        '  "watch_list_notes"   — 2-3 sentences on WATCH tickers collectively',
        '  "risk_factors"       — 2-3 sentences on macro or data risks that could invalidate these signals',
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_narratives(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(result, dict):
        return {}
    return result


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class StockfitLLMClient:
    """
    Provider-agnostic LLM client for ShockArb stock narrative generation.

    Example:
        client = StockfitLLMClient.from_env()
        narratives = client.generate_narratives(verdicts)
        # → {"executive_summary": "...", "picks_analysis": {"ETN": "..."}, ...}
    """

    def __init__(
        self,
        api_key:           str | None = None,
        *,
        google_api_key:    str | None = None,
        model:             str | None = None,
        daily_call_limit:  int = 10,
        call_pause:        float = 5.0,
    ):
        self.daily_call_limit = daily_call_limit
        self.call_pause       = call_pause
        self._budget          = _DailyBudget()
        self._backend         = self._select_backend(api_key, google_api_key, model)

    @classmethod
    def from_env(cls) -> "StockfitLLMClient":
        return cls(
            api_key        = os.environ.get("ANTHROPIC_API_KEY"),
            google_api_key = os.environ.get("GOOGLE_API_KEY"),
            model          = os.environ.get("SHOCKARB_LLM_MODEL"),
            daily_call_limit = int(os.environ.get("SHOCKARB_LLM_CALL_LIMIT", "10")),
            call_pause       = float(os.environ.get("SHOCKARB_LLM_CALL_PAUSE", "5.0")),
        )

    def generate_narratives(self, verdicts: list[StockVerdict]) -> dict[str, Any]:
        """
        Generate all narrative sections for the enhanced stock report.

        Returns {} on any failure — caller falls back to build() gracefully.

        Parameters
        ----------
        verdicts : output of rules.evaluate_all()

        Returns
        -------
        dict with keys: executive_summary, picks_analysis, watch_list_notes, risk_factors
        """
        include_count = sum(1 for v in verdicts if v.tier == "INCLUDE")
        if include_count == 0:
            logger.info("No INCLUDE-tier tickers — skipping LLM call.")
            return {}

        if not self._budget.can_call(self.daily_call_limit):
            logger.warning(
                f"LLM daily call limit reached ({self.daily_call_limit}/day). "
                "Falling back to basic report."
            )
            return {}

        prompt = _build_prompt(verdicts)
        logger.info(f"Calling LLM for stock narratives ({self._backend.__class__.__name__})…")

        try:
            raw_text, cost = self._backend.call(prompt)
            self._budget.calls_today += 1
            narratives = _parse_narratives(raw_text)
            logger.info(
                f"LLM returned {len(narratives)} sections"
                + (f" (est. ${cost:.4f})" if cost > 0 else "")
            )
            if self.call_pause > 0:
                time.sleep(self.call_pause)
            return narratives
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return {}

    @staticmethod
    def _select_backend(api_key, google_api_key, model):
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_key    = google_api_key or os.environ.get("GOOGLE_API_KEY", "")
        if anthropic_key:
            return _AnthropicBackend(api_key=anthropic_key, model=model)
        if gemini_key:
            return _GeminiBackend(api_key=gemini_key, model=model)
        raise RuntimeError(
            "No API key configured. Set ANTHROPIC_API_KEY or GOOGLE_API_KEY."
        )
