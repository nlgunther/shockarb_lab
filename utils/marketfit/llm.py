"""
marketfit.llm — Provider-agnostic LLM client for ShockArb narrative generation.

Adapted from StatementGuard's llm_client.py.  Generates the <!-- LEARN --> narrative
sections of the enhanced market report from structured market data.

Provider support (auto-selected from environment):
  - Google Gemini   — set GOOGLE_API_KEY   (default; free tier available)
  - Anthropic Claude — set ANTHROPIC_API_KEY (preferred when both keys present)

Usage
-----
    client = MarketfitLLMClient.from_env()
    narratives = client.generate_narratives(snapshot, verdict)
    # narratives: {"executive_summary": "...", "sector_rotation_story": "...", ...}

Environment variables
---------------------
    GOOGLE_API_KEY          Gemini API key
    ANTHROPIC_API_KEY       Anthropic API key (preferred over Gemini when both set)
    SHOCKARB_LLM_MODEL      Override model (e.g. "gemini-2.0-flash", "claude-haiku-4-5")
    SHOCKARB_LLM_CALL_LIMIT Max API calls per day (default 10)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from loguru import logger

from trading_calendar import session_label, market_open_at_fetch, et_datetime


# ---------------------------------------------------------------------------
# Retry constants (shared across backends)
# ---------------------------------------------------------------------------

_RETRYABLE_CODES = {"503", "429"}
_MAX_RETRIES     = 3


def _parse_retry_delay(exc: Exception, default: float = 60.0) -> float:
    """
    Extract retry delay from a 429/503 exception message.
    Gemini includes e.g. 'retryDelay: "47s"' or 'Please retry in 47.8s'.
    """
    match = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else default


# ---------------------------------------------------------------------------
# System prompt — ShockArb analyst voice
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a ShockArb market analyst writing the narrative sections of a
daily market report. ShockArb is a quantitative strategy that identifies stocks temporarily
mispriced by macro panic. The factor model decomposes each stock's return into a
macro-explained part and a residual; stocks with large positive residuals (fell more than
factors imply) are mean-reversion candidates. The key signal is confidence_delta = delta_rel × r².

Writing style — strictly enforced:
- Direct and concise. No preamble, no "it is worth noting that", no LLM padding.
- Open with the key observation, not with context-setting.
- Use specific numbers inline: "Dow +1.75%" not "the Dow rose significantly".
- 2–5 sentences per section. Judgment sections may run to 3 short paragraphs.
- Always close the ShockArb-relevant sections with a sentence on what the conditions
  mean for finding dislocated picks — positive or negative.
- Do not repeat numbers that are already in the table above the section.
- The input gives SESSION_LABEL ("today" or "yesterday") for which trading session
  the numbers describe. Use that exact word — never say "today" unless SESSION_LABEL
  says "today". If MARKET_STATUS is not "OPEN", never describe the session as still
  in progress or use present-tense phrasing like "is falling" — it has already closed.
- Any catalyst headline prefixed with a tag in brackets (e.g. "[GUIDANCE]", "[RATING]",
  "[LEADERSHIP]", "[LEGAL]") was flagged as a real, material story, not routine noise.
  If catalyst_summary covers that ticker at all, it must reflect what the tagged
  headline actually says rather than a more flattering unrelated headline for the
  same name.

Return ONLY a JSON object mapping section names to narrative strings.
No markdown fences, no extra keys, no preamble. Example structure:
{"executive_summary": "...", "broad_market_interpretation": "...", ...}"""


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class _LLMBackend(Protocol):
    def call(self, prompt: str) -> tuple[str, float]:
        """Returns (raw_text, estimated_cost_usd). Raises on failure."""
        ...


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

class _AnthropicBackend:
    """
    Calls Anthropic Claude via the `anthropic` package.
    Cost estimate: ~$0.25/MTok input + $1.25/MTok output (Haiku).
    """
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
            # Haiku: $0.25/MTok in, $1.25/MTok out (rough combined estimate)
            cost = (response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000
        return text, cost


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

class _GeminiBackend:
    """
    Calls Google Gemini via the `google-genai` package.
    Retries on 429/503, honouring the API's reported retry delay.
    Cost: treated as $0 on free tier; update if billing applies.
    """
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
                    logger.warning(
                        f"Gemini {code.group(1)} (attempt {attempt}/{_MAX_RETRIES}) — "
                        f"retrying in {delay:.0f}s"
                    )
                    time.sleep(delay)
                    last_exc = exc
                else:
                    raise

        raise last_exc


# ---------------------------------------------------------------------------
# Budget tracker — daily call limit (free tier protection)
# ---------------------------------------------------------------------------

@dataclass
class _DailyBudget:
    """In-process daily call counter. Resets when the date changes."""
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

def _describe_session(snapshot: dict[str, Any]) -> tuple[str, str]:
    """
    Return (session_label, market_status) describing which trading session
    this snapshot's numbers are from and whether the market was open at
    fetch time — the two facts the LLM needs to avoid asserting a session
    is "today" when it's actually a completed prior session (the bug this
    exists to prevent; root-caused 2026-08-07).

    Intraday mode is a special case: live prices are overlaid on top of the
    last cached daily close, so session_date (from the daily cache) does NOT
    describe what's actually being shown — the live numbers are always
    "today, in progress" regardless of what session_date says.

    Example:
        _describe_session({"mode": "daily", "session_date": "2026-08-06",
                            "fetched_at": "2026-08-07T11:46:00+00:00"})
        # -> ("yesterday", "CLOSED")
    """
    if snapshot.get("mode") == "intraday":
        return "today (live, in progress)", "OPEN (intraday snapshot)"

    session_desc = "unknown"
    try:
        fetched_at    = snapshot.get("fetched_at")
        session_date  = snapshot.get("session_date")
        if fetched_at and session_date:
            today_et = et_datetime(datetime.fromisoformat(fetched_at)).date()
            session_desc = session_label(date.fromisoformat(session_date), today_et)
    except Exception:
        pass

    market_status = "OPEN" if market_open_at_fetch(snapshot) else "CLOSED"
    return session_desc, market_status


def _build_prompt(
    snapshot:        dict[str, Any],
    verdict:         Any,
    picks_df:        Any | None = None,   # pandas DataFrame from live_alpha_us.csv
    news_dict:       dict | None = None,  # {ticker: [headline, ...]}
    fundamentals_df: Any | None = None,   # pandas DataFrame from fundamentals.csv
) -> str:
    """
    Build the structured user prompt for narrative generation.

    All computed values are spelled out explicitly so the LLM doesn't have
    to infer numbers from tables — it receives them as structured inputs.
    """
    by_ticker = {t["ticker"]: t for t in snapshot.get("tickers", [])}

    def chg(ticker: str) -> str:
        v = by_ticker.get(ticker, {}).get("chg_pct")
        return f"{v:+.2f}%" if v is not None else "N/A"

    def close(ticker: str) -> str:
        v = by_ticker.get(ticker, {}).get("close")
        return f"{v:,.2f}" if v is not None else "N/A"

    # Sector sort
    sector_tickers = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLRE","XLB","XLC"]
    sector_names   = {"XLK":"Tech","XLF":"Financials","XLE":"Energy","XLV":"Health Care",
                      "XLI":"Industrials","XLY":"Consumer Disc.","XLP":"Consumer Staples",
                      "XLU":"Utilities","XLRE":"Real Estate","XLB":"Materials","XLC":"Comm. Services"}
    sectors_sorted = sorted(
        sector_tickers,
        key=lambda t: by_ticker.get(t, {}).get("chg_pct") or -999,
        reverse=True,
    )

    # Overseas summary
    overseas = [t for t in snapshot.get("tickers", []) if t.get("group") == "overseas"]
    overseas_lines = [f"  {t['label']}: {t['chg_pct']:+.2f}%" for t in overseas if t.get("chg_pct") is not None]

    # VIX close
    vix_row    = by_ticker.get("^VIX", {})
    vix_level  = vix_row.get("close", "N/A")
    vix_chg_v  = vix_row.get("chg_pct", "N/A")

    session_desc, market_status = _describe_session(snapshot)

    parts = [
        f"DATE: {snapshot.get('fetched_at_local', 'unknown')}  "
        f"MODE: {snapshot.get('mode','daily')}  "
        f"BASELINE: {snapshot.get('baseline_date','unknown')}",
        f"SESSION_LABEL: {session_desc}  "
        f"SESSION_DATE: {snapshot.get('session_date', 'unknown')}  "
        f"MARKET_STATUS: {market_status}",
        "",
        "BROAD MARKET:",
        f"  SPY: {chg('SPY')}  QQQ: {chg('QQQ')}  DIA: {chg('DIA')}  IWM: {chg('IWM')}",
        f"  QQQ vs SPY: {_rel(by_ticker,'QQQ','SPY'):+.2f}pp  "
        f"IWM vs SPY: {_rel(by_ticker,'IWM','SPY'):+.2f}pp",
        "",
        "SECTORS (best→worst):",
    ]
    for t in sectors_sorted:
        parts.append(f"  {sector_names.get(t, t)}: {chg(t)}")

    parts += [
        "",
        "BONDS:",
        f"  TLT: {chg('TLT')}  IEF: {chg('IEF')}  HYG: {chg('HYG')}  LQD: {chg('LQD')}",
        "",
        "RISK GAUGES:",
        f"  VIX: {vix_level} ({vix_chg_v:+.2f}%)"
        if isinstance(vix_chg_v, float) else f"  VIX: {vix_level}",
        f"  Gold: {chg('GLD')}  Oil (USO): {chg('USO')}",
        "",
        "OVERSEAS:",
    ] + overseas_lines + [
        "",
        "SHOCKARB VERDICT:",
        f"  Overall: {verdict.overall}  Score: {verdict.score}/11  Trend: {verdict.trend_status}",
        f"  Breadth: {verdict.breadth_status}  VIX: {verdict.vix_status}  "
        f"Dispersion: {verdict.dispersion_status}",
        f"  Tech: {verdict.tech_status}  Bond: {verdict.bond_status}",
    ]

    if picks_df is not None and not picks_df.empty:
        parts += ["", "SHOCKARB PICKS (top signals):"]
        for _, row in picks_df.head(10).iterrows():
            ticker = str(row.get("Ticker", row.get("ticker", "")))
            conf   = row.get("confidence_delta", row.get("Conf.Δ", ""))
            r2     = row.get("r_squared", row.get("R²", ""))
            parts.append(f"  {ticker}: conf_delta={conf}  r²={r2}")

    if fundamentals_df is not None and not fundamentals_df.empty:
        parts += ["", "FUNDAMENTALS (top picks):"]
        for _, row in fundamentals_df.iterrows():
            ticker  = str(row.get("Ticker", ""))
            price   = row.get("Price", "—")
            fwd_pe  = row.get("Fwd P/E", "—")
            tgt     = row.get("Analyst Tgt", "—")
            ex_div  = row.get("Ex-Div", "—")
            parts.append(f"  {ticker}: price={price}  fwd_pe={fwd_pe}x  target={tgt}  ex_div={ex_div}")

    if news_dict:
        parts += ["", "CATALYST HEADLINES (from news feed):"]
        for ticker, headlines in list(news_dict.items())[:10]:
            for h in headlines[:2]:
                parts.append(f"  [{ticker}] {h}")

    # Section specs
    parts += [
        "",
        "Generate these sections as a JSON object {section_name: narrative_text}:",
        '  "executive_summary"          — 3–4 sentences: day\'s story, what\'s driving it, ShockArb relevance',
        '  "broad_market_interpretation" — 2–3 sentences: what the index spread tells us',
        '  "sector_rotation_story"       — 3–5 sentences: why sectors moved as they did, ShockArb implications',
        '  "bond_signal_interpretation"  — 2–3 sentences: what the bond complex signals (use SESSION_LABEL, not "today")',
        '  "overseas_read"               — 2–3 sentences: global picture, any factor risks for US picks',
        '  "risk_gauge_read"             — 2–3 sentences: VIX/Gold/Oil as a combined signal',
        '  "shockarb_fit_analysis"       — 3–5 sentences: deeper ShockArb conditions read',
        '  "watch_list"                  — 3–5 bullet-style sentences on what to monitor into the close/next session',
    ]
    if picks_df is not None and not picks_df.empty:
        parts.append('  "picks_commentary"           — 3–5 sentences: tiered conviction commentary on top picks using fundamentals and news')

    return "\n".join(parts)


def _rel(by_ticker: dict, a: str, b: str) -> float:
    """Return chg_pct(a) - chg_pct(b), or 0.0 if either is missing."""
    ca = by_ticker.get(a, {}).get("chg_pct")
    cb = by_ticker.get(b, {}).get("chg_pct")
    if ca is None or cb is None:
        return 0.0
    return ca - cb


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _try_json_object(text: str) -> dict | None:
    """Parse text as a JSON object; return None (not {}) on any failure."""
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


def _parse_narratives(text: str) -> dict[str, str]:
    """
    Parse LLM JSON response into {section_name: narrative_text}.
    Returns {} if malformed — LLM failures must never crash the pipeline.

    Tries the full trimmed text first. If the model added preamble or
    trailing commentary around the JSON — a common formatting slip that
    gets more likely as the prompt grows (e.g. a large catalyst-headline
    block) — falls back to the first {...} span in the text before
    giving up.

    Example:
        _parse_narratives('{"executive_summary": "Stocks fell on...", ...}')
        # → {"executive_summary": "Stocks fell on...", ...}
        _parse_narratives('Here is the report:\\n{"executive_summary": "..."}')
        # → {"executive_summary": "..."}  (preamble stripped)
    """
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    result = _try_json_object(text)
    if result is None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        result = _try_json_object(match.group(0)) if match else None
    if result is None:
        return {}

    # Keep only string values
    return {k: v for k, v in result.items() if isinstance(v, str) and v.strip()}


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class MarketfitLLMClient:
    """
    Provider-agnostic LLM client for ShockArb market report narrative generation.

    Backend auto-selected: Anthropic if ANTHROPIC_API_KEY is set (preferred),
    Gemini if GOOGLE_API_KEY is set.  Set SHOCKARB_LLM_MODEL to override model.

    Example:
        client = MarketfitLLMClient.from_env()
        narratives = client.generate_narratives(snapshot, verdict)
        # → {"executive_summary": "...", "sector_rotation_story": "...", ...}
    """

    def __init__(
        self,
        api_key:         str | None = None,
        *,
        google_api_key:  str | None = None,
        model:           str | None = None,
        daily_call_limit: int = 10,
        call_pause:       float = 5.0,
    ):
        self.daily_call_limit = daily_call_limit
        self.call_pause       = call_pause
        self._budget          = _DailyBudget()
        self._backend         = self._select_backend(api_key, google_api_key, model)

    @classmethod
    def from_env(cls) -> "MarketfitLLMClient":
        """Construct from environment variables; auto-selects backend."""
        return cls(
            api_key        = os.environ.get("ANTHROPIC_API_KEY"),
            google_api_key = os.environ.get("GOOGLE_API_KEY"),
            model          = os.environ.get("SHOCKARB_LLM_MODEL"),
            daily_call_limit = int(os.environ.get("SHOCKARB_LLM_CALL_LIMIT", "10")),
            call_pause       = float(os.environ.get("SHOCKARB_LLM_CALL_PAUSE", "5.0")),
        )

    def generate_narratives(
        self,
        snapshot:        dict[str, Any],
        verdict:         Any,
        picks_df:        Any | None = None,
        news_dict:       dict | None = None,
        fundamentals_df: Any | None = None,
    ) -> dict[str, str]:
        """
        Generate all narrative sections for the enhanced market report.

        Returns {} on any failure — callers should fall back to build() gracefully.

        Parameters
        ----------
        snapshot        : market_snapshot.json dict
        verdict         : rules.Verdict from rules.evaluate()
        picks_df        : DataFrame from live_alpha_us.csv (optional)
        news_dict       : {ticker: [headline, ...]} from news.txt (optional)
        fundamentals_df : DataFrame from fundamentals.csv (optional)

        Returns
        -------
        dict mapping section names to narrative strings, e.g.:
            {"executive_summary": "...", "sector_rotation_story": "...", ...}
        """
        if not self._budget.can_call(self.daily_call_limit):
            logger.warning(
                f"LLM daily call limit reached ({self.daily_call_limit}/day). "
                "Falling back to basic report. Set SHOCKARB_LLM_CALL_LIMIT to increase."
            )
            return {}

        prompt = _build_prompt(snapshot, verdict, picks_df, news_dict, fundamentals_df)
        logger.info(f"Calling LLM for market report narratives ({self._backend.__class__.__name__})…")

        try:
            raw_text, cost = self._backend.call(prompt)
            self._budget.calls_today += 1
            narratives = _parse_narratives(raw_text)
            logger.info(
                f"LLM returned {len(narratives)} sections"
                + (f" (est. ${cost:.4f})" if cost > 0 else "")
            )
            if not narratives:
                # A clean API call that didn't parse to any usable section is
                # otherwise a black box — the report just silently falls back
                # to build(). Log what actually came back so a recurrence is
                # diagnosable instead of another guessing exercise.
                preview = raw_text[:500].replace("\n", " ")
                logger.warning(f"LLM response had no usable sections — raw text (first 500 chars): {preview!r}")
            if self.call_pause > 0:
                time.sleep(self.call_pause)
            return narratives
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _select_backend(
        api_key: str | None,
        google_api_key: str | None,
        model: str | None,
    ) -> _LLMBackend:
        """Anthropic wins when both keys are present."""
        anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        gemini_key    = google_api_key or os.environ.get("GOOGLE_API_KEY", "")

        if anthropic_key:
            return _AnthropicBackend(api_key=anthropic_key, model=model)
        if gemini_key:
            return _GeminiBackend(api_key=gemini_key, model=model)

        raise RuntimeError(
            "No API key configured. Set ANTHROPIC_API_KEY (Anthropic) "
            "or GOOGLE_API_KEY (Gemini). See docs/ENVIRONMENT_VARIABLES.md."
        )
