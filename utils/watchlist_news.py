"""
watchlist_news.py -- Fundamental news digest for a configured ticker watchlist.

Uses Gemini's Google Search grounding to surface deep fundamental news
(SEC filings, fund redemption caps, workforce cuts, regulatory actions) that
the standard yfinance-based news_scanner.py misses.

Ticker resolution (in priority order):
  1. --tickers X Y Z --override   -> only these tickers (watchlist + CSV ignored)
  2. --tickers X Y Z              -> additive: watchlist U top-N alpha CSV U explicit, deduplicated
  3. (default)                    -> watchlist U top-N alpha CSV, deduplicated

One Gemini call is made per ticker so the model is forced to search for each
name individually rather than falling back to stale training data for a batch.

Output: reports/watchlist_news_YYYYMMDD_HHMM.md

Usage
-----
    python utils/watchlist_news.py
    python utils/watchlist_news.py --tickers BLK ORCL
    python utils/watchlist_news.py --tickers BLK ORCL --override
    python utils/watchlist_news.py --top 10 --no-out

Environment variables
---------------------
    GOOGLE_API_KEY          Gemini API key (required)
    SHOCKARB_LLM_MODEL      Override model (default: gemini-2.5-flash)
    WATCHLIST_CALL_LIMIT    Max API calls per day (default: 20; one call per ticker)
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from paths import LIVE_ALPHA_US, REPORTS, WATCHLIST


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL      = "gemini-2.5-flash"
_DEFAULT_TOP_N      = 5
_DEFAULT_CALL_LIMIT = 20   # one call per ticker; typical watchlist is 5-10
_MAX_RETRIES        = 3
_RETRYABLE_CODES    = {"503", "429"}


# ---------------------------------------------------------------------------
# Retry helper (mirrors stockfit/llm.py)
# ---------------------------------------------------------------------------

def _parse_retry_delay(exc: Exception, default: float = 60.0) -> float:
    match = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else default


# ---------------------------------------------------------------------------
# Daily budget (mirrors stockfit/llm.py)
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
# Ticker resolution
# ---------------------------------------------------------------------------

def _load_watchlist(path: Path) -> list[str]:
    """Return uppercase tickers from a watchlist file; skip blank lines and comments."""
    if not path.exists():
        return []
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _load_alpha_top(path: Path, top_n: int) -> list[str]:
    """Return top-N tickers by confidence_delta from a ShockArb alpha CSV."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        if "confidence_delta" not in df.columns or "Ticker" not in df.columns:
            return []
        return df.nlargest(top_n, "confidence_delta")["Ticker"].str.upper().tolist()
    except Exception as exc:
        logger.warning(f"Could not load alpha CSV ({path}): {exc}")
        return []


def resolve_tickers(
    explicit:       list[str] | None,
    override:       bool,
    watchlist_path: Path = WATCHLIST,
    alpha_path:     Path = LIVE_ALPHA_US,
    top_n:          int  = _DEFAULT_TOP_N,
) -> list[str]:
    """
    Return a sorted, deduplicated, uppercase ticker list.

    Resolution rules:
      override=True  -> only explicit tickers (--tickers required)
      override=False -> watchlist U alpha top-N U explicit, deduplicated
    """
    if override:
        if not explicit:
            raise SystemExit("--override requires --tickers")
        return sorted(set(t.upper() for t in explicit))

    pool: set[str] = set()
    pool.update(_load_watchlist(watchlist_path))
    pool.update(_load_alpha_top(alpha_path, top_n))
    if explicit:
        pool.update(t.upper() for t in explicit)
    return sorted(pool)


# ---------------------------------------------------------------------------
# System prompt -- passed as system_instruction (not prepended to user content)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial analyst writing a fundamental news digest for equity investors.

For the ticker in the user's query, use Google Search to find news published in the
last 7 days. Report on events that affect the fundamental investment case:
  - SEC filings (8-K, S-3/ATM equity offerings, proxy statements)
  - Earnings surprises or guidance changes
  - Workforce reductions, restructuring, or leadership changes
  - Fund redemption caps or liquidity restrictions
  - Regulatory, legal, or antitrust actions
  - M&A, spin-offs, or strategic pivots
  - Credit rating changes or significant new debt issuance
  - Unusual insider buying or selling

Output format -- Markdown for this ticker only:

## {TICKER} -- {Company Name}
**Key takeaway:** One sentence on the most important development this week,
or "No material news this week." if nothing was found after searching.

2-3 additional sentences of context if material news was found.

**Stories:**
- **[Exact headline](URL)** -- one sentence on why this matters to a long equity holder.
(One bullet per story. Omit this subsection entirely if no stories were found.)
"""


# ---------------------------------------------------------------------------
# Grounded Gemini backend
# ---------------------------------------------------------------------------

def _extract_sources(response) -> list[dict]:
    """Extract {title, url} pairs from Gemini grounding metadata."""
    sources: list[dict] = []
    try:
        for candidate in response.candidates:
            meta = getattr(candidate, "grounding_metadata", None)
            if not meta:
                continue
            for chunk in getattr(meta, "grounding_chunks", []):
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    sources.append({
                        "title": getattr(web, "title", None) or web.uri,
                        "url":   web.uri,
                    })
    except Exception as exc:
        logger.warning(f"Could not extract grounding sources: {exc}")
    return sources


class _GeminiGroundedBackend:
    """
    Gemini backend with Google Search grounding enabled.

    Makes one grounded call per ticker so the model is forced to search
    for each name individually rather than falling back to training data
    for a large batch. system_instruction is passed in config (not prepended
    to contents) so it does not suppress the model's search decision.
    """

    DEFAULT_MODEL = _DEFAULT_MODEL

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model   = model or self.DEFAULT_MODEL

    def call(self, tickers: list[str]) -> tuple[str, list[dict]]:
        """
        Make one grounded search call per ticker.
        Return (combined_markdown, all_sources).
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google-genai package required: pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        sections:    list[str]  = []
        all_sources: list[dict] = []

        for ticker in tickers:
            prompt   = _build_ticker_prompt(ticker)
            last_exc: Exception | None = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    response = client.models.generate_content(
                        model=self.model, contents=prompt, config=config,
                    )
                    srcs = _extract_sources(response)
                    sections.append(response.text.strip())
                    all_sources.extend(srcs)
                    logger.info(f"  {ticker}: {len(response.text)} chars, {len(srcs)} sources")
                    break
                except Exception as exc:
                    code = re.search(r"^(\d+)", str(exc))
                    if code and code.group(1) in _RETRYABLE_CODES:
                        if "PerDay" in str(exc) or "per_day" in str(exc).lower():
                            raise RuntimeError(
                                f"Daily quota exhausted for {self.model!r}. "
                                "Try again tomorrow or set SHOCKARB_LLM_MODEL to an alternative."
                            ) from exc
                        delay = _parse_retry_delay(exc)
                        logger.warning(
                            f"Gemini {code.group(1)} -- {ticker}"
                            f" (attempt {attempt}/{_MAX_RETRIES}), retrying in {delay:.0f}s"
                        )
                        time.sleep(delay)
                        last_exc = exc
                    else:
                        raise
            else:
                raise last_exc  # type: ignore[misc]

        return "\n\n".join(sections), all_sources


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_ticker_prompt(ticker: str) -> str:
    """Minimal, unambiguous prompt -- lets system_instruction carry the format contract."""
    return (
        f"Search Google for material fundamental news about {ticker} "
        f"from the past 7 days and write the digest section for this ticker."
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _build_overall_digest(combined_text: str) -> str:
    """
    Extract the Key takeaway line from each ticker section and render
    them as a top-level Overall Digest. Zero extra API calls.
    """
    takeaways = re.findall(r"\*\*Key takeaway:\*\* (.+)", combined_text)
    if not takeaways:
        return ""
    lines = ["## Overall Digest", ""]
    lines.extend(f"- {t}" for t in takeaways)
    return "\n".join(lines) + "\n\n---\n\n"


def _render_report(text: str, sources: list[dict], tickers: list[str]) -> str:
    """
    Assemble final report: header + overall digest + per-ticker sections
    + deduplicated sources from grounding metadata.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        "# Watchlist News Digest\n"
        f"**{timestamp}** -- {', '.join(tickers)}\n\n"
        "*Fundamental events only. Powered by Gemini Google Search grounding.*\n\n"
        "---\n\n"
    )
    digest = _build_overall_digest(text)

    # Append any grounding sources not already linked inline
    seen_urls = set(re.findall(r"\(https?://[^\)]+\)", text))
    extra = [s for s in sources if f"({s['url']})" not in seen_urls]
    sources_section = ""
    if extra:
        deduped: set[str] = set()
        lines = ["\n---", "\n## Sources\n"]
        for s in extra:
            if s["url"] not in deduped:
                deduped.add(s["url"])
                lines.append(f"- [{s['title']}]({s['url']})")
        sources_section = "\n".join(lines)

    return header + digest + text + sources_section


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class WatchlistNewsClient:
    """
    Fundamental news digest client using Gemini Google Search grounding.

    Example:
        client = WatchlistNewsClient.from_env()
        report = client.run(["BLK", "ORCL", "MSFT"])
        # -> Markdown string, ready to save or print
    """

    def __init__(
        self,
        api_key:          str | None = None,
        model:            str | None = None,
        daily_call_limit: int = _DEFAULT_CALL_LIMIT,
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
        self._backend = _GeminiGroundedBackend(api_key=key, model=model)
        self._limit   = daily_call_limit
        self._budget  = _DailyBudget()

    @classmethod
    def from_env(cls) -> "WatchlistNewsClient":
        return cls(
            api_key          = os.environ.get("GOOGLE_API_KEY"),
            model            = os.environ.get("SHOCKARB_LLM_MODEL"),
            daily_call_limit = int(os.environ.get("WATCHLIST_CALL_LIMIT", str(_DEFAULT_CALL_LIMIT))),
        )

    def run(self, tickers: list[str]) -> str:
        """
        Search for fundamental news on tickers and return a Markdown report.
        Returns empty string on budget exhaustion or failure.
        """
        if not tickers:
            logger.warning("No tickers resolved -- nothing to scan.")
            return ""
        if self._budget.calls_today + len(tickers) > self._limit:
            logger.warning(
                f"Daily call limit ({self._limit}) would be exceeded "
                f"({self._budget.calls_today} used + {len(tickers)} needed)."
            )
            return ""

        logger.info(f"Scanning watchlist news: {', '.join(tickers)}")
        try:
            text, sources = self._backend.call(tickers)
            self._budget.calls_today += len(tickers)
            logger.info(f"Response: {len(text)} chars, {len(sources)} grounding sources")
            return _render_report(text, sources, tickers)
        except Exception as exc:
            logger.error(f"Watchlist news scan failed: {exc}")
            return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save_report(report: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path  = out_dir / f"watchlist_news_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    logger.success(f"Report saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fundamental news digest via Gemini Google Search grounding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Explicit ticker list (additive by default; use --override to replace)",
    )
    parser.add_argument(
        "--override", action="store_true",
        help="Use only --tickers; ignore watchlist.txt and alpha CSV",
    )
    parser.add_argument(
        "--top", type=int, default=_DEFAULT_TOP_N,
        help="Top-N tickers from alpha CSV (default: %(default)s; ignored with --override)",
    )
    parser.add_argument(
        "--watchlist", type=Path, default=WATCHLIST,
        help="Watchlist file path (default: data/watchlist.txt; ignored with --override)",
    )
    parser.add_argument(
        "--out", type=Path, default=REPORTS,
        help="Output directory (default: reports/)",
    )
    parser.add_argument(
        "--no-out", action="store_true",
        help="Print report to stdout; do not save file",
    )
    args = parser.parse_args()

    tickers = resolve_tickers(
        explicit       = args.tickers,
        override       = args.override,
        watchlist_path = args.watchlist,
        alpha_path     = LIVE_ALPHA_US,
        top_n          = args.top,
    )
    if not tickers:
        raise SystemExit(
            "No tickers resolved. Add entries to data/watchlist.txt or pass --tickers."
        )

    client = WatchlistNewsClient.from_env()
    report = client.run(tickers)
    if not report:
        raise SystemExit("Report generation failed -- check logs above.")

    if args.no_out:
        print(report)
    else:
        _save_report(report, args.out)


if __name__ == "__main__":
    main()
