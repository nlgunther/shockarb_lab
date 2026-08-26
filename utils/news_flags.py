"""
news_flags — shared helpers for cross-ticker headline attachment and
severity-tag / narrative-coverage checking.

Two related gaps in the CPRT-MISSING-FUNDAMENTAL-CONTEXT pattern (see
HIL_todo.md) live here:

1. news_scanner.py fetches headlines per-ticker and files each one under
   whichever ticker's query returned it. A headline naming several tracked
   tickers (e.g. "ADBE, CRM, NOW, MSFT: Software Stocks Fall...") only lands
   on one of them — root-caused 2026-07-15. cross_attach_headlines() copies
   a headline to every OTHER already-tracked ticker whose symbol appears in
   it as a standalone word.

2. Even when a real, negative headline IS correctly attached, the LLM
   narrative can still pick a more flattering angle and simply not mention
   it (observed on HON, 2026-08-12 — the report's own catalyst feed carried
   the negative headline, but the narrative built its case around two
   unrelated positive ones instead). flagged_headlines_missing_from_narrative()
   checks whether any severity-tagged headline (the "[RATING]"/"[GUIDANCE]"/
   etc. prefix news_scanner.py already writes — see news_scanner._flag_severity)
   is entirely absent from the generated narrative text, so a report can
   surface that gap as a Data Quality Flag instead of relying solely on a
   human final_review to catch it.

Deliberately deferred: cross-attachment here only matches on ticker SYMBOL
(e.g. "ZTS"), not company NAME (e.g. "Zoetis") — the 2026-08-10 ZTS/IDXX
misattachment needed a name match instead. Symbol matching is simpler, has
no file-I/O dependency, and covers the more common "TICKER1, TICKER2: ..."
digest-headline pattern seen repeatedly in this project's news feed. Name
matching (via shockarb.names.TickerReferenceResolver) is a reasonable next
step if symbol matching proves insufficient in practice.
"""

from __future__ import annotations

import re

# Keep in sync with news_scanner.py::_MAX_HEADLINES and the mirrored cap in
# stockfit/features.py and marketfit/cli.py's _load_news — no shared
# constant module exists across all of these yet.
_MAX_HEADLINES_PER_TICKER = 5

# news_scanner.py's _flag_severity() prefixes a matched headline with
# "[CATEGORY] " (e.g. "[GUIDANCE] Honeywell Slides 5%..."); that tag survives
# into news.txt and from there into news_headlines. Matching the tag here
# rather than re-deriving severity from keywords keeps this module decoupled
# from news_scanner's private keyword list.
_TAG_RE = re.compile(r"^\[([A-Z]+)\]\s+(.*)$")

_WORD_RE = re.compile(r"[A-Za-z]{5,}")

# Common long-ish words that would otherwise create false "the narrative
# addressed it" matches on generic overlap alone.
_STOPWORDS = {
    "about", "after", "their", "which", "would", "could", "should", "other",
    "these", "those", "where", "being", "still", "while", "today", "yesterday",
    "market", "stock", "stocks", "shares", "share", "quarter", "report",
}


def cross_attach_headlines(news: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Copy each headline to every other already-tracked ticker it names.

    Only tickers already present as keys in *news* are considered — that's
    the same top-N set news_scanner.py scans each day, so nothing outside
    that day's actual catalyst run gets pulled in. A ticker's own list is
    capped at _MAX_HEADLINES_PER_TICKER after cross-attachment, same as the
    original per-ticker cap.

    Example:
        news = {
            "NOW":  ["ADBE, CRM, NOW, MSFT: Software Stocks Fall After IBM Warns..."],
            "CRM":  ["Salesforce announces new AI product"],
            "ADBE": [],
        }
        cross_attach_headlines(news)["ADBE"]
        # → ["ADBE, CRM, NOW, MSFT: Software Stocks Fall After IBM Warns..."]
    """
    tickers = list(news.keys())
    if len(tickers) < 2:
        return news

    result = {ticker: list(headlines) for ticker, headlines in news.items()}

    for source_ticker, headlines in news.items():
        for headline in headlines:
            for other in tickers:
                if other == source_ticker or len(other) < 3:
                    continue
                if headline in result[other]:
                    continue
                if len(result[other]) >= _MAX_HEADLINES_PER_TICKER:
                    continue
                if re.search(rf"\b{re.escape(other)}\b", headline):
                    result[other].append(headline)

    return result


def flagged_headlines_missing_from_narrative(headlines: list[str], narrative: str) -> list[str]:
    """
    Return the severity-tagged headlines whose *substance* never appears in
    *narrative* — i.e. real flagged news the narrative is silent on.

    A single overlapping word is not enough evidence of coverage: the
    ticker's own company name (e.g. "Honeywell") appears in virtually any
    narrative about that ticker, flagged or not, so single-word overlap
    alone produced false "covered" verdicts — a headline about a guidance
    cut was judged "covered" by a narrative that only shared the word
    "Honeywell" with it. Requiring at least two overlapping content stems
    (or, for a short headline with only one significant word, requiring
    that one word) makes the check actually test whether the narrative
    engages with what the headline says, not just who it's about.

    Example:
        flagged_headlines_missing_from_narrative(
            ["[GUIDANCE] Honeywell Slides 5% As Growth Guidance Disappoints",
             "Quantinuum Posts First Earnings Since IPO"],
            "Honeywell looks undervalued after the recent Aerospace spinoff.",
        )
        # → ["[GUIDANCE] Honeywell Slides 5% As Growth Guidance Disappoints"]
        # ("Honeywell" alone overlaps, but that's not evidence the guidance
        #  cut itself was addressed.)
    """
    if not headlines or not narrative:
        return []

    narrative_stems = _content_stems(narrative)
    missing = []
    for headline in headlines:
        match = _TAG_RE.match(headline)
        if not match:
            continue
        title_stems = _content_stems(match.group(2))
        if not title_stems:
            continue
        overlap = title_stems & narrative_stems
        required = min(2, len(title_stems))
        if len(overlap) < required:
            missing.append(headline)
    return missing


# First-N-characters is a crude but adequate stand-in for a real stemmer
# here — good enough to fold "upgrades"/"upgraded" or "analyst"/"analysts"
# together without pulling in an NLP dependency for a two-function module.
_STEM_LEN = 6


def _content_stems(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS
    return {w[:_STEM_LEN] for w in words}
