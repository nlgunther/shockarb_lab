"""
ShockArb Session Log Summarizer.

Reads the tail of data/shockarb.log for the requested time window and optionally
calls the LLM (via the same Anthropic/Gemini backend used by marketfit) to produce
a structured session summary.

Usage
-----
    # Print recent log lines (last 2 hours, no LLM)
    python scripts/session_log.py --hours 2

    # Summarize the last 3 hours of activity with LLM
    python scripts/session_log.py --hours 3 --summarize

    # Save summary to logs/ in addition to printing
    python scripts/session_log.py --hours 3 --summarize --save

    # Summarize last N lines instead of a time window
    python scripts/session_log.py --lines 200 --summarize

Output
------
    Console: structured summary with sections:
      - What was scored (regimes, tickers, any drops)
      - Signals found (INCLUDE / WATCH highlights)
      - Errors / warnings seen
      - Suggested CHEATSHEET additions or workflow notes
    File (--save): logs/session_YYYYMMDD_HHMM.txt

Requirements
------------
    ANTHROPIC_API_KEY or GOOGLE_API_KEY in environment (for --summarize).
    Run from the project root.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_PATH    = Path("data/shockarb.log")
_LOGS_DIR    = Path("logs")
_DEFAULT_HRS = 3
_DEFAULT_LINES = 500   # fallback when --lines given instead of --hours

_SYSTEM_PROMPT = """\
You are ShockArb session analyst. You receive raw log output from a quantitative
trading pipeline and produce a concise, structured session summary.

Output format (Markdown, no preamble):

## Session Summary — {date}

### What Was Run
<bullet list: regimes scored, commands executed, key flags used>

### Signals Found
<bullet list: INCLUDE tickers with r², conf.Δ; WATCH tickers; EXCLUDE count>

### Errors / Warnings
<bullet list: coverage drops, missing files, API errors, stale data warnings>
(Write "None" if the session was clean.)

### Suggested CHEATSHEET / Workflow Notes
<bullet list: anything the user should add to their cheatsheet or bat file,
workflow steps that were awkward, flags that would have helped>
(Write "None" if nothing stands out.)

Keep each section concise. Omit sections that are empty / not applicable.
Do not add commentary outside these sections.
"""


# ---------------------------------------------------------------------------
# Log reading
# ---------------------------------------------------------------------------

def _read_log_lines(hours: float | None, max_lines: int) -> list[str]:
    """Return log lines from the last `hours` hours (or last `max_lines` lines)."""
    if not _LOG_PATH.exists():
        print(f"❌  Log not found: {_LOG_PATH}")
        print("    Run a scoring session first: python -m shockarb score")
        sys.exit(1)

    raw = _LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()

    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        filtered = []
        for line in raw:
            # loguru format: "2026-06-18 14:32:01.123 | INFO  | ..."
            ts_part = line[:23].strip()
            try:
                ts = datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S.%f")
                ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    filtered.append(line)
            except ValueError:
                # Non-timestamped continuation line — include if we're already in window
                if filtered:
                    filtered.append(line)
        return filtered

    # Fallback: last N lines
    return raw[-max_lines:]


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------

def _summarize(log_lines: list[str], date_str: str) -> str:
    """Call the LLM to produce a structured session summary."""
    log_text = "\n".join(log_lines)

    # Reuse marketfit's provider-agnostic LLM backend
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
        from marketfit.llm import MarketfitLLMClient  # type: ignore
    except ImportError as exc:
        print(f"❌  Cannot import marketfit.llm: {exc}")
        print("    Ensure utils/ is on the path and dependencies are installed.")
        sys.exit(1)

    try:
        client = MarketfitLLMClient.from_env()
    except RuntimeError as exc:
        print(f"❌  LLM not available: {exc}")
        print("    Set ANTHROPIC_API_KEY or GOOGLE_API_KEY to enable --summarize.")
        sys.exit(1)

    system = _SYSTEM_PROMPT.format(date=date_str)
    user_msg = (
        f"Below is the ShockArb pipeline log for the session ending {date_str}.\n\n"
        f"```\n{log_text[:40_000]}\n```\n\n"   # truncate to ~40k chars to stay in context
        "Produce the session summary as specified."
    )

    # Use the client's low-level call (bypass the narrative-specific generate_narratives)
    try:
        summary = client._call(system=system, user=user_msg)   # type: ignore[attr-defined]
    except AttributeError:
        # Fallback: try calling the generate_narratives pathway with a sentinel snapshot
        summary = _fallback_summarize(client, system, user_msg)

    return summary


def _fallback_summarize(client, system: str, user_msg: str) -> str:
    """
    If the client doesn't expose _call(), try constructing a minimal prompt
    via the Anthropic or Google SDK directly.
    """
    provider = getattr(client, "_provider", None)
    model    = getattr(client, "_model", None)

    if provider == "anthropic":
        import anthropic  # type: ignore
        ac = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = ac.messages.create(
            model   = model or "claude-haiku-4-5",
            max_tokens = 1024,
            system  = system,
            messages = [{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    if provider == "google":
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        m = genai.GenerativeModel(
            model_name = model or "gemini-2.0-flash",
            system_instruction = system,
        )
        resp = m.generate_content(user_msg)
        return resp.text

    raise RuntimeError(f"Unknown provider '{provider}' — cannot call LLM.")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_summary(summary: str, date_str: str) -> Path:
    """Write summary to logs/session_YYYYMMDD_HHMM.txt."""
    _LOGS_DIR.mkdir(exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = _LOGS_DIR / f"session_{ts}.txt"
    path.write_text(summary, encoding="utf-8")
    return path


def _print_lines(lines: list[str], hours: float | None, max_lines: int) -> None:
    """Print log lines with a header."""
    window = f"last {hours:.0f}h" if hours else f"last {max_lines} lines"
    print(f"\n{'='*70}")
    print(f"  shockarb.log — {window}  ({len(lines)} lines)")
    print(f"{'='*70}")
    for line in lines:
        print(line)
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a ShockArb pipeline session from shockarb.log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hours", type=float, default=None,
        help=f"How many hours back to scan (default: {_DEFAULT_HRS}). "
             "Mutually exclusive with --lines.",
    )
    parser.add_argument(
        "--lines", type=int, default=None,
        help=f"Read last N lines instead of a time window (default: {_DEFAULT_LINES}).",
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Call LLM to produce a structured session summary (requires API key).",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the summary to logs/session_YYYYMMDD_HHMM.txt.",
    )
    parser.add_argument(
        "--log", default=str(_LOG_PATH),
        help=f"Path to shockarb.log (default: {_LOG_PATH}).",
    )
    args = parser.parse_args()

    # Override log path if requested
    global _LOG_PATH
    _LOG_PATH = Path(args.log)

    # Default: --hours 3 if neither --hours nor --lines given
    hours     = args.hours if (args.hours is not None or args.lines is not None) else float(_DEFAULT_HRS)
    max_lines = args.lines if args.lines is not None else _DEFAULT_LINES

    lines    = _read_log_lines(hours=hours, max_lines=max_lines)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if not args.summarize:
        _print_lines(lines, hours, max_lines)
        print("Tip: pass --summarize to generate an LLM session summary.")
        return

    if not lines:
        print("No log lines found in the requested window.")
        sys.exit(0)

    print(f"Summarizing {len(lines)} log lines with LLM…")
    summary = _summarize(lines, date_str)

    print(f"\n{'='*70}")
    print(summary)
    print(f"{'='*70}\n")

    if args.save:
        path = _save_summary(summary, date_str)
        print(f"📁  Saved to: {path}")


if __name__ == "__main__":
    main()
