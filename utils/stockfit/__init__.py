"""
stockfit — Automated ShockArb stock opportunity report.

Reads live_alpha_us.csv + fundamentals.csv + news.txt, applies signal-quality
filters, and produces a ranked Markdown report with optional LLM narratives.

Usage
-----
    cd utils && python -m stockfit report
    cd utils && python -m stockfit report --llm --timestamp

Modules
-------
    features   — pure: per-ticker feature extraction from CSV + news inputs
    rules      — pure: feature dict → StockVerdict (INCLUDE / WATCH / EXCLUDE)
    report     — pure: candidates list → Markdown string
    llm        — provider-agnostic LLM client for per-stock narrative (Gemini / Anthropic)
    cli        — argparse CLI entry point
"""

from stockfit.features import extract_all
from stockfit.rules import evaluate_all
from stockfit.report import build, build_enhanced

__all__ = ["extract_all", "evaluate_all", "build", "build_enhanced"]
