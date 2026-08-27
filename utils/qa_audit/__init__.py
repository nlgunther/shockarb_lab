"""
qa_audit — QA / health-check tooling for ShockArb.

Two independent layers, meant to be read together:

  stats_checks   — fast, deterministic, no-LLM sanity checks (cache
                   alignment, return-magnitude outliers, r² distribution,
                   pick-count-vs-history, upside sanity, cluster
                   concentration). Runs in milliseconds, catches the class
                   of bug this project has hit repeatedly (cache
                   corruption, misaligned dates, stale overrides).

  llm_validator  — the "gold standard" layer: an independent LLM call,
                   given the same evidence a human analyst would have
                   (ShockArb's own numbers, current price/target, recent
                   headlines, market-wide context), asked to form its own
                   judgment on whether each pick's mean-reversion thesis
                   holds up — explicitly instructed not to just rubber-
                   stamp the quant signal. See llm_validator.py's module
                   docstring for the full prompt design and its stated
                   limitations (no live web search — reasons from provided
                   evidence + general knowledge, not real-time facts).

report / cli / __main__ wire the two together into a single Markdown
report and a `python -m qa_audit run` entry point.
"""
