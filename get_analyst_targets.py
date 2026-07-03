import argparse
import json
import re
import pandas as pd
import requests
import yfinance as yf
import os
import sys
import time
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from dataclasses import dataclass
from loguru import logger

_REQUEST_TIMEOUT = 10  # seconds

# Retry tuning shared by GeminiSearchProvider — same constants/approach as
# utils/stockfit/llm.py's _GeminiBackend. Duplicated rather than imported:
# that module lives in a different package (utils/stockfit) that isn't on
# sys.path for this top-level script, and the logic is ~15 lines, so a
# private cross-package import wasn't worth the coupling.
_RETRYABLE_CODES = {"503", "429"}
_MAX_RETRIES     = 3


def _parse_retry_delay(exc: Exception, default: float = 60.0) -> float:
    match = re.search(r"retry[^0-9]*(\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else default


def _is_permission_denied(exc: Exception) -> bool:
    """
    True if `exc` is a 403 from the Gemini API.

    Unlike a 429/503, a 403 here is never per-request — it means the
    GOOGLE_API_KEY's project has an account-level problem (e.g. a billing
    "dunning" enforcement action, a disabled API, a restricted key). Every
    subsequent call in the run will fail the exact same way, so this is
    treated as fatal for the whole batch rather than retried per ticker.

    Example:
        _is_permission_denied(Exception("403 PERMISSION_DENIED. {...}"))
        # -> True
    """
    code = re.search(r"^(\d+)", str(exc))
    return bool(code and code.group(1) == "403")


@dataclass
class _DailyBudget:
    """Caps LLM calls per day. Same pattern as stockfit.llm._DailyBudget."""
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


def _extract_json(text: str) -> dict | None:
    """
    Parse a JSON object out of `text`, tolerating markdown fences and
    surrounding prose.

    Tries a plain `json.loads` first (the common case when the model
    followed instructions exactly). Falls back to scanning for the first
    *balanced* {...} block via brace counting. A naive greedy regex
    (`\\{.*\\}`) would splice from the first "{" to the very last "}" in
    the whole reply — if the model's prose mentions more than one brace
    group, that produces a garbled, unparseable span instead of the
    actual answer.

    Example:
        _extract_json('Here you go:\\n```json\\n{"a": 1}\\n```')
        # -> {"a": 1}
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None

# ==========================================
# 1. PROVIDER INTERFACE & IMPLEMENTATIONS
# ==========================================

class BaseProvider(ABC):
    @abstractmethod
    def fetch_target(self, symbol: str) -> dict:
        pass

class FMPProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable is not set.")
        self.base_url = "https://financialmodelingprep.com/api/v4/price-target-consensus"

    def fetch_target(self, symbol: str) -> dict:
        url = f"{self.base_url}?symbol={symbol}&apikey={self.api_key}"
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                pt_data = data[0]
                return {
                    "Symbol": symbol,
                    "Target_Mean": pt_data.get("targetConsensus"),
                    "Target_Median": pt_data.get("targetMedian"),
                    "Target_High": pt_data.get("targetHigh"),
                    "Target_Low": pt_data.get("targetLow")
                }
            return None

        elif response.status_code in [402, 403]:
            raise PermissionError(f"HTTP {response.status_code} on {symbol}: {response.text}")
        else:
            logger.warning(f"FMP failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class YFinanceProvider(BaseProvider):
    def fetch_target(self, symbol: str) -> dict:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            mean_target = info.get('targetMeanPrice')

            if mean_target:
                return {
                    "Symbol": symbol,
                    "Target_Mean": mean_target,
                    "Target_Median": info.get('targetMedianPrice'),
                    "Target_High": info.get('targetHighPrice'),
                    "Target_Low": info.get('targetLowPrice')
                }
            return None
        except Exception as e:
            logger.warning(f"yfinance failed to fetch {symbol}: {e}")
            return None

class AlphaVantageProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("AV_API_KEY")
        if not self.api_key:
            raise ValueError("AV_API_KEY environment variable is not set.")
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_target(self, symbol: str) -> dict:
        params = {"function": "EARNINGS", "symbol": symbol, "apikey": self.api_key}
        response = requests.get(self.base_url, params=params, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if "Information" in data and "rate limit" in data["Information"].lower():
                raise PermissionError(f"Alpha Vantage rate limit hit on {symbol} (Max 25/day or 5/min on free tier).")

            quarterly_earnings = data.get("quarterlyEarnings", [])
            if quarterly_earnings:
                latest = quarterly_earnings[0]
                return {
                    "Symbol": symbol,
                    "Estimated_EPS": latest.get("estimatedEPS"),
                    "Reported_EPS": latest.get("reportedEPS"),
                    "Surprise": latest.get("surprise"),
                    "Surprise_Pct": latest.get("surprisePercentage")
                }
            return None
        else:
            logger.warning(f"Alpha Vantage failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class FinnhubProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY environment variable is not set.")
        self.base_url = "https://finnhub.io/api/v1/stock/price-target"

    def fetch_target(self, symbol: str) -> dict:
        params = {"symbol": symbol, "token": self.api_key}
        response = requests.get(self.base_url, params=params, timeout=_REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            if data and "targetMean" in data:
                time.sleep(1.1)  # Respect the 60 calls/min limit
                return {
                    "Symbol": symbol,
                    "Target_Mean": data.get("targetMean"),
                    "Target_Median": data.get("targetMedian"),
                    "Target_High": data.get("targetHigh"),
                    "Target_Low": data.get("targetLow")
                }
            return None
        elif response.status_code == 429:
            raise PermissionError(f"Finnhub rate limit hit on {symbol}. Try slowing down the loop.")
        else:
            logger.warning(f"Finnhub failed to fetch {symbol}: HTTP {response.status_code}")
            return None

class FinvizProvider(BaseProvider):
    def fetch_target(self, symbol: str) -> dict:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = f"https://finviz.com/quote.ashx?t={symbol}"

        try:
            response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                target_td = soup.find('td', string='Target Price')

                if target_td:
                    target_price_str = target_td.find_next_sibling('td').text.strip()
                    if target_price_str != '-':
                        time.sleep(1)  # Be kind to their servers
                        return {
                            "Symbol": symbol,
                            "Target_Consensus": float(target_price_str)
                        }
            logger.info(f"No Target Price found on Finviz for {symbol}")
            return None
        except Exception as e:
            logger.warning(f"Finviz scraping failed for {symbol}: {e}")
            return None

def _extract_sources(response) -> list[dict]:
    """
    Pull {title, uri} citation pairs out of a grounded Gemini response.

    Defensive by necessity: grounding_metadata is an SDK response object,
    not a dict, and its shape isn't something this project can exercise
    in CI without a live API key, so every access is get-attr'd rather
    than assumed.
    """
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    metadata = getattr(candidates[0], "grounding_metadata", None)
    chunks = getattr(metadata, "grounding_chunks", None) or []
    sources = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is not None:
            sources.append({"title": getattr(web, "title", None), "uri": getattr(web, "uri", None)})
    return sources


class GeminiSearchProvider(BaseProvider):
    """
    Analyst price targets via Gemini's native Google Search grounding.

    The API/scrape providers above return one blended consensus number with
    no provenance. This asks Gemini to search for current analyst estimates
    and report each one individually with the date it was issued and a
    source citation — the piece those providers can't give us.

    Calls google-genai directly (the same SDK stockfit.llm._GeminiBackend
    already depends on) rather than going through the Agent Development Kit.
    A single grounded Q&A exchange doesn't need ADK's multi-turn
    session/runner machinery, and calling the SDK directly means this
    inherits the same retry/budget shape as the rest of the project's LLM
    calls instead of a thinner, ADK-flavored reimplementation of it.

    Requires GOOGLE_API_KEY (https://aistudio.google.com/app/apikey).
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    _INSTRUCTION = (
        "You are a financial research assistant. Use Google Search to find "
        "current analyst price targets for the given stock ticker. Reply "
        "with ONLY a JSON object, no other text, in this exact shape:\n"
        '{"target_mean": <float or null>, "target_high": <float or null>, '
        '"target_low": <float or null>, "num_analysts": <int or null>, '
        '"estimates": [{"firm": <string>, "target": <float>, '
        '"date": <string YYYY-MM-DD or null>}, ...]}\n'
        "Include a `date` for each individual estimate whenever the source "
        "states one. Use null rather than guessing at a value or date."
    )

    def __init__(
        self,
        model: str | None = None,
        daily_call_limit: int | None = None,
        call_pause: float | None = None,
    ):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        self.model = model or os.getenv("SHOCKARB_ANALYST_LLM_MODEL", self.DEFAULT_MODEL)
        self.daily_call_limit = daily_call_limit or int(
            os.getenv("SHOCKARB_ANALYST_LLM_CALL_LIMIT", "100")
        )
        self.call_pause = (
            call_pause if call_pause is not None
            else float(os.getenv("SHOCKARB_ANALYST_LLM_CALL_PAUSE", "2.0"))
        )
        self._budget = _DailyBudget()

    def fetch_target(self, symbol: str) -> dict:
        if not self._budget.can_call(self.daily_call_limit):
            raise PermissionError(
                f"Gemini analyst-target daily call limit reached ({self.daily_call_limit}/day)."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=self._INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        prompt = f"Analyst price targets for {symbol}, with dates."

        response = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=prompt, config=config)
                self._budget.calls_today += 1
                break
            except Exception as exc:
                if _is_permission_denied(exc):
                    raise PermissionError(
                        f"Gemini permission/billing error fetching {symbol}: {exc}"
                    ) from exc
                code = re.search(r"^(\d+)", str(exc))
                if not (code and code.group(1) in _RETRYABLE_CODES):
                    logger.warning(f"Gemini search-grounded call failed for {symbol}: {exc}")
                    return None
                if "PerDay" in str(exc) or "per_day" in str(exc).lower():
                    raise PermissionError(
                        f"Daily Gemini quota exhausted fetching {symbol}. "
                        "Set SHOCKARB_ANALYST_LLM_MODEL to an alternative or try tomorrow."
                    ) from exc
                delay = _parse_retry_delay(exc)
                logger.warning(
                    f"Gemini {code.group(1)} for {symbol} (attempt {attempt}/{_MAX_RETRIES}) — retrying in {delay:.0f}s"
                )
                time.sleep(delay)
        else:
            logger.warning(f"Gemini search-grounded call failed for {symbol} after {_MAX_RETRIES} attempts.")
            return None

        parsed = _extract_json(response.text)
        if parsed is None:
            logger.warning(f"Gemini returned no parseable JSON for {symbol}: {response.text[:200]!r}")
            return None

        if self.call_pause > 0:
            time.sleep(self.call_pause)

        return {
            "Symbol": symbol,
            "Target_Mean": parsed.get("target_mean"),
            "Target_High": parsed.get("target_high"),
            "Target_Low": parsed.get("target_low"),
            "Num_Analysts": parsed.get("num_analysts"),
            "Estimates_JSON": json.dumps(parsed.get("estimates", [])),
            "Sources_JSON": json.dumps(_extract_sources(response)),
        }

# ==========================================
# 2. MAIN EXECUTION LOGIC
# ==========================================

def get_provider(provider_name: str) -> BaseProvider:
    providers = {
        "fmp": FMPProvider,
        "yfinance": YFinanceProvider,
        "alpha_advantage": AlphaVantageProvider,
        "finnhub": FinnhubProvider,
        "finviz": FinvizProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider '{provider_name}'.")
    return providers[provider_name]()

def _load_tickers(args: argparse.Namespace) -> list[str]:
    """
    Return the ticker list for this run.

    Either from --tickers directly, or from column `args.column` of the
    CSV at `args.file`. Raises on a bad CSV path/column so main() can
    report the error and exit.
    """
    if args.tickers:
        return [t.strip().upper() for t in args.tickers]

    filepath = args.file if args.file.endswith('.csv') else f"{args.file}.csv"
    df = pd.read_csv(filepath)
    tickers = df.iloc[:, args.column].dropna().unique().tolist()
    logger.info(f"Loaded {len(tickers)} unique tickers from '{filepath}' (column {args.column})")
    return tickers



def _best_target(row: dict) -> float | None:
    """Return the best available target price from a provider result dict."""
    for key in ("Target_Consensus", "Target_Mean", "Target_Median"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _fetch_llm_estimates(tickers: list[str]) -> dict[str, dict | str]:
    """
    Cross-check `tickers` via GeminiSearchProvider.

    Returns {ticker: result_dict} on success, or {ticker: reason_str} when
    that ticker's lookup failed. A daily-quota PermissionError means every
    remaining ticker would fail the same way, so on that specific failure
    we stop calling and label the rest with the same reason rather than
    burning further attempts — but every requested ticker still gets an
    entry, so the caller can fall back to the main provider's value with
    a note instead of silently dropping rows.

    Example:
        _fetch_llm_estimates(["KLAC"])
        # -> {"KLAC": {"Symbol": "KLAC", "Target_Mean": 277.34, ...}}
    """
    try:
        provider = GeminiSearchProvider()
    except ValueError as e:
        return {t: str(e) for t in tickers}

    results: dict[str, dict | str] = {}
    remaining = list(tickers)
    while remaining:
        symbol = remaining.pop(0)
        try:
            data = provider.fetch_target(symbol)
            results[symbol] = data if data is not None else "no data returned"
        except PermissionError as e:
            logger.warning(f"LLM cross-check stopped early: {e}")
            for t in (symbol, *remaining):
                results[t] = str(e)
            break
        except Exception as e:
            results[symbol] = str(e)
    return results


def _format_llm_note(provider_name: str, main_value: float | None, llm_result: dict | str) -> str:
    """Plain-English, non-judgmental comparison of the main provider's target vs. the LLM's. Never picks a winner."""
    if isinstance(llm_result, str):
        tail = f"showing {provider_name} value only" if main_value is not None else "no target available"
        return f"LLM cross-check unavailable ({llm_result}) — {tail}."

    llm_mean = llm_result.get("Target_Mean")
    if llm_mean is None:
        return (
            f"{provider_name}: ${main_value:.2f}; Gemini search returned no target."
            if main_value is not None else
            "Neither source returned a target."
        )

    dates = [e.get("date") for e in json.loads(llm_result.get("Estimates_JSON") or "[]") if e.get("date")]
    n = llm_result.get("Num_Analysts")
    detail = ", ".join(filter(None, [f"n={n}" if n else "", f"latest {max(dates)}" if dates else ""]))
    llm_desc = f"${llm_mean:.2f}" + (f" ({detail})" if detail else "")

    if main_value is None:
        return f"No {provider_name} target; Gemini search: {llm_desc}."
    delta_pct = (llm_mean - main_value) / main_value * 100
    return f"{provider_name}: ${main_value:.2f}; Gemini search: {llm_desc} — differ by {delta_pct:+.1f}%."


def _combine_llm_results(
    out_df: pd.DataFrame,
    llm_tickers: list[str],
    llm_results: dict[str, dict | str],
    provider_name: str,
) -> pd.DataFrame:
    """
    Add LLM cross-check columns (+ a comparison Note) onto out_df.

    Only rows for tickers named in `llm_tickers` get anything written —
    every other row is untouched. A --llm ticker the main provider didn't
    return gets appended as its own row rather than silently dropped,
    since the user asked for it explicitly.
    """
    combined = out_df.copy()
    if "Symbol" not in combined.columns:
        combined["Symbol"] = pd.Series(dtype=str)

    missing = [t for t in llm_tickers if t not in combined["Symbol"].values]
    if missing:
        combined = pd.concat([combined, pd.DataFrame({"Symbol": missing})], ignore_index=True)

    llm_col_names = (
        "LLM_Target_Mean", "LLM_Target_High", "LLM_Target_Low",
        "LLM_Num_Analysts", "LLM_Estimates_JSON", "LLM_Sources_JSON",
    )
    columns: dict[str, list] = {name: [] for name in llm_col_names}
    columns["Note"] = []

    for _, row in combined.iterrows():
        symbol = row["Symbol"]
        if symbol not in llm_tickers:
            for name in llm_col_names:
                columns[name].append(None)
            columns["Note"].append(None)
            continue

        llm_result = llm_results.get(symbol, "not attempted")
        main_value = _best_target(row.to_dict())
        if pd.isna(main_value):
            main_value = None

        if isinstance(llm_result, dict):
            columns["LLM_Target_Mean"].append(llm_result.get("Target_Mean"))
            columns["LLM_Target_High"].append(llm_result.get("Target_High"))
            columns["LLM_Target_Low"].append(llm_result.get("Target_Low"))
            columns["LLM_Num_Analysts"].append(llm_result.get("Num_Analysts"))
            columns["LLM_Estimates_JSON"].append(llm_result.get("Estimates_JSON"))
            columns["LLM_Sources_JSON"].append(llm_result.get("Sources_JSON"))
        else:
            for name in llm_col_names:
                columns[name].append(None)
        columns["Note"].append(_format_llm_note(provider_name, main_value, llm_result))

    for name, values in columns.items():
        combined[name] = values
    return combined


def _update_fundamentals(results: list[dict], fundamentals_path: str) -> None:
    """Patch the Analyst Tgt column in fundamentals.csv with freshly fetched targets."""
    try:
        fund_df = pd.read_csv(fundamentals_path)
    except FileNotFoundError:
        print(f"[Error] fundamentals.csv not found at: {fundamentals_path}")
        return

    if "Ticker" not in fund_df.columns or "Analyst Tgt" not in fund_df.columns:
        print(f"[Error] Expected columns 'Ticker' and 'Analyst Tgt' in {fundamentals_path}")
        return

    updated = []
    skipped = []
    for row in results:
        ticker = row.get("Symbol", "")
        target = _best_target(row)
        if target is None:
            skipped.append(ticker)
            continue
        mask = fund_df["Ticker"].str.upper() == ticker.upper()
        if mask.any():
            fund_df.loc[mask, "Analyst Tgt"] = round(target, 2)
            updated.append(ticker)
        else:
            logger.warning(f"{ticker} not found in {fundamentals_path} — row not updated")

    fund_df.to_csv(fundamentals_path, index=False)
    print(f"\nUpdated {len(updated)} ticker(s) in {fundamentals_path}: {updated}")
    if skipped:
        print(f"Skipped (no target value): {skipped}")

def main():
    parser = argparse.ArgumentParser(description="Fetch consensus analyst data for a list of tickers.")

    ticker_source = parser.add_mutually_exclusive_group()
    ticker_source.add_argument(
        "--tickers", "-t", type=str, nargs="+",
        help="One or more ticker symbols to fetch directly, e.g. --tickers AAPL MSFT GOOGL.",
    )
    ticker_source.add_argument(
        "--file", "-f", type=str, default="fundamentals.csv",
        help="Path to input CSV (used if --tickers is not given).",
    )
    parser.add_argument("--column", "-c", type=int, default=0, help="Zero-indexed column containing tickers.")

    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["yfinance", "fmp", "alpha_advantage", "finnhub", "finviz"],
        default="finviz",
        help="Data provider to use. Defaults to 'finviz' — the only provider that needs no API key.",
    )
    parser.add_argument(
        "--llm",
        type=str, nargs="+", metavar="TICKER",
        default=None,
        help=(
            "Also cross-check these tickers via a Gemini search-grounded LLM lookup "
            "(GeminiSearchProvider), alongside --provider. Scoped only to the tickers "
            "named here — independent of --tickers/--file — to bound API cost. "
            "Requires GOOGLE_API_KEY. Never overrides --provider's value; adds "
            "LLM_* columns and a comparison Note instead."
        ),
    )
    parser.add_argument(
        "--update-fundamentals", "-u",
        nargs="?",
        const="data/fundamentals.csv",
        metavar="PATH",
        help=(
            "Patch Analyst Tgt in fundamentals.csv with fetched targets. "
            "Optionally supply a path (default: data/fundamentals.csv)."
        ),
    )

    args = parser.parse_args()
    llm_tickers = [t.strip().upper() for t in args.llm] if args.llm else []

    try:
        tickers = _load_tickers(args)
    except Exception as e:
        print(f"[Error] Failed to load tickers: {e}")
        sys.exit(1)

    try:
        provider = get_provider(args.provider)
        logger.info(f"Initialized provider: {args.provider.upper()}")
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    results = []
    logger.info(f"Fetching data for {len(tickers)} ticker(s)...")

    for symbol in tickers:
        try:
            data = provider.fetch_target(symbol)
            if data:
                results.append(data)
            else:
                logger.info(f"No data found for {symbol}")
        except PermissionError as e:
            print(f"\n[CRITICAL ERROR] {e}")
            print("Stopping execution to prevent further API blocks.")
            break
        except Exception as e:
            logger.warning(f"Unexpected error on {symbol}: {e}")

    if not results and not llm_tickers:
        print("\nNo analyst data was retrieved.")
        return

    out_df = pd.DataFrame(results) if results else pd.DataFrame({"Symbol": []})

    if llm_tickers:
        logger.info(f"Cross-checking {len(llm_tickers)} ticker(s) via Gemini search grounding...")
        llm_results = _fetch_llm_estimates(llm_tickers)
        out_df = _combine_llm_results(out_df, llm_tickers, llm_results, args.provider)

    print(out_df.to_string(index=False))
    output_file = f"{args.provider}{'_vs_llm' if llm_tickers else ''}_analyst_data.csv"
    out_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved data for {len(out_df)} ticker row(s) to '{output_file}'.")

    if args.update_fundamentals:
        _update_fundamentals(results, args.update_fundamentals)

if __name__ == "__main__":
    main()
