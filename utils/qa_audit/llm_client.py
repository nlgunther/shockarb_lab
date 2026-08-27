"""
qa_audit.llm_client — minimal, generic LLM client for the QA validation layer.

Deliberately NOT a third copy of stockfit/llm.py and marketfit/llm.py's
_AnthropicBackend/_GeminiBackend/_DailyBudget machinery. Those two files are
already near-identical duplicates of each other (see their own comments —
"reuse same pattern as marketfit.llm"); adding a third copy-paste here would
compound a known smell rather than fix it. This client is intentionally
smaller: no daily call budget (a human runs this audit on demand, not on an
unattended schedule processing dozens of tickers a minute), no multi-attempt
retry loop (one clear failure beats a silent multi-minute hang for a tool a
person is sitting in front of) — just "send a system+user prompt, get text
back, fail loudly and specifically if it doesn't work."

If stockfit/marketfit's LLM clients are ever consolidated into one shared
module, this one is the natural third caller to fold in at that point.

Environment
-----------
    ANTHROPIC_API_KEY   preferred, matches stockfit/marketfit convention
    GOOGLE_API_KEY       fallback if ANTHROPIC_API_KEY is unset
    SHOCKARB_LLM_MODEL   overrides the default model for whichever backend is used
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger


class LLMUnavailableError(RuntimeError):
    """Raised by from_env() when neither ANTHROPIC_API_KEY nor GOOGLE_API_KEY is set."""


@dataclass
class LLMClient:
    """
    Thin wrapper around one LLM backend, chosen once at construction.

    Parameters
    ----------
    backend : str
        "anthropic" or "gemini".
    api_key : str
        Key for the chosen backend.
    model : str
        Model name for the chosen backend.
    call_fn : callable, optional
        Injected (system, user) -> str override, entirely bypassing the
        real API. Tests set this; production code leaves it None and gets
        the real anthropic/google-genai call. This is the seam that makes
        llm_validator.py testable without ever touching a real API key.

    Example
    -------
        client = LLMClient.from_env()
        text = client.complete("You are a skeptical analyst.", "Is AAPL cheap?")

        # In tests:
        fake = LLMClient(backend="anthropic", api_key="x", model="x",
                          call_fn=lambda system, user: '{"verdict": "UNCERTAIN"}')
    """
    backend: str
    api_key: str
    model:   str
    call_fn: Optional[Callable[[str, str], str]] = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        """
        Pick a backend from environment variables, Anthropic preferred.

        Raises
        ------
        LLMUnavailableError
            If neither ANTHROPIC_API_KEY nor GOOGLE_API_KEY is set.
        """
        model_override = os.environ.get("SHOCKARB_LLM_MODEL")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            return cls(
                backend="anthropic", api_key=anthropic_key,
                model=model_override or "claude-haiku-4-5-20251001",
            )

        google_key = os.environ.get("GOOGLE_API_KEY")
        if google_key:
            return cls(
                backend="gemini", api_key=google_key,
                model=model_override or "gemini-2.5-flash",
            )

        raise LLMUnavailableError(
            "Neither ANTHROPIC_API_KEY nor GOOGLE_API_KEY is set — "
            "qa_audit's LLM validation layer has nothing to call. "
            "The stats_checks layer still runs without an LLM key."
        )

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """
        Send a system+user prompt, return the raw text response.

        Raises whatever the underlying SDK raises (ImportError if the
        package isn't installed, the SDK's own exception on an API error) —
        callers decide how to handle a failed validation call; this method
        does not swallow errors the way stockfit/marketfit's
        generate_narratives() does, because a validation tool silently
        returning nothing is worse than a validation tool visibly failing.
        """
        if self.call_fn is not None:
            return self.call_fn(system, user)

        if self.backend == "anthropic":
            return self._call_anthropic(system, user, max_tokens)
        if self.backend == "gemini":
            return self._call_gemini(system, user, max_tokens)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def _call_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package required: pip install anthropic")
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model      = self.model,
            max_tokens = max_tokens,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _call_gemini(self, system: str, user: str, max_tokens: int) -> str:
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("google-genai package required: pip install google-genai")
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model, contents=system + "\n\n" + user,
        )
        return response.text
