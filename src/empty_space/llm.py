"""Thin LLM client wrapper over the google-genai SDK.

GeminiClient supports both Flash (default — roles / Judge / retrieval /
rubric) and Pro (Composer) via the `model` parameter.

Normalized response (GeminiResponse) exposes:
    - content: main text
    - raw: original SDK response (for debugging)
    - tokens_in: int (prompt token count)
    - tokens_out: int (candidate/response token count)
    - model: str
    - latency_ms: int
"""
import os
import time
from dataclasses import dataclass
from typing import Any

from google import genai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeminiResponse:
    content: str
    raw: Any
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: int


class GeminiClient:
    """Wrapper over google-genai Client."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ["GEMINI_API_KEY"]
        self._client = genai.Client(api_key=key)

    def generate(
        self,
        system: str,
        user: str,
        model: str = "gemini-2.5-flash",
    ) -> GeminiResponse:
        """Call Gemini with system_instruction + user content."""
        start = time.monotonic()
        response = self._client.models.generate_content(
            model=model,
            contents=user,
            config=genai.types.GenerateContentConfig(system_instruction=system),
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        return GeminiResponse(
            content=response.text,
            raw=response,
            tokens_in=response.usage_metadata.prompt_token_count,
            tokens_out=response.usage_metadata.candidates_token_count,
            model=model,
            latency_ms=latency_ms,
        )
