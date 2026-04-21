"""Smoke test: verify Gemini 2.5 Flash and Gemini 2.5 Pro work end-to-end.

Usage:
    uv run python scripts/hello.py
"""
from empty_space.llm import GeminiClient


def main():
    client = GeminiClient()

    print("Testing Gemini 2.5 Flash (roles / Judge / retrieval / rubric)...")
    f_resp = client.generate(
        system="Reply in one short sentence.",
        user="What is the smallest unit of theatre according to Peter Brook?",
        model="gemini-2.5-flash",
    )
    print(f"  Response: {f_resp.content}")
    print(f"  Tokens out: {f_resp.tokens_out}, latency: {f_resp.latency_ms}ms")

    print("\nTesting Gemini 2.5 Pro (Composer)...")
    p_resp = client.generate(
        system="Reply in one short sentence.",
        user="What is the smallest unit of theatre according to Peter Brook?",
        model="gemini-2.5-pro",
    )
    print(f"  Response: {p_resp.content}")
    print(f"  Tokens out: {p_resp.tokens_out}, latency: {p_resp.latency_ms}ms")

    print("\nOK — Gemini Flash + Pro both work.")


if __name__ == "__main__":
    main()
