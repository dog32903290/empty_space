# 空的空間 (The Empty Space)

雙角色觀察實驗台 — 兩個演員 + 一個看著的人 = 最小戲劇單位。

- Spec: `docs/superpowers/specs/2026-04-21-空的空間-design.md`
- Plans: `docs/superpowers/plans/`

## Setup

    uv sync --all-extras
    cp .env.example .env    # then fill in API keys

## Test

    uv run pytest

## Smoke test (needs real API keys)

    uv run python scripts/hello.py
