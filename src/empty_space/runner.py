"""Session runner — orchestrates the Phase 2 turn loop.

run_session(config, llm_client) -> SessionResult

For each turn:
  1. Determine speaker (odd=protagonist / even=counterpart).
  2. Trigger director event if scheduled.
  3. Build system + user prompts.
  4. Call LLM.
  5. Parse response.
  6. Append Turn to in-memory state.
  7. Write turn_NNN.yaml + conversation append.

Termination (Phase 2): max_turns only.
Errors: LLM exceptions propagate; partial turn files are kept; meta.yaml is not written.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from empty_space.loaders import load_persona, load_setting
from empty_space.llm import GeminiResponse
from empty_space.parser import parse_response
from empty_space.paths import RUNS_DIR
from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.schemas import (
    ExperimentConfig,
    Persona,
    SessionResult,
    Setting,
    Turn,
)
from empty_space.writer import append_turn, init_run, write_meta


class LLMClient(Protocol):
    """Duck-typed interface that both GeminiClient and MockLLMClient satisfy."""
    def generate(self, *, system: str, user: str, model: str = ...) -> GeminiResponse: ...


@dataclass
class SessionState:
    """Runner-internal state. Not persisted to schemas.py."""
    config: ExperimentConfig
    protagonist: Persona
    counterpart: Persona
    setting: Setting
    turns: list[Turn] = field(default_factory=list)
    active_events: list[tuple[int, str]] = field(default_factory=list)


def run_session(
    *, config: ExperimentConfig, llm_client: LLMClient
) -> SessionResult:
    """Run one experiment session end-to-end."""
    protagonist = load_persona(config.protagonist.path, version=config.protagonist.version)
    counterpart = load_persona(config.counterpart.path, version=config.counterpart.version)
    setting = load_setting(config.setting.path)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RUNS_DIR / config.exp_id / timestamp
    init_run(out_dir, config)

    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
    )

    start_time = time.monotonic()
    events_triggered: list[tuple[int, str]] = []

    for n in range(1, config.max_turns + 1):
        # 1. speaker
        speaker_role = "protagonist" if n % 2 == 1 else "counterpart"
        speaker_persona = protagonist if speaker_role == "protagonist" else counterpart
        other_party_name = counterpart.name if speaker_role == "protagonist" else protagonist.name

        # 2. director event trigger
        if n in config.director_events:
            new_event = (n, config.director_events[n])
            state.active_events.append(new_event)
            events_triggered.append(new_event)

        # 3. prompts
        system_prompt = build_system_prompt(
            persona=speaker_persona,
            counterpart_name=other_party_name,
            setting=setting,
            scene_premise=config.scene_premise,
            initial_state=config.initial_state,
            active_events=state.active_events,
        )
        user_message = build_user_message(history=state.turns)

        # 4. LLM call
        resp = llm_client.generate(
            system=system_prompt,
            user=user_message,
            model="gemini-2.5-flash",
        )

        # 5. parse
        main_content, impressions, parse_err = parse_response(resp.content)

        # 6. turn record
        turn = Turn(
            turn_number=n,
            speaker=speaker_role,  # type: ignore[arg-type]
            persona_name=speaker_persona.name,
            content=main_content,
            candidate_impressions=impressions,
            prompt_system=system_prompt,
            prompt_user=user_message,
            raw_response=resp.content,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            model=resp.model,
            latency_ms=resp.latency_ms,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            director_events_active=list(state.active_events),
            parse_error=parse_err,
        )
        state.turns.append(turn)

        # 7. append
        append_turn(out_dir, turn)

    duration = time.monotonic() - start_time
    termination_reason = "max_turns"

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    total_tokens_out = sum(t.tokens_out for t in state.turns)
    total_candidate_impressions = sum(len(t.candidate_impressions) for t in state.turns)
    turns_with_parse_error = sum(1 for t in state.turns if t.parse_error is not None)
    models_used = sorted({t.model for t in state.turns})

    write_meta(
        out_dir=out_dir,
        config=config,
        total_turns=len(state.turns),
        termination_reason=termination_reason,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_candidate_impressions=total_candidate_impressions,
        turns_with_parse_error=turns_with_parse_error,
        director_events_triggered=events_triggered,
        models_used=models_used,
        duration_seconds=duration,
    )

    return SessionResult(
        exp_id=config.exp_id,
        out_dir=out_dir,
        total_turns=len(state.turns),
        termination_reason=termination_reason,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        duration_seconds=duration,
    )
