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

from empty_space import ledger
from empty_space.composer import run_composer
from empty_space.loaders import load_persona, load_setting
from empty_space.llm import GeminiResponse
from empty_space.parser import parse_response
from empty_space.paths import RUNS_DIR
from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.retrieval import load_synonym_map, run_session_start_retrieval
from empty_space.judge import (
    STAGE_ORDER,
    apply_stage_target,
    is_basin_lock,
    is_fire_release,
    run_judge,
)
from empty_space.schemas import (
    ComposerSessionResult,
    ExperimentConfig,
    JudgeState,
    Persona,
    RetrievalResult,
    SessionResult,
    Setting,
    Turn,
)
from empty_space.writer import append_turn, init_run, write_meta, write_retrieval


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
    retrieval_protagonist: RetrievalResult | None = None
    retrieval_counterpart: RetrievalResult | None = None
    # Level 4:
    judge_state_protagonist: JudgeState | None = None
    judge_state_counterpart: JudgeState | None = None
    director_injections: list[dict] = field(default_factory=list)


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

    # Level 2: Session-start retrieval (once per role)
    synonym_map = load_synonym_map()
    relationship = f"{protagonist.name}_x_{counterpart.name}"

    retrieval_protagonist = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name=protagonist.name,
        query_text=_compose_query(config.scene_premise, config.protagonist_prelude),
        relationship=relationship,
        other_persona_name=counterpart.name,
        synonym_map=synonym_map,
        llm_client=llm_client,
        top_n=3,
    )
    retrieval_counterpart = run_session_start_retrieval(
        speaker_role="counterpart",
        persona_name=counterpart.name,
        query_text=_compose_query(config.scene_premise, config.counterpart_prelude),
        relationship=relationship,
        other_persona_name=protagonist.name,
        synonym_map=synonym_map,
        llm_client=llm_client,
        top_n=3,
    )
    write_retrieval(
        out_dir,
        protagonist=retrieval_protagonist,
        counterpart=retrieval_counterpart,
    )

    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
        retrieval_protagonist=retrieval_protagonist,
        retrieval_counterpart=retrieval_counterpart,
    )

    state.judge_state_protagonist = _init_judge_state("protagonist", config.initial_state)
    state.judge_state_counterpart = _init_judge_state("counterpart", config.initial_state)

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

        # Select this role's retrieval + prelude
        role_retrieval = (
            state.retrieval_protagonist if speaker_role == "protagonist"
            else state.retrieval_counterpart
        )
        role_prelude = (
            config.protagonist_prelude if speaker_role == "protagonist"
            else config.counterpart_prelude
        )

        # 3. prompts
        active_judge_state = (
            state.judge_state_protagonist if speaker_role == "protagonist"
            else state.judge_state_counterpart
        )
        active_persona_contexts = (
            speaker_persona.stage_mode_contexts_parsed
            if speaker_persona.stage_mode_contexts_parsed
            else None
        )
        system_prompt = build_system_prompt(
            persona=speaker_persona,
            counterpart_name=other_party_name,
            setting=setting,
            scene_premise=config.scene_premise,
            initial_state=config.initial_state,
            active_events=state.active_events,
            prelude=role_prelude,
            retrieved_impressions=role_retrieval.impressions,
            judge_state=active_judge_state,
            stage_mode_contexts=active_persona_contexts,
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
            retrieved_impressions=list(role_retrieval.impressions),
        )
        state.turns.append(turn)

        # Level 4: run Judge for both speakers after this turn
        judge_out_p, judge_out_c = _run_judges_post_turn(state, llm_client)

        # 7. append
        append_turn(
            out_dir, turn,
            judge_output_protagonist=judge_out_p,
            judge_output_counterpart=judge_out_c,
        )

    duration = time.monotonic() - start_time
    termination_reason = "max_turns"

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    total_tokens_out = sum(t.tokens_out for t in state.turns)
    total_candidate_impressions = sum(len(t.candidate_impressions) for t in state.turns)
    turns_with_parse_error = sum(1 for t in state.turns if t.parse_error is not None)
    models_used = sorted({t.model for t in state.turns})

    # Level 2: Session-end ledger append
    source_run = f"{config.exp_id}/{timestamp}"
    ledger_appends, new_raw_ids = _append_session_ledgers(
        relationship=relationship,
        protagonist_persona=protagonist,
        counterpart_persona=counterpart,
        turns=state.turns,
        source_run=source_run,
    )

    # Level 3: Composer Pro bake at session end
    composer_result = _run_composer_at_session_end(
        relationship=relationship,
        protagonist=protagonist,
        counterpart=counterpart,
        out_dir=out_dir,
        turns=state.turns,
        new_raw_ids=new_raw_ids,
        source_run=source_run,
        llm_client=llm_client,
    )

    # Update models_used to include Pro if Composer ran
    if composer_result.tokens_in > 0:
        models_used = sorted(set(models_used) | {"gemini-2.5-pro"})

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
        retrieval_total_tokens_in=(
            retrieval_protagonist.flash_tokens_in + retrieval_counterpart.flash_tokens_in
        ),
        retrieval_total_tokens_out=(
            retrieval_protagonist.flash_tokens_out + retrieval_counterpart.flash_tokens_out
        ),
        ledger_appends=ledger_appends,
        # Level 3 new:
        composer_tokens_in=composer_result.tokens_in,
        composer_tokens_out=composer_result.tokens_out,
        composer_latency_ms=composer_result.latency_ms,
        protagonist_refined_added=composer_result.protagonist_refined_added,
        counterpart_refined_added=composer_result.counterpart_refined_added,
        composer_parse_error=composer_result.parse_error,
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


def _compose_query(scene_premise: str | None, prelude: str | None) -> str:
    """Join scene_premise + prelude for retrieval query (both optional)."""
    parts = [scene_premise, prelude]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _run_composer_at_session_end(
    *,
    relationship: str,
    protagonist: Persona,
    counterpart: Persona,
    out_dir: Path,
    turns: list[Turn],
    new_raw_ids: dict[str, list[str]],
    source_run: str,
    llm_client,
) -> ComposerSessionResult:
    """Wrap run_composer — exception-safe. On any error, returns result
    with parse_error set, refined ledgers untouched.

    run_composer itself already catches exceptions; this wrapper is
    double protection for anything that somehow escapes.
    """
    try:
        return run_composer(
            relationship=relationship,
            protagonist_name=protagonist.name,
            counterpart_name=counterpart.name,
            out_dir=out_dir,
            session_turns=turns,
            new_raw_ids=new_raw_ids,
            source_run=source_run,
            llm_client=llm_client,
        )
    except Exception as e:
        return ComposerSessionResult(
            tokens_in=0, tokens_out=0, latency_ms=0,
            protagonist_refined_added=0, counterpart_refined_added=0,
            parse_error=f"composer exception: {type(e).__name__}: {e}",
        )


def _init_judge_state(
    speaker_role: str, initial_state
) -> JudgeState:
    """Seed JudgeState from experiment's initial_state (InitialState).

    initial_state.mode is already v3-migrated by the pydantic validator
    (legacy '基線' → '在').
    """
    return JudgeState(
        speaker_role=speaker_role,  # type: ignore[arg-type]
        stage=initial_state.stage,
        mode=initial_state.mode,
        last_why="",
        last_verdict="",
        move_history=[],
        verdict_history=[],
        hits_history=[],
    )


def _should_run_judge(persona: Persona) -> bool:
    """Judge only runs for personas with both v3 files present."""
    return bool(persona.judge_principles_text) and bool(persona.stage_mode_contexts_parsed)


def _stage_mode_contexts_text(persona: Persona) -> str:
    """Render persona.stage_mode_contexts_parsed as readable text for Judge prompt."""
    if not persona.stage_mode_contexts_parsed:
        return ""
    lines: list[str] = []
    for key, cell in persona.stage_mode_contexts_parsed.items():
        lines.append(f"{key}:")
        for field_name in ("身體傾向", "語聲傾向", "注意力"):
            if cell.get(field_name):
                lines.append(f"  {field_name}: {cell[field_name]}")
    return "\n".join(lines)


def _format_recent_turns(turns: list[Turn], n: int = 3) -> str:
    """Render last n turns as readable text for Judge prompt."""
    if not turns:
        return "（尚無對話）"
    tail = turns[-n:]
    return "\n".join(
        f"[Turn {t.turn_number} {t.persona_name}] {t.content}"
        for t in tail
    )


def _run_judges_post_turn(
    state: "SessionState",
    llm_client,
) -> tuple[dict, dict]:
    """Run Judge for both speakers (if eligible). Returns (p_output_dict, c_output_dict)
    suitable for writing to turn yaml's judge_output_protagonist / _counterpart.

    Updates state.judge_state_protagonist and state.judge_state_counterpart
    in place via apply_stage_target.
    """
    recent = _format_recent_turns(state.turns, n=3)
    outputs: dict[str, dict] = {}

    for role, persona, attr in (
        ("protagonist", state.protagonist, "judge_state_protagonist"),
        ("counterpart", state.counterpart, "judge_state_counterpart"),
    ):
        last = getattr(state, attr)
        if not _should_run_judge(persona):
            outputs[role] = {"skipped": True, "reason": "no_v3_config"}
            continue
        jr = run_judge(
            last_state=last,
            principles_text=persona.judge_principles_text,
            stage_mode_contexts_text=_stage_mode_contexts_text(persona),
            recent_turns_text=recent,
            speaker_role=role,
            persona_name=persona.name,
            llm_client=llm_client,
        )
        new_state, move = apply_stage_target(
            last_state=last,
            proposed_stage=jr.proposed_stage,
            proposed_mode=jr.proposed_mode,
            proposed_verdict=jr.proposed_verdict,
            why=jr.why,
            hits=jr.hits,
        )
        setattr(state, attr, new_state)
        outputs[role] = {
            "proposed": {
                "stage": jr.proposed_stage,
                "mode": jr.proposed_mode,
                "verdict": jr.proposed_verdict,
                "why": jr.why,
            },
            "applied": {
                "stage": new_state.stage,
                "mode": new_state.mode,
                "move": move,
            },
            "hits": list(jr.hits),
            "meta": dict(jr.meta),
        }
    return outputs["protagonist"], outputs["counterpart"]


def _append_session_ledgers(
    *,
    relationship: str,
    protagonist_persona: Persona,
    counterpart_persona: Persona,
    turns: list[Turn],
    source_run: str,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Append session candidates to raw ledgers.

    Returns (meta_appends, new_raw_ids) where new_raw_ids is
    {"protagonist": ["imp_045", ...], "counterpart": ["imp_012", ...]}
    for Composer provenance tracking.
    Empty buckets skipped (no ledger file created for that side).
    """
    p_candidates = [
        (t.turn_number, imp) for t in turns
        if t.speaker == "protagonist"
        for imp in t.candidate_impressions
    ]
    c_candidates = [
        (t.turn_number, imp) for t in turns
        if t.speaker == "counterpart"
        for imp in t.candidate_impressions
    ]

    appends: list[dict] = []
    new_ids: dict[str, list[str]] = {"protagonist": [], "counterpart": []}

    if p_candidates:
        p_new_ids = ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="protagonist",
            persona_name=protagonist_persona.name,
            candidates=p_candidates,
            source_run=source_run,
        )
        new_ids["protagonist"] = p_new_ids
        new_ledger = ledger.read_ledger(
            relationship=relationship, persona_name=protagonist_persona.name,
        )
        appends.append({
            "relationship": relationship,
            "speaker": "protagonist",
            "persona_name": protagonist_persona.name,
            "candidates_added": len(p_candidates),
            "new_ledger_version": new_ledger.ledger_version,
        })

    if c_candidates:
        c_new_ids = ledger.append_session_candidates(
            relationship=relationship,
            speaker_role="counterpart",
            persona_name=counterpart_persona.name,
            candidates=c_candidates,
            source_run=source_run,
        )
        new_ids["counterpart"] = c_new_ids
        new_ledger = ledger.read_ledger(
            relationship=relationship, persona_name=counterpart_persona.name,
        )
        appends.append({
            "relationship": relationship,
            "speaker": "counterpart",
            "persona_name": counterpart_persona.name,
            "candidates_added": len(c_candidates),
            "new_ledger_version": new_ledger.ledger_version,
        })

    return appends, new_ids
