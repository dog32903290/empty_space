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
from empty_space.ledger import read_refined_ledger
from empty_space.loaders import load_persona, load_setting
from empty_space.llm import GeminiResponse
from empty_space.parser import parse_response
from empty_space.paths import RUNS_DIR
from empty_space.prompt_assembler import build_system_prompt, build_user_message
from empty_space.retrieval import load_synonym_map, run_session_start_retrieval
from empty_space.judge import (
    STAGE_ORDER,
    apply_stage_target,
    infer_initial_state,
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
    # Bookkeeping for meta.yaml aggregation:
    judge_history_protagonist: dict = field(
        default_factory=lambda: {"stages": [], "modes": [], "verbs": []}
    )
    judge_history_counterpart: dict = field(
        default_factory=lambda: {"stages": [], "modes": [], "verbs": []}
    )
    judge_health_events: dict = field(
        default_factory=lambda: {"protagonist": [], "counterpart": []}
    )


def run_session(
    *, config: ExperimentConfig, llm_client: LLMClient, interactive: bool = False
) -> SessionResult:
    """Run one experiment session end-to-end."""
    protagonist = load_persona(config.protagonist.path, version=config.protagonist.version)
    counterpart = load_persona(config.counterpart.path, version=config.counterpart.version)
    setting = load_setting(config.setting.path)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RUNS_DIR / config.exp_id / timestamp
    init_run(out_dir, config)

    synonym_map = load_synonym_map()
    relationship = f"{protagonist.name}_x_{counterpart.name}"

    # Create SessionState early so _init_judge_state can mutate it
    state = SessionState(
        config=config,
        protagonist=protagonist,
        counterpart=counterpart,
        setting=setting,
    )

    # Level 4: infer initial Judge states BEFORE retrieval so retrieval has current state
    state.judge_state_protagonist = _init_judge_state(
        "protagonist", protagonist, counterpart.name, config, llm_client,
    )
    state.judge_state_counterpart = _init_judge_state(
        "counterpart", counterpart, protagonist.name, config, llm_client,
    )

    # Level 2: Session-start retrieval — NOW with inferred state as current_state
    retrieval_protagonist = run_session_start_retrieval(
        speaker_role="protagonist",
        persona_name=protagonist.name,
        query_text=_compose_query(config.scene_premise, config.protagonist_prelude),
        relationship=relationship,
        other_persona_name=counterpart.name,
        synonym_map=synonym_map,
        llm_client=llm_client,
        current_state=_judge_state_as_dict(state.judge_state_protagonist),
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
        current_state=_judge_state_as_dict(state.judge_state_counterpart),
        top_n=3,
    )
    write_retrieval(
        out_dir,
        protagonist=retrieval_protagonist,
        counterpart=retrieval_counterpart,
    )

    state.retrieval_protagonist = retrieval_protagonist
    state.retrieval_counterpart = retrieval_counterpart

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

        # Level 4: interactive peak hook (may inject director event for next turn)
        director_injection = None
        if interactive:
            peak, triggered_by = _is_peak(state)
            if peak:
                event_text = _prompt_for_director_event(
                    turn_number=n,
                    state_p=state.judge_state_protagonist,
                    state_c=state.judge_state_counterpart,
                    triggered_by=triggered_by,
                )
                if event_text:
                    next_turn = n + 1
                    state.config.director_events[next_turn] = event_text
                    director_injection = {
                        "event": event_text,
                        "triggered_by": triggered_by,
                        "applied_to_turn": next_turn,
                    }
                    state.director_injections.append({
                        "turn": n, "event": event_text, "triggered_by": triggered_by,
                    })

        # 7. append
        append_turn(
            out_dir, turn,
            judge_output_protagonist=judge_out_p,
            judge_output_counterpart=judge_out_c,
            director_injection=director_injection,
        )

        # Level 4: termination check
        should_stop, reason = _should_terminate(state)
        if should_stop:
            termination_reason = reason
            break
    else:
        termination_reason = "max_turns"

    duration = time.monotonic() - start_time

    total_tokens_in = sum(t.tokens_in for t in state.turns)
    total_tokens_out = sum(t.tokens_out for t in state.turns)
    total_candidate_impressions = sum(len(t.candidate_impressions) for t in state.turns)
    turns_with_parse_error = sum(1 for t in state.turns if t.parse_error is not None)
    models_used = sorted({t.model for t in state.turns})

    # Level 2: Session-end ledger append
    source_run = f"{config.exp_id}/{timestamp}"
    ledger_appends, new_raw_ids, state_maps, candidate_states = _append_session_ledgers(
        relationship=relationship,
        protagonist_persona=protagonist,
        counterpart_persona=counterpart,
        turns=state.turns,
        source_run=source_run,
        state=state,
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
        state_maps=state_maps,
        candidate_states=candidate_states,
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
        # Level 4:
        judge_trajectories=_build_judge_trajectories(state),
        judge_health=_build_judge_health(state),
        termination_turn=len(state.turns),
        director_injections=list(state.director_injections),
        interactive_mode=interactive,
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
    state_maps: dict[str, dict[str, dict]] | None = None,
    candidate_states: dict[str, list[dict]] | None = None,
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
            state_maps=state_maps,
            candidate_states=candidate_states,
        )
    except Exception as e:
        return ComposerSessionResult(
            tokens_in=0, tokens_out=0, latency_ms=0,
            protagonist_refined_added=0, counterpart_refined_added=0,
            parse_error=f"composer exception: {type(e).__name__}: {e}",
        )


def _init_judge_state(
    speaker_role: str,
    persona: Persona,
    other_persona_name: str,
    config: ExperimentConfig,
    llm_client,
) -> JudgeState:
    """Seed JudgeState for one speaker.

    If persona has v3 config (judge_principles + stage_mode_contexts), runs
    a turn-0 Judge inference over scene_premise + prelude + persona to place
    the character into stage/mode/verb.

    If v3 config is missing OR inference fails, falls back to
    config.initial_state (legacy path).
    """
    initial_state = config.initial_state
    fallback = JudgeState(
        speaker_role=speaker_role,  # type: ignore[arg-type]
        stage=initial_state.stage,
        mode=initial_state.mode,
        current_verb=initial_state.verb,
    )

    if not _should_run_judge(persona):
        return fallback

    prelude = (
        config.protagonist_prelude if speaker_role == "protagonist"
        else config.counterpart_prelude
    )
    relationship_text = persona.relationship_texts.get(other_persona_name, "")
    return infer_initial_state(
        speaker_role=speaker_role,
        persona_name=persona.name,
        persona_core_text=persona.core_text,
        persona_relationship_text=relationship_text,
        principles_text=persona.judge_principles_text,
        scene_premise=config.scene_premise or "",
        prelude=prelude or "",
        other_persona_name=other_persona_name,
        fallback_stage=initial_state.stage,
        fallback_mode=initial_state.mode,
        fallback_verb=initial_state.verb,
        llm_client=llm_client,
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


def _build_refined_excerpt(persona: Persona, relationship: str) -> str:
    """Return top-3 most recent refined impressions as formatted text.

    Reads speaker's refined ledger; impressions are chronologically appended,
    so we take the last 3. Returns "" if no refined ledger exists yet.
    """
    try:
        ledger_obj = read_refined_ledger(
            relationship=relationship,
            persona_name=persona.name,
        )
    except Exception:
        return ""
    if not ledger_obj.impressions:
        return ""
    recent = ledger_obj.impressions[-3:]
    lines = []
    for imp in reversed(recent):  # most recent first
        symbols_str = ", ".join(imp.symbols) if imp.symbols else ""
        lines.append(f"- {imp.id}: {imp.text} [symbols: {symbols_str}]")
    return "\n".join(lines)


def _run_judges_post_turn(
    state: "SessionState",
    llm_client,
) -> tuple[dict, dict]:
    """Run Judge for both speakers (if eligible). Returns (p_output_dict, c_output_dict).

    Updates state.judge_state_protagonist and state.judge_state_counterpart via
    apply_stage_target. Also appends to bookkeeping fields:
      state.judge_history_{protagonist,counterpart}.{stages,modes}
      state.judge_health_events.{protagonist,counterpart}  (parse_status tag per call)
    """
    recent = _format_recent_turns(state.turns, n=3)
    relationship = f"{state.protagonist.name}_x_{state.counterpart.name}"
    outputs: dict[str, dict] = {}

    for role, persona, attr, hist_attr in (
        ("protagonist", state.protagonist, "judge_state_protagonist", "judge_history_protagonist"),
        ("counterpart", state.counterpart, "judge_state_counterpart", "judge_history_counterpart"),
    ):
        last = getattr(state, attr)
        if not _should_run_judge(persona):
            outputs[role] = {"skipped": True, "reason": "no_v3_config"}
            # Track stage/mode/verb unchanged for trajectory length consistency
            getattr(state, hist_attr)["stages"].append(last.stage)
            getattr(state, hist_attr)["modes"].append(last.mode)
            getattr(state, hist_attr)["verbs"].append(last.current_verb)
            state.judge_health_events[role].append({"parse_status": "no_judge"})
            continue
        refined_excerpt = _build_refined_excerpt(persona, relationship)
        jr = run_judge(
            last_state=last,
            principles_text=persona.judge_principles_text,
            stage_mode_contexts_text=_stage_mode_contexts_text(persona),
            recent_turns_text=recent,
            speaker_role=role,
            persona_name=persona.name,
            llm_client=llm_client,
            refined_excerpt=refined_excerpt,
        )
        new_state, move = apply_stage_target(
            last_state=last,
            proposed_stage=jr.proposed_stage,
            proposed_mode=jr.proposed_mode,
            proposed_verdict=jr.proposed_verdict,
            proposed_verb=jr.proposed_verb,
            why=jr.why,
            hits=jr.hits,
        )
        setattr(state, attr, new_state)
        getattr(state, hist_attr)["stages"].append(new_state.stage)
        getattr(state, hist_attr)["modes"].append(new_state.mode)
        getattr(state, hist_attr)["verbs"].append(new_state.current_verb)
        state.judge_health_events[role].append({"parse_status": jr.meta.get("parse_status", "ok")})
        outputs[role] = {
            "proposed": {
                "stage": jr.proposed_stage,
                "mode": jr.proposed_mode,
                "verb": jr.proposed_verb,
                "verdict": jr.proposed_verdict,
                "why": jr.why,
            },
            "applied": {
                "stage": new_state.stage,
                "mode": new_state.mode,
                "verb": new_state.current_verb,
                "move": move,
            },
            "hits": list(jr.hits),
            "meta": dict(jr.meta),
        }
    return outputs["protagonist"], outputs["counterpart"]


def _build_judge_trajectories(state: "SessionState") -> dict:
    """Pull per-speaker (stages, modes, verbs, moves, verdicts) lists from final state."""
    return {
        "protagonist": {
            "stages": list(state.judge_history_protagonist["stages"]),
            "modes": list(state.judge_history_protagonist["modes"]),
            "verbs": list(state.judge_history_protagonist["verbs"]),
            "moves": list(state.judge_state_protagonist.move_history) if state.judge_state_protagonist else [],
            "verdicts": list(state.judge_state_protagonist.verdict_history) if state.judge_state_protagonist else [],
        },
        "counterpart": {
            "stages": list(state.judge_history_counterpart["stages"]),
            "modes": list(state.judge_history_counterpart["modes"]),
            "verbs": list(state.judge_history_counterpart["verbs"]),
            "moves": list(state.judge_state_counterpart.move_history) if state.judge_state_counterpart else [],
            "verdicts": list(state.judge_state_counterpart.verdict_history) if state.judge_state_counterpart else [],
        },
    }


def _build_judge_health(state: "SessionState") -> dict:
    """Aggregate per-speaker call counts by parse_status tag.

    Categories: total_calls, ok, parse_fallback (partial|fallback_used), llm_error, no_judge.
    """
    def pack(events: list[dict]) -> dict:
        total = len(events)
        ok = sum(1 for e in events if e.get("parse_status") == "ok")
        parse_fallback = sum(1 for e in events if e.get("parse_status") in ("partial", "fallback_used"))
        llm_error = sum(1 for e in events if e.get("parse_status") == "llm_error")
        no_judge = sum(1 for e in events if e.get("parse_status") == "no_judge")
        return {
            "total_calls": total,
            "ok": ok,
            "parse_fallback": parse_fallback,
            "llm_error": llm_error,
            "no_judge": no_judge,
        }
    return {
        "protagonist": pack(state.judge_health_events["protagonist"]),
        "counterpart": pack(state.judge_health_events["counterpart"]),
    }


def _should_terminate(state: "SessionState", consecutive_required: int = 2) -> tuple[bool, str]:
    """Check whether session should stop after the current turn.

    Returns (should_stop, reason). Reason is "dual_basin_lock" or "" (no stop).
    Requires BOTH speakers' verdict_history to end with `consecutive_required`
    basin_lock verdicts. Single-speaker basin_lock does not terminate.
    """
    jp, jc = state.judge_state_protagonist, state.judge_state_counterpart
    if jp is None or jc is None:
        return False, ""

    def last_n_all_basin(s: JudgeState, n: int) -> bool:
        h = s.verdict_history
        if len(h) < n:
            return False
        return all(v == "basin_lock" for v in h[-n:])

    if last_n_all_basin(jp, consecutive_required) and last_n_all_basin(jc, consecutive_required):
        return True, "dual_basin_lock"
    return False, ""


def _is_peak(state: "SessionState") -> tuple[bool, str]:
    """Return (is_peak, triggered_by) for the most recently completed turn.

    Peak = any speaker's last_verdict is fire_release or basin_lock.
    Priority: fire_release > basin_lock (fire more urgent); protagonist > counterpart (tiebreak).
    """
    jp, jc = state.judge_state_protagonist, state.judge_state_counterpart
    # Fire first (urgency), then basin
    if jp and is_fire_release(jp):
        return True, "fire_release on protagonist"
    if jc and is_fire_release(jc):
        return True, "fire_release on counterpart"
    if jp and is_basin_lock(jp):
        return True, "basin_lock on protagonist"
    if jc and is_basin_lock(jc):
        return True, "basin_lock on counterpart"
    return False, ""


def _prompt_for_director_event(
    *,
    turn_number: int,
    state_p: JudgeState,
    state_c: JudgeState,
    triggered_by: str,
) -> str | None:
    """Block stdin for director input. Empty line or EOF → None (skip)."""
    print(f"\n{'='*60}")
    print(f"[PEAK @ turn {turn_number}] {triggered_by}")
    print(f"  protagonist: stage={state_p.stage}, mode={state_p.mode}, verdict={state_p.last_verdict}")
    print(f"  counterpart: stage={state_c.stage}, mode={state_c.mode}, verdict={state_c.last_verdict}")
    print(f"{'='*60}")
    print("導演介入？輸入事件（空行=跳過）：")
    try:
        line = input().strip()
    except EOFError:
        return None
    return line if line else None


def _judge_state_as_dict(judge_state: "JudgeState | None") -> dict | None:
    """Convert a JudgeState to a plain dict for current_state kwarg in retrieval."""
    if judge_state is None:
        return None
    return {
        "stage": judge_state.stage,
        "mode": judge_state.mode,
        "verb": judge_state.current_verb,
    }


def _build_state_map_for_speaker(
    state: "SessionState",
    speaker_role: str,
    raw_id_to_turn: dict[str, int],
) -> dict[str, dict]:
    """Given raw_id → turn_number, return raw_id → state_dict.

    Looks up the judge history at that turn to get (stage, mode, verb, verdict).
    """
    hist = (
        state.judge_history_protagonist
        if speaker_role == "protagonist"
        else state.judge_history_counterpart
    )
    judge_state = (
        state.judge_state_protagonist
        if speaker_role == "protagonist"
        else state.judge_state_counterpart
    )
    verdicts = judge_state.verdict_history if judge_state else []

    result: dict[str, dict] = {}
    for raw_id, turn_num in raw_id_to_turn.items():
        idx = turn_num - 1  # turn 1 → index 0
        if 0 <= idx < len(hist["stages"]):
            result[raw_id] = {
                "turn": turn_num,
                "stage": hist["stages"][idx],
                "mode": hist["modes"][idx],
                "verb": hist["verbs"][idx],
                "verdict": verdicts[idx] if idx < len(verdicts) else "N/A",
            }
    return result


def _append_session_ledgers(
    *,
    relationship: str,
    protagonist_persona: Persona,
    counterpart_persona: Persona,
    turns: list[Turn],
    source_run: str,
    state: "SessionState",
) -> tuple[list[dict], dict[str, list[str]], dict[str, dict[str, dict]], dict[str, list[dict]]]:
    """Append session candidates to raw ledgers.

    Returns (meta_appends, new_raw_ids, state_maps, candidate_states) where:
    - new_raw_ids: {"protagonist": ["imp_045", ...], "counterpart": ["imp_012", ...]}
    - state_maps: {"protagonist": {raw_id: state_dict}, "counterpart": {...}}
    - candidate_states: {"protagonist": [state_dict, ...], "counterpart": [...]}
      (parallel list to new_raw_ids, used by Composer for salience weighting)

    state_maps enables Composer to attach source_states to refined drafts.
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

    # Build raw_id → turn_number maps for each role
    p_raw_id_to_turn: dict[str, int] = {}
    for raw_id, (turn_num, _) in zip(new_ids["protagonist"], p_candidates):
        p_raw_id_to_turn[raw_id] = turn_num

    c_raw_id_to_turn: dict[str, int] = {}
    for raw_id, (turn_num, _) in zip(new_ids["counterpart"], c_candidates):
        c_raw_id_to_turn[raw_id] = turn_num

    state_maps: dict[str, dict[str, dict]] = {
        "protagonist": _build_state_map_for_speaker(state, "protagonist", p_raw_id_to_turn),
        "counterpart": _build_state_map_for_speaker(state, "counterpart", c_raw_id_to_turn),
    }

    # Build candidate_states: parallel list to new_raw_ids[role]
    candidate_states: dict[str, list[dict]] = {"protagonist": [], "counterpart": []}
    for role in ("protagonist", "counterpart"):
        for raw_id in new_ids[role]:
            state_dict = state_maps[role].get(raw_id, {})
            candidate_states[role].append(state_dict)

    return appends, new_ids, state_maps, candidate_states
