"""Atomic per-turn persistence for Phase 2 runs.

Writes to runs/<exp_id>/<timestamp>/:
  - config.yaml (deep copy of ExperimentConfig)
  - turns/turn_NNN.yaml (one file per turn, atomic rename)
  - conversation.md (append per turn, markdown)
  - conversation.jsonl (append per turn, one JSON object per line)
  - meta.yaml (written once at the end)

Atomicity for yaml files: write to .tmp, os.replace → final.
"""
import json
import os
from pathlib import Path

import yaml

from empty_space.schemas import ExperimentConfig


def init_run(out_dir: Path, config: ExperimentConfig) -> None:
    """Create the run directory skeleton and write config.yaml + conversation init."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "turns").mkdir(exist_ok=True)

    # config.yaml — deep copy via pydantic's model_dump
    config_dump = config.model_dump()
    _atomic_write_yaml(out_dir / "config.yaml", config_dump)

    # conversation.md header
    scene = config.scene_premise or ""
    header_lines = [
        f"# {config.exp_id} @ {out_dir.name}",
        "",
    ]
    if scene:
        header_lines.extend([f"**場景**：{scene.rstrip()}", "", "---", ""])
    else:
        header_lines.extend(["---", ""])
    (out_dir / "conversation.md").write_text(
        "\n".join(header_lines), encoding="utf-8"
    )

    # conversation.jsonl — empty file, appended per turn
    (out_dir / "conversation.jsonl").write_text("", encoding="utf-8")


def _atomic_write_yaml(path: Path, data: object) -> None:
    """Write YAML via .tmp + os.replace for atomicity."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from empty_space.schemas import Turn


def append_turn(out_dir: Path, turn: "Turn") -> None:
    """Write turn_NNN.yaml atomically; append director_event marker (if new) and
    turn entry to conversation.md + conversation.jsonl.
    """
    turn_file = out_dir / "turns" / f"turn_{turn.turn_number:03d}.yaml"
    _atomic_write_yaml(turn_file, _turn_to_yaml_dict(turn))

    new_event = _new_event_this_turn(turn)
    _append_conversation_md(out_dir, turn, new_event)
    _append_conversation_jsonl(out_dir, turn, new_event)


def _new_event_this_turn(turn: "Turn") -> tuple[int, str] | None:
    """Return the event triggered AT this turn, if any.

    Convention: runner appends director_events[turn_number] to active_events
    just before the LLM call, so if the last active event's turn equals
    turn.turn_number, that event was triggered this turn.
    """
    if turn.director_events_active and turn.director_events_active[-1][0] == turn.turn_number:
        return turn.director_events_active[-1]
    return None


def _turn_to_yaml_dict(turn: "Turn") -> dict:
    return {
        "turn": turn.turn_number,
        "speaker": turn.speaker,
        "persona_name": turn.persona_name,
        "timestamp": turn.timestamp,
        "prompt_assembled": {
            "system": turn.prompt_system,
            "user": turn.prompt_user,
            "prompt_tokens": turn.tokens_in,  # total prompt tokens reported by API
        },
        "response": {
            "content": turn.content,
            "raw": turn.raw_response,
            "tokens_out": turn.tokens_out,
            "model": turn.model,
            "latency_ms": turn.latency_ms,
        },
        "candidate_impressions": [
            {"text": imp.text, "symbols": list(imp.symbols)}
            for imp in turn.candidate_impressions
        ],
        "director_events_active": [
            {"turn": t, "content": c} for t, c in turn.director_events_active
        ],
        "parse_error": turn.parse_error,
        "retrieved_impressions": [
            {
                "id": imp.id,
                "text": imp.text,
                "symbols": list(imp.symbols),
                "speaker": imp.speaker,
                "persona_name": imp.persona_name,
                "from_run": imp.from_run,
                "from_turn": imp.from_turn,
                "score": imp.score,
                "matched_symbols": list(imp.matched_symbols),
            }
            for imp in turn.retrieved_impressions
        ],
    }


def _append_conversation_md(
    out_dir: Path, turn: "Turn", new_event: tuple[int, str] | None
) -> None:
    parts: list[str] = []
    if new_event is not None:
        parts.append(f"**[世界] Turn {new_event[0]}：{new_event[1]}**\n")
    parts.append(f"**Turn {turn.turn_number} · {turn.persona_name}**\n{turn.content}\n")
    with (out_dir / "conversation.md").open("a", encoding="utf-8") as f:
        f.write("".join(parts) + "\n")


def _append_conversation_jsonl(
    out_dir: Path, turn: "Turn", new_event: tuple[int, str] | None
) -> None:
    lines: list[str] = []
    if new_event is not None:
        lines.append(json.dumps(
            {"type": "director_event", "turn": new_event[0], "content": new_event[1]},
            ensure_ascii=False,
        ))
    lines.append(json.dumps(
        {
            "turn": turn.turn_number,
            "speaker": turn.speaker,
            "name": turn.persona_name,
            "content": turn.content,
            "timestamp": turn.timestamp,
        },
        ensure_ascii=False,
    ))
    with (out_dir / "conversation.jsonl").open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def write_meta(
    *,
    out_dir: Path,
    config: ExperimentConfig,
    total_turns: int,
    termination_reason: str,
    total_tokens_in: int,
    total_tokens_out: int,
    total_candidate_impressions: int,
    turns_with_parse_error: int,
    director_events_triggered: list[tuple[int, str]],
    models_used: list[str],
    duration_seconds: float,
    retrieval_total_tokens_in: int = 0,
    retrieval_total_tokens_out: int = 0,
    ledger_appends: list[dict] | None = None,
) -> None:
    """Write meta.yaml with session-level summary."""
    meta = {
        "exp_id": config.exp_id,
        "run_timestamp": out_dir.name,
        "total_turns": total_turns,
        "termination_reason": termination_reason,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "duration_seconds": duration_seconds,
        "total_candidate_impressions": total_candidate_impressions,
        "turns_with_parse_error": turns_with_parse_error,
        "director_events_triggered": [
            {"turn": t, "content": c} for t, c in director_events_triggered
        ],
        "models_used": models_used,
        "retrieval_total_tokens_in": retrieval_total_tokens_in,
        "retrieval_total_tokens_out": retrieval_total_tokens_out,
        "ledger_appends": ledger_appends or [],
    }
    _atomic_write_yaml(out_dir / "meta.yaml", meta)


def write_retrieval(
    out_dir: Path,
    *,
    protagonist,
    counterpart,
) -> None:
    """Write retrieval.yaml with both roles' session-start retrieval outcomes."""
    data = {
        "protagonist": _retrieval_to_yaml_dict(protagonist),
        "counterpart": _retrieval_to_yaml_dict(counterpart),
    }
    _atomic_write_yaml(out_dir / "retrieval.yaml", data)


def _retrieval_to_yaml_dict(r) -> dict:
    return {
        "speaker_role": r.speaker_role,
        "persona_name": r.persona_name,
        "query_text": r.query_text,
        "query_symbols": list(r.query_symbols),
        "expanded_symbols": list(r.expanded_symbols),
        "impressions": [
            {
                "id": imp.id,
                "text": imp.text,
                "symbols": list(imp.symbols),
                "speaker": imp.speaker,
                "persona_name": imp.persona_name,
                "from_run": imp.from_run,
                "from_turn": imp.from_turn,
                "score": imp.score,
                "matched_symbols": list(imp.matched_symbols),
            }
            for imp in r.impressions
        ],
        "flash_latency_ms": r.flash_latency_ms,
        "flash_tokens_in": r.flash_tokens_in,
        "flash_tokens_out": r.flash_tokens_out,
    }
