"""Cross-session impression ledger — append-only, per speaker.

Files land at ledgers/<relationship>.from_<persona_name>.yaml.
Maintains symbol_index (reverse lookup) and cooccurrence (1-hop graph edges)
incrementally on each append.

Atomic write via .tmp + os.replace.
"""
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import yaml

from empty_space.paths import LEDGERS_DIR
from empty_space.schemas import (
    CandidateImpression,
    Ledger,
    LedgerEntry,
    RefinedImpression,
    RefinedImpressionDraft,
    RefinedLedger,
)


def ledger_path(*, relationship: str, persona_name: str) -> Path:
    """Returns <LEDGERS_DIR>/<relationship>.from_<persona_name>.yaml"""
    return LEDGERS_DIR / f"{relationship}.from_{persona_name}.yaml"


def read_ledger(*, relationship: str, persona_name: str) -> Ledger:
    """Read ledger file; if absent, return empty Ledger (do not raise).

    Note: when the file is absent, speaker is set to 'protagonist' as a
    placeholder. Callers should not rely on the .speaker field of an empty
    ledger since the speaker_role isn't knowable from the file path alone.
    """
    path = ledger_path(relationship=relationship, persona_name=persona_name)
    if not path.exists():
        return Ledger(
            relationship=relationship,
            speaker="protagonist",  # placeholder; overwritten on first append
            persona_name=persona_name,
            ledger_version=0,
            candidates=[],
            symbol_index={},
            cooccurrence={},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Ledger(
        relationship=data["relationship"],
        speaker=data["speaker"],
        persona_name=data["persona_name"],
        ledger_version=data["ledger_version"],
        candidates=[
            LedgerEntry(
                id=c["id"],
                text=c["text"],
                symbols=list(c["symbols"]),
                from_run=c["from_run"],
                from_turn=c["from_turn"],
                created=c["created"],
            )
            for c in (data.get("candidates") or [])
        ],
        symbol_index={k: list(v) for k, v in (data.get("symbol_index") or {}).items()},
        cooccurrence={
            k: dict(v) for k, v in (data.get("cooccurrence") or {}).items()
        },
    )


def append_session_candidates(
    *,
    relationship: str,
    speaker_role: str,
    persona_name: str,
    candidates: list[tuple[int, CandidateImpression]],
    source_run: str,
) -> list[str]:
    """Append one session's worth of candidates to a ledger. Atomic write.

    candidates: list of (turn_number, CandidateImpression) tuples.
    Updates symbol_index (reverse) and cooccurrence (symmetric pair counts).
    Increments ledger_version.

    Empty candidates list still creates/updates the file and bumps version.

    Returns list of newly-appended LedgerEntry ids (for Composer provenance
    tracking; existing callers can ignore).
    """
    # Read existing (may be empty)
    existing = read_ledger(relationship=relationship, persona_name=persona_name)

    # Determine next id
    next_id_num = len(existing.candidates) + 1
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build new entries and merge
    new_entries: list[LedgerEntry] = []
    for turn_number, imp in candidates:
        entry = LedgerEntry(
            id=f"imp_{next_id_num:03d}",
            text=imp.text,
            symbols=list(imp.symbols),
            from_run=source_run,
            from_turn=turn_number,
            created=now_iso,
        )
        new_entries.append(entry)
        next_id_num += 1

    all_candidates = existing.candidates + new_entries

    # Update symbol_index incrementally
    symbol_index = {k: list(v) for k, v in existing.symbol_index.items()}
    for entry in new_entries:
        for sym in entry.symbols:
            symbol_index.setdefault(sym, []).append(entry.id)

    # Update cooccurrence (symmetric)
    cooccurrence = {k: dict(v) for k, v in existing.cooccurrence.items()}
    for entry in new_entries:
        for sym_a, sym_b in combinations(entry.symbols, 2):
            cooccurrence.setdefault(sym_a, {})
            cooccurrence[sym_a][sym_b] = cooccurrence[sym_a].get(sym_b, 0) + 1
            cooccurrence.setdefault(sym_b, {})
            cooccurrence[sym_b][sym_a] = cooccurrence[sym_b].get(sym_a, 0) + 1

    # Construct final Ledger
    new_ledger = Ledger(
        relationship=relationship,
        speaker=speaker_role,
        persona_name=persona_name,
        ledger_version=existing.ledger_version + 1,
        candidates=all_candidates,
        symbol_index=symbol_index,
        cooccurrence=cooccurrence,
    )

    _atomic_write_ledger(new_ledger)
    return [entry.id for entry in new_entries]


def _atomic_write_ledger(ledger: Ledger) -> None:
    """Serialize ledger to YAML via .tmp + os.replace."""
    path = ledger_path(relationship=ledger.relationship, persona_name=ledger.persona_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "relationship": ledger.relationship,
        "speaker": ledger.speaker,
        "persona_name": ledger.persona_name,
        "ledger_version": ledger.ledger_version,
        "candidates": [
            {
                "id": e.id,
                "text": e.text,
                "symbols": list(e.symbols),
                "from_run": e.from_run,
                "from_turn": e.from_turn,
                "created": e.created,
            }
            for e in ledger.candidates
        ],
        "symbol_index": ledger.symbol_index,
        "cooccurrence": ledger.cooccurrence,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# --- Level 3: refined ledger I/O ---


def refined_ledger_path(*, relationship: str, persona_name: str) -> Path:
    """Returns <LEDGERS_DIR>/<relationship>.refined.from_<persona_name>.yaml"""
    return LEDGERS_DIR / f"{relationship}.refined.from_{persona_name}.yaml"


def read_refined_ledger(*, relationship: str, persona_name: str) -> RefinedLedger:
    """Read refined ledger file; if absent, return empty RefinedLedger (do not raise).

    Note: when the file is absent, speaker is set to 'protagonist' as a
    placeholder (callers should override or not rely on it for an empty ledger).
    """
    path = refined_ledger_path(relationship=relationship, persona_name=persona_name)
    if not path.exists():
        return RefinedLedger(
            relationship=relationship,
            speaker="protagonist",  # placeholder
            persona_name=persona_name,
            ledger_version=0,
            impressions=[],
            symbol_index={},
            cooccurrence={},
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RefinedLedger(
        relationship=data["relationship"],
        speaker=data["speaker"],
        persona_name=data["persona_name"],
        ledger_version=data["ledger_version"],
        impressions=[
            RefinedImpression(
                id=i["id"],
                text=i["text"],
                symbols=list(i["symbols"]),
                speaker=i.get("speaker", data["speaker"]),
                persona_name=i.get("persona_name", data["persona_name"]),
                from_run=i["from_run"],
                source_raw_ids=list(i.get("source_raw_ids") or []),
                created=i["created"],
                source_states=list(i.get("source_states") or []),
            )
            for i in (data.get("impressions") or [])
        ],
        symbol_index={k: list(v) for k, v in (data.get("symbol_index") or {}).items()},
        cooccurrence={
            k: dict(v) for k, v in (data.get("cooccurrence") or {}).items()
        },
    )


def append_refined_impressions(
    *,
    relationship: str,
    speaker_role: str,
    persona_name: str,
    drafts: list[RefinedImpressionDraft],
    source_run: str,
) -> int:
    """Append refined drafts to ledger. Returns new ledger_version.

    Empty drafts: no-op, returns 0 (file not created/touched).
    Updates symbol_index and cooccurrence incrementally.
    Atomic write via .tmp + os.replace.
    """
    if not drafts:
        return 0

    existing = read_refined_ledger(
        relationship=relationship, persona_name=persona_name,
    )

    next_id_num = len(existing.impressions) + 1
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    new_entries: list[RefinedImpression] = []
    for draft in drafts:
        entry = RefinedImpression(
            id=f"ref_{next_id_num:03d}",
            text=draft.text,
            symbols=list(draft.symbols),
            speaker=speaker_role,  # type: ignore[arg-type]
            persona_name=persona_name,
            from_run=source_run,
            source_raw_ids=list(draft.source_raw_ids),
            created=now_iso,
            source_states=list(draft.source_states),
        )
        new_entries.append(entry)
        next_id_num += 1

    all_impressions = existing.impressions + new_entries

    # Update symbol_index incrementally
    symbol_index = {k: list(v) for k, v in existing.symbol_index.items()}
    for entry in new_entries:
        for sym in entry.symbols:
            symbol_index.setdefault(sym, []).append(entry.id)

    # Update cooccurrence (symmetric)
    cooccurrence = {k: dict(v) for k, v in existing.cooccurrence.items()}
    for entry in new_entries:
        for sym_a, sym_b in combinations(entry.symbols, 2):
            cooccurrence.setdefault(sym_a, {})
            cooccurrence[sym_a][sym_b] = cooccurrence[sym_a].get(sym_b, 0) + 1
            cooccurrence.setdefault(sym_b, {})
            cooccurrence[sym_b][sym_a] = cooccurrence[sym_b].get(sym_a, 0) + 1

    new_ledger = RefinedLedger(
        relationship=relationship,
        speaker=speaker_role,  # type: ignore[arg-type]
        persona_name=persona_name,
        ledger_version=existing.ledger_version + 1,
        impressions=all_impressions,
        symbol_index=symbol_index,
        cooccurrence=cooccurrence,
    )

    _atomic_write_refined_ledger(new_ledger)
    return new_ledger.ledger_version


def _atomic_write_refined_ledger(ledger: RefinedLedger) -> None:
    """Serialize refined ledger to YAML via .tmp + os.replace."""
    path = refined_ledger_path(
        relationship=ledger.relationship, persona_name=ledger.persona_name,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "relationship": ledger.relationship,
        "speaker": ledger.speaker,
        "persona_name": ledger.persona_name,
        "ledger_version": ledger.ledger_version,
        "impressions": [
            {
                "id": e.id,
                "text": e.text,
                "symbols": list(e.symbols),
                "speaker": e.speaker,
                "persona_name": e.persona_name,
                "from_run": e.from_run,
                "source_raw_ids": list(e.source_raw_ids),
                "created": e.created,
                **( {"source_states": list(e.source_states)} if e.source_states else {}),
            }
            for e in ledger.impressions
        ],
        "symbol_index": ledger.symbol_index,
        "cooccurrence": ledger.cooccurrence,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
