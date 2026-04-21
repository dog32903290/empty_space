"""Session-start retrieval: extract symbols → expand via co-occurrence →
score candidates in both ledgers → return top-3 per role.
"""
import re
from pathlib import Path

import yaml

from empty_space.paths import PROJECT_ROOT

DEFAULT_SYNONYMS_PATH = PROJECT_ROOT / "config" / "symbol_synonyms.yaml"


# --- canonicalization ---

def canonicalize(symbol: str, synonym_map: dict[str, str]) -> str:
    """Return synonym_map[symbol] if mapped, else symbol."""
    return synonym_map.get(symbol, symbol)


def load_synonym_map(path: Path | None = None) -> dict[str, str]:
    """Load config/symbol_synonyms.yaml. Returns symbol→canonical dict.

    If file missing, returns {}.
    If file present but groups is empty or absent, returns {}.
    Within each group, the first element is canonical; all elements map to it.
    """
    if path is None:
        path = DEFAULT_SYNONYMS_PATH
    if not Path(path).exists():
        return {}

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    groups = data.get("groups") or []

    result: dict[str, str] = {}
    for group in groups:
        if not group:
            continue
        canonical = group[0]
        for sym in group:
            result[sym] = canonical
    return result


# --- co-occurrence expansion ---

def expand_with_cooccurrence(
    *,
    seed_symbols: list[str],
    cooccurrence: dict[str, dict[str, int]],
    top_neighbors_per_seed: int = 2,
) -> list[str]:
    """For each seed, add its top-K most-cooccurring neighbors.

    Preserve seed order; append neighbors after seeds. Dedup globally.
    Tiebreak: count desc, then alphabetical asc.
    """
    seen = set(seed_symbols)
    result = list(seed_symbols)
    for seed in seed_symbols:
        neighbors = cooccurrence.get(seed, {})
        # Sort by count desc, then key asc
        sorted_neighbors = sorted(
            neighbors.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:top_neighbors_per_seed]
        for sym, _ in sorted_neighbors:
            if sym not in seen:
                result.append(sym)
                seen.add(sym)
    return result


def merge_cooccurrence(
    a: dict[str, dict[str, int]],
    b: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    """Sum two cooccurrence maps. Does not mutate inputs."""
    result = {k: dict(v) for k, v in a.items()}
    for sym_a, neighbors in b.items():
        if sym_a not in result:
            result[sym_a] = dict(neighbors)
        else:
            for sym_b, count in neighbors.items():
                result[sym_a][sym_b] = result[sym_a].get(sym_b, 0) + count
    return result


# --- Flash-based symbol extraction ---

def _parse_yaml_block_list(text: str) -> list[str] | None:
    """Parse a strict YAML block sequence by line matching.

    Accepts only lines matching ``^- (.*)$`` (top-level list items, no indent).
    Returns a list of stripped values on success, or None if any non-blank line
    does not match the pattern (malformed / flow sequence / mapping, etc.).
    """
    lines = text.splitlines()
    items: list[str] = []
    for line in lines:
        if line.strip() == "":
            continue
        m = re.match(r"^- (.*)$", line)
        if m:
            items.append(m.group(1).strip())
        else:
            return None
    return items

_EXTRACT_SYMBOLS_SYSTEM_PROMPT = """\
你負責從一段中文敘述中提取「感受符號」——能作為記憶檢索 key 的關鍵詞。
規則：
- 每個 symbol 是一個具體的名詞、動詞或感官詞
- 不要抽象名詞（「痛苦」「關係」這類太大）
- 不要連詞、助詞、時間副詞
- 偏好單字或兩字詞，不用長片語
- 輸出 3-10 個 symbols，YAML list 格式
- 只輸出 YAML，不加解釋

範例：
輸入：
「你昨晚和女朋友分手。她說『我不能等你。』」

輸出：
- 分手
- 女朋友
- 等
- 拒絕
"""


def extract_symbols(
    *,
    text: str,
    llm_client,
) -> tuple[list[str], int, int, int]:
    """Ask Flash to extract semantic symbols from a query text.

    Returns:
        (symbols, tokens_in, tokens_out, latency_ms)

    If text is empty/whitespace, skips the LLM call and returns empty.
    On YAML parse failure or non-list response, returns empty symbols list
    (tokens and latency still report the actual call).
    """
    if not text or not text.strip():
        return [], 0, 0, 0

    resp = llm_client.generate(
        system=_EXTRACT_SYMBOLS_SYSTEM_PROMPT,
        user=text,
        model="gemini-2.5-flash",
    )

    parsed = _parse_yaml_block_list(resp.content)
    if parsed is None:
        return [], resp.tokens_in, resp.tokens_out, resp.latency_ms

    symbols = [s for s in parsed if s]
    return symbols, resp.tokens_in, resp.tokens_out, resp.latency_ms


# --- scoring & top-N retrieval ---

from empty_space.schemas import (
    Ledger,
    LedgerEntry,
    RetrievedImpression,
    RetrievalResult,
)
from empty_space.ledger import read_ledger


def retrieve_top_n(
    *,
    query_symbols: list[str],
    ledger_a: Ledger,
    ledger_b: Ledger,
    synonym_map: dict[str, str],
    top_n: int = 3,
) -> list[RetrievedImpression]:
    """Score candidates in both ledgers via symbol hit count under canonical
    equivalence. Return top N by (score desc, created desc). Dedup by
    (speaker, id).
    """
    canon_q = {canonicalize(s, synonym_map) for s in query_symbols}
    if not canon_q:
        return []

    # Score every entry in both ledgers that has at least one match
    scored: list[tuple[int, str, LedgerEntry, Ledger, list[str]]] = []
    # (score, created, entry, ledger, matched_canonicals_sorted)
    for ledger in (ledger_a, ledger_b):
        for entry in ledger.candidates:
            canon_e = {canonicalize(s, synonym_map) for s in entry.symbols}
            matched = canon_q & canon_e
            if matched:
                scored.append((
                    len(matched),
                    entry.created,
                    entry,
                    ledger,
                    sorted(matched),
                ))

    # Multi-key sort via Python's stable sort: apply tiebreaker first, primary last
    scored.sort(key=lambda t: t[1], reverse=True)   # created desc (tiebreaker)
    scored.sort(key=lambda t: t[0], reverse=True)   # score desc (primary)

    # Dedup by (speaker, id)
    seen_keys: set[tuple[str, str]] = set()
    result: list[RetrievedImpression] = []
    for score, _, entry, ledger, matched in scored:
        key = (ledger.speaker, entry.id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(RetrievedImpression(
            id=entry.id,
            text=entry.text,
            symbols=tuple(entry.symbols),
            speaker=ledger.speaker,
            persona_name=ledger.persona_name,
            from_run=entry.from_run,
            from_turn=entry.from_turn,
            score=score,
            matched_symbols=tuple(matched),
        ))
        if len(result) >= top_n:
            break

    return result


# --- session-start orchestrator ---

def run_session_start_retrieval(
    *,
    speaker_role: str,
    persona_name: str,
    query_text: str,
    relationship: str,
    other_persona_name: str,
    synonym_map: dict[str, str],
    llm_client,
    top_n: int = 3,
) -> RetrievalResult:
    """Full session-start retrieval pipeline for one role.

    1. Flash extract symbols from query_text.
    2. Load both ledgers (self + other).
    3. Expand with merged cooccurrence (1-hop).
    4. Score candidates in both ledgers, return top N.
    5. Package as RetrievalResult (with debug info).
    """
    # Step 1: extract
    query_symbols, tokens_in, tokens_out, latency_ms = extract_symbols(
        text=query_text, llm_client=llm_client,
    )

    # Step 2: load ledgers
    ledger_self = read_ledger(relationship=relationship, persona_name=persona_name)
    ledger_other = read_ledger(relationship=relationship, persona_name=other_persona_name)

    # Set correct speaker on ledgers (read_ledger uses placeholder when file missing)
    ledger_self = _with_speaker(ledger_self, speaker_role)
    other_role = "counterpart" if speaker_role == "protagonist" else "protagonist"
    ledger_other = _with_speaker(ledger_other, other_role)

    # Step 3: expand
    merged_cooc = merge_cooccurrence(ledger_self.cooccurrence, ledger_other.cooccurrence)
    expanded_symbols = expand_with_cooccurrence(
        seed_symbols=query_symbols,
        cooccurrence=merged_cooc,
        top_neighbors_per_seed=2,
    )

    # Step 4: retrieve top N (use expanded symbols so co-occurrence neighbors count)
    impressions = retrieve_top_n(
        query_symbols=expanded_symbols,
        ledger_a=ledger_self,
        ledger_b=ledger_other,
        synonym_map=synonym_map,
        top_n=top_n,
    )

    return RetrievalResult(
        speaker_role=speaker_role,
        persona_name=persona_name,
        query_text=query_text,
        query_symbols=query_symbols,
        expanded_symbols=expanded_symbols,
        impressions=impressions,
        flash_latency_ms=latency_ms,
        flash_tokens_in=tokens_in,
        flash_tokens_out=tokens_out,
    )


def _with_speaker(ledger: Ledger, speaker: str) -> Ledger:
    """Return a new Ledger with .speaker overridden.

    Used for empty ledgers read from missing files (read_ledger uses a
    placeholder speaker since the speaker_role isn't knowable from path alone).
    """
    if ledger.speaker == speaker:
        return ledger
    return Ledger(
        relationship=ledger.relationship,
        speaker=speaker,  # type: ignore[arg-type]
        persona_name=ledger.persona_name,
        ledger_version=ledger.ledger_version,
        candidates=ledger.candidates,
        symbol_index=ledger.symbol_index,
        cooccurrence=ledger.cooccurrence,
    )
