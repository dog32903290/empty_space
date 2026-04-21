"""Session-start retrieval: extract symbols → expand via co-occurrence →
score candidates in both ledgers → return top-3 per role.
"""
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
