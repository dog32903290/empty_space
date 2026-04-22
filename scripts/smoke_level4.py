"""Level 4 smoke test: 6-turn hospital session with real Gemini Flash.

Runs batch mode (no interactive) and prints key Level 4 artifacts for
human eyeball check.

Usage:
    uv run python scripts/smoke_level4.py

Prerequisites:
    GEMINI_API_KEY env var set.
"""
import sys
from pathlib import Path

import yaml

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    config = load_experiment("mother_x_son_act1_hospital")
    # Force short session for smoke
    config.max_turns = 6

    client = GeminiClient()
    result = run_session(config=config, llm_client=client, interactive=False)

    print(f"\n✓ Completed {result.exp_id}")
    print(f"  Out: {result.out_dir}")
    print(f"  Turns: {result.total_turns}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Duration: {result.duration_seconds:.1f}s")

    meta = yaml.safe_load((result.out_dir / "meta.yaml").read_text(encoding="utf-8"))
    print("\n--- Judge Trajectories ---")
    for role in ("protagonist", "counterpart"):
        t = meta["judge_trajectories"][role]
        print(f"\n{role}:")
        print(f"  stages:  {t['stages']}")
        print(f"  modes:   {t['modes']}")
        print(f"  moves:   {t['moves']}")
        print(f"  verdicts:{t['verdicts']}")

    print("\n--- Judge Health ---")
    for role in ("protagonist", "counterpart"):
        print(f"{role}: {meta['judge_health'][role]}")

    print("\n--- Sample turn 3 judge_output ---")
    turn_3 = yaml.safe_load(
        (result.out_dir / "turns" / "turn_003.yaml").read_text(encoding="utf-8")
    )
    print("protagonist:")
    print(yaml.safe_dump(turn_3.get("judge_output_protagonist", {}), allow_unicode=True))
    print("counterpart:")
    print(yaml.safe_dump(turn_3.get("judge_output_counterpart", {}), allow_unicode=True))

    print("\n--- Sample turn 3 system prompt (此刻 block) ---")
    sys_prompt = turn_3["prompt_assembled"]["system"]
    # Extract 此刻 block
    start = sys_prompt.find("## 此刻")
    end = sys_prompt.find("## 現場")
    if start >= 0 and end > start:
        print(sys_prompt[start:end].strip())

    return 0


if __name__ == "__main__":
    sys.exit(main())
