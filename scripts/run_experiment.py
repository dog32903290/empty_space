"""Run a single experiment session.

Usage:
    uv run python scripts/run_experiment.py <exp_id>

Example:
    uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
"""
import sys

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_experiment.py <exp_id>", file=sys.stderr)
        return 2

    exp_id = sys.argv[1]
    config = load_experiment(exp_id)
    client = GeminiClient()

    result = run_session(config=config, llm_client=client)

    print(f"✓ Completed {result.exp_id}")
    print(f"  Output: {result.out_dir}")
    print(f"  Turns: {result.total_turns}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Tokens in/out: {result.total_tokens_in} / {result.total_tokens_out}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
