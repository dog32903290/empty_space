"""Run a single experiment session.

Usage:
    uv run python scripts/run_experiment.py <exp_id> [--interactive]

Examples:
    uv run python scripts/run_experiment.py mother_x_son_hospital_v3_001
    uv run python scripts/run_experiment.py mother_x_son_act1_hospital --interactive
"""
import argparse
import sys

from empty_space.llm import GeminiClient
from empty_space.loaders import load_experiment
from empty_space.runner import run_session


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one empty-space experiment session.")
    ap.add_argument("exp_id", help="experiment id (matches experiments/<exp_id>.yaml)")
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive director hook at fire_release/basin_lock peaks",
    )
    args = ap.parse_args()

    config = load_experiment(args.exp_id)
    client = GeminiClient()

    result = run_session(config=config, llm_client=client, interactive=args.interactive)

    print(f"✓ Completed {result.exp_id}")
    print(f"  Output: {result.out_dir}")
    print(f"  Turns: {result.total_turns}")
    print(f"  Termination: {result.termination_reason}")
    print(f"  Tokens in/out: {result.total_tokens_in} / {result.total_tokens_out}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
