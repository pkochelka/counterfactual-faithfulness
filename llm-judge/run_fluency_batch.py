"""Thin wrapper around run_judge_batch.py that judges fluency instead of faithfulness.

It forwards every argument to run_judge_batch, but injects --fluency and a
fluency-appropriate default model so a full batch is a single command:

    python llm-judge/run_fluency_batch.py --token-env-vars TOK1,TOK2 --sample-size all

Output defaults to data/judged_fluency/ (see run_judge_batch for the switch).
"""

from __future__ import annotations

import sys

import run_judge_batch

DEFAULT_FLUENCY_MODEL = "deepseek-v4-pro"


def main() -> None:
    argv = sys.argv[1:]
    if "--fluency" not in argv:
        argv = ["--fluency", *argv]
    if not any(arg == "--model" or arg.startswith("--model=") for arg in argv):
        argv = [*argv, "--model", DEFAULT_FLUENCY_MODEL]
    sys.argv = [sys.argv[0], *argv]
    run_judge_batch.main()


if __name__ == "__main__":
    main()
