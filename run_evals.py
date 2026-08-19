#!/usr/bin/env python3
"""Run inspect-ai evals.

By default, runs every active eval against every active model. Narrow with
--eval and/or --model to run a subset (both together runs just that one
pair). Pairs that already have a run in results.json are skipped — pass
--force to rerun them anyway. Extra args are forwarded straight to
`inspect eval`. To pin a specific OpenRouter sub-provider:

    run_evals.py --eval my-eval --model openrouter/moonshotai/kimi-k2-thinking \\
        -M "provider={'order':['moonshotai']}"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Models currently being tracked for comparison across evals. `extra_args` are
# always applied for that model (e.g. pinning an OpenRouter sub-provider);
# `solver` is the default agentic-eval harness for that model.
ACTIVE_MODELS = {
    "anthropic/claude-opus-5": {
        "extra_args": [],
        "solver": "claude_code",
    },
    "anthropic/claude-haiku-4-5": {
        "extra_args": [],
        "solver": "claude_code",
    },
    "openai/gpt-5.6-luna": {
        "extra_args": [],
        "solver": "codex",
    },
    # "google/gemini-3.1-pro-preview": {
    #     "extra_args": [],
    #     "solver": "pi",
    # },
    # "openrouter/x-ai/grok-4.5": {
    #     "extra_args": ["-M", "provider={'order':['grok']}"],
    #     "solver": "pi",
    # },
    "openrouter/qwen/qwen3.8-27b": {
        "extra_args": ["-M", "provider={'order':['venice/fp8']}"],
        "solver": "pi",
    },
    "openrouter/qwen/qwen3.6-35b-a3b": {
        "extra_args": ["-M", "provider={'order':['venice/fp8']}"],
        "solver": "pi",
    },
}

ROOT = Path(__file__).parent
EVALS_DIR = ROOT / "src" / "evals"

DRY_RUN = False


def frontmatter(index_md: Path) -> dict:
    text = index_md.read_text()
    if not text.startswith("---"):
        return {}
    _, fm, _ = text.split("---", 2)
    return yaml.safe_load(fm) or {}


def active_evals(agentic: bool) -> list[str]:
    """Non-archived eval names, filtered to agentic vs. non-agentic type."""
    names = []
    for eval_dir in sorted(EVALS_DIR.iterdir()):
        index_md = eval_dir / "index.md"
        if not index_md.is_file():
            continue
        fm = frontmatter(index_md)
        if fm.get("archived"):
            continue
        if (fm.get("type") == "agentic") == agentic:
            names.append(eval_dir.name)
    return names


def run(cmd: list[str]):
    print("+", " ".join(cmd))
    if DRY_RUN:
        return
    # Eval files do `from agentic import ...`, which needs the project root
    # on PYTHONPATH since we invoke `inspect`/`extract_results.py` directly
    # rather than through `just` (which used to export this).
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(cmd, check=True, env=env)


def run_eval(name: str, model: str, extra: list[str]):
    model_extra_args = ACTIVE_MODELS.get(model, {}).get("extra_args", [])
    run(
        [
            "uv",
            "run",
            "inspect",
            "eval",
            f"src/evals/{name}/eval.py",
            "--model",
            model,
            "--log-dir",
            "logs",
            *model_extra_args,
            *extra,
        ]
    )
    run(["uv", "run", "python", "extract_results.py", name])


def eval_type(name: str) -> str | None:
    index_md = EVALS_DIR / name / "index.md"
    return frontmatter(index_md).get("type") if index_md.is_file() else None


def has_result(name: str, model: str) -> bool:
    """Whether results.json already has a run for this eval+model, regardless
    of solver (matches the dashboard's own dedup: one run per model)."""
    results_json = EVALS_DIR / name / "results" / "results.json"
    if not results_json.is_file():
        return False
    try:
        data = json.loads(results_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return any(r.get("full_model") == model for r in data.get("runs", []))


def resolve_solver(model: str, solver: str | None) -> str:
    if solver:
        return solver
    solver = ACTIVE_MODELS.get(model, {}).get("solver")
    if not solver:
        sys.exit(f"no default solver for {model!r} — pass --solver explicitly")
    return solver


def run_target(name: str, model: str, solver: str | None, extra: list[str]):
    if eval_type(name) == "agentic":
        extra = ["--solver", resolve_solver(model, solver), *extra]
    run_eval(name, model, extra)


def print_list(args):
    print("active evals:", *active_evals(agentic=False))
    print("active agentic evals:", *active_evals(agentic=True))
    print("active models:")
    for model, cfg in ACTIVE_MODELS.items():
        print(f"  {model} (default solver: {cfg['solver']})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", help="limit to this model (default: every model in ACTIVE_MODELS)"
    )
    parser.add_argument(
        "--eval",
        dest="eval_name",
        help="limit to this eval (default: every active eval)",
    )
    parser.add_argument(
        "--solver",
        help="override the agentic solver (default: the model's entry in ACTIVE_MODELS)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list active evals and models, then exit"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun even if results.json already has a run for that eval+model",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without running them"
    )
    # Unrecognized args (e.g. `-M "provider=..."` meant for `inspect eval`)
    # pass through instead of erroring.
    args, extra = parser.parse_known_args()

    global DRY_RUN
    DRY_RUN = args.dry_run

    if args.list:
        print_list(args)
        return

    models = [args.model] if args.model else list(ACTIVE_MODELS)
    evals = (
        [args.eval_name]
        if args.eval_name
        else [*active_evals(agentic=False), *active_evals(agentic=True)]
    )
    try:
        for model in models:
            for name in evals:
                if not args.force and has_result(name, model):
                    print(f"skip (already have a result): {name} {model}")
                    continue
                run_target(name, model, args.solver, extra)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
