#!/usr/bin/env python3
"""
Export Inspect evaluation logs to results.json for the Astro dashboard.

Reads all .eval logs for the given eval, keeping the most recent version per
model, and writes results/results.json with this shape:

{
  "eval": "grab-bag",
  "generated": "2026-07-06T...",
  "tests": [{"id": "test-1", "input": "...", "expected": "..."}],
  "runs": [{
    "provider_id": "claude-sonnet-4-0",
    "model": "claude-sonnet-4-0",
    "full_model": "anthropic/claude-sonnet-4-0",
    "solver": "",
    "openrouter_provider": "",
    "timestamp": "...",
    "pass_rate": 0.9,
    "samples": [{
      "id": "test-1",
      "output": "...",
      "passed": true,
      "duration_ms": 1234.0,
      "input_tokens": 512,
      "output_tokens": 128,
      "scores": {"includes": {"value": "C", "explanation": null}},
      "diff": null,
      "checks": [{"name": "has_py_file", "passed": true}]
    }]
  }]
}

Tests (input/expected) are stored once and referenced by sample id, so large
inputs aren't duplicated across model runs.

The frontend treats this file as ready to display: pass_rate is the single
authoritative rate for a run (computed here, from the primary scorer), the
agentic git diff is the sample's "diff" field, and check results are
structured — nothing downstream needs to know scorer naming conventions.
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log

# The git_diff scorer marks "a diff was captured", not "the agent succeeded".
# It is lifted out of scores into the sample's diff field and never counts
# toward pass/fail.
DIFF_SCORER = "git_diff"

CHECK_LINE = re.compile(r"^\s*([✓✗])\s*(.*)$")


def extract_openrouter_provider(log):
    """Return the OpenRouter sub-provider if one was specified, else empty string."""
    args = getattr(log.eval, 'model_args', None) or {}
    order = (args.get('provider') or {}).get('order') or []
    return order[0] if order else ""


def extract_provider_id_from_log(log):
    """Return a dedup key that includes the model leaf name and, for OpenRouter
    runs with an explicit provider, a disambiguating suffix."""
    model = log.eval.model
    leaf = model.split('/')[-1] if '/' in model else model
    sub = extract_openrouter_provider(log)
    return f"{leaf} ({sub})" if sub else leaf


def task_matches_eval(task_name: str, eval_name: str) -> bool:
    """Inspect task names use underscores where eval dirs use hyphens."""
    return task_name.replace("_", "-") == eval_name


def read_latest_logs_per_model(log_files, eval_name):
    """Read all logs for eval_name, keeping only the most recent per provider_id.

    Returns a list of EvalLog objects sorted by provider_id. Logs whose task
    doesn't match eval_name are skipped — the filename glob is only a
    prefilter, and one eval's name can be a substring of another's.
    """
    logs_by_provider = defaultdict(list)

    for log_file in log_files:
        try:
            log = read_eval_log(str(log_file))
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
            continue
        if not task_matches_eval(log.eval.task, eval_name):
            continue
        provider_id = extract_provider_id_from_log(log)
        logs_by_provider[provider_id].append((log.eval.created, log))

    latest = []
    for provider_id, logs in sorted(logs_by_provider.items()):
        logs.sort(reverse=True, key=lambda x: x[0])
        latest.append(logs[0][1])

    return latest


def normalize_text(value) -> str:
    """Normalize a sample input/target to plain text.

    Inputs may be a string or a list of chat messages whose content mixes
    text and images; images become an "[image]" marker.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            content = getattr(item, "content", item)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    text = getattr(c, "text", None)
                    if text is not None:
                        parts.append(text)
                    elif getattr(c, "type", "") == "image":
                        parts.append("[image]")
        return "\n".join(p for p in parts if p)
    return str(value)


def extract_output(sample) -> str:
    """Return the model's text output for a sample."""
    if not (sample.output and sample.output.choices):
        return ""
    content = sample.output.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.text for item in content if hasattr(item, "text"))
    return str(content) if content else ""


def json_safe(value) -> Any:
    """Coerce a score value to something JSON-serializable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def extract_tokens(sample) -> tuple[int | None, int | None]:
    """Return (input_tokens, output_tokens) for a sample, None when the log
    didn't record usage (older logs, cached generations, external harnesses)."""
    inp = out = 0
    for usage in (getattr(sample, "model_usage", None) or {}).values():
        inp += usage.input_tokens or 0
        out += usage.output_tokens or 0
    if not (inp or out):
        usage = sample.output.usage if sample.output else None
        if usage:
            inp = usage.input_tokens or 0
            out = usage.output_tokens or 0
    return (inp or None, out or None)


def primary_scorer_name(log) -> str | None:
    """The scorer whose results count: first non-diff scorer declared in the
    run's results (deterministic, unlike sample dict order)."""
    if log.results and log.results.scores:
        for score_data in log.results.scores:
            if score_data.name != DIFF_SCORER:
                return score_data.name
    return None


def score_to_passed(value) -> bool | None:
    """Map a score value to pass/fail ("C" or 1.0 passes)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value == "C"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1.0
    return False


def derive_passed(scores, primary: str | None) -> bool | None:
    """Derive pass/fail for a sample from its primary score, or None if
    unscored. Falls back to the first non-diff score when the run-level
    primary is missing from this sample."""
    if not scores:
        return None
    score = scores.get(primary) if primary else None
    if score is None:
        score = next(
            (s for name, s in scores.items() if name != DIFF_SCORER), None
        )
    if score is None:
        return None
    return score_to_passed(json_safe(score.value))


def extract_checks(scores) -> list[dict]:
    """Return structured [{name, passed}] check results for a sample.

    Prefers Score.metadata["checks"] (scorers emit this going forward);
    falls back to parsing ✓/✗ lines from explanations for older logs.
    """
    checks = []
    for name, score in scores.items():
        if name == DIFF_SCORER:
            continue
        meta = getattr(score, "metadata", None) or {}
        if isinstance(meta.get("checks"), dict):
            checks.extend(
                {"name": k, "passed": bool(v)} for k, v in meta["checks"].items()
            )
            continue
        for line in (score.explanation or "").split("\n"):
            m = CHECK_LINE.match(line)
            if m and m.group(2):
                checks.append({"name": m.group(2), "passed": m.group(1) == "✓"})
    return checks


def compute_pass_rate(log, samples) -> float | None:
    """The run's single authoritative rate.

    Prefers the primary scorer's run-level accuracy/mean metric (handles
    partial credit); falls back to the scored samples' pass ratio. None when
    nothing comparable was scored (diff-only agentic runs).
    """
    primary = primary_scorer_name(log)
    if primary and log.results and log.results.scores:
        for score_data in log.results.scores:
            if score_data.name != primary or not score_data.metrics:
                continue
            for metric in ("accuracy", "mean"):
                if metric in score_data.metrics:
                    return score_data.metrics[metric].value

    scored = [s for s in samples if s["passed"] is not None]
    if not scored:
        return None
    return sum(1 for s in scored if s["passed"]) / len(scored)


def generate_results_json(log_files, eval_name, output_path):
    """Build results.json from the most recent log per model."""
    logs = read_latest_logs_per_model(log_files, eval_name)

    tests: dict[str, dict] = {}
    runs = []

    for log in logs:
        if not log.samples:
            continue

        provider_id = extract_provider_id_from_log(log)
        leaf = log.eval.model.split('/')[-1] if '/' in log.eval.model else log.eval.model
        primary = primary_scorer_name(log)

        samples = []
        for sample in log.samples:
            test_id = f"test-{sample.id}"

            if test_id not in tests:
                tests[test_id] = {
                    "id": test_id,
                    "input": normalize_text(sample.input),
                    "expected": normalize_text(sample.target),
                }

            sample_scores = dict(sample.scores or {})
            diff_score = sample_scores.pop(DIFF_SCORER, None)
            diff = diff_score.explanation if diff_score else None

            scores = {
                name: {
                    "value": json_safe(s.value),
                    "explanation": s.explanation or None,
                }
                for name, s in sample_scores.items()
            }

            duration_ms = None
            if getattr(sample, "total_time", None) is not None:
                duration_ms = round(sample.total_time * 1000, 1)

            input_tokens, output_tokens = extract_tokens(sample)

            samples.append({
                "id": test_id,
                "output": extract_output(sample),
                "passed": derive_passed(sample_scores, primary),
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "scores": scores,
                "diff": diff or None,
                "checks": extract_checks(sample_scores),
            })

        runs.append({
            "provider_id": provider_id,
            "model": leaf,
            "full_model": log.eval.model,
            "solver": getattr(log.eval, "solver", None) or "",
            "openrouter_provider": extract_openrouter_provider(log),
            "timestamp": log.eval.created,
            "pass_rate": compute_pass_rate(log, samples),
            "samples": samples,
        })

    data = {
        "eval": eval_name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tests": list(tests.values()),
        "runs": runs,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")

    n_samples = sum(len(r["samples"]) for r in runs)
    print(f"✅ Generated {output_path} with {len(runs)} runs, {n_samples} samples")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Export Inspect log results to results.json"
    )
    parser.add_argument(
        "eval_name",
        help="Name of the eval (e.g., 'grab-bag')"
    )
    parser.add_argument(
        "--log-file",
        help="Specific log file to export (default: all logs for eval)"
    )
    args = parser.parse_args()

    eval_dir = Path("src/evals") / args.eval_name
    if not eval_dir.exists():
        print(f"❌ Error: Eval directory not found: {eval_dir}")
        return 1

    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)

    logs_dir = Path("logs")
    if args.log_file:
        log_files = [Path(args.log_file)]
    else:
        log_files = list(logs_dir.glob(f"*{args.eval_name}*.eval"))

    if not log_files:
        print(f"❌ Error: No log files found for eval '{args.eval_name}'")
        print(f"    Looked in: {logs_dir}")
        return 1

    print(f"🔄 Extracting results for {args.eval_name} ({len(log_files)} log files)")

    generate_results_json(log_files, args.eval_name, results_dir / "results.json")

    # results.json replaces the old CSV outputs
    for stale in ("results.csv", "aggregate.csv"):
        stale_path = results_dir / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"🗑  Removed {stale_path}")

    return 0


if __name__ == "__main__":
    exit(main())
