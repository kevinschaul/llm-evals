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
    "metrics": {"includes": {"accuracy": 0.9}},
    "samples": [{
      "id": "test-1",
      "output": "...",
      "passed": true,
      "duration_ms": 1234.0,
      "scores": {"includes": {"value": "C", "explanation": null}}
    }]
  }]
}

Tests (input/expected) are stored once and referenced by sample id, so large
inputs aren't duplicated across model runs. Aggregates are computed by the
frontend from samples.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


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


def read_latest_logs_per_model(log_files):
    """Read all logs, keeping only the most recent per provider_id.

    Returns a list of EvalLog objects sorted by provider_id.
    """
    logs_by_provider = defaultdict(list)

    for log_file in log_files:
        try:
            log = read_eval_log(str(log_file))
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
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


def derive_passed(scores) -> bool | None:
    """Derive a pass/fail bool from a sample's scores, or None if unscored.

    Uses the first non-git_diff score ("C" or 1.0 means pass), matching the
    old CSV extraction behavior.
    """
    if not scores:
        return None
    primary = next(
        (s for name, s in scores.items() if name != "git_diff"),
        next(iter(scores.values())),
    )
    if primary.value is None:
        return None
    if isinstance(primary.value, str):
        return primary.value == "C"
    if isinstance(primary.value, (int, float)):
        return primary.value == 1.0
    return False


def extract_run_metrics(log) -> dict:
    """Return {scorer_name: {metric_name: value}} from run-level results."""
    metrics = {}
    if log.results and log.results.scores:
        for score_data in log.results.scores:
            if score_data.metrics:
                metrics[score_data.name] = {
                    name: m.value for name, m in score_data.metrics.items()
                }
    return metrics


def generate_results_json(log_files, output_path):
    """Build results.json from the most recent log per model."""
    logs = read_latest_logs_per_model(log_files)

    tests: dict[str, dict] = {}
    runs = []

    for log in logs:
        if not log.samples:
            continue

        provider_id = extract_provider_id_from_log(log)
        leaf = log.eval.model.split('/')[-1] if '/' in log.eval.model else log.eval.model

        samples = []
        for sample in log.samples:
            test_id = f"test-{sample.id}"

            if test_id not in tests:
                tests[test_id] = {
                    "id": test_id,
                    "input": normalize_text(sample.input),
                    "expected": normalize_text(sample.target),
                }

            scores = {}
            if sample.scores:
                for name, s in sample.scores.items():
                    scores[name] = {
                        "value": json_safe(s.value),
                        "explanation": s.explanation or None,
                    }

            duration_ms = None
            if getattr(sample, "total_time", None) is not None:
                duration_ms = round(sample.total_time * 1000, 1)

            samples.append({
                "id": test_id,
                "output": extract_output(sample),
                "passed": derive_passed(sample.scores),
                "duration_ms": duration_ms,
                "scores": scores,
            })

        runs.append({
            "provider_id": provider_id,
            "model": leaf,
            "full_model": log.eval.model,
            "solver": getattr(log.eval, "solver", None) or "",
            "openrouter_provider": extract_openrouter_provider(log),
            "timestamp": log.eval.created,
            "metrics": extract_run_metrics(log),
            "samples": samples,
        })

    data = {
        "eval": output_path.parent.parent.name,
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

    generate_results_json(log_files, results_dir / "results.json")

    # results.json replaces the old CSV outputs
    for stale in ("results.csv", "aggregate.csv"):
        stale_path = results_dir / stale
        if stale_path.exists():
            stale_path.unlink()
            print(f"🗑  Removed {stale_path}")

    return 0


if __name__ == "__main__":
    exit(main())
