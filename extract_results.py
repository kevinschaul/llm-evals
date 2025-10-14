#!/usr/bin/env python3
"""
Export Inspect evaluation logs to CSVs for Observable Framework dashboard.

Reads all .eval logs for the given eval, keeping the most recent version per model, and writes results.csv and aggregate.csv
"""
import argparse
import csv
from pathlib import Path
from typing import Any
from collections import defaultdict

from inspect_ai.log import read_eval_log


def extract_provider_id_from_log(log):
    """Extract provider ID from the model name"""
    model = log.eval.model
    # Try to extract a clean provider ID
    # Examples: "openai/gpt-4o-mini" -> "gpt-4o-mini"
    #          "anthropic/claude-opus-4-0" -> "claude-opus-4-0"
    if '/' in model:
        return model.split('/')[-1]
    return model


def get_latest_logs_per_model(log_files):
    """Keep only the most recent log file per model

    Args:
        log_files: List of Path objects pointing to .eval log files

    Returns:
        List of Path objects for the most recent log per model
    """
    # Group log files by model
    logs_by_model = defaultdict(list)

    for log_file in log_files:
        try:
            log = read_eval_log(str(log_file))
            provider_id = extract_provider_id_from_log(log)
            timestamp = log.eval.created
            logs_by_model[provider_id].append((timestamp, log_file))
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
            continue

    # Keep only the most recent log per model
    latest_logs = []
    for provider_id, logs in logs_by_model.items():
        # Sort by timestamp (most recent first) and take the first one
        logs.sort(reverse=True, key=lambda x: x[0])
        latest_logs.append(logs[0][1])

    return latest_logs


def generate_results_csv(log_files, output_path):
    """Generate detailed results.csv from eval logs"""

    results = []

    # Deduplicate: keep only the most recent log file per model
    log_files = get_latest_logs_per_model(log_files)

    for log_file in sorted(log_files):
        try:
            log = read_eval_log(str(log_file))

            if not log.samples or not log.results:
                continue

            provider_id = extract_provider_id_from_log(log)

            for sample in log.samples:
                # Extract basic info
                test_id = f"test-{sample.id}"
                prompt = sample.input[:200] if sample.input else ""  # Truncate long prompts

                # Get model output
                result = ""
                if sample.output and sample.output.choices:
                    result = sample.output.choices[0].message.content or ""

                # Get error if any
                error = ""

                # Get score/passed status
                passed = ""
                expected = sample.target or ""

                if sample.scores:
                    # Get the first score
                    score = list(sample.scores.values())[0]
                    if score.value is not None:
                        # Handle different score formats:
                        # - String: 'C' (correct) or 'I' (incorrect)
                        # - Numeric: 1.0 (pass) or 0.0 (fail)
                        if isinstance(score.value, str):
                            passed = "True" if score.value == 'C' else "False"
                        elif isinstance(score.value, (int, float)):
                            passed = "True" if score.value == 1.0 else "False"
                        else:
                            passed = "False"

                # Duration (not available in migrated logs)
                duration_ms = None

                # Timestamp
                timestamp = log.eval.created

                results.append({
                    'provider_id': provider_id,
                    'prompt_id': 'prompt-1',
                    'test': test_id,
                    'prompt': prompt,
                    'result': result,
                    'error': error,
                    'duration_ms': duration_ms,
                    'passed': passed,
                    'expected': expected,
                    'timestamp': timestamp,
                })

        except Exception as e:
            print(f"Warning: Could not process {log_file}: {e}")
            continue

    # Write results.csv
    if results:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['provider_id', 'prompt_id', 'test', 'prompt', 'result',
                         'error', 'duration_ms', 'passed', 'expected', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"✅ Generated {output_path} with {len(results)} results")
    else:
        print(f"⚠️  No results found")

    return results


def generate_aggregate_csv(log_files, output_path):
    """Generate aggregated aggregate.csv from eval logs"""

    aggregates = defaultdict(lambda: {
        'total_tests': 0,
        'assertions': 0,
        'passed': 0,
        'failed': 0,
        'no_assertions': 0,
        'errors': 0,
        'durations': []
    })

    # Deduplicate: keep only the most recent log file per model
    log_files = get_latest_logs_per_model(log_files)

    for log_file in sorted(log_files):
        try:
            log = read_eval_log(str(log_file))

            if not log.results:
                continue

            provider_id = extract_provider_id_from_log(log)
            key = (provider_id, 'prompt-1')

            agg = aggregates[key]

            # Get stats from results
            if log.results.scores:
                score_data = log.results.scores[0]
                agg['total_tests'] += log.results.total_samples
                agg['assertions'] += score_data.scored_samples

                # Calculate passed/failed from accuracy
                if score_data.metrics and 'accuracy' in score_data.metrics:
                    accuracy = score_data.metrics['accuracy'].value
                    passed_count = int(accuracy * score_data.scored_samples)
                    agg['passed'] += passed_count
                    agg['failed'] += score_data.scored_samples - passed_count

                agg['no_assertions'] += score_data.unscored_samples or 0

        except Exception as e:
            print(f"Warning: Could not aggregate {log_file}: {e}")
            continue

    # Convert to list and calculate rates
    aggregate_rows = []
    for (provider_id, prompt_id), agg in sorted(aggregates.items()):
        pass_rate = round(agg['passed'] / agg['assertions'] * 100, 1) if agg['assertions'] > 0 else 0
        avg_duration = round(sum(agg['durations']) / len(agg['durations']), 1) if agg['durations'] else None
        min_duration = min(agg['durations']) if agg['durations'] else None
        max_duration = max(agg['durations']) if agg['durations'] else None

        aggregate_rows.append({
            'provider_id': provider_id,
            'prompt_id': prompt_id,
            'total_tests': agg['total_tests'],
            'assertions': agg['assertions'],
            'passed': agg['passed'],
            'failed': agg['failed'],
            'no_assertions': agg['no_assertions'],
            'pass_rate': pass_rate,
            'errors': agg['errors'],
            'avg_duration_ms': avg_duration,
            'min_duration_ms': min_duration,
            'max_duration_ms': max_duration,
        })

    # Write aggregate.csv
    if aggregate_rows:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['provider_id', 'prompt_id', 'total_tests', 'assertions',
                         'passed', 'failed', 'no_assertions', 'pass_rate', 'errors',
                         'avg_duration_ms', 'min_duration_ms', 'max_duration_ms']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_rows)

        print(f"✅ Generated {output_path} with {len(aggregate_rows)} aggregate rows")
    else:
        print(f"⚠️  No aggregate data found")


def main():
    parser = argparse.ArgumentParser(
        description="Export Inspect log results to CSV"
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

    # Find eval directory
    eval_dir = Path("src/evals") / args.eval_name
    if not eval_dir.exists():
        print(f"❌ Error: Eval directory not found: {eval_dir}")
        return 1

    # Create results directory
    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Find log files
    logs_dir = Path("logs")
    eval_task_name = args.eval_name

    if args.log_file:
        log_files = [Path(args.log_file)]
    else:
        log_files = list(logs_dir.glob(f"*{eval_task_name}*.eval"))

    if not log_files:
        print(f"❌ Error: No log files found for eval '{args.eval_name}' (task: {eval_task_name})")
        print(f"    Looked in: {logs_dir}")
        return 1

    print(f"🔄 Extracting results for {args.eval_name}...")
    print(f"   Found {len(log_files)} log files")
    print("=" * 70)

    # Generate results.csv
    results_path = results_dir / "results.csv"
    generate_results_csv(log_files, results_path)

    # Generate aggregate.csv
    aggregate_path = results_dir / "aggregate.csv"
    generate_aggregate_csv(log_files, aggregate_path)

    print("=" * 70)
    print("✨ Extraction complete!")
    print(f"\n💡 Files generated:")
    print(f"   {results_path}")
    print(f"   {aggregate_path}")

    return 0


if __name__ == "__main__":
    exit(main())
