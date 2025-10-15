#!/usr/bin/env python3
"""
Cleanup old log files from the logs directory.

Keeps only the most recent log file for each eval+model combination,
removing older duplicates to save disk space.

Uses the same logic as extract_results.py to determine which logs to keep.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log


def extract_provider_id_from_log(log):
    """Extract provider ID from the model name (imported from extract_results.py)"""
    model = log.eval.model
    # Try to extract a clean provider ID
    # Examples: "openai/gpt-4o-mini" -> "gpt-4o-mini"
    #          "anthropic/claude-opus-4-0" -> "claude-opus-4-0"
    if '/' in model:
        return model.split('/')[-1]
    return model


def extract_eval_name_from_log(log):
    """Extract eval name from the log task"""
    # The task name is typically the eval name
    return log.eval.task


def find_logs_to_delete(logs_dir: Path) -> list[Path]:
    """
    Find log files that should be deleted.

    For each eval+model combination, keeps only the most recent file based on
    the timestamp stored in the log metadata (same logic as extract_results.py).
    Returns a list of Path objects to delete.
    """
    log_files = list(logs_dir.glob('*.eval'))

    # Group files by eval_name + model
    grouped = defaultdict(list)
    unparseable = []

    for log_file in log_files:
        try:
            log = read_eval_log(str(log_file))
            eval_name = extract_eval_name_from_log(log)
            provider_id = extract_provider_id_from_log(log)
            timestamp = log.eval.created

            key = (eval_name, provider_id)
            grouped[key].append((timestamp, log_file))
        except Exception as e:
            unparseable.append((log_file, str(e)))
            continue

    if unparseable:
        print(f"Warning: {len(unparseable)} files couldn't be parsed:")
        for f, error in unparseable[:5]:  # Show first 5
            print(f"  - {f.name}: {error}")
        if len(unparseable) > 5:
            print(f"  ... and {len(unparseable) - 5} more")

    # For each group, keep only the most recent
    to_delete = []
    for key, files in grouped.items():
        if len(files) <= 1:
            continue

        # Sort by timestamp (newest first)
        files.sort(key=lambda x: x[0], reverse=True)

        # Keep the first (newest), delete the rest
        eval_name, provider_id = key
        print(f"\n{eval_name} / {provider_id}: keeping {files[0][1].name}")
        for timestamp, log_file in files[1:]:
            to_delete.append(log_file)
            print(f"  → deleting {log_file.name}")

    return to_delete


def main():
    parser = argparse.ArgumentParser(
        description='Cleanup old log files, keeping only the most recent per eval+model'
    )
    parser.add_argument(
        '--delete',
        action='store_true',
        help='Actually delete files (default is dry-run)'
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='Directory containing log files (default: logs)'
    )

    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: Log directory '{args.log_dir}' does not exist")
        return 1

    to_delete = find_logs_to_delete(args.log_dir)

    if not to_delete:
        print("No old log files to delete. All logs are the most recent for their eval+model.")
        return 0

    # Calculate size savings
    total_size = sum(f.stat().st_size for f in to_delete)
    size_mb = total_size / (1024 * 1024)

    print(f"Found {len(to_delete)} old log files to delete (saves {size_mb:.1f} MB)")

    if not args.delete:
        print("\nDry run - files that would be deleted:")
        for f in to_delete[:10]:  # Show first 10
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")
        print("\nRun with --delete to actually remove these files")
        return 0

    # Actually delete files
    deleted_count = 0
    for f in to_delete:
        try:
            f.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {f.name}: {e}")

    print(f"Deleted {deleted_count} files (saved {size_mb:.1f} MB)")
    return 0


if __name__ == '__main__':
    exit(main())
