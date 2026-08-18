# openai-api/llama-cpp/XX
export LLAMA_CPP_API_KEY := env_var_or_default("LLAMA_CPP_API_KEY", "KEY")
export LLAMA_CPP_BASE_URL := env_var_or_default("LLAMA_CPP_BASE_URL", "http://box.local:1112/v1")

# Make the project root importable so evals can `from agentic import ...`
export PYTHONPATH := justfile_directory()

default:
    @just --list

# Install dependencies and git hooks
install:
    uv sync
    npm install
    uv run pre-commit install

# Start Astro dashboard dev server
dev:
    npm run dev

# View Inspect logs in web interface
inspect:
    uv run inspect view start --log-dir logs



# Remove outdated log files, keeping only the most recent per eval+model combination
cleanup-logs:
    uv run python cleanup_old_logs.py --delete

# Scan repo for secrets, including inside .eval zip archives
scan-secrets:
    #!/usr/bin/env bash
    set -euo pipefail
    gitleaks -v dir
    tmpdir=$(mktemp -d)
    trap "rm -rf $tmpdir" EXIT
    for f in logs/*.eval; do
        cp "$f" "$tmpdir/$(basename $f .eval).zip"
    done
    gitleaks -v dir --max-archive-depth 1 "$tmpdir"
