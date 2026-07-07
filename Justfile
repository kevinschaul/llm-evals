# openai-api/llama-cpp/XX
export LLAMA_CPP_API_KEY := env_var_or_default("LLAMA_CPP_API_KEY", "KEY")
export LLAMA_CPP_BASE_URL := env_var_or_default("LLAMA_CPP_BASE_URL", "http://box.local:1112/v1")

# Make the project root importable so evals can `from agentic import ...`
export PYTHONPATH := justfile_directory()

# To specify a specific openrouter provider:
# openrouter/moonshotai/kimi-k2-thinking -M "provider={'order':['moonshotai']}"

default:
    @just --list

# Install dependencies and git hooks
install:
    uv sync
    npm install
    uv run pre-commit install

# Run all evals against a model. Example: just eval-all anthropic/claude-3-5-sonnet-20241022
eval-all model *ARGS:
    just eval grab-bag {{model}} {{ARGS}}
    just eval extract-fema-incidents {{model}} {{ARGS}}
    just eval article-tracking-trump {{model}} {{ARGS}}
    just eval article-tracking-categories {{model}} {{ARGS}}
    just eval political-fundraising-emails {{model}} {{ARGS}}

# Run a single eval against a model. Example: just eval grab-bag anthropic/claude-3-5-sonnet-20241022
eval name model *ARGS:
    uv run inspect eval src/evals/{{name}}/eval.py --model {{model}} --log-dir logs {{ARGS}}
    just extract {{name}}

# Export results from a single eval to results.json for the dashboard
extract name:
    uv run python extract_results.py {{name}}

# Remove outdated log files, keeping only the most recent per eval+model combination
cleanup-logs:
    uv run python cleanup_old_logs.py --delete

# View Inspect logs in web interface
inspect:
    uv run inspect view start --log-dir logs

# Start Astro dashboard dev server
dev:
    npm run dev

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
