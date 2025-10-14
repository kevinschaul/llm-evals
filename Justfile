# openai-api/llmstudio/XX
export LMSTUDIO_BASE_URL := "http://127.0.0.1:1234/v1"
export LMSTUDIO_API_KEY := "KEY"

default:
    @just --list

# Run all evals against a model. Example: just eval-all anthropic/claude-3-5-sonnet-20241022
eval-all model *ARGS:
    # just eval grab-bag {{model}} {{ARGS}}
    # TODO errors out
    # just eval extract-fema-incidents {{model}} {{ARGS}}
    just eval article-tracking-trump {{model}} {{ARGS}}
    just eval political-fundraising-emails {{model}} {{ARGS}}

# Run a single eval against a model. Example: just eval grab-bag anthropic/claude-3-5-sonnet-20241022
eval name model *ARGS:
    uv run inspect eval src/evals/{{name}}/eval.py --model {{model}} --log-dir logs {{ARGS}}
    just extract {{name}}

# Export results from a single eval to CSVs for Observable dashboard
extract name:
    uv run python extract_results.py {{name}}

# Remove outdated log files, keeping only the most recent per model
cleanup-logs:
    uv run python cleanup_old_logs.py --delete

# View Inspect logs in web interface
inspect:
    uv run inspect view start --log-dir logs

# Start Observable Framework dashboard
dev:
    npm run dev
