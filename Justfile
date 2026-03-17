# Local models via llama-swap on :1112
# Usage: just eval grab-bag openai-api/llama-swap/<model-name>
export LLAMA_SWAP_BASE_URL := "http://127.0.0.1:1112/v1"
export LLAMA_SWAP_API_KEY := "dummy"

# Current models worth testing (as of 2026-03):
#   anthropic/claude-sonnet-4-6          (default - best balance)
#   anthropic/claude-opus-4-6            (most capable, slower/pricier)
#   anthropic/claude-haiku-4-5-20251001  (fastest, cheapest)
#   openai/gpt-4o
#   openai/o3-mini
#   google/gemini-2.0-flash
#   openai-api/llama-swap/<model-name>   (local via llama-swap on :1112)

# To specify a specific openrouter provider:
# openrouter/moonshotai/kimi-k2-thinking -M "provider={'order':['moonshotai']}"

default_model := "anthropic/claude-sonnet-4-6"

default:
    @just --list

# Run all evals against a model. Example: just eval-all anthropic/claude-opus-4-6
eval-all model=default_model *ARGS:
    just eval grab-bag {{model}} {{ARGS}}
    just eval extract-fema-incidents {{model}} {{ARGS}}
    just eval article-tracking-trump {{model}} {{ARGS}}
    just eval article-tracking-categories {{model}} {{ARGS}}
    just eval political-fundraising-emails {{model}} {{ARGS}}

# Run a single eval against a model. Example: just eval grab-bag
eval name model=default_model *ARGS:
    uv run inspect eval src/evals/{{name}}/eval.py --model {{model}} --log-dir logs {{ARGS}}
    just extract {{name}}

# Export results from a single eval to CSVs for Observable dashboard
extract name:
    uv run python extract_results.py {{name}}

# Remove outdated log files, keeping only the most recent per eval+model combination
cleanup-logs:
    uv run python cleanup_old_logs.py --delete

# View Inspect logs in web interface
inspect:
    uv run inspect view start --log-dir logs

# Start Observable Framework dashboard
dev:
    npm run dev
