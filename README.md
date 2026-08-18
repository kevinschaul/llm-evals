# llm-evals

Because we should all have our own set of LLM evals. [Blog post](https://kschaul.com/post/2025/04/10/2025-04-10-your-own-llm-evals/)

[Explore my leaderboard](https://kschaul.com/llm-evals/)

## Installation

```
brew install just gitleaks
just install
```

## Running the evals

Run every active eval against every active model:
```
uv run python run_evals.py
```

Narrow to one model, one eval, or both:
```
uv run python run_evals.py --model anthropic/claude-sonnet-5
uv run python run_evals.py --eval political-bias
uv run python run_evals.py --model anthropic/claude-sonnet-5 --eval political-bias
```

Pairs that already have a result are skipped automatically; add `--force` to rerun them anyway. Add `--dry-run` to see what would run without running it. See `uv run python run_evals.py --help` for all options (including `--solver` for agentic evals, and `--list` to see the active evals/models).

To view the dashboard (the version published at [https://kschaul.com/llm-evals/](https://kschaul.com/llm-evals/)):
```
just dev
```

