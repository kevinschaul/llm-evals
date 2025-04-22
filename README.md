# llm-evals

Because we should all have our own set of LLM evals. [Blog post](https://kschaul.com/post/2025/04/10/2025-04-10-your-own-llm-evals/)

[Explore my leaderboard](https://kschaul.com/llm-evals/)

## Installation

[llm library](https://llm.datasette.io/en/stable/) and plugins:

```
uv sync
```

[Observable framework](https://observablehq.com/framework/)
```
npm install
```

## Running the evals

```
# Create a new evaluation
uv run cli/cli.py create my-new-eval

# Run an evaluation with a config file
uv run cli/cli.py run src/evals/my-new-eval/llm-evals-config.yaml

# View results in browser
uv run cli/cli.py view
```

