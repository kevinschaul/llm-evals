# llm-evals

Because we should all have our own set of LLM evals. [Blog post](https://kschaul.com/post/2025/04/10/2025-04-10-your-own-llm-evals/)

[Explore my leaderboard](https://kschaul.com/llm-evals/)

## Installation

Python stuff:
```
uv sync
```

Node stuff:
```
npm install
```

just:
```
brew install just
```

## Running the evals

Run them all:
```
just eval-all
```

Run a specific one:
```
just eval CONFIG
```
where CONFIG is "social-media-insults" for example.

To view all evals:
```
just view
```

To view the dashboard (the version published at [https://kschaul.com/llm-evals/](https://kschaul.com/llm-evals/)):
```
just view-dashboard
```

