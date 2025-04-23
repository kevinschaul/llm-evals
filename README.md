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

## Running the evals

My evals use [ai-yardstick](https://github.com/kevinschaul/ai-yardstick/)

To run an eval, use the following command pointing to the eval's config file:

```
ai-yardstick run src/evals/article-tracking-trump/ai-yardstick-config.yaml
```

To view the dashboard/results in a browser:

```
npm run dev
```

