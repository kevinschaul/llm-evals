# Political Neutrality

Evaluates whether LLMs provide equally helpful responses across opposing political stances. Based on [Anthropic's political-neutrality-eval](https://github.com/anthropics/political-neutrality-eval).

This evaluation contains 1,350 pairs of prompts representing opposing political positions across 60 broad categories (education policy, police reform, abortion, political figures, etc.). Each pair tests whether a model responds with similar quality and engagement regardless of the political stance requested.

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/political-neutrality)

```js
import AggregateTable from "../../components/AggregateTable.js"
import ResultsTable from "../../components/ResultsTable.js"
import SelectionDetails from "../../components/SelectionDetails.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

## About This Evaluation

The test dataset includes six task categories:

- **Reasoning**: "Argue that..." and "Explain why some believe..."
- **Formal writing**: "Write a persuasive essay..."
- **Narratives**: Creative storytelling
- **Analysis**: "Evaluate the evidence..."
- **Opinion**: Opinion-based requests
- **Humor**: Comedy and satire

The original Anthropic evaluation measures three dimensions:

1. **Even-handedness**: Whether responses are equally helpful across opposing stances
2. **Refusals**: Level of engagement vs. declining to answer
3. **Hedging**: Amount of argumentative hedging (acknowledgment of counterarguments)

**Note**: This implementation collects model responses without automated scoring. Full evaluation would require an LLM-as-judge approach to compare response quality across paired prompts.

## Aggregate

```js
AggregateTable(aggregate)
```

## Results

```js
const selection = view(ResultsTable(results))
```

```js
SelectionDetails(selection, display)
```
