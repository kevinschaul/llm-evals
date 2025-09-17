# Social media insults

Whether the given social media post contains an insult. Contributed by [Jeremy B. Merill](https://jeremybmerrill.com/).

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/social-media-insults)

```js
import AggregateTable from "../../components/AggregateTable.js"
import ResultsTable from "../../components/ResultsTable.js"
import SelectionDetails from "../../components/SelectionDetails.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

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
