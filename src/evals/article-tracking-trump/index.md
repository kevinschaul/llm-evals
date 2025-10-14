# Article tracking: Trump

Determine whether an article is describing a new action/policy by the Trump administration.

[Read full blog post](https://kschaul.com/post/2025/03/05/2025-03-05-use-llm-to-keep-trackers-updated/)

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/article-tracking-trump)

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
