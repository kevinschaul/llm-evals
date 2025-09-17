# Political fundraising emails

A random sample of 100 fundraising emails from [Derek Willis](https://thescoop.org/archives/2025/01/27/llm-extraction-challenge-fundraising-emails/index.html).

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/political-fundraising-emails)

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
