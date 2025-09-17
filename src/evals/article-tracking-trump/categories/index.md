# Article tracking: Trump - categories

How well do the llm categories match up with mine?

```js
import AggregateTable from "../../../components/AggregateTable.js"
import ResultsTable from "../../../components/ResultsTable.js"
import SelectionDetails from "../../../components/SelectionDetails.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

## Aggregate

```js
Inputs.table(aggregate, AggregateTable(aggregate))
```

## Results

```js
const selection = view(Inputs.table(results, ResultsTable(results)))
```

```js
SelectionDetails(selection, display, Inputs)
```
