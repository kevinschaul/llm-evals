# Social media insults

Whether the given social media post contains an insult. Contributed by [Jeremy B. Merill](https://jeremybmerrill.com/).

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/social-media-insults)

```js
import AggregateTable from "../../components/AggregateTable.js"
import ResultsTable from "../../components/ResultsTable.js"
import getSelectionDetailsConfig from "../../components/SelectionDetails.js"
const results = FileAttachment("results/results.csv").csv({ typed: true })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

## Aggregate

```js
Inputs.table(aggregate, AggregateTable())
```

## Results

```js
const selection = view(Inputs.table(results, ResultsTable()))
```

```js
if (selection) {
  display(htl.html`<h3>Selection details</h3>`)
  const config = getSelectionDetailsConfig()
  const allKeys = Object.keys(selection)
  const testVarKeys = allKeys.filter(k => !config.coreKeys.includes(k) && k !== "prompt")
  const orderedKeys = config.coreKeys.concat(testVarKeys)
  
  for (const key of orderedKeys) {
    if (selection[key] !== undefined && selection[key] !== null && selection[key] !== "") {
      display(
        Inputs.textarea({ 
          label: key, 
          value: String(selection[key]), 
          readonly: true,
          rows: config.getRows(key, selection[key])
        }),
      )
      display(htl.html`<br/>`)
    }
  }
} else {
  display(htl.html`<i>Click a row above to see all details</i>`)
}
```
