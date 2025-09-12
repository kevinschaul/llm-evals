# Article tracking: Trump

Determine whether an article is describing a new action/policy by the Trump administration.

[Read full blog post](https://kschaul.com/post/2025/03/05/2025-03-05-use-llm-to-keep-trackers-updated/)

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/article-tracking-trump)

See also: [categories/](categories/)

```js
import AggregateTable from "../../components/AggregateTable.js"
import ResultsTable from "../../components/ResultsTable.js"
import getSelectionDetailsConfig from "../../components/SelectionDetails.js"
const results = FileAttachment("results/results.csv").csv({ typed: true })
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
