# Code: Rust feature implementation

Implement the mode=git feature for the "use" subcommand in the jump-start-tools Rust CLI.

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/code-rust-feature)

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

## Git diff

```js
if (selection) {
  const result = selection.result;
  if (result) {
    display(html`<pre><code>${result}</code></pre>`);
  } else {
    display(html`<p><em>No result available for this selection</em></p>`);
  }
} else {
  display(html`<p><em>Select a row above to see the git diff</em></p>`);
}
```