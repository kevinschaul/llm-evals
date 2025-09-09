# NHTSA recalls 

Whether an NHTSA investigation is newsworthy. Contributed by [Jeremy B. Merill](https://jeremybmerrill.com/).

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/nhtsa-recalls)

```js
import sparkBar from "../../components/sparkBar.js"
const results = FileAttachment("results/results.csv").csv({ typed: true })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

## Aggregate

```js
Inputs.table(aggregate, {
  sort: "share_correct",
  reverse: true,
  format: {
    share_correct: sparkBar(1),
  },
  align: {
    share_correct: "right",
  },
})
```

## Results

```js
const resultColumns = Object.keys(results[0]).filter((d) => d.startsWith("["))
const resultsFormatters = resultColumns.reduce((p, v) => {
  p[v] = (x) =>
    x.includes("PASS")
      ? htl.html`<div style="background: #d5edca;">✔</div>`
      : htl.html`<div style="background: #f9dddb;">✗</div>`
  return p
}, {})
const resultsAligns = resultColumns.reduce((p, v) => {
  p[v] = "center"
  return p
}, {})
```

```js
const selection = view(
  Inputs.table(results, {
    widths: {
      "input.body": 80,
    },
    format: resultsFormatters,
    align: resultsAligns,
    layout: "fixed",
    required: false,
    multiple: false,
  }),
)
```

```js
if (selection) {
  display(htl.html`<h3>Selection details</h3>`)
  const keys = Object.keys(selection)
  for (const key of keys) {
    display(
      Inputs.textarea({ label: key, value: selection[key], readonly: true }),
    )
    display(htl.html`<br/>`)
  }
} else {
  display(htl.html`<i>Click a row above to see all details</i>`)
}
```
