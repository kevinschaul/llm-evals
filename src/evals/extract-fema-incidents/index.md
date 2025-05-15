# Extract FEMA incidents

Extracting data from a .pdf or .jpg of a table. Suggested by Simon Wilison at his [PyCon 2025 talk about building software with LLMs](https://building-with-llms-pycon-2025.readthedocs.io/en/latest/structured-data-extraction.html#something-a-bit-more-impressive).

The table is on page 9 of [this pdf](fema-daily-operation-brief.pdf). Here is the .jpg version:

![Screenshot of the table to parse](fema-daily-operation-brief-p9.jpg)

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
    share_correct: "center",
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
