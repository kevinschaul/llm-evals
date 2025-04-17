# Trump storylines

```js
const results = FileAttachment(
  "results/results.csv",
).csv({ typed: true })
const aggregate = FileAttachment(
  "results/aggregate.csv",
).csv({ typed: true })
```

## Aggregate

```js
Inputs.table(aggregate, {
  sort: "equals_expected_rate",
  reverse: true,
  format: {
    equals_expected_rate: (x) => (x * 100).toFixed(0) + "%",
  },
})
```

## Results

```js
Inputs.table(results, {
  columns: [
    "attributes.model",
    "attributes.prompt",
    "input.headline",
    "input.content",
    "expected_output",
    "output",
    "assertions_passed_rate",
  ],
  widths: {
    "input.body": 80,
  },
  format: {
    assertions_passed_rate: (x) =>
      x === 1
        ? htl.html`<div style="background: #d5edca;">✔</div>`
        : htl.html`<div style="background: #f9dddb;">✗</div>`,
  },
  align: {
    assertions_passed_rate: "center",
  },
  layout: "fixed",
})
```
