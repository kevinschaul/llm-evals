# Political fundraising emails

A random sample of 100 fundraising emails from [Derek Willis](https://thescoop.org/archives/2025/01/27/llm-extraction-challenge-fundraising-emails/index.html).

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
Inputs.table(results, {
  columns: [
    "attributes.model",
    "attributes.prompt",
    "input.body",
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
