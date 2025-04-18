# Article tracking: Trump

Determine whether an article is describing a new action/policy by the Trump administration.

[Read full blog post](https://kschaul.com/post/2025/03/05/2025-03-05-use-llm-to-keep-trackers-updated/)

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/article-tracking-trump)

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
const selection = view(
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
