# Code: AI inventory search & consolidation

The harder version of the [AI inventory consolidation eval](../ai-inventory/): agents must find, download, and consolidate federal agency AI use case inventories from scratch — no files provided. Inspired by [this Washington Post story](https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/).

**Prompt:** Federal agencies are required by executive order to publish AI use case inventories. Search the web to find inventory spreadsheets from at least 4 agencies, download them, and consolidate into a single `consolidated.csv`.

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/ai-inventory-search)

```js
import { table } from "npm:@observablehq/inputs"
import highlight from "npm:highlight.js"
import * as d3 from "npm:d3"
const results = FileAttachment("results/results.csv").csv({ typed: false })
```

## Results

```js
const selection = view(
  table(results, {
    columns: ["provider_id", "solver", "prompt", "result", "duration_ms"],
    format: {
      duration_ms: (d) => (d != null ? `${Math.round(d)}ms` : ""),
      result: (d) => d ? d.slice(0, 80) + "…" : "",
    },
    align: {
      duration_ms: "right",
    },
    layout: "fixed",
    required: false,
    multiple: false,
  }),
)
```

## Consolidated CSV

```js
if (selection) {
  const result = selection.result
  if (result) {
    const parsed = d3.csvParse(result)
    if (parsed.length > 0) {
      display(table(parsed))
    } else {
      const pre = html`<pre><code></code></pre>`
      pre.querySelector("code").innerHTML = highlight.highlight(result, { language: "csv" }).value
      display(pre)
    }
  } else {
    display(html`<p><em>No output for this run</em></p>`)
  }
} else {
  display(html`<p><em>Select a row above to see the consolidated CSV</em></p>`)
}
```
