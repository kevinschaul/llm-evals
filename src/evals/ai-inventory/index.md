# Code: AI inventory consolidation

Compare how different agentic coding tools consolidate heterogeneous federal agency AI use case inventories into a single CSV. Inspired by [this Washington Post story](https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/).

**Prompt:** I have downloaded AI use case inventory spreadsheets from several federal agencies. They are in the current directory with different formats and inconsistent column names. Write a Python script to: 1. Read each CSV file in the current directory 2. Examine the column names in each file 3. Map them to these standardized columns: agency, use_case_name, description, development_stage, topic_area 4. Add an 'agency' column identifying which agency the data came from (infer from the filename if not in the data) 5. Consolidate all rows into a single file called consolidated.csv. Then run the script and print a summary of how many rows came from each file.

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/ai-inventory)

```js
import { table } from "npm:@observablehq/inputs"
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
    let parsed = null
    try {
      parsed = d3.csvParse(result)
    } catch (e) {
      parsed = null
    }

    if (parsed && parsed.length > 0) {
      display(html`<p><strong>${parsed.length} rows</strong> in consolidated output</p>`)
      display(
        table(parsed, {
          layout: "auto",
          required: false,
          multiple: false,
        })
      )
    } else {
      display(html`<pre>${result}</pre>`)
    }
  } else {
    display(html`<p><em>No result available for this selection</em></p>`)
  }
} else {
  display(html`<p><em>Select a row above to see the consolidated CSV</em></p>`)
}
```
