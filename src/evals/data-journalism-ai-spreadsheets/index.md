# Data journalism: consolidate federal AI use case spreadsheets

Inspired by [How I used Claude Code in a real data journalism project](https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/). The agent gets a directory of (synthetic) federal-agency AI use case inventories in different formats and column names, and is asked to consolidate them into a single normalized CSV.

**Starting state:** [`fixture/`](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/data-journalism-ai-spreadsheets/fixture) — four inventories from four (fictional) agencies, in CSV, TSV, and JSON.

**Expected output:** `consolidated_use_cases.csv` at the root of the working directory with columns `agency,name,description,status,contact,year_deployed`, status normalized to `Pilot`/`Production`/`Retired`, sorted by agency then name.

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/data-journalism-ai-spreadsheets)

```js
import { table } from "npm:@observablehq/inputs"
import highlight from "npm:highlight.js"
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

## Git diff

```js
function parseDiffSummary(diffString) {
  const lines = diffString.split("\n")
  const files = []
  let currentFile = null
  let additions = 0
  let deletions = 0

  for (const line of lines) {
    if (line.startsWith("diff --git")) {
      if (currentFile) files.push(currentFile)
      const match = line.match(/diff --git a\/(.*?) b\/(.*)/)
      currentFile = {
        name: match ? match[2] : "unknown",
        additions: 0,
        deletions: 0,
      }
    } else if (line.startsWith("+") && !line.startsWith("+++")) {
      if (currentFile) currentFile.additions++
      additions++
    } else if (line.startsWith("-") && !line.startsWith("---")) {
      if (currentFile) currentFile.deletions++
      deletions++
    }
  }

  if (currentFile) files.push(currentFile)

  return {
    files,
    totalAdditions: additions,
    totalDeletions: deletions,
    filesChanged: files.length,
  }
}
```

```js
if (selection) {
  const result = selection.result
  if (result) {
    const summary = parseDiffSummary(result)

    display(html`
      <div>
        <strong>Summary:</strong>
        ${summary.filesChanged} file(s) changed,
        <span>+${summary.totalAdditions}</span>
        insertions,
        <span>-${summary.totalDeletions}</span> deletions
        <details style="margin-top: 10px;">
          <summary>Files</summary>
          <ul>
            ${summary.files.map(
              (f) => html`
                <li>
                  ${f.name}:
                  <span>+${f.additions}</span>
                  <span>-${f.deletions}</span>
                </li>
              `,
            )}
          </ul>
        </details>
      </div>
    `)

    const highlighted = highlight.highlight(result, { language: "diff" }).value
    const pre = html`<pre><code></code></pre>`
    pre.querySelector("code").innerHTML = highlighted
    display(pre)
  } else {
    display(html`<p><em>No result available for this selection</em></p>`)
  }
} else {
  display(html`<p><em>Select a row above to see the diff</em></p>`)
}
```
