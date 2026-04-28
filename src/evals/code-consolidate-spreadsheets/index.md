# Code: consolidate spreadsheets

A subset of a real data journalism task described in the blog post [How I used Claude Code in a real data journalism project](https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/). The agent gets a directory of federal-agency AI use case inventories in different formats and column names, and is asked to consolidate them into a single normalized CSV.

```js
import { table } from "npm:@observablehq/inputs"
import highlight from "npm:highlight.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
```

## Results

```js
const selection = view(
  table(results, {
    columns: ["provider_id", "solver", "score_check_output", "result", "duration_ms"],
    header: {
      score_check_output: "checks",
    },
    format: {
      duration_ms: (d) => (d != null ? `${Math.round(d)}ms` : ""),
    },
    align: {
      duration_ms: "right",
      score_check_output: "center",
    },
    layout: "fixed",
    required: false,
    multiple: false,
  }),
)
```

## Checks

```js
if (selection) {
  const explanation = selection.score_check_output_explanation
  if (explanation) {
    display(html`<pre style="font-size: 13px">${explanation}</pre>`)
  }
} else {
  display(html`<p><em>Select a row above to see check details</em></p>`)
}
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
