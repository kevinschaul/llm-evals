# Code: consolidate spreadsheets

A subset of a real data journalism task described in the blog post [How I used Claude Code in a real data journalism project](https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/). The agent gets a directory of federal-agency AI use case inventories in different formats and column names, and is asked to consolidate them into a single normalized CSV.

```js
import { table } from "npm:@observablehq/inputs"
import highlight from "npm:highlight.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
```

## Results

```js
function parseChecks(explanation) {
  if (!explanation) return { passed: 0, failed: 0, total: 0 }
  const lines = explanation.split("\n")
  const passed = lines.filter((l) => l.startsWith("✓")).length
  const failed = lines.filter((l) => l.startsWith("✗")).length
  return { passed, failed, total: passed + failed }
}

const rows = results.map((r) => {
  const checks = parseChecks(r.score_check_output_explanation)
  return {
    ...r,
    assertions: checks.total || null,
    passed: checks.passed || null,
    failed: checks.failed || null,
    pass_rate: checks.total > 0 ? checks.passed / checks.total : null,
  }
})

const selection = view(
  table(rows, {
    columns: ["model", "solver", "assertions", "passed", "failed", "pass_rate", "duration_ms"],
    header: { pass_rate: "pass rate" },
    format: {
      pass_rate: (d) => (d != null ? `${Math.round(d * 100)}%` : ""),
      duration_ms: (d) => (d != null ? `${Math.round(d)}ms` : ""),
    },
    align: {
      assertions: "right",
      passed: "right",
      failed: "right",
      pass_rate: "right",
      duration_ms: "right",
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
  const parts = [selection.full_model, selection.openrouter_provider].filter(Boolean)
  display(html`<p style="font-size:13px;color:#666;margin-bottom:8px">${parts.join(" · ")}</p>`)
}
```

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
