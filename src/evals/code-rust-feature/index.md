# Code: Rust feature implementation

Compare how different agentic coding tools and models implement a rust feature. Works on my [`jump-start-tools`](https://github.com/kevinschaul/jump-start-tools) project.

**Prompt:** Implement the mode=git feature for the "use" subcommand in the jump-start-tools Rust CLI.

**Starting codebase:** https://github.com/kevinschaul/jump-start-tools/tree/de15260b46318c29f6b2435ff068fb9c42fc2806

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/code-rust-feature)

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
    // New file
    if (line.startsWith("diff --git")) {
      if (currentFile) files.push(currentFile)
      const match = line.match(/diff --git a\/(.*?) b\/(.*)/)
      currentFile = {
        name: match ? match[2] : "unknown",
        additions: 0,
        deletions: 0,
      }
    }
    // Count changes
    else if (line.startsWith("+") && !line.startsWith("+++")) {
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
