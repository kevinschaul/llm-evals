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
      duration_ms: (d) => (d != null ? `${Math.round(d / 1000)}s` : ""),
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

## Agent Activity & Code Changes

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

    // Score summary card
    display(html`
      <div style="background: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 16px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong style="font-size: 18px;">Grade: ${selection.value || "N/A"}</strong>
            <div style="margin-top: 8px; color: #64748b;">
              ${summary.filesChanged} file(s) changed,
              <span style="color: #059669;">+${summary.totalAdditions}</span> insertions,
              <span style="color: #dc2626;">-${summary.totalDeletions}</span> deletions
            </div>
          </div>
          <div style="text-align: right; color: #64748b;">
            <div>Duration: ${Math.round(selection.duration_ms / 1000)}s</div>
            <div>${summary.filesChanged} files modified</div>
          </div>
        </div>
      </div>
    `)

    // Agent activity
    display(html`
      <h3 style="margin-top: 32px;">Agent Activity</h3>
      <div style="background: #fafafa; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 14px;">
        <div style="margin-bottom: 8px;">✓ Cloned repository at commit ${selection.prompt.includes('de15260') ? 'de15260' : 'specific commit'}</div>
        <div style="margin-bottom: 8px;">✓ Implemented mode=git feature in Rust CLI</div>
        <div style="margin-bottom: 8px;">✓ Modified ${summary.filesChanged} file(s)</div>
        <div style="color: #059669;">✓ Changes captured via git diff</div>
      </div>
    `)

    // Files changed summary
    display(html`
      <h3 style="margin-top: 32px;">Files Changed</h3>
      <div style="background: #f8fafc; padding: 16px; border-radius: 8px;">
        <ul style="list-style: none; padding: 0; margin: 0; font-family: monospace; font-size: 14px;">
          ${summary.files.map(
            (f) => html`
              <li style="padding: 8px; border-bottom: 1px solid #e2e8f0;">
                <strong>${f.name}</strong>
                <span style="float: right;">
                  <span style="color: #059669;">+${f.additions}</span>
                  <span style="color: #dc2626; margin-left: 12px;">-${f.deletions}</span>
                </span>
              </li>
            `,
          )}
        </ul>
      </div>
    `)

    // Full diff
    display(html`<h3 style="margin-top: 32px;">Git Diff</h3>`)

    const highlighted = highlight.highlight(result, { language: "diff" }).value
    const pre = html`<pre style="max-height: 600px; overflow: auto; background: #1e293b; padding: 20px; border-radius: 8px;"><code></code></pre>`
    pre.querySelector("code").innerHTML = highlighted
    display(pre)

    // Technical details
    display(html`
      <details style="margin-top: 16px; padding: 16px; background: #f0fdf4; border-radius: 8px;">
        <summary style="cursor: pointer; font-weight: bold;">Implementation Details</summary>
        <ul style="margin-top: 12px; line-height: 1.8;">
          <li>Repository: <code>jump-start-tools</code></li>
          <li>Starting commit: <code>de15260b46</code></li>
          <li>Feature: Add <code>mode=git</code> support to <code>use</code> subcommand</li>
          <li>Language: Rust</li>
          <li>Total changes: ${summary.totalAdditions + summary.totalDeletions} lines</li>
        </ul>
      </details>
    `)
  } else {
    display(html`<p><em>No result available for this selection</em></p>`)
  }
} else {
  display(html`<p><em>Select a row above to see the code changes and agent activity</em></p>`)
}
```
