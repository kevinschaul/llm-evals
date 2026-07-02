# Illustrator: mobile promo chart

An agentic graphics eval based on an Illustrator promo workflow. The starting
file has two artboards: a larger desktop artboard with a political-bias stacked
bar chart and a smaller mobile artboard with placeholder Ukraine-aid art. The
agent must replace the placeholder with a mobile version of the larger chart
while preserving the smaller artboard's bold headline treatment.

The scorer exports before/after JPGs for both artboards and checks the edited
Illustrator file for the required chart text, placeholder removal and headline
style preservation. Set `ILLUSTRATOR_VISION_JUDGE_MODEL` to also run a
multimodal judge over those JPGs.

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
  const checks = parseChecks(r.score_check_illustrator_result_explanation)
  return {
    ...r,
    assertions: checks.total || null,
    passed_checks: checks.passed || null,
    failed_checks: checks.failed || null,
    pass_rate: checks.total > 0 ? checks.passed / checks.total : null,
  }
})

const selection = view(
  table(rows, {
    columns: ["model", "solver", "assertions", "passed_checks", "failed_checks", "pass_rate", "duration_ms"],
    header: { passed_checks: "passed", failed_checks: "failed", pass_rate: "pass rate" },
    format: {
      pass_rate: (d) => (d != null ? `${Math.round(d * 100)}%` : ""),
      duration_ms: (d) => (d != null ? `${Math.round(d)}ms` : ""),
    },
    align: {
      assertions: "right",
      passed_checks: "right",
      failed_checks: "right",
      pass_rate: "right",
      duration_ms: "right",
    },
    layout: "fixed",
    required: false,
    multiple: false,
  }),
)
```

## JPG Export Checks

```js
if (selection) {
  display(html`<pre style="font-size: 13px">${selection.score_check_illustrator_result_explanation || ""}</pre>`)
} else {
  display(html`<p><em>Select a row above to see check details and exported JPG paths.</em></p>`)
}
```

## Vision Judge

```js
if (selection) {
  display(html`<pre style="font-size: 13px">${selection.score_vision_judge_explanation || ""}</pre>`)
}
```

## Git Diff

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
  return { files, totalAdditions: additions, totalDeletions: deletions, filesChanged: files.length }
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
        <span>+${summary.totalAdditions}</span> insertions,
        <span>-${summary.totalDeletions}</span> deletions
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
  display(html`<p><em>Select a row above to see the diff.</em></p>`)
}
```
