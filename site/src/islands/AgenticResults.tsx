import { useMemo, useState } from "react"
import {
  formatDuration,
  type CheckResult,
  type Run,
  type SampleResult,
} from "../lib/types"
import { useResults } from "./useResults"
import DiffViewer from "./DiffViewer"

interface AgenticRow {
  run: Run
  sample: SampleResult
  key: string
  diff: string
  checks: CheckResult[]
  passed: number
}

function RunBlock({ row, defaultOpen }: { row: AgenticRow; defaultOpen: boolean }) {
  // Render the (potentially huge) diff only once expanded
  const [open, setOpen] = useState(defaultOpen)

  return (
    <details
      className="test-block"
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        {row.checks.length > 0 && (
          <span
            className={`badge ${row.passed === row.checks.length ? "badge-pass" : "badge-fail"}`}
          >
            {row.passed}/{row.checks.length}
          </span>
        )}
        <span className="test-block-input">
          <strong>{row.run.provider_id}</strong> via{" "}
          {row.run.solver || "unknown harness"}
        </span>
        <span className="test-block-meta">
          {formatDuration(row.sample.duration_ms)}
        </span>
      </summary>
      {open && (
        <div className="test-block-body">
          {row.checks.length > 0 && (
            <ul className="checks-list">
              {row.checks.map((check) => (
                <li key={check.name}>
                  {check.passed ? "✓" : "✗"} {check.name}
                </li>
              ))}
            </ul>
          )}
          <DiffViewer diff={row.diff} />
        </div>
      )}
    </details>
  )
}

export default function AgenticResults({ url }: { url: string }) {
  const { data, error } = useResults(url)

  const rows = useMemo(() => {
    if (!data) return []
    const out: AgenticRow[] = []
    for (const run of data.runs) {
      for (const sample of run.samples) {
        out.push({
          run,
          sample,
          key: `${run.provider_id}|${run.solver}|${sample.id}`,
          diff: sample.diff ?? "",
          checks: sample.checks,
          passed: sample.checks.filter((c) => c.passed).length,
        })
      }
    }
    // Leaderboard order: most checks passed, then fastest
    return out.sort((a, b) => {
      const ra = a.checks.length ? a.passed / a.checks.length : -1
      const rb = b.checks.length ? b.passed / b.checks.length : -1
      if (rb !== ra) return rb - ra
      return (a.sample.duration_ms ?? Infinity) - (b.sample.duration_ms ?? Infinity)
    })
  }, [data])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  return (
    <div className="test-blocks">
      {rows.map((row, i) => (
        <RunBlock key={row.key} row={row} defaultOpen={i === 0} />
      ))}
    </div>
  )
}
