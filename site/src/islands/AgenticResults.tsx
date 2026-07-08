import { useMemo, useState } from "react"
import {
  formatDuration,
  formatTokens,
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
  files: number | null
}

function RunRow({ row }: { row: AgenticRow }) {
  // Render the (potentially huge) diff only once expanded
  const [open, setOpen] = useState(false)

  return (
    <>
      <tr className="results-toggle-row" onClick={() => setOpen(!open)}>
        <td className="cell-truncate">
          <span className="row-caret">{open ? "▾" : "▸"}</span>
          {row.run.provider_id}
        </td>
        <td className="cell-truncate">
          {row.run.solver || "unknown harness"}
        </td>
        <td className="num">
          {row.checks.length > 0 ? (
            <span
              className={`badge ${row.passed === row.checks.length ? "badge-pass" : "badge-fail"}`}
            >
              {row.passed}/{row.checks.length}
            </span>
          ) : (
            "–"
          )}
        </td>
        <td className="num">{row.files ?? "–"}</td>
        <td className="num">{formatTokens(row.sample.output_tokens)}</td>
        <td className="num">{formatDuration(row.sample.duration_ms)}</td>
      </tr>
      {open && (
        <tr className="results-detail-row">
          <td colSpan={6}>
            <div className="results-detail-body">
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
          </td>
        </tr>
      )}
    </>
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
          files: sample.diff
            ? (sample.diff.match(/^diff --git /gm) || []).length
            : null,
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
    <table className="leaderboard results-table">
      <colgroup>
        <col style={{ width: "32%" }} />
        <col style={{ width: "18%" }} />
        <col style={{ width: "12%" }} />
        <col style={{ width: "12%" }} />
        <col style={{ width: "12%" }} />
        <col style={{ width: "14%" }} />
      </colgroup>
      <thead>
        <tr>
          <th>Model</th>
          <th>Harness</th>
          <th className="num">Checks</th>
          <th className="num">Files</th>
          <th className="num">Tokens</th>
          <th className="num">Duration</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <RunRow key={row.key} row={row} />
        ))}
      </tbody>
    </table>
  )
}
