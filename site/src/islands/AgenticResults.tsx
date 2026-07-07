import { useMemo, useState } from "react"
import {
  formatDuration,
  type CheckResult,
  type Run,
  type SampleResult,
} from "../lib/types"
import { useResults } from "./useResults"
import DiffViewer, { parseDiff } from "./DiffViewer"

interface AgenticRow {
  run: Run
  sample: SampleResult
  key: string
  diff: string
  checks: CheckResult[]
  passed: number
  filesChanged: number
}

export default function AgenticResults({ url }: { url: string }) {
  const { data, error } = useResults(url)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)

  const rows = useMemo(() => {
    if (!data) return []
    const out: AgenticRow[] = []
    for (const run of data.runs) {
      for (const sample of run.samples) {
        const diff = sample.diff ?? ""
        out.push({
          run,
          sample,
          key: `${run.provider_id}|${run.solver}|${sample.id}`,
          diff,
          checks: sample.checks,
          passed: sample.checks.filter((c) => c.passed).length,
          filesChanged: diff ? parseDiff(diff).length : 0,
        })
      }
    }
    // Most checks passed first, then fastest
    return out.sort((a, b) => {
      const ra = a.checks.length ? a.passed / a.checks.length : -1
      const rb = b.checks.length ? b.passed / b.checks.length : -1
      if (rb !== ra) return rb - ra
      return (a.sample.duration_ms ?? Infinity) - (b.sample.duration_ms ?? Infinity)
    })
  }, [data])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  const selected = rows.find((r) => r.key === selectedKey) ?? null
  const hasChecks = rows.some((r) => r.checks.length > 0)

  const toggle = (key: string) =>
    setSelectedKey(selectedKey === key ? null : key)

  return (
    <>
      <table className="runs-table leaderboard">
        <thead>
          <tr>
            <th>Model</th>
            <th>Harness</th>
            {hasChecks && <th className="num">Checks</th>}
            <th className="num">Files changed</th>
            <th className="num">Duration</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.key}
              className={`selectable ${selectedKey === row.key ? "selected" : ""}`}
              role="button"
              tabIndex={0}
              aria-expanded={selectedKey === row.key}
              onClick={() => toggle(row.key)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  toggle(row.key)
                }
              }}
            >
              <td>{row.run.provider_id}</td>
              <td>{row.run.solver}</td>
              {hasChecks && (
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
              )}
              <td className="num">{row.diff ? row.filesChanged : "–"}</td>
              <td className="num">{formatDuration(row.sample.duration_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {selected ? (
        <div>
          <h3>
            {selected.run.provider_id}{" "}
            <span style={{ color: "var(--muted)", fontWeight: "normal" }}>
              via {selected.run.solver || "unknown harness"}
            </span>
          </h3>
          {selected.checks.length > 0 && (
            <ul className="checks-list">
              {selected.checks.map((check) => (
                <li key={check.name}>
                  {check.passed ? "✓" : "✗"} {check.name}
                </li>
              ))}
            </ul>
          )}
          <DiffViewer diff={selected.diff} />
        </div>
      ) : (
        <p style={{ color: "var(--muted)" }}>
          <em>Select a run above to see what the agent changed.</em>
        </p>
      )}
    </>
  )
}
