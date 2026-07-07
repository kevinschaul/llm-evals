import { useMemo, useState } from "react"
import {
  formatDuration,
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
  checks: { passed: number; failed: number; lines: string[] }
  filesChanged: number
}

function parseChecks(sample: SampleResult) {
  const lines: string[] = []
  for (const [name, score] of Object.entries(sample.scores)) {
    if (name === "git_diff" || !score.explanation) continue
    lines.push(
      ...score.explanation.split("\n").filter((l) => /^[✓✗]/.test(l.trim())),
    )
  }
  return {
    passed: lines.filter((l) => l.trim().startsWith("✓")).length,
    failed: lines.filter((l) => l.trim().startsWith("✗")).length,
    lines,
  }
}

export default function AgenticResults({ url }: { url: string }) {
  const { data, error } = useResults(url)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)

  const rows = useMemo(() => {
    if (!data) return []
    const out: AgenticRow[] = []
    for (const run of data.runs) {
      for (const sample of run.samples) {
        const diff = (sample.scores.git_diff?.explanation as string) || ""
        out.push({
          run,
          sample,
          key: `${run.provider_id}|${run.solver}|${sample.id}`,
          diff,
          checks: parseChecks(sample),
          filesChanged: diff ? parseDiff(diff).length : 0,
        })
      }
    }
    // Most checks passed first, then fastest
    return out.sort((a, b) => {
      const ra = a.checks.passed + a.checks.failed
        ? a.checks.passed / (a.checks.passed + a.checks.failed)
        : -1
      const rb = b.checks.passed + b.checks.failed
        ? b.checks.passed / (b.checks.passed + b.checks.failed)
        : -1
      if (rb !== ra) return rb - ra
      return (a.sample.duration_ms ?? Infinity) - (b.sample.duration_ms ?? Infinity)
    })
  }, [data])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  const selected = rows.find((r) => r.key === selectedKey) ?? null
  const hasChecks = rows.some((r) => r.checks.passed + r.checks.failed > 0)

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
              onClick={() =>
                setSelectedKey(selectedKey === row.key ? null : row.key)
              }
            >
              <td>{row.run.provider_id}</td>
              <td>{row.run.solver}</td>
              {hasChecks && (
                <td className="num">
                  {row.checks.passed + row.checks.failed > 0 ? (
                    <span
                      className={`badge ${row.checks.failed === 0 ? "badge-pass" : "badge-fail"}`}
                    >
                      {row.checks.passed}/{row.checks.passed + row.checks.failed}
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
          {selected.checks.lines.length > 0 && (
            <ul className="checks-list">
              {selected.checks.lines.map((line, i) => (
                <li key={i}>{line}</li>
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
