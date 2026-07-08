import { useMemo, useState } from "react"
import {
  formatTokens,
  sortRunsByPassRate,
  type Run,
  type SampleResult,
  type TestCase,
} from "../lib/types"
import { useResults } from "./useResults"

interface ModelResult {
  run: Run
  sample: SampleResult
}

interface TestRow {
  test: TestCase
  results: ModelResult[] // leaderboard order
  passed: number
  scored: number
}

function passBadge(sample: SampleResult) {
  if (sample.passed === null) return "–"
  return sample.passed ? (
    <span className="badge badge-pass">Pass</span>
  ) : (
    <span className="badge badge-fail">Fail</span>
  )
}

function excerpt(text: string, len = 90) {
  const trimmed = text.trim().replace(/\s+/g, " ")
  return trimmed.length > len ? trimmed.slice(0, len) + "…" : trimmed
}

function ModelResultRow({ run, sample }: ModelResult) {
  // Render the (potentially long) output only once expanded
  const [open, setOpen] = useState(false)
  const notes = Object.entries(sample.scores).filter(
    ([, s]) => s.explanation,
  )
  return (
    <>
      <tr className="results-toggle-row" onClick={() => setOpen(!open)}>
        <td className="cell-truncate">
          <span className="row-caret">{open ? "▾" : "▸"}</span>
          {run.provider_id}
        </td>
        <td className="cell-truncate">
          {sample.output ? excerpt(sample.output) : <em>(no output)</em>}
        </td>
        <td className="num">{passBadge(sample)}</td>
        <td className="num">{formatTokens(sample.output_tokens)}</td>
      </tr>
      {open && (
        <tr className="results-detail-row">
          <td colSpan={4}>
            <div className="results-detail-body">
              <div className="compare-card-body">
                {sample.output || <em>(no output)</em>}
              </div>
              {notes.length > 0 && (
                <details className="score-note">
                  <summary>score detail</summary>
                  {notes.map(([name, s]) => (
                    <pre key={name}>
                      {name}: {JSON.stringify(s.value)}
                      {"\n"}
                      {s.explanation}
                    </pre>
                  ))}
                </details>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

function TestRowBlock({ row }: { row: TestRow }) {
  // Render the (potentially many) output cards only once expanded
  const [open, setOpen] = useState(false)
  const { test, results, passed, scored } = row

  return (
    <>
      <tr className="results-toggle-row" onClick={() => setOpen(!open)}>
        <td className="cell-truncate">
          <span className="row-caret">{open ? "▾" : "▸"}</span>
          {test.input || test.id}
        </td>
        <td className="num">
          {scored > 0 ? (
            <span
              className={`badge ${
                passed === scored
                  ? "badge-pass"
                  : passed === 0
                    ? "badge-fail"
                    : ""
              }`}
            >
              {passed}/{scored}
            </span>
          ) : (
            "–"
          )}
        </td>
      </tr>
      {open && (
        <tr className="results-detail-row">
          <td colSpan={2}>
            <div className="results-detail-body">
              <div className="compare-context">
                <p>
                  <strong>Input:</strong>{" "}
                  {test.input.length > 500
                    ? test.input.slice(0, 500) + " ..."
                    : test.input}
                </p>
                {test.expected && (
                  <p>
                    <strong>Expected:</strong> {test.expected}
                  </p>
                )}
              </div>
              <p className="results-subhead">Results by model</p>
              <div className="results-table-nested-wrap">
                <table className="leaderboard results-table results-table-nested">
                  <colgroup>
                    <col style={{ width: "28%" }} />
                    <col style={{ width: "37%" }} />
                    <col style={{ width: "15%" }} />
                    <col style={{ width: "20%" }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Result</th>
                      <th className="num">Pass/Fail</th>
                      <th className="num">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => (
                      <ModelResultRow key={r.run.provider_id} {...r} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function ResultsByTest({ url }: { url: string }) {
  const { data, error } = useResults(url)

  // Tests stay in eval order; models within a test in leaderboard order
  const rows = useMemo(() => {
    if (!data) return []
    const runs = sortRunsByPassRate(data.runs)
    return data.tests.map((test) => {
      const results: ModelResult[] = []
      for (const run of runs) {
        const sample = run.samples.find((s) => s.id === test.id)
        if (sample) results.push({ run, sample })
      }
      const scoredResults = results.filter((r) => r.sample.passed !== null)
      return {
        test,
        results,
        scored: scoredResults.length,
        passed: scoredResults.filter((r) => r.sample.passed).length,
      }
    })
  }, [data])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  return (
    <table className="leaderboard results-table">
      <colgroup>
        <col style={{ width: "80%" }} />
        <col style={{ width: "20%" }} />
      </colgroup>
      <thead>
        <tr>
          <th>Test</th>
          <th className="num">Pass rate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <TestRowBlock key={row.test.id} row={row} />
        ))}
      </tbody>
    </table>
  )
}
