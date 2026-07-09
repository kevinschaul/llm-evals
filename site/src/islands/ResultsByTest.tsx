import { useMemo, useState, type ReactNode } from "react"
import { formatTokens, type SampleResult } from "../lib/types"
import {
  buildTestRows,
  excerpt,
  type ModelResult,
  type TestRow,
} from "../lib/resultsTable"
import { useResults } from "./useResults"

// The default cells show pass/fail; evals scored some other way wrap this
// component in their own island and pass custom renderers (islands can't
// receive functions from Astro, so the wrapper must be the hydration root —
// see components/evals/PoliticalBiasResults.tsx).
interface Props {
  url: string
  summaryHeader?: string
  resultHeader?: string
  renderSummaryCell?: (results: ModelResult[]) => ReactNode
  renderResultCell?: (sample: SampleResult) => ReactNode
}

function passBadge(sample: SampleResult): ReactNode {
  if (sample.passed === null) return "–"
  return sample.passed ? (
    <span className="badge badge-pass">Pass</span>
  ) : (
    <span className="badge badge-fail">Fail</span>
  )
}

function passSummaryBadge(results: ModelResult[]): ReactNode {
  const scored = results.filter((r) => r.sample.passed !== null)
  if (scored.length === 0) return "–"
  const passed = scored.filter((r) => r.sample.passed).length
  return (
    <span
      className={`badge ${
        passed === scored.length
          ? "badge-pass"
          : passed === 0
            ? "badge-fail"
            : ""
      }`}
    >
      {passed}/{scored.length}
    </span>
  )
}

function ModelResultRow({
  run,
  sample,
  renderResultCell,
}: ModelResult & { renderResultCell: (sample: SampleResult) => ReactNode }) {
  // Render the (potentially long) output only once expanded
  const [open, setOpen] = useState(false)
  const notes = Object.entries(sample.scores).filter(([, s]) => s.explanation)
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
        <td className="num">{renderResultCell(sample)}</td>
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

function TestRowBlock({
  row,
  resultHeader,
  renderSummaryCell,
  renderResultCell,
}: {
  row: TestRow
  resultHeader: string
  renderSummaryCell: (results: ModelResult[]) => ReactNode
  renderResultCell: (sample: SampleResult) => ReactNode
}) {
  // Render the (potentially many) output cards only once expanded
  const [open, setOpen] = useState(false)
  const { test, results } = row

  return (
    <>
      <tr className="results-toggle-row" onClick={() => setOpen(!open)}>
        <td className="cell-truncate">
          <span className="row-caret">{open ? "▾" : "▸"}</span>
          {test.input || test.id}
        </td>
        <td className="num">{renderSummaryCell(results)}</td>
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
                      <th className="num">{resultHeader}</th>
                      <th className="num">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r) => (
                      <ModelResultRow
                        key={r.run.provider_id}
                        {...r}
                        renderResultCell={renderResultCell}
                      />
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

export default function ResultsByTest({
  url,
  summaryHeader = "Pass rate",
  resultHeader = "Pass/Fail",
  renderSummaryCell = passSummaryBadge,
  renderResultCell = passBadge,
}: Props) {
  const { data, error } = useResults(url)

  const rows = useMemo(() => (data ? buildTestRows(data) : []), [data])

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
          <th className="num">{summaryHeader}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <TestRowBlock
            key={row.test.id}
            row={row}
            resultHeader={resultHeader}
            renderSummaryCell={renderSummaryCell}
            renderResultCell={renderResultCell}
          />
        ))}
      </tbody>
    </table>
  )
}
