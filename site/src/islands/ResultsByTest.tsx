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
  if (sample.passed === null) return null
  return sample.passed ? (
    <span className="badge badge-pass">Pass</span>
  ) : (
    <span className="badge badge-fail">Fail</span>
  )
}

function ModelCard({ run, sample }: ModelResult) {
  const notes = Object.entries(sample.scores).filter(
    ([, s]) => s.explanation,
  )
  return (
    <div className="compare-card">
      <div className="compare-card-head">
        <strong>{run.provider_id}</strong>
        {passBadge(sample)}
        <span className="meta">{formatTokens(sample.output_tokens)}</span>
      </div>
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
  )
}

function TestBlock({ row, defaultOpen }: { row: TestRow; defaultOpen: boolean }) {
  // Render the (potentially many) output cards only once expanded
  const [open, setOpen] = useState(defaultOpen)
  const { test, results, passed, scored } = row

  return (
    <details
      className="test-block"
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        {scored > 0 && (
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
        )}
        <span className="test-block-input" title={test.input}>
          {test.input || test.id}
        </span>
      </summary>
      {open && (
        <div className="test-block-body">
          <div className="compare-context">
            <p>
              <strong>Input:</strong> {test.input}
            </p>
            {test.expected && (
              <p>
                <strong>Looking for:</strong> {test.expected}
              </p>
            )}
          </div>
          <div className="compare-grid">
            {results.map((r) => (
              <ModelCard key={r.run.provider_id} {...r} />
            ))}
          </div>
        </div>
      )}
    </details>
  )
}

const INITIAL_BLOCKS = 12

export default function ResultsByTest({ url }: { url: string }) {
  const { data, error } = useResults(url)
  const [showAll, setShowAll] = useState(false)

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

  // Small evals are meant to be read with your eyes — open everything.
  // Larger ones start with just the first test open, longest ones
  // truncated behind a show-all button.
  const visible = showAll ? rows : rows.slice(0, INITIAL_BLOCKS)
  return (
    <div className="test-blocks">
      {visible.map((row, i) => (
        <TestBlock
          key={row.test.id}
          row={row}
          defaultOpen={i === 0 || rows.length <= 8}
        />
      ))}
      {rows.length > visible.length && (
        <button className="show-all-tests" onClick={() => setShowAll(true)}>
          Show all {rows.length} tests
        </button>
      )}
    </div>
  )
}
