import { useMemo, useState } from "react"
import {
  formatDuration,
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
    <span className="badge badge-pass">pass</span>
  ) : (
    <span className="badge badge-fail">fail</span>
  )
}

// Failures first — they're what you're here to read; passing outputs
// mostly agree with each other
function cardOrder(a: ModelResult, b: ModelResult): number {
  const rank = (s: SampleResult) =>
    s.passed === false ? 0 : s.passed === null ? 1 : 2
  return rank(a.sample) - rank(b.sample)
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
        <span className="meta">{formatDuration(sample.duration_ms)}</span>
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
        {scored > 0 && (
          <span className="dot-strip" aria-hidden="true">
            {results.map(({ run, sample }) => (
              <span
                key={run.provider_id}
                className={`dot ${
                  sample.passed === true
                    ? "pass"
                    : sample.passed === false
                      ? "fail"
                      : "none"
                }`}
                title={`${run.provider_id}: ${
                  sample.passed === null
                    ? "unscored"
                    : sample.passed
                      ? "pass"
                      : "fail"
                }`}
              />
            ))}
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
            {[...results].sort(cardOrder).map((r) => (
              <ModelCard key={r.run.provider_id} {...r} />
            ))}
          </div>
        </div>
      )}
    </details>
  )
}

export default function ResultsByTest({ url }: { url: string }) {
  const { data, error } = useResults(url)

  const rows = useMemo(() => {
    if (!data) return []
    const runs = sortRunsByPassRate(data.runs)
    const out: TestRow[] = data.tests.map((test) => {
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
    // Hardest tests first; unscored (freeform) keep dataset order
    return out.sort((a, b) => {
      const ra = a.scored ? a.passed / a.scored : 2
      const rb = b.scored ? b.passed / b.scored : 2
      return ra - rb
    })
  }, [data])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  // Small evals are meant to be read with your eyes — open everything.
  // Larger ones start with just the hardest test open.
  return (
    <div className="test-blocks">
      {rows.map((row, i) => (
        <TestBlock
          key={row.test.id}
          row={row}
          defaultOpen={i === 0 || rows.length <= 8}
        />
      ))}
    </div>
  )
}
