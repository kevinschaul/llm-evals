import { useMemo, useState } from "react"
import {
  formatDuration,
  sortRunsByPassRate,
  type SampleResult,
} from "../lib/types"
import { useResults } from "./useResults"

function passBadge(sample: SampleResult) {
  if (sample.passed === null) return null
  return sample.passed ? (
    <span className="badge badge-pass">pass</span>
  ) : (
    <span className="badge badge-fail">fail</span>
  )
}

export default function OutputCompare({ url }: { url: string }) {
  const { data, error } = useResults(url)
  const [testIndex, setTestIndex] = useState(0)

  const runs = useMemo(
    () => (data ? sortRunsByPassRate(data.runs) : []),
    [data],
  )

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  const test = data.tests[testIndex]
  if (!test) return <p>No test cases found.</p>

  return (
    <>
      {data.tests.length > 1 && (
        <div className="compare-picker">
          {data.tests.map((t, i) => (
            <button
              key={t.id}
              className={i === testIndex ? "active" : ""}
              onClick={() => setTestIndex(i)}
              title={t.input}
            >
              {t.id}
            </button>
          ))}
        </div>
      )}
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
        {runs.map((run) => {
          const sample = run.samples.find((s) => s.id === test.id)
          if (!sample) return null
          return (
            <div className="compare-card" key={run.provider_id}>
              <div className="compare-card-head">
                <strong>{run.provider_id}</strong>
                {passBadge(sample)}
                <span className="meta">
                  {formatDuration(sample.duration_ms)}
                </span>
              </div>
              <div className="compare-card-body">
                {sample.output || <em>(no output)</em>}
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}
