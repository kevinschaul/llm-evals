import { useMemo, useState } from "react"
import {
  formatDuration,
  formatPercent,
  sortRunsByPassRate,
  type Run,
  type SampleResult,
  type TestCase,
} from "../lib/types"
import { useResults } from "./useResults"
import DetailDrawer, { type DrawerField } from "./DetailDrawer"

interface Selection {
  run: Run
  test: TestCase
  sample: SampleResult
}

function cellGlyph(sample: SampleResult): string {
  if (sample.passed === true) return "✓"
  if (sample.passed === false) return "✗"
  return "·"
}

function cellClass(sample: SampleResult): string {
  if (sample.passed === true) return "pass"
  if (sample.passed === false) return "fail"
  return "none"
}

function selectionFields(sel: Selection): DrawerField[] {
  const fields: DrawerField[] = [
    { label: "Input", value: sel.test.input },
    { label: "Expected", value: sel.test.expected },
    { label: "Output", value: sel.sample.output },
  ]
  for (const [name, score] of Object.entries(sel.sample.scores)) {
    fields.push({
      label: `Score: ${name}`,
      value: [`value: ${JSON.stringify(score.value)}`, score.explanation]
        .filter(Boolean)
        .join("\n"),
    })
  }
  if (sel.sample.duration_ms !== null) {
    fields.push({
      label: "Duration",
      value: formatDuration(sel.sample.duration_ms),
    })
  }
  return fields
}

export default function ResultsTable({ url }: { url: string }) {
  const { data, error } = useResults(url)
  const [selected, setSelected] = useState<Selection | null>(null)

  const runs = useMemo(
    () => (data ? sortRunsByPassRate(data.runs) : []),
    [data],
  )
  const samplesByRun = useMemo(() => {
    const m = new Map<Run, Map<string, SampleResult>>()
    for (const run of runs) {
      m.set(run, new Map(run.samples.map((s) => [s.id, s])))
    }
    return m
  }, [runs])

  if (error) return <p>Failed to load results: {error}</p>
  if (!data) return <p>Loading results…</p>

  return (
    <>
      <div className="results-wrap breakout">
        <table className="results-grid">
          <thead>
            <tr>
              <th></th>
              {runs.map((run) => (
                <th
                  key={run.provider_id}
                  className="results-model"
                  title={`${run.provider_id} — ${formatPercent(run.pass_rate)}`}
                >
                  {run.provider_id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.tests.map((test) => (
              <tr key={test.id}>
                <td className="results-test" title={test.input}>
                  {test.input || test.id}
                </td>
                {runs.map((run) => {
                  const sample = samplesByRun.get(run)?.get(test.id)
                  const isSelected =
                    selected?.run === run && selected?.test === test
                  return (
                    <td key={run.provider_id}>
                      {sample ? (
                        <button
                          className={`results-cell ${cellClass(sample)} ${isSelected ? "selected" : ""}`}
                          aria-label={`${run.provider_id} on ${test.id}: ${sample.passed === null ? "unscored" : sample.passed ? "pass" : "fail"}`}
                          onClick={() =>
                            setSelected(
                              isSelected ? null : { run, test, sample },
                            )
                          }
                        >
                          {cellGlyph(sample)}
                        </button>
                      ) : null}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected && (
        <DetailDrawer
          title={`${selected.run.provider_id} · ${selected.test.id}`}
          badge={
            selected.sample.passed === null
              ? { label: "unscored", kind: "none" }
              : selected.sample.passed
                ? { label: "pass", kind: "pass" }
                : { label: "fail", kind: "fail" }
          }
          fields={selectionFields(selected)}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  )
}
