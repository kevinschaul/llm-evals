import {
  sortRunsByPassRate,
  type EvalResults,
  type Run,
  type SampleResult,
  type TestCase,
} from "./types"

export interface ModelResult {
  run: Run
  sample: SampleResult
}

export interface TestRow {
  test: TestCase
  results: ModelResult[] // leaderboard order
}

// Tests stay in eval order; models within a test in leaderboard order
export function buildTestRows(results: EvalResults): TestRow[] {
  const runs = sortRunsByPassRate(results.runs)
  return results.tests.map((test) => {
    const rows: ModelResult[] = []
    for (const run of runs) {
      const sample = run.samples.find((s) => s.id === test.id)
      if (sample) rows.push({ run, sample })
    }
    return { test, results: rows }
  })
}

export function excerpt(text: string, len = 90): string {
  const trimmed = text.trim().replace(/\s+/g, " ")
  return trimmed.length > len ? trimmed.slice(0, len) + "…" : trimmed
}
