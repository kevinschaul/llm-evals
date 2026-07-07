export interface ScoreEntry {
  value: unknown
  explanation: string | null
}

export interface SampleResult {
  id: string
  output: string
  passed: boolean | null
  duration_ms: number | null
  scores: Record<string, ScoreEntry>
}

export interface TestCase {
  id: string
  input: string
  expected: string
}

export interface Run {
  provider_id: string
  model: string
  full_model: string
  solver: string
  openrouter_provider: string
  timestamp: string
  metrics: Record<string, Record<string, number>>
  samples: SampleResult[]
}

export interface EvalResults {
  eval: string
  generated: string
  tests: TestCase[]
  runs: Run[]
}

export interface RunAggregate {
  total: number
  scored: number
  passed: number
  failed: number
  passRate: number | null
  avgDurationMs: number | null
}

export function aggregateRun(run: Run): RunAggregate {
  const scoredSamples = run.samples.filter((s) => s.passed !== null)
  const passed = scoredSamples.filter((s) => s.passed).length

  // Prefer the run-level accuracy/mean metric (handles partial-credit
  // scorers); fall back to counting per-sample pass/fail.
  let passRate: number | null = null
  for (const metrics of Object.values(run.metrics || {})) {
    for (const key of ["accuracy", "mean"]) {
      if (metrics[key] !== undefined) {
        passRate = metrics[key]
        break
      }
    }
    if (passRate !== null) break
  }
  if (passRate === null && scoredSamples.length > 0) {
    passRate = passed / scoredSamples.length
  }

  const durations = run.samples
    .map((s) => s.duration_ms)
    .filter((d): d is number => d !== null)

  return {
    total: run.samples.length,
    scored: scoredSamples.length,
    passed,
    failed: scoredSamples.length - passed,
    passRate,
    avgDurationMs: durations.length
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : null,
  }
}

export function sortRunsByPassRate(runs: Run[]): Run[] {
  return [...runs].sort((a, b) => {
    const ra = aggregateRun(a).passRate ?? -1
    const rb = aggregateRun(b).passRate ?? -1
    if (rb !== ra) return rb - ra
    return a.provider_id.localeCompare(b.provider_id)
  })
}

export function formatDuration(ms: number | null): string {
  if (ms === null) return ""
  if (ms >= 60000) return `${(ms / 60000).toFixed(1)}m`
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.round(ms)}ms`
}

export function formatPercent(rate: number | null): string {
  return rate === null ? "–" : `${Math.round(rate * 100)}%`
}
