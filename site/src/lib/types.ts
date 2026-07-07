export interface ScoreEntry {
  value: unknown
  explanation: string | null
}

export interface CheckResult {
  name: string
  passed: boolean
}

export interface SampleResult {
  id: string
  output: string
  passed: boolean | null
  duration_ms: number | null
  scores: Record<string, ScoreEntry>
  diff: string | null
  checks: CheckResult[]
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
  // The single authoritative rate for this run, computed at extraction
  // from the primary scorer (null for diff-only agentic runs)
  pass_rate: number | null
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
  passRate: number | null
  avgDurationMs: number | null
}

export function aggregateRun(run: Run): RunAggregate {
  const durations = run.samples
    .map((s) => s.duration_ms)
    .filter((d): d is number => d !== null)

  return {
    total: run.samples.length,
    passRate: run.pass_rate,
    avgDurationMs: durations.length
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : null,
  }
}

export function sortRunsByPassRate(runs: Run[]): Run[] {
  return [...runs].sort((a, b) => {
    const ra = a.pass_rate ?? -1
    const rb = b.pass_rate ?? -1
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
