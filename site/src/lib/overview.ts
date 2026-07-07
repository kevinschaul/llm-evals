import type { Run, SampleResult } from "./types"

// Pass rates for the overview must ignore git_diff, which is "C" whenever a
// diff was captured — it marks "the agent ran", not "the agent succeeded".
function hasComparableScore(sample: SampleResult): boolean {
  return Object.keys(sample.scores).some((name) => name !== "git_diff")
}

export function comparableRate(run: Run): number | null {
  for (const [name, metrics] of Object.entries(run.metrics || {})) {
    if (name === "git_diff") continue
    for (const key of ["accuracy", "mean"]) {
      if (metrics[key] !== undefined) return metrics[key]
    }
  }
  const scored = run.samples.filter(
    (s) => s.passed !== null && hasComparableScore(s),
  )
  if (scored.length === 0) return null
  return scored.filter((s) => s.passed).length / scored.length
}

// Fold host variants ("gpt-oss-120b (deepinfra)") into one model row
export function normalizeModel(providerId: string): string {
  return providerId.replace(/\s*\(.+\)$/, "")
}

export interface OverviewCell {
  rate: number | null
  runs: number
}

export interface OverviewRow {
  model: string
  cells: Map<string, OverviewCell>
  covered: number
  mean: number | null
  latest: number // ms since epoch of the model's most recent run
}

export function buildOverviewRows(
  resultsByEval: Map<string, Run[]>,
): OverviewRow[] {
  const rows = new Map<string, OverviewRow>()

  for (const [evalId, runs] of resultsByEval) {
    for (const run of runs) {
      const model = normalizeModel(run.provider_id)
      let row = rows.get(model)
      if (!row) {
        row = { model, cells: new Map(), covered: 0, mean: null, latest: 0 }
        rows.set(model, row)
      }
      row.latest = Math.max(row.latest, Date.parse(run.timestamp) || 0)
      const rate = comparableRate(run)
      const cell = row.cells.get(evalId)
      if (!cell) {
        row.cells.set(evalId, { rate, runs: 1 })
      } else {
        // Same model via multiple harnesses/hosts: average the rates
        cell.runs += 1
        if (rate !== null) {
          cell.rate = cell.rate === null ? rate : (cell.rate * (cell.runs - 1) + rate) / cell.runs
        }
      }
    }
  }

  for (const row of rows.values()) {
    const rates = [...row.cells.values()]
      .map((c) => c.rate)
      .filter((r): r is number => r !== null)
    row.covered = row.cells.size
    row.mean = rates.length
      ? rates.reduce((a, b) => a + b, 0) / rates.length
      : null
  }

  return [...rows.values()].sort((a, b) => {
    if (b.covered !== a.covered) return b.covered - a.covered
    return (b.mean ?? -1) - (a.mean ?? -1)
  })
}

// Models run within this window of the dataset's newest run stay in the main
// table (relative to the data, so the split doesn't rot as the repo ages).
const RECENT_WINDOW_MS = 240 * 24 * 60 * 60 * 1000
const MIN_COVERAGE = 3

export function splitByRelevance(rows: OverviewRow[]): {
  main: OverviewRow[]
  older: OverviewRow[]
} {
  const newest = Math.max(...rows.map((r) => r.latest), 0)
  const cutoff = newest - RECENT_WINDOW_MS
  const main: OverviewRow[] = []
  const older: OverviewRow[] = []
  for (const row of rows) {
    if (row.latest >= cutoff || row.covered >= MIN_COVERAGE) main.push(row)
    else older.push(row)
  }
  return { main, older }
}
