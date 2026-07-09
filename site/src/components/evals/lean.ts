// Political-bias lean helpers, shared by the chart data prep and the
// custom results-table cells.
import type { SampleResult } from "../../lib/types"

export type Lean = "left" | "right" | "both"

export function leanOf(sample: SampleResult): Lean | "" {
  const v = sample.scores?.llm_judge?.value
  if (v === "left" || v === "right" || v === "both") return v
  return ""
}

export const LEANS: Lean[] = ["left", "both", "right"]

export const LEAN_LABELS: Record<Lean, string> = {
  left: "D",
  both: "B",
  right: "R",
}

// Percentage of samples judged each lean, excluding unparseable ones
export function leanPercents(
  samples: SampleResult[],
): Record<Lean, number> | null {
  const counts: Record<Lean, number> = { left: 0, both: 0, right: 0 }
  let total = 0
  for (const s of samples) {
    const lean = leanOf(s)
    if (!lean) continue
    counts[lean]++
    total++
  }
  if (total === 0) return null
  return {
    left: Math.round((counts.left / total) * 100),
    both: Math.round((counts.both / total) * 100),
    right: Math.round((counts.right / total) * 100),
  }
}
