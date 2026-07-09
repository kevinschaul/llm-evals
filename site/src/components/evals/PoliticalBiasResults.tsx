import "./political-bias.css"
import ResultsByTest from "../../islands/ResultsByTest"
import type { ModelResult } from "../../lib/resultsTable"
import type { SampleResult } from "../../lib/types"
import { leanOf, leanPercents, LEAN_LABELS, LEANS } from "./lean"

// Wrapper island: render functions can't cross the Astro island boundary,
// so the lean-flavored cells are closed over here instead.
function LeanBadge({ sample }: { sample: SampleResult }) {
  const lean = leanOf(sample)
  return lean ? (
    <span className={`badge badge-lean-${lean}`}>{LEAN_LABELS[lean]}</span>
  ) : (
    <>–</>
  )
}

function LeanBar({ results }: { results: ModelResult[] }) {
  const pcts = leanPercents(results.map((r) => r.sample))
  if (!pcts) return <>–</>
  return (
    <span className="lean-bar">
      <span className="lean-bar-track">
        {LEANS.map((lean) => (
          <span
            key={lean}
            className={`lean-bar-fill lean-bar-${lean}`}
            style={{ width: `${pcts[lean]}%` }}
            title={`${LEAN_LABELS[lean]} ${pcts[lean]}%`}
          />
        ))}
      </span>
    </span>
  )
}

export default function PoliticalBiasResults({ url }: { url: string }) {
  return (
    <ResultsByTest
      url={url}
      summaryHeader="Lean"
      resultHeader="Lean"
      renderSummaryCell={(results) => <LeanBar results={results} />}
      renderResultCell={(sample) => <LeanBadge sample={sample} />}
    />
  )
}
