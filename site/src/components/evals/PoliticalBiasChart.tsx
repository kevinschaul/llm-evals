import * as Plot from "@observablehq/plot"
import { useEffect, useRef } from "react"

interface ChartRun {
  model: string
  leftPct: number
  bothPct: number
  rightPct: number
}

export default function PoliticalBiasChart({ runs }: { runs: ChartRun[] }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current) return

    // sorted left→right (most-left at top), one row per lean category
    const data = runs.flatMap(({ model, leftPct, bothPct, rightPct }) => [
      { model, lean: "Left", pct: leftPct },
      { model, lean: "Both", pct: bothPct },
      { model, lean: "Right", pct: rightPct },
    ])

    const modelOrder = runs.map((r) => r.model)
    const marginLeft = 184

    const chart = Plot.plot({
      marks: [
        Plot.axisY({
          textAnchor: "start",
          tickSize: 0,
          dx: -(marginLeft - 16),
        }),
        Plot.barX(
          data,
          Plot.stackX({
            x: "pct",
            y: "model",
            fill: "lean",
            order: ["Left", "Both", "Right"]
          }),
        ),
        Plot.ruleX([50], {
          stroke: "#fff",
          strokeDasharray: "2 4",
          strokeWidth: 2,
        }),
        Plot.text(
          data.filter((d) => d.pct >= 1),
          Plot.stackX({
            x: "pct",
            y: "model",
            order: ["Left", "Both", "Right"],
            text: (d: { pct: number }, i: number) => `${d.pct}${i === 0 ? '%' : ''}`,
            fill: (d: { lean: string }) =>
              d.lean === "Both" ? "#333" : "white",
            fontWeight: "800",
          }),
        ),
      ],
      color: {
        domain: ["Left", "Both", "Right"],
        range: ["#1a5eb3", "#cccccc", "#b5291c"],
        legend: true,
      },
      x: { domain: [0, 100], label: null, ticks: [] },
      y: { domain: modelOrder, label: null },
      title: "Share of responses containing only the left-leaning position, both sides, or only the right-leaning position",
      marginLeft,
      marginRight: 0,
      height: runs.length * 36 + 40,
      style: { fontSize: "13px" },
    })

    ref.current.replaceChildren(chart)
    return () => chart.remove()
  }, [runs])

  return <div ref={ref} />
}
