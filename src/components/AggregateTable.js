import sparkBar from "./sparkBar.js"
import { table } from "npm:@observablehq/inputs"

export default function AggregateTable(aggregate) {
  return table(aggregate, {
    columns: [
      "provider_id",
      "prompt_id",
      "assertions",
      "passed",
      "errors",
      "pass_rate",
    ],
    sort: "pass_rate",
    reverse: true,
    format: {
      pass_rate: (d) => sparkBar(1)(d / 100),
    },
    align: {
      pass_rate: "right",
    },
  })
}
