import * as htl from "htl"
import { table } from "npm:@observablehq/inputs"

export default function ResultsTable(results) {
  return table(results, {
    columns: [
      "provider_id",
      "prompt_id",
      "test",
      "passed",
      "result",
      "expected",
      "error",
      "duration_ms",
    ],
    format: {
      passed: (d) => {
        // Handle tri-state: true, false, or null/undefined (no assertions)
        if (d === true || d === "True" || d === "true") {
          return htl.html`<div style="background: #d5edca; padding: 2px 6px; text-align: center;">✔</div>`
        } else if (d === false || d === "False" || d === "false") {
          return htl.html`<div style="background: #f9dddb; padding: 2px 6px; text-align: center;">✗</div>`
        } else {
          // null/undefined - no assertions
          return htl.html`<div style="background: #e3f2fd; padding: 2px 6px; text-align: center;">?</div>`
        }
      },
      duration_ms: (d) => `${d}ms`,
      result: (d) =>
        d && d.length > 0
          ? d.substring(0, 50) + (d.length > 50 ? "..." : "")
          : "",
      error: (d) =>
        d && d.length > 0
          ? d.substring(0, 50) + (d.length > 50 ? "..." : "")
          : "",
    },
    align: {
      passed: "center",
      duration_ms: "right",
    },
    widths: {
      result: 200,
      error: 200,
    },
    layout: "fixed",
    required: false,
    multiple: false,
  })
}
