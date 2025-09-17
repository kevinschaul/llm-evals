import * as htl from "htl"
import { textarea } from "npm:@observablehq/inputs"

export default function SelectionDetails(selection, display) {
  if (selection) {
    display(htl.html`<h3>Selection details</h3>`)
    const coreKeys = [
      "provider_id",
      "prompt_id",
      "test",
      "passed",
      "result",
      "expected",
      "error",
      "duration_ms",
    ]
    const allKeys = Object.keys(selection)
    const testVarKeys = allKeys.filter(
      (k) => !coreKeys.includes(k) && k !== "__expected",
    )
    const orderedKeys = coreKeys.concat(testVarKeys)

    for (const key of orderedKeys) {
      if (
        selection[key] !== undefined &&
        selection[key] !== null &&
        selection[key] !== ""
      ) {
        display(
          textarea({
            label: key,
            value: String(selection[key]),
            readonly: true,
          }),
        )
        display(htl.html`<br/>`)
      }
    }
  } else {
    display(htl.html`<i>Click a row above to see all details</i>`)
  }
}
