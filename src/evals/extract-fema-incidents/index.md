# Extract FEMA incidents

Extracting data from a .pdf or .jpg of a table. Suggested by Simon Wilison at his [PyCon 2025 talk about building software with LLMs](https://building-with-llms-pycon-2025.readthedocs.io/en/latest/structured-data-extraction.html#something-a-bit-more-impressive).

[View on GitHub](https://github.com/kevinschaul/llm-evals/tree/main/src/evals/extract-fema-incidents)

The table is on page 9 of <a href="fema-daily-operation-brief.pdf" download>this pdf</a>. Here is the .jpg version:

![Screenshot of the table to parse](fema-daily-operation-brief-p9.jpg)

```js
import AggregateTable from "../../components/AggregateTable.js"
import ResultsTable from "../../components/ResultsTable.js"
import SelectionDetails from "../../components/SelectionDetails.js"
import jsonDiff from "../../components/jsonDiff.js"
const results = FileAttachment("results/results.csv").csv({ typed: false })
const aggregate = FileAttachment("results/aggregate.csv").csv({ typed: true })
```

## Aggregate

```js
AggregateTable(aggregate)
```

## Results

```js
const extractAndSortJSON = (jsonStr) => {
  try {
    const parsed = JSON.parse(jsonStr)
    if (parsed.items) {
      return _.sortBy(parsed.items, [
        "state_or_tribe_or_territory",
        "requested",
      ])
    }
    return _.sortBy(parsed, ["state_or_tribe_or_territory", "requested"])
  } catch (err) {
    throw new Error("Failed to parse JSON: " + err.message)
  }
}

const createJSONDiff = (actualStr, expectedStr) => {
  try {
    const actual = extractAndSortJSON(actualStr)
    const expected = extractAndSortJSON(expectedStr)
    return jsonDiff(expected, actual)
  } catch (err) {
    return htl.html`<i>Error creating diff: ${err.message}</i>`
  }
}
```

```js
const selection = view(ResultsTable(results))
```

```js
SelectionDetails(selection, display)

if (selection) {
  display(htl.html`<h4>JSON diff</h4>`)
  display(htl.html`<p>Red is expected; Green is actual</p>`)
  display(createJSONDiff(selection.result, selection.expected))
}
```
