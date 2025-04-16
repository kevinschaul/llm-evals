# Political fundraising emails

```js
const results = FileAttachment(
  "data/political-fundraising-emails-results.csv",
).csv({ typed: true })
const aggregate = FileAttachment(
  "data/political-fundraising-emails-aggregate.csv",
).csv({ typed: true })
```

## Aggregate

```js
Inputs.table(aggregate, {
  sort: "equals_expected_rate",
  reverse: true,
  format: {
    equals_expected_rate: (x) => (x * 100).toFixed(0) + "%",
  },
})
```

## Results

```js
Inputs.table(results, {
  columns: [
    "attributes.model",
    "attributes.prompt",
    "input.body",
    "expected_output",
    "output",
    "assertions_passed_rate",
  ],
  widths: {
    "input.body": 80,
  },
  format: {
    assertions_passed_rate: (x) =>
      x === 1
        ? htl.html`<div style="background: #d5edca;">✔</div>`
        : htl.html`<div style="background: #f9dddb;">✗</div>`,
  },
  align: {
    assertions_passed_rate: "center",
  },
  layout: "fixed",
})
```

---

Custom table code below

```js
Object.keys(results[0])
```

```js
const columns = [
  "attributes.model",
  "attributes.prompt",
  "input.body",
  "input.date",
  "input.email",
  "input.subject",
  "expected_output",
  "output",
  "assertions_passed_rate",
]

const table = d3.create("table").attr("class", "table-aggregate")

const thead = table.append("thead")
thead
  .append("tr")
  .selectAll("th")
  .data(columns)
  .join("th")
  .text((d) => d)

const tbody = table.append("tbody")

const tr = tbody.selectAll("tr").data(results).join("tr")

tr.selectAll("td")
  .data((d) => columns.map((c) => d[c]))
  .join("td")
  .text((d) => d)

display(table.node())
```

<style>
.table-aggregate tr {
}
.table-aggregate td {
  max-width: 140px;
  overflow: hidden;
}
</style>
