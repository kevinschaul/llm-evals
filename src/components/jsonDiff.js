import * as jsonDiff from "https://esm.sh/json-diff@1.0.6"

export default function diff(left, right) {
  const diffContainer = document.createElement("pre")

  diffContainer.innerHTML = jsonDiff.colorize(jsonDiff.diff(left, right), {
    theme: {
      " ": (d) => d,
      "+": (d) =>
        `<span style="color: var(--syntax-markup-inserted); background: var(--syntax-markup-inserted-background);">${d}</span>`,
      "-": (d) =>
        `<span style="color: var(--syntax-markup-deleted); background: var(--syntax-markup-deleted-background);">${d}</span>`,
    },
  })

  diffContainer.innerHTML += `<style>pre { background: none; }</style>`

  return diffContainer
}
