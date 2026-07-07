import fs from "node:fs"
import path from "node:path"
import type { EvalResults } from "./types"

// Build-time loader; astro runs with cwd at site/, evals live one level up
export function loadResults(slug: string): EvalResults | null {
  const file = path.resolve(
    process.cwd(),
    "..",
    "src/evals",
    slug,
    "results/results.json",
  )
  if (!fs.existsSync(file)) return null
  return JSON.parse(fs.readFileSync(file, "utf-8"))
}
