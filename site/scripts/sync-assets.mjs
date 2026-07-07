/*
 * Copy publishable files from ../src/evals/<name>/ into site/public/evals/<name>/:
 * results/results.json plus any top-level images/PDFs referenced by prose.
 *
 * Strict allowlist — eval dirs also hold caches, logs and fixtures
 * (.db, .log, .eval, fixture/) that must never be published.
 */
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(fileURLToPath(import.meta.url))
const evalsDir = path.resolve(root, "../../src/evals")
const publicDir = path.resolve(root, "../public/evals")

const ASSET_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf"])

fs.rmSync(publicDir, { recursive: true, force: true })

let copied = 0
for (const name of fs.readdirSync(evalsDir)) {
  const evalDir = path.join(evalsDir, name)
  if (!fs.statSync(evalDir).isDirectory()) continue

  const outDir = path.join(publicDir, name)

  const resultsJson = path.join(evalDir, "results", "results.json")
  if (fs.existsSync(resultsJson)) {
    fs.mkdirSync(outDir, { recursive: true })
    fs.copyFileSync(resultsJson, path.join(outDir, "results.json"))
    copied++
  }

  for (const file of fs.readdirSync(evalDir)) {
    if (!ASSET_EXTENSIONS.has(path.extname(file).toLowerCase())) continue
    fs.mkdirSync(outDir, { recursive: true })
    fs.copyFileSync(path.join(evalDir, file), path.join(outDir, file))
    copied++
  }
}

console.log(`sync-assets: copied ${copied} files to site/public/evals/`)
