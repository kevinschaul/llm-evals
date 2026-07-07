/*
 * Copy publishable files from ../src/evals/<name>/ into site/public/evals/<name>/:
 * results/results.json plus any top-level images/PDFs referenced by prose.
 *
 * Strict allowlist — eval dirs also hold caches, logs and fixtures
 * (.db, .log, .eval, fixture/) that must never be published.
 *
 * Files are overwritten in place and stale entries pruned individually;
 * deleting the whole output dir breaks a running dev server's static
 * file serving.
 */
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(fileURLToPath(import.meta.url))
const evalsDir = path.resolve(root, "../../src/evals")
const publicDir = path.resolve(root, "../public/evals")

const ASSET_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf"])

fs.mkdirSync(publicDir, { recursive: true })

const syncedDirs = new Set()
let copied = 0

for (const name of fs.readdirSync(evalsDir)) {
  const evalDir = path.join(evalsDir, name)
  if (!fs.statSync(evalDir).isDirectory()) continue

  const outDir = path.join(publicDir, name)
  const wanted = new Set()

  const resultsJson = path.join(evalDir, "results", "results.json")
  if (fs.existsSync(resultsJson)) {
    fs.mkdirSync(outDir, { recursive: true })
    fs.copyFileSync(resultsJson, path.join(outDir, "results.json"))
    wanted.add("results.json")
    copied++
  }

  for (const file of fs.readdirSync(evalDir)) {
    if (!ASSET_EXTENSIONS.has(path.extname(file).toLowerCase())) continue
    fs.mkdirSync(outDir, { recursive: true })
    fs.copyFileSync(path.join(evalDir, file), path.join(outDir, file))
    wanted.add(file)
    copied++
  }

  if (wanted.size > 0) syncedDirs.add(name)

  if (fs.existsSync(outDir)) {
    for (const file of fs.readdirSync(outDir)) {
      if (!wanted.has(file)) fs.rmSync(path.join(outDir, file), { recursive: true, force: true })
    }
  }
}

for (const name of fs.readdirSync(publicDir)) {
  if (!syncedDirs.has(name)) fs.rmSync(path.join(publicDir, name), { recursive: true, force: true })
}

console.log(`sync-assets: copied ${copied} files to site/public/evals/`)
