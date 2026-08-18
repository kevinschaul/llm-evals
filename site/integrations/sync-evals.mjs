/*
 * Astro integration: copy publishable files from ../src/evals/<name>/ into
 * site/public/evals/<name>/ (results/results.json plus any top-level
 * images/PDFs referenced by prose), and keep them in sync automatically.
 *
 * Runs once on config setup (covers both `astro dev` and `astro build`), and
 * during dev additionally watches ../src/evals and re-syncs + triggers a
 * full reload whenever a result gets re-extracted, so the dashboard reflects
 * new eval runs without a manual `npm run sync` or browser refresh.
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

const ASSET_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf"])

function syncEvals(evalsDir, publicDir) {
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

  return copied
}

export default function syncEvalsIntegration() {
  let evalsDir, publicDir

  return {
    name: "sync-evals",
    hooks: {
      "astro:config:setup": ({ config, logger }) => {
        const root = fileURLToPath(config.root)
        evalsDir = path.resolve(root, "../src/evals")
        publicDir = path.resolve(root, "public/evals")
        const copied = syncEvals(evalsDir, publicDir)
        logger.info(`copied ${copied} files to public/evals/`)
      },
      "astro:server:setup": ({ server, logger }) => {
        server.watcher.add(evalsDir)

        let debounce
        server.watcher.on("all", (event, changedPath) => {
          if (!changedPath.startsWith(evalsDir)) return
          clearTimeout(debounce)
          debounce = setTimeout(() => {
            const copied = syncEvals(evalsDir, publicDir)
            logger.info(
              `resynced ${copied} files (${event}: ${path.relative(evalsDir, changedPath)})`,
            )
            server.ws.send({ type: "full-reload" })
          }, 200)
        })
      },
    },
  }
}
