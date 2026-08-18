import { defineConfig } from "astro/config"
import react from "@astrojs/react"
import syncEvals from "./integrations/sync-evals.mjs"

// GITHUB_PAGES_BASE_PATH is set by the deploy workflow (e.g. "/llm-evals")
const base = process.env.GITHUB_PAGES_BASE_PATH || "/"

export default defineConfig({
  site: "https://kschaul.com",
  base,
  integrations: [react(), syncEvals()],
  vite: {
    server: {
      fs: {
        // eval content and the theme package live outside site/
        allow: [".."],
      },
    },
  },
})
