import { defineCollection, z } from "astro:content"
import { glob } from "astro/loaders"

// Eval prose stays colocated with each eval in ../src/evals/<name>/index.md
const evals = defineCollection({
  loader: glob({
    pattern: "*/index.md",
    base: "../src/evals",
    generateId: ({ entry }) => entry.split("/")[0],
  }),
  schema: z.object({
    title: z.string(),
    type: z.enum(["tests", "freeform", "agentic"]),
    blurb: z.string().optional(),
    archived: z.boolean().default(false),
  }),
})

export const collections = { evals }
