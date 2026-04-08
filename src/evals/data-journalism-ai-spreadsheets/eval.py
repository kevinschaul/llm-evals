"""Agentic eval: consolidate federal AI use case spreadsheets.

Inspired by Kevin Schaul's "How I used Claude Code in a real data journalism
project" (https://kschaul.com/post/2026/02/09/2026-02-09-ai-data-journalism/),
where the task was to consolidate AI use case inventories that different
federal agencies publish in different locations, file formats, and column
names. This eval gives an agent a synthetic version of that mess and asks it
to produce a single normalized CSV.

User must specify a solver via the CLI, e.g.:

    just eval data-journalism-ai-spreadsheets anthropic/claude-sonnet-4-5 --solver claude_code
    just eval data-journalism-ai-spreadsheets openai/gpt-5-codex          --solver codex
    just eval data-journalism-ai-spreadsheets openrouter/openai/gpt-4o-mini --solver pi
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from agentic import (
    claude_code,
    cleanup_workdir,
    codex,
    copy_fixture,
    git_diff,
    pi,
)

# Re-exported so `--solver claude_code` / `--solver codex` / `--solver pi`
# resolves them.
__all__ = ["claude_code", "codex", "pi", "data_journalism_ai_spreadsheets"]

PROMPT = """\
The directory `data/raw/` contains AI use case inventories that several
(fictional) federal agencies have published. The agencies do not coordinate,
so each agency's file is in a different format (CSV, TSV, JSON) and uses
different column names for the same concepts.

Write a Python script at `scripts/consolidate_inventories.py` that reads
every raw agency file and writes a single consolidated CSV. Requirements:

  * The script should discover agency files under `data/raw/` on its own —
    don't hard-code the list of agencies.
  * All output rows should share a consistent set of columns. The columns
    should contain the same information across agencies even though the
    raw files name them differently. Pick sensible column names.
  * Normalize equivalent status values (e.g. "Operational", "Deployed",
    "In Production") to a consistent vocabulary so the column is
    comparable across agencies.
  * Keep the script easy for a human to audit. The Python standard library
    is fine — don't reach for heavy dependencies.
  * The script must be safe to rerun. Running it twice in a row should
    leave the workspace in the same state, not error out or duplicate
    rows.
  * While processing, keep a log of any questions or potentially confusing
    situations (ambiguous status values, missing fields, rows you had to
    guess on, etc.) so a human can double-check them later. Write the log
    to a file under `data/` that a reviewer can read.
  * Run the script once and verify the output CSV looks well-formed
    before finishing.
"""


@task
def data_journalism_ai_spreadsheets() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
