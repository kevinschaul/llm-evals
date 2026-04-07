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
The directory `inventories/` contains AI use case inventories that several
(fictional) federal agencies have published. The agencies do not coordinate, so
the files are in different formats (CSV, TSV, JSON) and use different column
names for the same concepts.

Your job is to consolidate them into a single CSV file at the root of this
working directory called `consolidated_use_cases.csv` with these normalized
columns, in this exact order:

    agency,name,description,status,contact,year_deployed

Notes:
  * `agency` should be a short human-readable agency name inferred from the
    inventory's directory (e.g. "Department of Energy", "HHS", "DOD",
    "Treasury").
  * `status` should be normalized to one of: Pilot, Production, Retired.
    Map equivalent terms (Operational, Deployed, In Production, etc.) onto
    these three values.
  * `year_deployed` should be a 4-digit integer.
  * Sort the output rows by agency, then by name.
  * Read every file under `inventories/` — do not skip any rows.

Write any helper code you need, run it, and verify the output is well-formed
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
