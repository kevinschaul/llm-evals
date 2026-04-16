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
# TODO what?
__all__ = ["claude_code", "codex", "pi", "code_consolidate_spreadsheets"]

PROMPT = """\
The directory `data/raw/` contains AI use case inventories that several
federal agencies published as part of the 2025 OMB reporting cycle. Write and execute a Python script at `scripts/consolidate_inventories.py` that reads
every raw agency file and writes a single consolidated CSV data/clean/consolidated.csv 

* All output rows should share a consistent set of columns
* Normalize equivalent status values
* While processing, keep a log of any questions or potentially confusing situations (ambiguous status values, missing fields, rows you had to guess on, etc.) so a human can double-check them later. Write the log to a file under `data/build` that a reviewer can read.
"""


@task
def code_consolidate_spreadsheets() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
