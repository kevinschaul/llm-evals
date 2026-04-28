from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from agentic import (
    claude_code,
    cleanup_workdir,
    codex,
    copy_fixture,
    git_diff,
    pi,
    require_solver,
)

PROMPT = """\
The directory `data/raw/` contains AI use case inventories that several
federal agencies published as part of the 2025 OMB reporting cycle. Write and execute a Python script at `scripts/consolidate_inventories.py` that reads
every raw agency file and writes a single consolidated CSV data/clean/consolidated.csv

* All output rows should share a consistent set of columns
* Normalize equivalent status values
* While processing, keep a log of any questions or potentially confusing situations (ambiguous status values, missing fields, rows you had to guess on, etc.) so a human can double-check them later. Write the log to a file under `data/build` that a reviewer can read.
"""

# Agriculture: 89
# Health and Human Services: 447
# Labor: 28
# Treasury: 129
EXPECTED_ROWS = 693


async def _capture(state: TaskState, work_dir: str) -> None:
    """Capture output file info into state.store before the work dir is deleted."""
    root = Path(work_dir)
    py_files = list(root.glob("**/*.py"))
    state.store.set("py_files", [str(p.relative_to(root)) for p in py_files])
    csv_path = root / "data" / "clean" / "consolidated.csv"
    if csv_path.exists():
        text = csv_path.read_text(errors="replace")
        # subtract 1 for the header row
        state.store.set("consolidated_csv_rows", len(text.splitlines()) - 1)
    else:
        state.store.set("consolidated_csv_rows", None)


@scorer(metrics=[])
def check_output() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        py_files = state.store.get("py_files", [])
        csv_rows = state.store.get("consolidated_csv_rows")

        checks = {
            "has_py_file": len(py_files) > 0,
            "has_consolidated_csv": csv_rows is not None,
            "row_count_correct": csv_rows == EXPECTED_ROWS,
        }
        passed = sum(checks.values())
        explanation = "\n".join(
            f"{'✓' if v else '✗'} {k}" for k, v in checks.items()
        )
        if csv_rows is not None and not checks["row_count_correct"]:
            explanation += f"\n  (got {csv_rows}, expected {EXPECTED_ROWS})"
        if py_files:
            explanation += f"\n  py files: {', '.join(py_files)}"

        return Score(value=f"{passed}/{len(checks)}", explanation=explanation)

    return score


@task
def code_consolidate_spreadsheets() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        solver=require_solver(),
        cleanup=cleanup_workdir(on_finish=_capture),
        scorer=[git_diff(), check_output()],
    )
