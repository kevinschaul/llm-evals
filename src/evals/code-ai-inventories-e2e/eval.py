import csv
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
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
The federal government requires agencies to publish AI use case inventories
as part of OMB reporting. For each of the four agencies below, search the web
to find their official AI use case inventory page, download the inventory file,
and save it to data/raw/<agency-slug>/<filename>:

  - Department of Agriculture (USDA)
  - Department of Health and Human Services (HHS)
  - Department of Labor (DOL)
  - Department of the Treasury

Once all files are downloaded, write and execute a Python script at
scripts/consolidate_inventories.py that:

* Reads every file under data/raw/
* Writes a single consolidated CSV to data/clean/consolidated.csv
* Uses a consistent set of columns across all rows
* Normalizes equivalent status values (e.g. "Deployed" vs "deployed")
* Logs any ambiguous values, missing fields, or decisions that a human should
  double-check to data/build/review.txt
"""

# Rough minimum: expect at least a few hundred rows if all 4 agencies are consolidated.
# Exact counts vary as agencies update their inventories.
MIN_ROWS = 100


async def _capture(state: TaskState, work_dir: str) -> None:
    """Capture output metrics into state.store before the work dir is deleted."""
    root = Path(work_dir)

    py_files = list(root.glob("**/*.py"))
    state.store.set("py_files", [str(p.relative_to(root)) for p in py_files])

    raw_dir = root / "data" / "raw"
    if raw_dir.exists():
        downloaded = [
            str(p.relative_to(root)) for p in raw_dir.rglob("*") if p.is_file()
        ]
    else:
        downloaded = []
    state.store.set("downloaded_files", downloaded)

    csv_path = root / "data" / "clean" / "consolidated.csv"
    if csv_path.exists():
        with csv_path.open(newline="", errors="replace") as f:
            state.store.set("consolidated_csv_rows", sum(1 for _ in csv.reader(f)) - 1)
    else:
        state.store.set("consolidated_csv_rows", None)


@scorer(metrics=[mean()])
def check_output() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        py_files = state.store.get("py_files", [])
        csv_rows = state.store.get("consolidated_csv_rows")
        downloaded = state.store.get("downloaded_files", [])

        checks = {
            "has_py_file": len(py_files) > 0,
            "downloaded_all_agencies": len(downloaded) >= 4,
            "has_consolidated_csv": csv_rows is not None,
            "has_substantial_rows": csv_rows is not None and csv_rows >= MIN_ROWS,
        }
        passed = sum(checks.values())
        explanation = "\n".join(
            f"{'✓' if v else '✗'} {k}" for k, v in checks.items()
        )
        if csv_rows is not None:
            explanation += f"\n  ({csv_rows} rows in consolidated CSV)"
        if downloaded:
            explanation += f"\n  downloaded: {', '.join(downloaded)}"
        if py_files:
            explanation += f"\n  py files: {', '.join(py_files)}"

        return Score(value=float(passed) / len(checks), explanation=explanation)

    return score


@task
def code_ai_inventories_e2e() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        solver=require_solver(),
        cleanup=cleanup_workdir(on_finish=_capture),
        scorer=[git_diff(), check_output()],
    )
