import csv
import io
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from agentic import (
    capture_files,
    claude_code,
    cleanup_workdir,
    codex,
    git_diff,
    pi,
    require_solver,
    serve_site_archive,
)

# Re-export so `--solver claude_code` / `--solver codex` / `--solver pi` resolves.
__all__ = ["claude_code", "codex", "pi", "code_atom_cumulative_downloads"]

HERE = Path(__file__).parent
OUTPUT = "cumulative_downloads.csv"

PROMPT = """\
Visit the site {url}

Find the line chart titled "Models Worldwide" (subtitle "Cumulative Downloads,
2023-present"), which plots cumulative model downloads in millions for three
series: USA, China, and EU. Extract that chart's data to a CSV file named
`cumulative_downloads.csv` in the current directory, with exactly this header:

    period,usa,china,eu
"""


def _rows(text: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(text or ""))
        if any(cell.strip() for cell in row)
    ]


def _same(a: list[str] | None, b: list[str] | None) -> bool:
    """Row equality that treats numeric cells by value ("7" == "7.0")."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            if float(x) != float(y):
                return False
        except ValueError:
            if x != y:
                return False
    return True


@scorer(metrics=[mean()])
def check_output() -> Scorer:
    """Pass iff the agent's CSV matches expected.csv row-for-row (numeric
    cells compared by value, so float formatting doesn't matter)."""
    expected = _rows((HERE / "expected.csv").read_text())

    async def score(state: TaskState, target: Target) -> Score:
        work_dir = state.store.get("work_dir")
        actual = _rows(capture_files(work_dir, [OUTPUT])[OUTPUT] if work_dir else None)
        if len(actual) == len(expected) and all(map(_same, actual, expected)):
            return Score(
                value=1.0,
                explanation="✓ CSV matches expected",
                metadata={"checks": {"CSV matches expected": True}},
            )

        lines = [
            "✗ CSV does not match expected",
            f"  expected {len(expected)} rows, got {len(actual)}",
        ]
        for i in range(max(len(expected), len(actual))):
            exp = expected[i] if i < len(expected) else None
            act = actual[i] if i < len(actual) else None
            if not _same(exp, act):
                lines.append(f"  row {i}: expected {exp}, got {act}")
            if len(lines) >= 8:
                lines.append("  ...")
                break
        return Score(
            value=0.0,
            explanation="\n".join(lines),
            metadata={"checks": {"CSV matches expected": False}},
        )

    return score


@task
def code_atom_cumulative_downloads() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=serve_site_archive(HERE / "site.tar.gz"),
        solver=require_solver(),
        cleanup=cleanup_workdir(),
        scorer=[git_diff(), check_output()],
    )
