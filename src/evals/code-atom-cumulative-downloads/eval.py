import csv
import io
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from agentic import (
    claude_code,
    cleanup_workdir,
    codex,
    git_diff,
    pi,
    require_solver,
    serve_site,
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


@scorer(metrics=[mean()])
def check_output() -> Scorer:
    """Pass iff the agent's CSV matches expected.csv row-for-row."""
    expected = _rows((HERE / "expected.csv").read_text())

    async def score(state: TaskState, target: Target) -> Score:
        actual = _rows((state.store.get("captured_files") or {}).get(OUTPUT))
        if actual == expected:
            return Score(value=1.0, explanation="✓ CSV matches expected")

        lines = [
            "✗ CSV does not match expected",
            f"  expected {len(expected)} rows, got {len(actual)}",
        ]
        for i in range(max(len(expected), len(actual))):
            exp = expected[i] if i < len(expected) else None
            act = actual[i] if i < len(actual) else None
            if exp != act:
                lines.append(f"  row {i}: expected {exp}, got {act}")
            if len(lines) >= 8:
                lines.append("  ...")
                break
        return Score(value=0.0, explanation="\n".join(lines))

    return score


@task
def code_atom_cumulative_downloads() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input=PROMPT)]),
        setup=serve_site(HERE / "site"),
        solver=require_solver(),
        cleanup=cleanup_workdir(capture=[OUTPUT]),
        scorer=[git_diff(), check_output()],
    )
