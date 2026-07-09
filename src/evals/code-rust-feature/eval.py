from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from agentic import (
    claude_code,
    cleanup_workdir,
    clone_git_repo,
    codex,
    git_diff,
    pi,
    require_solver,
)

REPO_URL = "https://github.com/kevinschaul/jump-start-tools.git"
COMMIT_ID = "de15260b46318c29f6b2435ff068fb9c42fc2806"


@task
def code_rust_feature() -> Task:
    return Task(
        dataset=MemoryDataset(
            [
                Sample(
                    input=(
                        "Implement the mode=git feature for the 'use' subcommand "
                        "in the jump-start-tools Rust CLI."
                    )
                )
            ]
        ),
        setup=clone_git_repo(REPO_URL, COMMIT_ID),
        solver=require_solver(),
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
