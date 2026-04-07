"""Agentic eval: implement a Rust feature in jump-start-tools.

User must specify a solver via the CLI, e.g.:

    just eval code-rust-feature anthropic/claude-sonnet-4-5 --solver claude_code
    just eval code-rust-feature openai/gpt-5-codex          --solver codex
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from agentic import claude_code, cleanup_workdir, clone_git_repo, codex, git_diff

# Re-exported so `--solver claude_code` / `--solver codex` resolves them.
__all__ = ["claude_code", "codex", "code_rust_feature"]

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
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
