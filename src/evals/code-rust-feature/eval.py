from inspect_ai import task, Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import solver, Solver, TaskState, Generate, generate
import subprocess
import tempfile
import os
import shutil

from inspect_ai.model import get_model


@solver
def claude_code() -> Solver:
    """Run Claude Code agent without sandbox"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the current model being evaluated
        model = get_model()

        work_dir = state.store.get("work_dir")

        # base options
        cmd = [
            "claude",
            "--print",  # run without interactions
            "--dangerously-skip-permissions",
            "--model",
            model.name,  # Use the model name from the eval
        ]

        # system message
        system_message = "\n\n".join(
            [m.content for m in state.messages if m.role == "system"]
        )
        if system_message:
            cmd.extend(["--append-system-prompt", system_message])

        # user prompt from input
        prompt = state.input_text
        cmd.append(prompt)

        # Run Claude Code in the cloned repository
        # Copy environment but strip ANTHROPIC_API_KEY
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        result = subprocess.run(
            cmd, cwd=work_dir, capture_output=True, text=True, env=env
        )

        if result.returncode == 0:
            # Store the output
            state.output.completion = result.stdout
        else:
            raise RuntimeError(
                f"Error executing claude code agent: {result.stdout}\n{result.stderr}"
            )

        return state

    return solve


def create_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(
                # input="Implement the mode=git feature for the 'use' subcommand in the jump-start-tools Rust CLI.",
                input="Add a joke to the readme file"
            )
        ]
    )


@solver
def clone_repo():
    async def solve(state, generate):
        # Create a persistent temporary directory
        tmpdir = tempfile.mkdtemp()

        repo_url = "https://github.com/kevinschaul/jump-start-tools.git"
        clone_result = subprocess.run(
            ["git", "clone", repo_url, "jump-start-tools"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if clone_result.returncode != 0:
            raise RuntimeError(
                f"Error cloning repository: {clone_result.stdout}\n{clone_result.stderr}"
            )

        state.output.completion = clone_result.stdout

        # Set working directory to the cloned repo
        state.store.set("work_dir", os.path.join(tmpdir, "jump-start-tools"))

        return state

    return solve


@solver
def git_diff():
    async def solve(state, generate):
        work_dir = state.store.get("work_dir")
        git_diff_result = subprocess.run(
            ["git", "diff"],
            cwd=work_dir,
            capture_output=True,
            text=True,
        )

        if git_diff_result.returncode != 0:
            raise RuntimeError(
                f"Error running git diff: {git_diff_result.stdout}\n{git_diff_result.stderr}"
            )

        state.output.completion = git_diff_result.stdout

        return state

    return solve


def cleanup_repo():
    async def cleanup(state):
        # Delete the temporary work directory if it exists
        if hasattr(state.metadata, "work_dir") and state.metadata.work_dir:
            work_dir = state.metadata.work_dir
            # Get the parent directory (the tmpdir)
            tmpdir = os.path.dirname(work_dir)
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)

    return cleanup


@task
def rust_feature() -> Task:
    return Task(
        dataset=create_dataset(),
        setup=clone_repo(),
        cleanup=cleanup_repo(),
        solver=[
            claude_code(),
            git_diff(),
        ],
    )
