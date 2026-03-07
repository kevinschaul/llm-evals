from inspect_ai import task, Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    get_model,
)
from inspect_ai.log import transcript
from inspect_ai.util import span
import asyncio
import tempfile
import os
import shutil
import json


REPO_URL = "https://github.com/kevinschaul/jump-start-tools.git"
COMMIT_ID = "de15260b46318c29f6b2435ff068fb9c42fc2806"


def create_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(
                input="Implement the mode=git feature for the 'use' subcommand in the jump-start-tools Rust CLI.",
            )
        ]
    )


@solver
def claude_code() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the current model being evaluated
        model = get_model()
        work_dir = state.store.get("work_dir")

        cmd = [
            "claude",
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            model.name,
        ]

        system_message = "\n\n".join(
            [m.content for m in state.messages if m.role == "system"]
        )
        if system_message:
            cmd.extend(["--append-system-prompt", system_message])

        # user prompt from input
        prompt = state.input_text
        cmd.append(prompt)

        # Copy environment but strip ANTHROPIC_API_KEY and Claude Code session
        # env vars that would prevent nested claude CLI execution
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        for key in list(env.keys()):
            if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_"):
                del env[key]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        tool_use_lookup = {}

        stderr_lines = []

        async def read_stderr():
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                decoded = line.decode().rstrip()
                stderr_lines.append(decoded)
                transcript().info(decoded, source="stderr")

        async def read_stdout():
            nonlocal tool_use_lookup

            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                try:
                    event = json.loads(line.decode())

                    # Log all events to the transcript
                    transcript().info(event)

                    # For message or tool events, add those to messages
                    event_type = event.get("type")

                    if event_type == "assistant":
                        message = event.get("message", {})
                        for item in message.get("content", []):
                            item_type = item.get("type")
                            if item_type == "text":
                                state.messages.append(
                                    ChatMessageAssistant(
                                        content=item.get("text"), metadata=message
                                    )
                                )
                            elif item_type == "tool_use":
                                # This includes the tool call but not the result
                                tool_use_lookup[item.get("id")] = item

                    elif event_type == "user":
                        message = event.get("message", {})
                        for item in message.get("content", []):
                            item_type = item.get("type")
                            if item_type == "tool_result":
                                tool_use_item = tool_use_lookup[item.get("tool_use_id")]
                                transcript().info(
                                    "User event!" + json.dumps(tool_use_item)
                                )

                                state.messages.append(
                                    ChatMessageTool(
                                        content=item.get("content", "a toll call"),
                                        function=tool_use_item.get("name"),
                                        metadata={
                                            "tool_use": tool_use_item,
                                            "tool_result": item,
                                        },
                                    )
                                )

                except json.JSONDecodeError:
                    transcript().info(line.decode().rstrip())

        await asyncio.gather(read_stdout(), read_stderr())
        await process.wait()

        if process.returncode != 0:
            stderr_text = "\n".join(stderr_lines)
            raise RuntimeError(
                f"Error executing claude code agent (exit code: {process.returncode})\n{stderr_text}"
            )

        return state

    return solve


@solver
def codex() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the current model being evaluated
        model = get_model()
        work_dir = state.store.get("work_dir")

        cmd = [
            "codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            model.name,
        ]

        # user prompt from input
        prompt = state.input_text
        cmd.append(prompt)

        # Copy environment but strip OPENAI_API_KEY
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        async def read_stderr():
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                transcript().info(line.decode().rstrip(), source="stderr")

        async def read_stdout():
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                try:
                    # https://github.com/openai/codex/blob/main/docs/exec.md#json-output-mode
                    event = json.loads(line.decode())

                    # Log all events to the transcript
                    transcript().info(event)

                    # For message or tool events, add those to messages
                    item = event.get("item", {})
                    item_type = item.get("type")

                    if item_type == "reasoning" or item_type == "agent_message":
                        state.messages.append(
                            ChatMessageAssistant(
                                content=item.get("text"), metadata=event
                            )
                        )
                    elif item_type == "command_execution":
                        state.messages.append(
                            ChatMessageTool(
                                content=item.get("aggregated_output"),
                                function=item.get("command"),
                                metadata=event,
                            )
                        )

                except json.JSONDecodeError:
                    transcript().info(line.decode().rstrip())

        await asyncio.gather(read_stdout(), read_stderr())
        await process.wait()

        if process.returncode != 0:
            raise RuntimeError(
                f"Error executing codex agent: {stdout.decode()}\n{stderr.decode()}"
            )

        return state

    return solve


@solver
def clone_repo():
    async def solve(state, generate):
        # Create a persistent temporary directory
        tmpdir = tempfile.mkdtemp()
        repo_path = os.path.join(tmpdir, "repo")

        async with span("clone_repo"):
            # Clone the repository
            transcript().info(f"Cloning repository: {REPO_URL}")
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                REPO_URL,
                "repo",
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(
                    f"Error cloning repository: {stdout.decode()}\n{stderr.decode()}"
                )

            transcript().info(f"✓ Repository cloned to {repo_path}")

            # If a specific commit is requested, check it out
            if COMMIT_ID:
                transcript().info(f"Checking out commit: {COMMIT_ID}")
                process = await asyncio.create_subprocess_exec(
                    "git",
                    "checkout",
                    COMMIT_ID,
                    cwd=repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    raise RuntimeError(
                        f"Error checking out commit {COMMIT_ID}: {stdout.decode()}\n{stderr.decode()}"
                    )

                transcript().info(f"✓ Checked out commit {COMMIT_ID}")

            # Set working directory to the cloned repo
            state.store.set("work_dir", repo_path)
            transcript().info({"work_dir": repo_path})

        return state

    return solve


@scorer(metrics=[])
def git_diff() -> Scorer:
    async def score(state: TaskState, target: Target):
        work_dir = state.store.get("work_dir")

        process = await asyncio.create_subprocess_exec(
            "git",
            "diff",
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"Error running git diff: {stdout.decode()}\n{stderr.decode()}"
            )

        return Score(value="C", explanation=stdout.decode())

    return score


def cleanup_repo():
    async def cleanup(state):
        # Delete the temporary work directory if it exists
        if hasattr(state.metadata, "work_dir") and state.metadata.work_dir:
            work_dir = state.metadata.work_dir
            # Get the parent directory (the tmpdir)
            tmpdir = os.path.dirname(work_dir)
            if os.path.exists(tmpdir):
                # Run in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    None, shutil.rmtree, tmpdir
                )

    return cleanup


@task
def code_rust_feature() -> Task:
    """
    User must specify a solver in the CLI
    e.g. just eval code-rust-feature openai/gpt-oss-20b --solver codex
    """
    return Task(
        dataset=create_dataset(),
        setup=clone_repo(),
        cleanup=cleanup_repo(),
        scorer=git_diff(),
    )
