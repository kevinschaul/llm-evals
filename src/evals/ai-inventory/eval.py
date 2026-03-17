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
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"

PROMPT = """I have downloaded AI use case inventory spreadsheets from several federal agencies. They are in the current directory with different formats and inconsistent column names.

Write a Python script to:
1. Read each CSV file in the current directory
2. Examine the column names in each file
3. Map them to these standardized columns: agency, use_case_name, description, development_stage, topic_area
4. Add an 'agency' column identifying which agency the data came from (infer from the filename if not in the data)
5. Consolidate all rows into a single file called consolidated.csv

Then run the script and print a summary of how many rows came from each file."""


def create_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(
                input=PROMPT,
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
                                tool_use_item = tool_use_lookup.get(item.get("tool_use_id"), {})
                                transcript().info(
                                    "User event!" + json.dumps(tool_use_item)
                                )
                                content = item.get("content", "a tool call")
                                if not isinstance(content, str):
                                    content = json.dumps(content)
                                state.messages.append(
                                    ChatMessageTool(
                                        content=content,
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

        # Stash output file content before cleanup can delete it
        work_dir = state.store.get("work_dir")
        consolidated_path = os.path.join(work_dir, "consolidated.csv") if work_dir else None
        if consolidated_path and os.path.exists(consolidated_path):
            with open(consolidated_path) as f:
                state.store.set("consolidated_csv", f.read())

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
            "--experimental-json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model",
            model.name,
        ]

        # user prompt from input
        prompt = state.input_text
        cmd.append(prompt)

        env = os.environ.copy()

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

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
            stderr_text = "\n".join(stderr_lines)
            raise RuntimeError(
                f"Error executing codex agent (exit code: {process.returncode})\n{stderr_text}"
            )

        # Stash output file content before cleanup can delete it
        work_dir = state.store.get("work_dir")
        consolidated_path = os.path.join(work_dir, "consolidated.csv") if work_dir else None
        if consolidated_path and os.path.exists(consolidated_path):
            with open(consolidated_path) as f:
                state.store.set("consolidated_csv", f.read())

        return state

    return solve


@solver
def setup_workspace():
    async def solve(state, generate):
        # Create a persistent temporary directory
        tmpdir = tempfile.mkdtemp()

        async with span("setup_workspace"):
            # Copy all fixture files into the temp directory
            for fixture_file in FIXTURES_DIR.iterdir():
                if fixture_file.is_file():
                    dest = os.path.join(tmpdir, fixture_file.name)
                    shutil.copy(fixture_file, dest)
                    transcript().info(f"Copied fixture: {fixture_file.name} -> {dest}")

            # Set working directory to the temp dir with fixture files
            state.store.set("work_dir", tmpdir)
            transcript().info({"work_dir": tmpdir})

        return state

    return solve


@scorer(metrics=[])
def output_csv() -> Scorer:
    async def score(state: TaskState, target: Target):
        content = state.store.get("consolidated_csv")
        if not content:
            return Score(value="I", explanation="No consolidated.csv was created")
        return Score(value="C", explanation=content)

    return score


def cleanup_workspace():
    async def cleanup(state):
        # Delete the temporary work directory if it exists
        work_dir = state.store.get("work_dir")
        if work_dir and os.path.exists(work_dir):
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.rmtree, work_dir
            )

    return cleanup


@task
def ai_inventory() -> Task:
    """
    User must specify a solver in the CLI
    e.g. just eval ai-inventory anthropic/claude-sonnet-4-5 --solver claude_code
    """
    return Task(
        dataset=create_dataset(),
        setup=setup_workspace(),
        cleanup=cleanup_workspace(),
        scorer=output_csv(),
    )
