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


PROMPT = """Federal agencies are required by executive order to publish AI use case inventories. Your task is to:

1. Search the web to find AI use case inventory spreadsheet files (CSV or Excel) from at least 4 different federal agencies. Look for links to downloadable files, not just web pages. Good places to look:
   - agency websites under sections like "AI", "Data", or "About"
   - data.gov
   - GitHub repositories (e.g. search for "federal agency AI inventory filetype:csv")

2. Download the files into the current directory.

3. Write a Python script to consolidate all the files into a single CSV called consolidated.csv with these standardized columns:
   - agency: the name of the federal agency
   - use_case_name: name of the AI use case
   - description: description of the use case
   - development_stage: stage of development (e.g. Production, Testing, Development)
   - topic_area: topic or domain area

4. Run the script.

Print a summary of how many rows came from each agency."""


def create_dataset() -> MemoryDataset:
    return MemoryDataset([Sample(input=PROMPT)])


@solver
def claude_code() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
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

        prompt = state.input_text
        cmd.append(prompt)

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
                    transcript().info(event)

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
                                tool_use_lookup[item.get("id")] = item

                    elif event_type == "user":
                        message = event.get("message", {})
                        for item in message.get("content", []):
                            item_type = item.get("type")
                            if item_type == "tool_result":
                                tool_use_item = tool_use_lookup.get(item.get("tool_use_id"), {})
                                content = item.get("content", "a tool call")
                                # tool_reference and other non-string content types aren't
                                # supported by ChatMessageTool — flatten to string
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
        tmpdir = tempfile.mkdtemp()

        async with span("setup_workspace"):
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
        work_dir = state.store.get("work_dir")
        if work_dir and os.path.exists(work_dir):
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.rmtree, work_dir
            )

    return cleanup


@task
def ai_inventory_search() -> Task:
    """
    Full pipeline eval: agent must find, download, and consolidate federal agency AI inventories.
    User must specify a solver in the CLI:
    e.g. just eval ai-inventory-search anthropic/claude-sonnet-4-6 --solver claude_code
    """
    return Task(
        dataset=create_dataset(),
        setup=setup_workspace(),
        cleanup=cleanup_workspace(),
        scorer=output_csv(),
    )
