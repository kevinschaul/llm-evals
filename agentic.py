"""Shared building blocks for agentic evals.

An agentic eval is one where an external coding agent (Claude Code, Codex, ...)
runs in a temporary work directory and makes file changes. We capture those
changes by `git diff`'ing the work directory at the end. A `git diff` is
usually enough to eyeball how well the agent did.

This module provides four pieces that compose into a Task:

  setup     prepare a fresh work directory and store its path in
            `state.store["work_dir"]`. Two flavors:
              * `clone_git_repo(url, commit)` — clone+checkout a remote repo
              * `copy_fixture(src_dir)`        — copy a local fixture directory

  solver    run an agent CLI inside `work_dir`. Two flavors:
              * `claude_code()` — runs `claude --print --output-format stream-json`
              * `codex()`        — runs `codex exec --json`
            The current model from `--model` is forwarded to the agent CLI,
            so the same task can be run against any model the agent supports.

  scorer    `git_diff()` — runs `git diff` in `work_dir` and stores the
            patch as the score explanation. `extract_results.py` already knows
            to surface the score explanation as the row's `result`, so the
            diff shows up in the dashboard for free.

  cleanup   `cleanup_workdir()` — recursively delete the temp dir.

To add a new agentic eval, create `src/evals/<name>/eval.py`:

    from pathlib import Path
    from inspect_ai import task, Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from agentic import copy_fixture, cleanup_workdir, git_diff

    @task
    def my_eval() -> Task:
        return Task(
            dataset=MemoryDataset([Sample(input="Do the thing.")]),
            setup=copy_fixture(Path(__file__).parent / "fixture"),
            cleanup=cleanup_workdir(),
            scorer=git_diff(),
        )

Then run it against any model + harness:

    just eval my-eval anthropic/claude-sonnet-4-5 --solver claude_code
    just eval my-eval openai/gpt-5-codex          --solver codex
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.util import span


# ---------------------------------------------------------------------------
# Setup helpers — prepare a work directory and stash its path in state.store
# ---------------------------------------------------------------------------


async def _run(*cmd: str, cwd: Optional[str] = None) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout, stderr


async def _git_init_commit(work_dir: str) -> None:
    """Initialize a git repo so `git diff` works after the agent runs."""
    for cmd in (
        ("git", "init", "-q", "-b", "main"),
        ("git", "-c", "user.email=eval@example.com", "-c", "user.name=eval",
         "add", "."),
        ("git", "-c", "user.email=eval@example.com", "-c", "user.name=eval",
         "commit", "-q", "--allow-empty", "-m", "initial"),
    ):
        rc, out, err = await _run(*cmd, cwd=work_dir)
        if rc != 0:
            raise RuntimeError(
                f"git setup failed ({' '.join(cmd)}):\n{out.decode()}\n{err.decode()}"
            )


def clone_git_repo(url: str, commit: Optional[str] = None) -> Solver:
    """Setup: clone `url` (optionally checking out `commit`) into a fresh tmpdir."""

    @solver
    def setup() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            tmpdir = tempfile.mkdtemp(prefix="agentic-eval-")
            repo_path = os.path.join(tmpdir, "repo")
            async with span("clone_git_repo"):
                transcript().info(f"Cloning {url}")
                rc, out, err = await _run("git", "clone", url, "repo", cwd=tmpdir)
                if rc != 0:
                    raise RuntimeError(
                        f"git clone failed:\n{out.decode()}\n{err.decode()}"
                    )
                if commit:
                    transcript().info(f"Checking out {commit}")
                    rc, out, err = await _run("git", "checkout", commit, cwd=repo_path)
                    if rc != 0:
                        raise RuntimeError(
                            f"git checkout failed:\n{out.decode()}\n{err.decode()}"
                        )
            state.store.set("work_dir", repo_path)
            transcript().info({"work_dir": repo_path})
            return state

        return solve

    return setup()


def copy_fixture(src_dir: str | Path) -> Solver:
    """Setup: copy a local fixture directory into a fresh tmpdir and `git init` it.

    The git init lets the `git_diff()` scorer capture whatever the agent
    creates or modifies, even when the starting state isn't a real repo.
    """
    src_path = Path(src_dir)

    @solver
    def setup() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            tmpdir = tempfile.mkdtemp(prefix="agentic-eval-")
            work_dir = os.path.join(tmpdir, "work")
            async with span("copy_fixture"):
                transcript().info(f"Copying fixture {src_path} -> {work_dir}")
                shutil.copytree(src_path, work_dir)
                await _git_init_commit(work_dir)
            state.store.set("work_dir", work_dir)
            transcript().info({"work_dir": work_dir})
            return state

        return solve

    return setup()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_workdir():
    """Cleanup: remove the temporary work directory created by a setup helper."""

    async def cleanup(state: TaskState) -> None:
        work_dir = state.store.get("work_dir")
        if not work_dir or not os.path.exists(work_dir):
            return
        tmpdir = os.path.dirname(work_dir)
        await asyncio.get_event_loop().run_in_executor(
            None, shutil.rmtree, tmpdir, True
        )

    return cleanup


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[])
def git_diff() -> Scorer:
    """Score by capturing `git diff` of the work directory."""

    async def score(state: TaskState, target: Target) -> Score:
        work_dir = state.store.get("work_dir")
        if not work_dir:
            return Score(value="I", explanation="(no work_dir set)")

        rc, out, err = await _run("git", "add", "-N", ".", cwd=work_dir)
        if rc != 0:
            raise RuntimeError(f"git add -N failed:\n{out.decode()}\n{err.decode()}")

        rc, out, err = await _run("git", "diff", cwd=work_dir)
        if rc != 0:
            raise RuntimeError(f"git diff failed:\n{out.decode()}\n{err.decode()}")

        return Score(value="C", explanation=out.decode(errors="replace"))

    return score


# ---------------------------------------------------------------------------
# Solvers — agent CLI harnesses
# ---------------------------------------------------------------------------


@solver
def claude_code() -> Solver:
    """Run the Claude Code CLI inside `state.store['work_dir']`.

    The model name from `--model` is forwarded as `claude --model …`.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        work_dir = state.store.get("work_dir")
        if not work_dir:
            raise RuntimeError("claude_code() requires a setup that sets work_dir")

        cmd = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model", model.name,
        ]

        system_message = "\n\n".join(
            m.content for m in state.messages if m.role == "system"
        )
        if system_message:
            cmd.extend(["--append-system-prompt", system_message])

        cmd.append(state.input_text)

        # Strip env vars that would confuse a nested Claude Code session.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        for key in list(env.keys()):
            if key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_"):
                del env[key]

        await _run_agent_cli(
            cmd, env=env, cwd=work_dir, state=state, parser=_parse_claude_event
        )
        return state

    return solve


@solver
def codex() -> Solver:
    """Run the OpenAI Codex CLI inside `state.store['work_dir']`.

    Auto-detects when the inspect model is an OpenRouter model and rewrites
    the `codex` invocation to route through `https://openrouter.ai/api/v1`
    using `OPENROUTER_API_KEY`. This makes it possible to test arbitrary
    OpenRouter-hosted models against the codex harness:

        export OPENROUTER_API_KEY=...
        just eval my-eval openrouter/anthropic/claude-3.5-sonnet --solver codex
        just eval my-eval openrouter/qwen/qwen3-coder            --solver codex

    The model name is passed through to codex unchanged, so it must be in the
    `vendor/model` form that OpenRouter uses.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        work_dir = state.store.get("work_dir")
        if not work_dir:
            raise RuntimeError("codex() requires a setup that sets work_dir")

        cmd = [
            "codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)

        if _is_openrouter(model):
            if not env.get("OPENROUTER_API_KEY"):
                raise RuntimeError(
                    "codex() detected an OpenRouter model but OPENROUTER_API_KEY"
                    " is not set in the environment."
                )
            cmd.extend(_codex_openrouter_overrides())

        cmd.extend(["--model", model.name, state.input_text])

        await _run_agent_cli(
            cmd, env=env, cwd=work_dir, state=state, parser=_parse_codex_event
        )
        return state

    return solve


def _is_openrouter(model) -> bool:
    """True if the inspect model is routed through OpenRouter.

    Checks both the API class name (`type(model.api).__name__`) and the
    `INSPECT_EVAL_MODEL` env var that inspect-ai sets when running an eval.
    """
    api = getattr(model, "api", None)
    if api is not None and "openrouter" in type(api).__name__.lower():
        return True
    full = os.environ.get("INSPECT_EVAL_MODEL", "")
    return full.startswith("openrouter/")


def _codex_openrouter_overrides() -> list[str]:
    """Return `-c key=value` codex flags that route through OpenRouter.

    Codex's `-c` flag accepts TOML key/value pairs, so string values must be
    quoted. These flags are equivalent to a `~/.codex/config.toml` containing:

        model_provider = "openrouter"

        [model_providers.openrouter]
        name     = "OpenRouter"
        base_url = "https://openrouter.ai/api/v1"
        env_key  = "OPENROUTER_API_KEY"
        wire_api = "chat"
    """
    return [
        "-c", 'model_provider="openrouter"',
        "-c", 'model_providers.openrouter.name="OpenRouter"',
        "-c", 'model_providers.openrouter.base_url="https://openrouter.ai/api/v1"',
        "-c", 'model_providers.openrouter.env_key="OPENROUTER_API_KEY"',
        "-c", 'model_providers.openrouter.wire_api="chat"',
    ]


# ---------------------------------------------------------------------------
# Agent CLI plumbing — shared between claude_code() and codex()
# ---------------------------------------------------------------------------


async def _run_agent_cli(cmd, env, cwd, state, parser) -> None:
    """Run an agent CLI subprocess, streaming stdout/stderr into the transcript.

    `parser(event_dict, state, lookup)` translates each JSON event into chat
    messages appended to `state.messages`. `lookup` is a dict the parser may
    use to thread tool_use_id -> tool_use across events.
    """
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    lookup: dict = {}
    stderr_lines: list[str] = []

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
                event = json.loads(line.decode())
                transcript().info(event)
                parser(event, state, lookup)
            except json.JSONDecodeError:
                transcript().info(line.decode().rstrip())

    await asyncio.gather(read_stdout(), read_stderr())
    await process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Agent CLI failed (exit {process.returncode}, cmd={cmd[0]}):\n"
            + "\n".join(stderr_lines)
        )


def _parse_claude_event(event: dict, state: TaskState, lookup: dict) -> None:
    event_type = event.get("type")
    if event_type == "assistant":
        message = event.get("message", {})
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "text":
                state.messages.append(
                    ChatMessageAssistant(content=item.get("text"), metadata=message)
                )
            elif item_type == "tool_use":
                lookup[item.get("id")] = item
    elif event_type == "user":
        message = event.get("message", {})
        for item in message.get("content", []):
            if item.get("type") == "tool_result":
                tool_use = lookup.get(item.get("tool_use_id"), {})
                state.messages.append(
                    ChatMessageTool(
                        content=item.get("content", ""),
                        function=tool_use.get("name"),
                        metadata={"tool_use": tool_use, "tool_result": item},
                    )
                )


def _parse_codex_event(event: dict, state: TaskState, lookup: dict) -> None:
    # https://github.com/openai/codex/blob/main/docs/exec.md#json-output-mode
    item = event.get("item", {})
    item_type = item.get("type")
    if item_type in ("reasoning", "agent_message"):
        state.messages.append(
            ChatMessageAssistant(content=item.get("text"), metadata=event)
        )
    elif item_type == "command_execution":
        state.messages.append(
            ChatMessageTool(
                content=item.get("aggregated_output"),
                function=item.get("command"),
                metadata=event,
            )
        )
