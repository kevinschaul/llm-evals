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
    """Initialize a git repo so `git diff` works after the agent runs.

    `commit.gpgsign=false` is forced because the eval's git history is
    throwaway — we don't want to depend on (or interact with) the host's
    commit-signing setup.
    """
    git_id = (
        "-c", "user.email=eval@example.com",
        "-c", "user.name=eval",
        "-c", "commit.gpgsign=false",
        "-c", "tag.gpgsign=false",
    )
    for cmd in (
        ("git", "init", "-q", "-b", "main"),
        ("git", *git_id, "add", "."),
        ("git", *git_id, "commit", "-q", "--allow-empty", "--no-gpg-sign",
         "-m", "initial"),
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
    """Cleanup: capture the work-dir diff, then remove temp dirs.

    inspect_ai runs Plan cleanup *before* the scorer (see
    `inspect_ai.solver._plan.Plan.__call__`'s `finally:` block), so a scorer
    that tries to read `work_dir` after cleanup will get FileNotFoundError.
    To work around that we capture `git diff` here while the dir still
    exists and stash it on `state.store["diff"]`. The `git_diff()` scorer
    just reads from the store.

    Solvers may also stash auxiliary tmp dirs in `state.store` (e.g. the
    `pi()` solver writes a temp `models.json` and stashes the dir under
    `pi_cfg_dir` so this cleanup hook can remove it).
    """

    def _rm(path: Optional[str]) -> None:
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    async def cleanup(state: TaskState) -> None:
        loop = asyncio.get_event_loop()

        work_dir = state.store.get("work_dir")
        if work_dir and os.path.exists(work_dir):
            # Capture diff before we wipe the dir.
            await _run("git", "add", "-N", ".", cwd=work_dir)
            rc, out, err = await _run("git", "diff", cwd=work_dir)
            if rc == 0:
                state.store.set("diff", out.decode(errors="replace"))
            else:
                state.store.set(
                    "diff",
                    f"(git diff failed: rc={rc}, stderr={err.decode(errors='replace')})",
                )
            tmpdir = os.path.dirname(work_dir)
            await loop.run_in_executor(None, _rm, tmpdir)

        pi_cfg_dir = state.store.get("pi_cfg_dir")
        if pi_cfg_dir:
            await loop.run_in_executor(None, _rm, pi_cfg_dir)

    return cleanup


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[])
def git_diff() -> Scorer:
    """Score by surfacing the `git diff` captured by `cleanup_workdir()`."""

    async def score(state: TaskState, target: Target) -> Score:
        diff = state.store.get("diff")
        if diff is None:
            return Score(
                value="I",
                explanation="(no diff in state.store — did cleanup_workdir() run?)",
            )
        return Score(value="C", explanation=diff)

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

    The model name is forwarded as `codex --model …` and must be one that
    the local `codex` install knows how to talk to (i.e. an OpenAI model with
    `OPENAI_API_KEY` set, or whatever provider you've configured in
    `~/.codex/config.toml`). For OpenRouter-hosted models, use the `pi`
    solver instead — pi has first-class OpenRouter support out of the box.
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
            "--model", model.name,
            state.input_text,
        ]

        await _run_agent_cli(
            cmd, env=os.environ.copy(), cwd=work_dir, state=state,
            parser=_parse_codex_event,
        )
        return state

    return solve


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
    event_error_lines: list[str] = []

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
                if event.get("type") == "error":
                    message = event.get("message")
                    if message:
                        event_error_lines.append(str(message))
                elif event.get("type") == "turn.failed":
                    error = event.get("error", {})
                    message = error.get("message")
                    if message:
                        event_error_lines.append(str(message))
                parser(event, state, lookup)
            except json.JSONDecodeError:
                transcript().info(line.decode().rstrip())

    await asyncio.gather(read_stdout(), read_stderr())
    await process.wait()

    if process.returncode != 0:
        error_lines = stderr_lines + event_error_lines
        raise RuntimeError(
            f"Agent CLI failed (exit {process.returncode}, cmd={cmd[0]}):\n"
            + "\n".join(error_lines)
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


@solver
def pi(base_url: Optional[str] = None, provider: str = "llama-swap") -> Solver:
    """Run the `pi` coding agent CLI inside `state.store['work_dir']`.

    https://github.com/badlogic/pi-mono — pi has first-class OpenRouter
    support out of the box (no provider config gymnastics needed) and a
    `--mode json` event stream we can hook into.

    Two modes:

    1. **Cloud / built-in provider** (default). The inspect model spec is
       forwarded as `pi --model <provider>/<model-id>`, where `<provider>`
       is derived from the inspect API class name (`OpenRouterAPI` →
       `openrouter`, `AnthropicAPI` → `anthropic`, ...).

           export OPENROUTER_API_KEY=...
           just eval my-eval openrouter/openai/gpt-4o-mini --solver pi
           just eval my-eval anthropic/claude-haiku-4-5    --solver pi

    2. **Custom OpenAI-compatible base URL** (llama-swap, llama-server,
       Ollama, vLLM, LM Studio, ...). When `base_url` is set (or the
       `PI_BASE_URL` env var is set at run time), the solver writes a
       throwaway `models.json` to a temp `pi-cfg-XXXX` directory and
       points pi at it via `PI_CODING_AGENT_DIR`, so we never touch the
       user's real `~/.pi/agent/`. The synthesized config has one
       provider (`provider`, default `llama-swap`) with one model entry
       whose id is the bare inspect model name. The compat block disables
       `developer` role and `reasoning_effort` because llama.cpp's server
       (the common llama-swap backend) supports neither.

           # in the eval:
           solver=pi(base_url="http://localhost:8080/v1")

           # at run time, --model picks which model llama-swap should swap to:
           just eval my-eval mockllm/qwen2.5-coder-7b --solver pi

           # or override the URL without editing the eval:
           PI_BASE_URL=http://other-host:8080/v1 \\
             just eval my-eval mockllm/qwen2.5-coder-7b --solver pi
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        work_dir = state.store.get("work_dir")
        if not work_dir:
            raise RuntimeError("pi() requires a setup that sets work_dir")

        env = os.environ.copy()

        effective_base_url = os.environ.get("PI_BASE_URL", base_url)
        if effective_base_url:
            # Custom base URL: synthesize a one-off models.json in a
            # throwaway PI_CODING_AGENT_DIR so we never touch ~/.pi/.
            model_id = model.name
            cfg_dir = tempfile.mkdtemp(prefix="pi-cfg-")
            models_json = {
                "providers": {
                    provider: {
                        "baseUrl": effective_base_url,
                        "api": "openai-completions",
                        "apiKey": "dummy",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                        "models": [{"id": model_id}],
                    }
                }
            }
            Path(cfg_dir, "models.json").write_text(json.dumps(models_json))
            env["PI_CODING_AGENT_DIR"] = cfg_dir
            state.store.set("pi_cfg_dir", cfg_dir)
            model_arg = f"{provider}/{model_id}"
        else:
            model_arg = _pi_model_arg(model)

        cmd = [
            "pi",
            "-p",                # non-interactive: process prompt and exit
            "--mode", "json",    # JSONL events on stdout
            "--no-session",      # ephemeral, don't write sessions/
            "--model", model_arg,
        ]

        system_message = "\n\n".join(
            m.content for m in state.messages if m.role == "system"
        )
        if system_message:
            cmd.extend(["--append-system-prompt", system_message])

        cmd.append(state.input_text)

        await _run_agent_cli(
            cmd, env=env, cwd=work_dir, state=state, parser=_parse_pi_event
        )
        return state

    return solve


def _pi_model_arg(model) -> str:
    """Compose the `provider/model-id` string that pi's `--model` flag expects.

    Inspect's `Model.api` is a provider-specific class (e.g. `OpenRouterAPI`,
    `AnthropicAPI`). We strip the trailing `API` and lowercase to get pi's
    provider name. `model.name` is already the bare model id within that
    provider, so concatenating gives the right `vendor/id` form.
    """
    api = getattr(model, "api", None)
    api_name = type(api).__name__.removesuffix("API").lower() if api else ""
    if not api_name:
        return model.name
    return f"{api_name}/{model.name}"


def _parse_pi_event(event: dict, state: TaskState, lookup: dict) -> None:
    """Translate a pi `--mode json` event into inspect chat messages.

    pi event types we care about:

      message_end          — emitted once for the user message and once for
                             each assistant message. Assistant messages have
                             a `content` array of `text` / `thinking` /
                             `toolCall` blocks.
      tool_execution_start — fires before a tool runs (`toolCallId`,
                             `toolName`, `args`).
      tool_execution_end   — fires after a tool finishes (`toolCallId`,
                             `toolName`, `result`, `isError`). We emit the
                             `ChatMessageTool` here so the result is captured.
    """
    event_type = event.get("type")

    if event_type == "message_end":
        message = event.get("message", {})
        if message.get("role") != "assistant":
            return
        for item in message.get("content", []) or []:
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text") or ""
                if text:
                    state.messages.append(
                        ChatMessageAssistant(content=text, metadata=message)
                    )
            elif item_type == "thinking":
                thinking = item.get("thinking") or ""
                if thinking:
                    state.messages.append(
                        ChatMessageAssistant(
                            content=f"<thinking>{thinking}</thinking>",
                            metadata=message,
                        )
                    )
            # toolCall blocks are surfaced via tool_execution_end below,
            # which has both the call args and the result in one place.

    elif event_type == "tool_execution_start":
        lookup[event.get("toolCallId")] = event

    elif event_type == "tool_execution_end":
        start = lookup.pop(event.get("toolCallId"), {})
        result = event.get("result")
        if not isinstance(result, str):
            try:
                result = json.dumps(result)
            except (TypeError, ValueError):
                result = str(result) if result is not None else ""
        state.messages.append(
            ChatMessageTool(
                content=result,
                function=event.get("toolName"),
                metadata={"start": start, "end": event},
            )
        )
