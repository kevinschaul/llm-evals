"""Helpers for evals that run external coding-agent CLIs in temp work dirs."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
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


async def _run(*cmd: str, cwd: Optional[str] = None) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout, stderr


async def git_init_commit(work_dir: str) -> None:
    """Initialize a throwaway git repo so cleanup can capture a diff."""
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
    src_path = Path(src_dir)

    @solver
    def setup() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            tmpdir = tempfile.mkdtemp(prefix="agentic-eval-")
            work_dir = os.path.join(tmpdir, "work")
            async with span("copy_fixture"):
                transcript().info(f"Copying fixture {src_path} -> {work_dir}")
                shutil.copytree(src_path, work_dir)
                await git_init_commit(work_dir)
            state.store.set("work_dir", work_dir)
            transcript().info({"work_dir": work_dir})
            return state

        return solve

    return setup()


def serve_site(site_dir: str | Path) -> Solver:
    """Serve a static site and substitute its local URL into the prompt."""
    site_path = Path(site_dir)

    @solver
    def setup() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            if not site_path.exists():
                raise RuntimeError(f"site dir not found: {site_path}")
            tmpdir = tempfile.mkdtemp(prefix="agentic-eval-")
            work_dir = os.path.join(tmpdir, "work")
            os.makedirs(work_dir)
            async with span("serve_site"):
                await git_init_commit(work_dir)
                with socket.socket() as s:
                    s.bind(("127.0.0.1", 0))
                    port = s.getsockname()[1]
                url = f"http://127.0.0.1:{port}/"
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "http.server",
                        str(port),
                        "--bind",
                        "127.0.0.1",
                        "--directory",
                        str(site_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                deadline = time.monotonic() + 5
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"HTTP server exited before serving {site_path}"
                        )
                    try:
                        with urllib.request.urlopen(url, timeout=0.2):
                            break
                    except OSError:
                        if time.monotonic() >= deadline:
                            proc.kill()
                            proc.wait(timeout=1)
                            raise RuntimeError(
                                f"Timed out waiting for HTTP server at {url}"
                            )
                        await asyncio.sleep(0.05)
                transcript().info(f"Serving {site_path} at {url} (pid {proc.pid})")
            state.store.set("work_dir", work_dir)
            state.store.set("server_pid", proc.pid)
            state._input = state.input_text.replace("{url}", url)
            return state

        return solve

    return setup()


async def _stop_pid(pid: int) -> None:
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + 1
    while True:
        try:
            waited_pid, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if waited_pid:
            return
        if time.monotonic() >= deadline:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                return
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            return
        await asyncio.sleep(0.05)


def _rm_tree(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _capture_files(work_dir: str, names: list[str]) -> dict[str, str | None]:
    captured = {}
    for name in names:
        path = Path(work_dir, name)
        captured[name] = path.read_text(errors="replace") if path.exists() else None
    return captured


def cleanup_workdir(on_finish=None, capture=None):
    """Capture eval artifacts and remove temp dirs before scoring runs."""

    async def cleanup(state: TaskState) -> None:
        loop = asyncio.get_event_loop()

        pid = state.store.get("server_pid")
        if pid:
            await _stop_pid(pid)

        work_dir = state.store.get("work_dir")
        if work_dir and os.path.exists(work_dir):
            await _run("git", "add", "-N", ".", cwd=work_dir)
            rc, out, err = await _run("git", "diff", cwd=work_dir)
            if rc == 0:
                state.store.set("diff", out.decode(errors="replace"))
            else:
                state.store.set(
                    "diff",
                    f"(git diff failed: rc={rc}, stderr={err.decode(errors='replace')})",
                )
            if capture:
                state.store.set("captured_files", _capture_files(work_dir, capture))
            if on_finish:
                await on_finish(state, work_dir)
            tmpdir = os.path.dirname(work_dir)
            await loop.run_in_executor(None, _rm_tree, tmpdir)

        pi_cfg_dir = state.store.get("pi_cfg_dir")
        if pi_cfg_dir:
            await loop.run_in_executor(None, _rm_tree, pi_cfg_dir)

    return cleanup


@scorer(metrics=[])
def git_diff() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        diff = state.store.get("diff")
        if diff is None:
            return Score(
                value="I",
                explanation="(no diff in state.store — did cleanup_workdir() run?)",
            )
        return Score(value="C", explanation=diff)

    return score


@solver
def require_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        raise RuntimeError(
            "No solver specified. Re-run with --solver claude_code, "
            "--solver codex, or --solver pi."
        )

    return solve


@solver
def claude_code() -> Solver:
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
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        work_dir = state.store.get("work_dir")
        if not work_dir:
            raise RuntimeError("codex() requires a setup that sets work_dir")

        is_openrouter = type(model.api).__name__ == "OpenRouterAPI"
        cmd = [
            "codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model", model.name,
        ]
        if is_openrouter:
            cmd += ["-c", "model_provider=openrouter"]
        cmd.append(state.input_text)

        await _run_agent_cli(
            cmd, env=os.environ.copy(), cwd=work_dir, state=state,
            parser=_parse_codex_event,
        )
        return state

    return solve


async def _run_agent_cli(cmd, env, cwd, state, parser) -> None:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=10 * 1024 * 1024,
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
                elif event.get("type") == "result" and event.get("is_error"):
                    message = event.get("result")
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
                        content=_coerce_tool_content(item.get("content", "")),
                        function=tool_use.get("name"),
                        metadata={"tool_use": tool_use, "tool_result": item},
                    )
                )


def _coerce_tool_content(content):
    """Convert Claude tool_result content blocks to inspect-safe text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                try:
                    parts.append(json.dumps(block))
                except (TypeError, ValueError):
                    parts.append(str(block))
        return "\n".join(parts)
    return str(content)


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
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        work_dir = state.store.get("work_dir")
        if not work_dir:
            raise RuntimeError("pi() requires a setup that sets work_dir")

        env = os.environ.copy()

        effective_base_url = os.environ.get("PI_BASE_URL", base_url)
        if effective_base_url:
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
            "-p",
            "--mode", "json",
            "--no-session",
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
    api = getattr(model, "api", None)
    api_name = type(api).__name__.removesuffix("API").lower() if api else ""
    if not api_name:
        return model.name
    return f"{api_name}/{model.name}"


def _parse_pi_event(event: dict, state: TaskState, lookup: dict) -> None:
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
