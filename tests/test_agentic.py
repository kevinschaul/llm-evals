"""Tests for `agentic.py`.

These exercise the pure event parsers, the `pi()` solver's command/env
construction in both cloud and base_url modes, the `cleanup_workdir()`
hook, and the `copy_fixture()` setup + `git_diff()` scorer pair against a
real on-disk git repo. The agent CLI subprocess is always mocked — we
never spawn `pi`, `claude`, or `codex` from a test.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import agentic
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import TaskState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubState:
    """Tiny stand-in for `TaskState` for parser tests.

    The parsers only touch `state.messages.append(...)`, so anything with a
    `messages` list works.
    """

    def __init__(self) -> None:
        self.messages: list = []


def make_task_state(work_dir: str | None = None) -> TaskState:
    """Build a real `TaskState` and optionally seed `work_dir` in the store."""
    state = TaskState(
        model="mockllm/test-model",
        sample_id="test",
        epoch=0,
        input="do the thing",
        messages=[],
    )
    if work_dir is not None:
        state.store.set("work_dir", work_dir)
    return state


def run_solver(solver, state: TaskState) -> None:
    """Drive a Solver to completion synchronously."""
    asyncio.run(solver(state, generate=lambda *a, **kw: None))


# ---------------------------------------------------------------------------
# _pi_model_arg
# ---------------------------------------------------------------------------


class OpenRouterAPI:
    """Stub provider class — name matters: `_pi_model_arg` strips trailing
    `API` and lowercases the result, so this becomes `openrouter`."""


class AnthropicAPI:
    """Stub provider class — becomes `anthropic`."""


class _StubModel:
    def __init__(self, name: str, api: object | None) -> None:
        self.name = name
        self.api = api


def test_pi_model_arg_strips_api_suffix_openrouter():
    model = _StubModel("openai/gpt-4o-mini", OpenRouterAPI())
    assert agentic._pi_model_arg(model) == "openrouter/openai/gpt-4o-mini"


def test_pi_model_arg_strips_api_suffix_anthropic():
    model = _StubModel("claude-sonnet-4-5", AnthropicAPI())
    assert agentic._pi_model_arg(model) == "anthropic/claude-sonnet-4-5"


def test_pi_model_arg_falls_back_when_no_api():
    model = _StubModel("bare-model", None)
    assert agentic._pi_model_arg(model) == "bare-model"


# ---------------------------------------------------------------------------
# _parse_pi_event
# ---------------------------------------------------------------------------


def test_parse_pi_event_assistant_text_appends_message():
    state = StubState()
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello world"}],
        },
    }
    agentic._parse_pi_event(event, state, {})
    assert len(state.messages) == 1
    assert isinstance(state.messages[0], ChatMessageAssistant)
    assert state.messages[0].text == "hello world"


def test_parse_pi_event_assistant_thinking_wrapped_in_tags():
    state = StubState()
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "step 1, step 2"}],
        },
    }
    agentic._parse_pi_event(event, state, {})
    assert len(state.messages) == 1
    assert state.messages[0].text == "<thinking>step 1, step 2</thinking>"


def test_parse_pi_event_user_message_skipped():
    state = StubState()
    event = {
        "type": "message_end",
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "ignored"}],
        },
    }
    agentic._parse_pi_event(event, state, {})
    assert state.messages == []


def test_parse_pi_event_empty_text_skipped():
    state = StubState()
    event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
        },
    }
    agentic._parse_pi_event(event, state, {})
    assert state.messages == []


def test_parse_pi_event_tool_execution_pair_emits_tool_message():
    state = StubState()
    lookup: dict = {}
    agentic._parse_pi_event(
        {
            "type": "tool_execution_start",
            "toolCallId": "call-1",
            "toolName": "bash",
            "args": {"cmd": "ls"},
        },
        state,
        lookup,
    )
    assert lookup["call-1"]["toolName"] == "bash"
    assert state.messages == []  # nothing emitted yet

    agentic._parse_pi_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "bash",
            "result": "file1\nfile2",
            "isError": False,
        },
        state,
        lookup,
    )
    assert "call-1" not in lookup  # popped
    assert len(state.messages) == 1
    msg = state.messages[0]
    assert isinstance(msg, ChatMessageTool)
    assert msg.text == "file1\nfile2"
    assert msg.function == "bash"


def test_parse_pi_event_tool_result_dict_is_json_serialized():
    state = StubState()
    lookup: dict = {}
    agentic._parse_pi_event(
        {
            "type": "tool_execution_end",
            "toolCallId": "call-99",
            "toolName": "read",
            "result": {"path": "/foo", "bytes": 42},
            "isError": False,
        },
        state,
        lookup,
    )
    msg = state.messages[0]
    assert json.loads(msg.text) == {"path": "/foo", "bytes": 42}


# ---------------------------------------------------------------------------
# _parse_codex_event
# ---------------------------------------------------------------------------


def test_parse_codex_event_reasoning_emits_assistant():
    state = StubState()
    agentic._parse_codex_event(
        {"item": {"type": "reasoning", "text": "thinking..."}}, state, {}
    )
    assert len(state.messages) == 1
    assert isinstance(state.messages[0], ChatMessageAssistant)
    assert state.messages[0].text == "thinking..."


def test_parse_codex_event_agent_message_emits_assistant():
    state = StubState()
    agentic._parse_codex_event(
        {"item": {"type": "agent_message", "text": "final answer"}}, state, {}
    )
    assert state.messages[0].text == "final answer"


def test_parse_codex_event_command_execution_emits_tool():
    state = StubState()
    agentic._parse_codex_event(
        {
            "item": {
                "type": "command_execution",
                "command": "ls -la",
                "aggregated_output": "drwx...",
            }
        },
        state,
        {},
    )
    assert isinstance(state.messages[0], ChatMessageTool)
    assert state.messages[0].text == "drwx..."
    assert state.messages[0].function == "ls -la"


# ---------------------------------------------------------------------------
# _parse_claude_event
# ---------------------------------------------------------------------------


def test_parse_claude_event_assistant_text():
    state = StubState()
    agentic._parse_claude_event(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "hi"}],
            },
        },
        state,
        {},
    )
    assert state.messages[0].text == "hi"


def test_parse_claude_event_tool_use_then_result_links_function_name():
    state = StubState()
    lookup: dict = {}
    agentic._parse_claude_event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Bash", "input": {}}
                ],
            },
        },
        state,
        lookup,
    )
    assert state.messages == []  # tool_use stashed, not emitted
    assert lookup["tu-1"]["name"] == "Bash"

    agentic._parse_claude_event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": "ok",
                    }
                ],
            },
        },
        state,
        lookup,
    )
    assert len(state.messages) == 1
    assert isinstance(state.messages[0], ChatMessageTool)
    assert state.messages[0].function == "Bash"
    assert state.messages[0].text == "ok"


# ---------------------------------------------------------------------------
# pi() solver — cloud mode
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_cli(monkeypatch):
    """Replace `_run_agent_cli` with a recorder; return the capture dict."""
    captured: dict = {}

    async def fake_run_agent_cli(cmd, env, cwd, state, parser):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        cfg_dir = env.get("PI_CODING_AGENT_DIR")
        if cfg_dir and (Path(cfg_dir) / "models.json").exists():
            captured["models_json"] = json.loads(
                (Path(cfg_dir) / "models.json").read_text()
            )

    monkeypatch.setattr(agentic, "_run_agent_cli", fake_run_agent_cli)
    return captured


@pytest.fixture
def tmp_work_dir():
    """Materialize a fresh `work_dir` and clean it up after."""
    parent = tempfile.mkdtemp(prefix="agentic-test-")
    work_dir = os.path.join(parent, "work")
    os.makedirs(work_dir)
    yield work_dir
    shutil.rmtree(parent, ignore_errors=True)


def test_pi_solver_cloud_mode_forwards_provider_prefix(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.delenv("PI_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentic,
        "get_model",
        lambda: _StubModel("openai/gpt-4o-mini", OpenRouterAPI()),
    )
    state = make_task_state(work_dir=tmp_work_dir)

    run_solver(agentic.pi(), state)

    cmd = captured_cli["cmd"]
    assert cmd[0] == "pi"
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "json"
    assert "--no-session" in cmd
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "openrouter/openai/gpt-4o-mini"
    assert cmd[-1] == "do the thing"
    # Cloud mode never sets PI_CODING_AGENT_DIR or stashes a cfg dir
    assert "PI_CODING_AGENT_DIR" not in captured_cli["env"]
    assert state.store.get("pi_cfg_dir") is None
    assert "models_json" not in captured_cli


def test_pi_solver_appends_system_prompt(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.delenv("PI_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("foo", AnthropicAPI())
    )
    state = make_task_state(work_dir=tmp_work_dir)
    state.messages.append(
        # role=system message that should flow into --append-system-prompt
        type("M", (), {"role": "system", "content": "Be terse."})()
    )

    run_solver(agentic.pi(), state)

    cmd = captured_cli["cmd"]
    assert "--append-system-prompt" in cmd
    assert cmd[cmd.index("--append-system-prompt") + 1] == "Be terse."


def test_pi_solver_raises_without_work_dir(monkeypatch, captured_cli):
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("foo", AnthropicAPI())
    )
    state = make_task_state(work_dir=None)
    with pytest.raises(RuntimeError, match="work_dir"):
        run_solver(agentic.pi(), state)


# ---------------------------------------------------------------------------
# pi() solver — base_url mode
# ---------------------------------------------------------------------------


def test_pi_solver_base_url_synthesizes_models_json(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.delenv("PI_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("qwen2.5-coder-7b", None)
    )
    state = make_task_state(work_dir=tmp_work_dir)

    try:
        run_solver(
            agentic.pi(base_url="http://localhost:8080/v1"),
            state,
        )

        # Command forces the llama-swap/<id> form
        cmd = captured_cli["cmd"]
        assert cmd[cmd.index("--model") + 1] == "llama-swap/qwen2.5-coder-7b"

        # Subprocess env has PI_CODING_AGENT_DIR pointing at a pi-cfg-* dir
        cfg_dir = captured_cli["env"]["PI_CODING_AGENT_DIR"]
        assert os.path.basename(cfg_dir).startswith("pi-cfg-")
        assert state.store.get("pi_cfg_dir") == cfg_dir

        # And that dir contains the right models.json
        models_json = captured_cli["models_json"]
        provider_cfg = models_json["providers"]["llama-swap"]
        assert provider_cfg["baseUrl"] == "http://localhost:8080/v1"
        assert provider_cfg["api"] == "openai-completions"
        assert provider_cfg["compat"] == {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        }
        assert provider_cfg["models"] == [{"id": "qwen2.5-coder-7b"}]
    finally:
        shutil.rmtree(state.store.get("pi_cfg_dir"), ignore_errors=True)


def test_pi_solver_base_url_from_env_var_overrides_default(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.setenv("PI_BASE_URL", "http://env-host:9000/v1")
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("model-x", None)
    )
    state = make_task_state(work_dir=tmp_work_dir)

    try:
        # No base_url in code — env var alone should activate base_url mode
        run_solver(agentic.pi(), state)
        models_json = captured_cli["models_json"]
        assert (
            models_json["providers"]["llama-swap"]["baseUrl"]
            == "http://env-host:9000/v1"
        )
    finally:
        cfg_dir = state.store.get("pi_cfg_dir")
        if cfg_dir:
            shutil.rmtree(cfg_dir, ignore_errors=True)


def test_pi_solver_env_var_overrides_constructor(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.setenv("PI_BASE_URL", "http://env-wins:9000/v1")
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("model-x", None)
    )
    state = make_task_state(work_dir=tmp_work_dir)

    try:
        run_solver(
            agentic.pi(base_url="http://constructor-loses:8080/v1"),
            state,
        )
        models_json = captured_cli["models_json"]
        assert (
            models_json["providers"]["llama-swap"]["baseUrl"]
            == "http://env-wins:9000/v1"
        )
    finally:
        cfg_dir = state.store.get("pi_cfg_dir")
        if cfg_dir:
            shutil.rmtree(cfg_dir, ignore_errors=True)


def test_pi_solver_custom_provider_name(
    monkeypatch, captured_cli, tmp_work_dir
):
    monkeypatch.delenv("PI_BASE_URL", raising=False)
    monkeypatch.setattr(
        agentic, "get_model", lambda: _StubModel("local-model", None)
    )
    state = make_task_state(work_dir=tmp_work_dir)

    try:
        run_solver(
            agentic.pi(
                base_url="http://localhost:8080/v1", provider="vllm-prod"
            ),
            state,
        )
        cmd = captured_cli["cmd"]
        assert cmd[cmd.index("--model") + 1] == "vllm-prod/local-model"
        assert "vllm-prod" in captured_cli["models_json"]["providers"]
    finally:
        cfg_dir = state.store.get("pi_cfg_dir")
        if cfg_dir:
            shutil.rmtree(cfg_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# cleanup_workdir()
# ---------------------------------------------------------------------------


def test_cleanup_workdir_removes_work_dir_parent_and_pi_cfg_dir():
    parent = tempfile.mkdtemp(prefix="agentic-cleanup-")
    work_dir = os.path.join(parent, "work")
    os.makedirs(work_dir)
    cfg_dir = tempfile.mkdtemp(prefix="pi-cfg-")

    state = make_task_state(work_dir=work_dir)
    state.store.set("pi_cfg_dir", cfg_dir)

    asyncio.run(agentic.cleanup_workdir()(state))

    assert not os.path.exists(parent)
    assert not os.path.exists(cfg_dir)


def test_cleanup_workdir_handles_missing_dirs_gracefully():
    state = make_task_state(work_dir="/nonexistent/path/to/nowhere")
    state.store.set("pi_cfg_dir", "/also/nonexistent")
    # Should not raise
    asyncio.run(agentic.cleanup_workdir()(state))


def test_cleanup_workdir_no_pi_cfg_dir_only_removes_workdir():
    parent = tempfile.mkdtemp(prefix="agentic-cleanup-")
    work_dir = os.path.join(parent, "work")
    os.makedirs(work_dir)
    state = make_task_state(work_dir=work_dir)

    asyncio.run(agentic.cleanup_workdir()(state))

    assert not os.path.exists(parent)


# ---------------------------------------------------------------------------
# copy_fixture() + git_diff() integration
# ---------------------------------------------------------------------------


def test_copy_fixture_and_git_diff_capture_modifications(tmp_path):
    # Build a tiny fixture
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "hello.txt").write_text("original\n")

    state = make_task_state(work_dir=None)
    setup = agentic.copy_fixture(fixture)
    asyncio.run(setup(state, generate=lambda *a, **kw: None))

    work_dir = state.store.get("work_dir")
    assert work_dir is not None
    assert os.path.exists(os.path.join(work_dir, "hello.txt"))
    assert os.path.exists(os.path.join(work_dir, ".git"))

    # Simulate the agent modifying a file and creating a new one
    Path(work_dir, "hello.txt").write_text("modified\n")
    Path(work_dir, "new.txt").write_text("created by agent\n")

    score = asyncio.run(agentic.git_diff()(state, target=None))
    assert score.value == "C"
    assert "modified" in score.explanation
    assert "new.txt" in score.explanation
    assert "hello.txt" in score.explanation

    # Cleanup the temp dir the fixture setup created
    asyncio.run(agentic.cleanup_workdir()(state))


def test_git_diff_returns_incorrect_when_no_workdir():
    state = make_task_state(work_dir=None)
    score = asyncio.run(agentic.git_diff()(state, target=None))
    assert score.value == "I"
    assert "no work_dir" in score.explanation
