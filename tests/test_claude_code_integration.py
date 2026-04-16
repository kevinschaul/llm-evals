"""End-to-end integration test for the `claude_code()` solver.

Spins up a tiny SSE-streaming Anthropic **Messages API** fake server on a
local port, points the `claude` CLI at it via `ANTHROPIC_BASE_URL`, and runs
the full solver -> claude CLI -> Bash exec -> cleanup -> git_diff loop.

Claude Code uses Anthropic's Messages API (`POST /v1/messages`), with SSE
events like `message_start`, `content_block_start/delta/stop`,
`message_delta`, `message_stop`. The shell tool is `Bash` (PascalCase) and
its input is `{"command": "...", "description": "..."}` — `command` is a
single shell string (not an array like codex's `shell`).

Skipped when the `claude` CLI isn't on PATH.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import agentic


# ---------------------------------------------------------------------------
# SSE helpers — Anthropic Messages API event format
# ---------------------------------------------------------------------------


def _sse(event_type: str, payload: dict) -> bytes:
    """Anthropic SSE frames carry an explicit `event:` line."""
    return (
        f"event: {event_type}\n"
        f"data: {json.dumps(payload)}\n\n"
    ).encode()


def _stream_tool_use(wfile, *, tool_use_id: str, name: str, args_obj: dict) -> None:
    """Stream a single tool_use content block end-to-end."""
    args_str = json.dumps(args_obj)

    wfile.write(_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "fake-model",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }))
    wfile.write(_sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": tool_use_id,
            "name": name,
            "input": {},
        },
    }))
    wfile.write(_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": args_str},
    }))
    wfile.write(_sse("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    }))
    wfile.write(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 10},
    }))
    wfile.write(_sse("message_stop", {"type": "message_stop"}))


def _stream_text(wfile, text: str) -> None:
    """Stream a single assistant text message end-to-end."""
    wfile.write(_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_test_final",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "fake-model",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }))
    wfile.write(_sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    wfile.write(_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": text},
    }))
    wfile.write(_sse("content_block_stop", {
        "type": "content_block_stop", "index": 0,
    }))
    wfile.write(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 5},
    }))
    wfile.write(_sse("message_stop", {"type": "message_stop"}))


def _pick_bash_tool(req: dict) -> str | None:
    """Pick the Bash-equivalent from claude's advertised tools.

    Claude Code advertises a tool literally named `Bash` (PascalCase).
    """
    for t in req.get("tools", []) or []:
        name = t.get("name", "")
        if name in {"Bash", "bash", "shell", "Shell"}:
            return name
    return None


class _MessagesHandler(BaseHTTPRequestHandler):
    """Two-turn scripted handler for the Anthropic Messages API.

    Per-test state lives on class attributes.
    """

    script_state: list = []
    bash_command: str = ""

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        type(self).script_state.append({
            "path": self.path,
            "messages": req.get("messages", []),
            "tools": [t.get("name") for t in (req.get("tools") or [])],
            "system": req.get("system"),
            "model": req.get("model"),
        })
        # Count only "real" turns: those that advertise our Bash tool.
        # Claude Code 2.x makes side-channel calls (auto-mode classifier,
        # hooks, etc.) that we want to ignore for scripting purposes.
        real_turns = [
            s for s in type(self).script_state
            if any(t in {"Bash", "bash"} for t in s["tools"])
        ]
        n = len(real_turns)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        tool_name = _pick_bash_tool(req)
        if tool_name is None:
            # Side-channel call (auto-mode classifier, etc.) — just emit a
            # minimal text response so claude moves on.
            _stream_text(self.wfile, "ok")
            return

        if n == 1:
            _stream_tool_use(
                self.wfile,
                tool_use_id="toolu_marker",
                name=tool_name,
                args_obj={
                    "command": type(self).bash_command,
                    "description": "create marker file",
                },
            )
        else:
            _stream_text(self.wfile, "done")

    def do_GET(self):
        # Some claude setup probes might hit /v1/models or similar.
        self.send_response(404)
        self.end_headers()

    def log_message(self, *a, **kw):
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_messages_server():
    port = _free_port()
    _MessagesHandler.script_state = []
    httpd = HTTPServer(("127.0.0.1", port), _MessagesHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _MessagesHandler.script_state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class _StubModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.api = None


def _make_state():
    from inspect_ai.solver import TaskState
    return TaskState(
        model="mockllm/fake-model",
        sample_id="test",
        epoch=0,
        input="please create marker.txt",
        messages=[],
    )


claude_required = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)


@claude_required
def test_claude_code_solver_e2e_against_fake_messages_server(
    monkeypatch, tmp_path, fake_messages_server,
):
    base_url, requests_seen = fake_messages_server

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README").write_text("starter\n")

    state = _make_state()
    asyncio.run(
        agentic.copy_fixture(fixture)(state, generate=lambda *a, **kw: None)
    )
    work_dir = state.store.get("work_dir")
    assert work_dir is not None and os.path.isdir(work_dir)

    # Point claude CLI at the fake server. ANTHROPIC_AUTH_TOKEN survives
    # the solver's env-stripping (it pops only ANTHROPIC_API_KEY).
    # IS_SANDBOX=1 is required to allow --dangerously-skip-permissions
    # when running as root.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-fake-test")
    monkeypatch.setenv("IS_SANDBOX", "1")
    monkeypatch.setattr(agentic, "get_model", lambda: _StubModel("fake-model"))

    marker_text = "hello from fake claude server"
    _MessagesHandler.bash_command = f"echo '{marker_text}' > marker.txt"

    asyncio.run(
        agentic.claude_code()(state, generate=lambda *a, **kw: None)
    )

    # Filter to just the "main" turns (those that advertise Bash).
    main_turns = [
        r for r in requests_seen
        if any(t in {"Bash", "bash"} for t in r["tools"])
    ]

    marker = Path(work_dir) / "marker.txt"
    assert marker.is_file(), (
        f"claude did not create marker.txt in {work_dir} — server saw "
        f"{len(requests_seen)} total request(s), {len(main_turns)} main; "
        f"first turn tools advertised: "
        f"{requests_seen[0]['tools'] if requests_seen else 'none'}"
    )
    assert marker_text in marker.read_text()

    # At least 2 main turns expected: prompt -> tool_use, then a follow-up
    # request whose user message reports the tool_result back to the model.
    # Claude Code may make additional summarization or auto-mode calls
    # after that, so we don't enforce a strict upper bound.
    assert len(main_turns) >= 2, (
        f"expected >=2 main turns, got {len(main_turns)} "
        f"(of {len(requests_seen)} total requests)"
    )

    # At least one follow-up turn must carry the tool_result back.
    def _has_tool_result(turn):
        for msg in turn["messages"]:
            content = msg.get("content")
            if isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "tool_result"
                for c in content
            ):
                return True
        return False

    assert any(_has_tool_result(t) for t in main_turns[1:]), (
        "expected a follow-up main turn carrying a tool_result block; "
        f"got {len(main_turns)} main turns with no tool_result"
    )

    # Verify cleanup + scorer round-trip.
    asyncio.run(agentic.cleanup_workdir()(state))
    assert not os.path.exists(work_dir)

    score = asyncio.run(agentic.git_diff()(state, target=None))
    assert score.value == "C"
    assert "marker.txt" in score.explanation
    assert marker_text in score.explanation
