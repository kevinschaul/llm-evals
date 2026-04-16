"""End-to-end integration test for the `pi()` solver.

Spins up a tiny SSE-streaming OpenAI Chat Completions fake server on a
local port, points pi at it via `base_url`, and runs the full solver →
pi CLI → tool execution → cleanup → git_diff loop.

What this proves that the unit tests don't:
  * the synthesized `models.json` + `PI_CODING_AGENT_DIR` actually makes
    pi talk to the configured base URL,
  * pi's expected wire format (SSE streaming, lowercase tool names, OpenAI
    Chat Completions tool_calls) round-trips through `_parse_pi_event`,
  * `cleanup_workdir()` captures a real diff before deletion and
    `git_diff()` surfaces it (i.e. the cleanup-vs-scorer ordering fix
    from inspect_ai.solver._plan.Plan.__call__ works on a real run).

Skipped when the `pi` CLI isn't installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import agentic


# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _stream_text(wfile, model: str, text: str) -> None:
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
    }))
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": text}}],
    }))
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))
    wfile.write(b"data: [DONE]\n\n")


def _stream_tool_call(wfile, model: str, tool_name: str, args_obj: dict) -> None:
    args_str = json.dumps(args_obj)
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
    }))
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{
                "index": 0, "id": "call_marker", "type": "function",
                "function": {"name": tool_name, "arguments": ""},
            }],
        }}],
    }))
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": args_str}}],
        }}],
    }))
    wfile.write(_sse({
        "id": "x", "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }))
    wfile.write(b"data: [DONE]\n\n")


def _pick_bash_tool(req: dict) -> str | None:
    """Pick the bash-equivalent from pi's advertised tools.

    Pi advertises lowercase names like `bash`, `read`, `edit`, `write`.
    """
    for t in req.get("tools", []) or []:
        fn = t.get("function") or t
        name = fn.get("name", "")
        if name.lower() in {"bash", "shell", "exec", "run_command", "execute"}:
            return name
    return None


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Two-turn scripted handler: tool_call then 'done'.

    `script_state` (a list shared via class attribute) tracks the turn.
    """

    script_state: list = []  # set per-test in the fixture
    bash_command: str = ""   # set per-test

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        type(self).script_state.append({
            "messages": req.get("messages", []),
            "tools": [(t.get("function") or t).get("name")
                      for t in (req.get("tools") or [])],
        })
        n = len(type(self).script_state)
        model = req.get("model", "fake-model")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if n == 1:
            bash_name = _pick_bash_tool(req)
            if bash_name is None:
                # Pi didn't advertise a shell tool — emit text and let the
                # test fail loudly on the assertion side.
                _stream_text(self.wfile, model, "no bash tool available")
                return
            _stream_tool_call(
                self.wfile, model, bash_name,
                {"command": type(self).bash_command},
            )
        else:
            _stream_text(self.wfile, model, "done")

    def do_GET(self):
        if self.path.endswith("/models"):
            data = json.dumps({
                "object": "list",
                "data": [{"id": "fake-model", "object": "model"}],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a, **kw):
        pass  # silence


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_sse_server():
    """Start the scripted SSE server on a free port; return (url, state)."""
    port = _free_port()
    # Reset script state for this test
    _ScriptedHandler.script_state = []
    httpd = HTTPServer(("127.0.0.1", port), _ScriptedHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1", _ScriptedHandler.script_state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Helpers (duplicated minimally from test_agentic.py to keep this file
# self-contained — these are throwaway test fixtures, not production code)
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


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


pi_required = pytest.mark.skipif(
    shutil.which("pi") is None,
    reason="pi CLI not installed",
)


@pi_required
def test_pi_solver_e2e_against_fake_sse_server(
    monkeypatch, tmp_path, fake_sse_server,
):
    base_url, requests_seen = fake_sse_server

    # Build a fixture and copy_fixture it into a work_dir.
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README").write_text("starter file\n")

    state = _make_state()
    asyncio.run(
        agentic.copy_fixture(fixture)(state, generate=lambda *a, **kw: None)
    )
    work_dir = state.store.get("work_dir")
    assert work_dir is not None and os.path.isdir(work_dir)

    # Force a deterministic model name and clear PI_BASE_URL so the
    # explicit base_url= path is exercised.
    monkeypatch.delenv("PI_BASE_URL", raising=False)
    monkeypatch.setattr(agentic, "get_model", lambda: _StubModel("fake-model"))

    # The bash command the fake server will tell pi to execute.
    marker_text = "hello from fake server"
    _ScriptedHandler.bash_command = (
        f"echo '{marker_text}' > marker.txt"
    )

    # Run the solver: it should hit the fake server, get a tool_call,
    # exec the bash command in work_dir, send the result back, and exit.
    asyncio.run(
        agentic.pi(base_url=base_url, provider="fake-provider")(
            state, generate=lambda *a, **kw: None
        )
    )

    # The bash tool should have created marker.txt inside work_dir.
    marker = Path(work_dir) / "marker.txt"
    assert marker.is_file(), (
        f"pi did not create marker.txt in {work_dir} — server saw "
        f"{len(requests_seen)} request(s); first tools advertised: "
        f"{requests_seen[0]['tools'] if requests_seen else 'none'}"
    )
    assert marker_text in marker.read_text()

    # The fake server should have seen exactly two requests:
    #   1. initial prompt -> we returned a tool_call
    #   2. tool result -> we returned 'done'
    assert len(requests_seen) == 2, (
        f"expected 2 turns, got {len(requests_seen)}"
    )
    last_msgs = requests_seen[-1]["messages"]
    assert last_msgs[-1]["role"] == "tool", (
        f"final turn's last message should be the tool result; "
        f"got role={last_msgs[-1]['role']}"
    )

    # Now run cleanup + scorer and verify the diff round-trips.
    asyncio.run(agentic.cleanup_workdir()(state))
    assert not os.path.exists(work_dir), "cleanup should remove work_dir"

    score = asyncio.run(agentic.git_diff()(state, target=None))
    assert score.value == "C"
    assert "marker.txt" in score.explanation
    assert marker_text in score.explanation
