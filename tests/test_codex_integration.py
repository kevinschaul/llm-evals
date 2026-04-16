"""End-to-end integration test for the `codex()` solver.

Spins up a tiny SSE-streaming OpenAI **Responses API** fake server on a
local port, points codex at it via `OPENAI_BASE_URL`, and runs the full
solver -> codex CLI -> shell exec -> cleanup -> git_diff loop.

Codex uses the OpenAI Responses API (`POST /v1/responses`), not the
Chat Completions API. The streaming events are different (`response.*`
event types), and the shell tool name is `shell` with a `command` array
parameter (passed to `execvp()`).

Skipped when the `codex` CLI isn't on PATH.
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
# SSE helpers — Responses API event format
# ---------------------------------------------------------------------------


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _stream_function_call(wfile, *, call_id: str, name: str, args_obj: dict) -> None:
    """Stream a single function_call output item end-to-end."""
    args_str = json.dumps(args_obj)
    response_id = "resp_test"
    item_id = f"fc_{call_id}"

    wfile.write(_sse({
        "type": "response.created",
        "response": {
            "id": response_id, "object": "response", "status": "in_progress",
            "model": "fake-model", "output": [],
        },
    }))
    wfile.write(_sse({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        },
    }))
    wfile.write(_sse({
        "type": "response.function_call_arguments.delta",
        "item_id": item_id, "output_index": 0, "delta": args_str,
    }))
    wfile.write(_sse({
        "type": "response.function_call_arguments.done",
        "item_id": item_id, "output_index": 0, "arguments": args_str,
    }))
    wfile.write(_sse({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": item_id, "call_id": call_id, "name": name,
            "arguments": args_str, "status": "completed",
        },
    }))
    wfile.write(_sse({
        "type": "response.completed",
        "response": {
            "id": response_id, "object": "response", "status": "completed",
            "model": "fake-model",
            "output": [{
                "type": "function_call",
                "id": item_id, "call_id": call_id, "name": name,
                "arguments": args_str, "status": "completed",
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }))
    wfile.write(b"data: [DONE]\n\n")


def _stream_text(wfile, text: str) -> None:
    """Stream a single assistant text message end-to-end."""
    response_id = "resp_test_final"
    item_id = "msg_done"
    wfile.write(_sse({
        "type": "response.created",
        "response": {
            "id": response_id, "object": "response", "status": "in_progress",
            "model": "fake-model", "output": [],
        },
    }))
    wfile.write(_sse({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "message",
            "id": item_id,
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    }))
    wfile.write(_sse({
        "type": "response.content_part.added",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }))
    wfile.write(_sse({
        "type": "response.output_text.delta",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "delta": text,
    }))
    wfile.write(_sse({
        "type": "response.output_text.done",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "text": text,
    }))
    wfile.write(_sse({
        "type": "response.content_part.done",
        "item_id": item_id, "output_index": 0, "content_index": 0,
        "part": {"type": "output_text", "text": text, "annotations": []},
    }))
    wfile.write(_sse({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "message", "id": item_id, "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        },
    }))
    wfile.write(_sse({
        "type": "response.completed",
        "response": {
            "id": response_id, "object": "response", "status": "completed",
            "model": "fake-model",
            "output": [{
                "type": "message", "id": item_id, "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }))
    wfile.write(b"data: [DONE]\n\n")


def _pick_shell_tool(req: dict) -> str | None:
    """Pick the shell-equivalent from codex's advertised tools.

    Codex advertises a tool literally named `shell` (lowercase).
    """
    for t in req.get("tools", []) or []:
        name = t.get("name") or (t.get("function") or {}).get("name", "")
        if name in {"shell", "local_shell"}:
            return name
    return None


class _ResponsesHandler(BaseHTTPRequestHandler):
    """Two-turn scripted handler for the Responses API.

    Per-test state lives on class attributes.
    """

    script_state: list = []
    bash_command: list = []  # ["bash", "-lc", "..."]

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        type(self).script_state.append({
            "input": req.get("input", []),
            "tools": [t.get("name") for t in (req.get("tools") or [])],
        })
        n = len(type(self).script_state)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if n == 1:
            shell_name = _pick_shell_tool(req) or "shell"
            _stream_function_call(
                self.wfile,
                call_id="call_marker",
                name=shell_name,
                args_obj={"command": type(self).bash_command,
                          "workdir": "."},
            )
        else:
            _stream_text(self.wfile, "done")

    def do_GET(self):
        # codex probes /v1/models?client_version=... at startup
        if "/models" in self.path:
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
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_responses_server():
    port = _free_port()
    _ResponsesHandler.script_state = []
    httpd = HTTPServer(("127.0.0.1", port), _ResponsesHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1", _ResponsesHandler.script_state
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


codex_required = pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI not installed",
)


@codex_required
def test_codex_solver_e2e_against_fake_responses_server(
    monkeypatch, tmp_path, fake_responses_server,
):
    base_url, requests_seen = fake_responses_server

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README").write_text("starter\n")

    state = _make_state()
    asyncio.run(
        agentic.copy_fixture(fixture)(state, generate=lambda *a, **kw: None)
    )
    work_dir = state.store.get("work_dir")
    assert work_dir is not None and os.path.isdir(work_dir)

    # Point codex at the fake server. OPENAI_API_KEY just needs to exist;
    # the fake server doesn't validate it.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setattr(agentic, "get_model", lambda: _StubModel("fake-model"))

    marker_text = "hello from fake codex server"
    _ResponsesHandler.bash_command = [
        "bash", "-lc", f"echo '{marker_text}' > marker.txt"
    ]

    asyncio.run(
        agentic.codex()(state, generate=lambda *a, **kw: None)
    )

    marker = Path(work_dir) / "marker.txt"
    assert marker.is_file(), (
        f"codex did not create marker.txt in {work_dir} — server saw "
        f"{len(requests_seen)} request(s); first turn tools advertised: "
        f"{requests_seen[0]['tools'] if requests_seen else 'none'}"
    )
    assert marker_text in marker.read_text()

    # Two turns expected: prompt -> function_call -> function_call_output -> done.
    assert len(requests_seen) == 2, (
        f"expected 2 turns, got {len(requests_seen)}"
    )
    last_input = requests_seen[-1]["input"]
    # Codex sends function_call_output items in the second request to
    # report the shell command's stdout/stderr back to the model.
    assert any(it.get("type") == "function_call_output" for it in last_input), (
        f"second turn should include a function_call_output item; "
        f"got types {[it.get('type') for it in last_input]}"
    )

    # Verify cleanup + scorer round-trip.
    asyncio.run(agentic.cleanup_workdir()(state))
    assert not os.path.exists(work_dir)

    score = asyncio.run(agentic.git_diff()(state, target=None))
    assert score.value == "C"
    assert "marker.txt" in score.explanation
    assert marker_text in score.explanation
