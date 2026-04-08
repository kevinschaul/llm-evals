# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a dual-system LLM evaluation framework that combines:

1. **Python evaluation engine** (`eval.py`) - Runs LLM tests against various providers with caching and assertion support
2. **Observable Framework dashboard** (`src/`) - Web interface for viewing results and interactive data exploration

### Key Components

**Python Evaluation System:**
- `eval.py` - Main evaluation runner that processes YAML configs, executes tests against LLM providers, and outputs CSV results
- Supports both CSV-based tests (`tests.csv` with `__expected` assertion columns) and inline YAML test definitions
- Includes caching via SQLite databases, result transforms (e.g., `parse_boolean`), and assertion types (`equals`, `contains`)
- Uses `llm` library for provider abstraction (OpenAI, Anthropic, Gemini, Ollama, etc.)

**Observable Framework Dashboard:**
- `src/index.md` - Landing page explaining the eval philosophy
- `src/evals/*/index.md` - Individual eval result pages with interactive tables
- `src/components/` - Shared UI components for aggregate tables, results tables, and selection details
- Real-time preview via `npm run dev`, builds static site for deployment

**Eval Structure:**
Each eval lives in `src/evals/[name]/` with:
- `eval.yaml` - Defines providers, prompts, test data source, and transforms
- `tests.csv` - Test cases with variables and `__expected` columns for assertions
- `index.md` - Observable page displaying results with interactive tables
- `results/` - Generated CSV files (results.csv, aggregate.csv)

## Development Commands

**Setup:**
```bash
uv sync                    # Install Python dependencies
npm install               # Install Node dependencies
```

**Running Evaluations:**
```bash
just eval [config-name]   # Run specific eval (e.g., just eval social-media-insults)
just eval-all            # Run all evaluations
uv run eval.py src/evals/[name]/eval.yaml --max-per-provider N  # Run with limit for testing
uv run eval.py src/evals/[name]/eval.yaml --provider NAME  # Run for just one provider
```

**Dashboard Development:**
```bash
just dev      # Start dev server with live reload
```

## Working with Evaluations

**CSV Test Format:**
- Regular columns become template variables (e.g., `{{content}}` in prompts)  
- Columns starting with `__` become assertions (e.g., `__expected` → equals assertion)
- Boolean comparisons automatically handle string/boolean type conversion

**Provider Configuration:**
- Model specified as `provider/model` (e.g., `openai/gpt-4o-mini`)
- Transforms like `parse_boolean` convert JSON `{"category": false}` to boolean `False`
- Temperature and other model parameters supported

**Observable Components:**
- Use shared components: `AggregateTable()`, `ResultsTable()`, `getSelectionDetailsConfig()`
- Components return configuration objects, not rendered elements (Observable Framework limitation)
- Keep display logic inline in `.md` files where `Inputs` and `display()` are available

## Agentic Evals

Agentic evals run an external coding agent (Claude Code, Codex, pi, ...)
inside a temporary work directory, then capture `git diff` of whatever the
agent changed. A diff is usually enough to eyeball how well the agent did, so
there's no golden-output assertion.

The shared building blocks live in `agentic.py` at the project root. The
Justfile exports `PYTHONPATH=justfile_directory()` so eval files can
`from agentic import ...` directly.

`agentic.py` exports:

- **Setups** (prepare a `work_dir` and stash it in `state.store`):
  - `clone_git_repo(url, commit=None)` — clone+checkout a remote git repo
  - `copy_fixture(src_dir)` — copy a local fixture directory and `git init` it
- **Solvers** (agent harnesses, run inside `work_dir`, forward `--model`):
  - `claude_code()` — runs the `claude` CLI with `stream-json` output
  - `codex()` — runs `codex exec --json`
  - `pi(base_url=None, provider="llama-swap")` — runs the `pi` CLI
    (https://github.com/badlogic/pi-mono) with `--mode json`. Pi has
    first-class OpenRouter support out of the box, so any
    `openrouter/<vendor>/<model>` from inspect's `--model` flag works.
    Set `base_url` (or `PI_BASE_URL`) to point at a local OpenAI-compatible
    server like llama-swap, llama-server, Ollama, vLLM, etc. — see below.
- **Scorer:** `git_diff()` — returns the diff in the score `explanation`,
  which `extract_results.py` surfaces as the `result` column.
- **Cleanup:** `cleanup_workdir()` — removes the temp work dir and any
  per-solver temp dirs (e.g. the synthesized pi config dir).

To add a new agentic eval, create `src/evals/<name>/eval.py`:

```python
from pathlib import Path
from inspect_ai import task, Task
from inspect_ai.dataset import MemoryDataset, Sample
from agentic import (
    claude_code, codex, pi, copy_fixture, cleanup_workdir, git_diff,
)

# Re-export so `--solver claude_code` / `--solver codex` / `--solver pi`
# resolves them.
__all__ = ["claude_code", "codex", "pi", "my_eval"]

@task
def my_eval() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input="Do the thing.")]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
```

Then run it against any model + harness:

```bash
just eval my-eval anthropic/claude-sonnet-4-5    --solver claude_code
just eval my-eval openai/gpt-5-codex             --solver codex
just eval my-eval openrouter/openai/gpt-4o-mini  --solver pi
```

The `pi` solver derives pi's `provider/model-id` form from inspect's
`Model.api` class name, so it works with any provider pi supports
(Anthropic, OpenAI, OpenRouter, Google, ...) — just set the matching
`*_API_KEY`. For OpenRouter-hosted models, prefer `pi` over `codex`. See
`agentic.py` for details.

### Local models via llama-swap (or any OpenAI-compatible server)

Pass `base_url` to `pi()` and the solver synthesizes a throwaway
`models.json` in a temp `pi-cfg-XXXX` dir, points pi at it via
`PI_CODING_AGENT_DIR`, and forces the `--model` arg to
`<provider>/<inspect-model-name>`. Nothing in `~/.pi/agent/` is touched.
The synthesized config sets `compat.supportsDeveloperRole=false` and
`compat.supportsReasoningEffort=false` because llama.cpp's HTTP server
(the common llama-swap backend) doesn't recognize the OpenAI `developer`
role or the `reasoning_effort` field.

```python
# in the eval, point at your local llama-swap:
@task
def my_eval() -> Task:
    return Task(
        dataset=MemoryDataset([Sample(input="Do the thing.")]),
        setup=copy_fixture(Path(__file__).parent / "fixture"),
        solver=pi(base_url="http://localhost:8080/v1"),
        cleanup=cleanup_workdir(),
        scorer=git_diff(),
    )
```

```bash
# pick the model at run time — inspect just needs *some* model spec, so
# `mockllm/<id>` works as a placeholder. The bare `<id>` is what gets
# forwarded to pi (and through llama-swap to the underlying server).
just eval my-eval mockllm/qwen2.5-coder-7b --solver pi

# override the URL without editing the eval:
PI_BASE_URL=http://other-host:8080/v1 \
  just eval my-eval mockllm/qwen2.5-coder-7b --solver pi
```

## File Patterns

**Eval Configuration (`eval.yaml`):**
```yaml
description: "Brief description"
providers:
  - id: provider-name
    model: provider/model-name
    temperature: 0.1
    transforms: ["parse_boolean"]
prompts:
  - "Template with {{variables}}"
tests: file://tests.csv
```

**Test Data (`tests.csv`):**
```csv
variable1,variable2,__expected
"test content","more data","expected_result"
```

**Results Page (`index.md`):**
Uses shared components for consistent UI across all evals. Import components and data, configure tables with shared functions, implement selection details with inline logic.

## Development Notes

- Use `uv run` for all Python commands to ensure proper virtual environment
- Observable Framework requires absolute imports and specific component patterns
- Results are cached in SQLite databases for faster re-runs
- Each eval is independent but follows consistent patterns for maintainability
