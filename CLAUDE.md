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
uv run eval.py src/evals/[name]/eval.yaml --limit N  # Run with limit for testing
```

**Dashboard Development:**
```bash
npm run dev              # Start dev server with live reload
just view-dashboard      # Alias for npm run dev
npm run build           # Build static site
npm run deploy          # Deploy to Observable
```

**Dashboard Viewing:**
```bash
just view               # View with promptfoo (legacy)
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