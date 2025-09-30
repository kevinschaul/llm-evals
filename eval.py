#!/usr/bin/env python

from dotenv import load_dotenv

load_dotenv()

from argparse import ArgumentParser
import csv
import json
import logging
import os
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union
from jinja2 import Template
import tempfile
import yaml
import llm
from llm.cli import logs_db_path, migrate
from llm.utils import make_schema_id, sqlite_utils
from pydantic import ValidationError

"""
Run the LLM eval suite specified in the passed-in config file.
"""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="eval.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

# Cache database paths
db_path_main = "eval_cache.db"
db_paths_search = [db_path_main, logs_db_path()]


class RawResponse:
    def __init__(self, db, response_row) -> None:
        self.db = db
        self.response_row = response_row

    def to_response(self):
        return llm.Response.from_row(self.db, self.response_row)

    def text(self):
        return self.response_row["response"]


def find_cached_response(
    db: sqlite_utils.Database,
    prompt_text: str,
    model_id_or_alias: str,
    system_prompt_text: Optional[str] = None,
    schema: Optional[Union[Dict[str, Any], type[llm.models.BaseModel]]] = None,
    fragments: Optional[List[str]] = None,
    system_fragments: Optional[List[str]] = None,
    attachments: Optional[List[llm.Attachment]] = None,
    **options: Any,
) -> Optional[RawResponse]:
    """
    Searches the LLM database for an exact cached query and returns the response row if found.
    """
    try:
        model_obj = llm.get_model(model_id_or_alias)
    except llm.UnknownModelError:
        # Model not found, so no cache entry can exist for it.
        return None

    canonical_model_id = model_obj.model_id

    # 1. Canonicalize options
    try:
        prompt_options_obj = model_obj.Options(**options)
    except ValidationError:
        # Invalid options for this model, so it couldn't have been logged like this.
        return None

    # Prepare options_json as it would be stored in the database
    options_dict_for_db = {
        k: v for k, v in dict(prompt_options_obj).items() if v is not None
    }
    options_json_to_match = json.dumps(options_dict_for_db or {}, sort_keys=True)

    # 2. Canonicalize schema
    schema_id_to_match: Optional[str] = None
    if schema:
        if not isinstance(schema, dict) and issubclass(schema, llm.models.BaseModel):
            current_schema_dict = schema.model_json_schema()
        else:
            current_schema_dict = schema
        schema_id_to_match, _ = make_schema_id(current_schema_dict)

    where_clauses = []
    params = {}

    where_clauses.append("model = :model_id")
    params["model_id"] = canonical_model_id

    where_clauses.append("prompt = :prompt_text")
    params["prompt_text"] = prompt_text

    if system_prompt_text is None:
        where_clauses.append("system IS NULL")
    else:
        where_clauses.append("system = :system_prompt_text")
        params["system_prompt_text"] = system_prompt_text

    where_clauses.append("options_json = :options_json")
    params["options_json"] = options_json_to_match

    if schema_id_to_match is None:
        where_clauses.append("schema_id IS NULL")
    else:
        where_clauses.append("schema_id = :schema_id")
        params["schema_id"] = schema_id_to_match

    candidate_responses = list(
        db["responses"].rows_where(
            where=" AND ".join(where_clauses),
            where_args=params,
            select="id",
            order_by="id DESC",
        )
    )

    if not candidate_responses:
        return None

    # Convert list of dicts to list of IDs
    candidate_response_ids = [row["id"] for row in candidate_responses]

    # 4. Canonicalize input fragments and attachments for comparison
    current_fragment_hashes = sorted([llm.Fragment(f).id() for f in (fragments or [])])
    current_system_fragment_hashes = sorted(
        [llm.Fragment(f).id() for f in (system_fragments or [])]
    )
    current_attachment_ids = sorted([att.id() for att in (attachments or [])])

    for response_id in candidate_response_ids:
        # Check fragments
        logged_prompt_fragments = list(
            db.query(
                """
            SELECT f.hash FROM fragments f
            JOIN prompt_fragments pf ON f.id = pf.fragment_id
            WHERE pf.response_id = ? ORDER BY f.hash
            """,
                [response_id],
            )
        )
        logged_prompt_fragment_hashes = [row["hash"] for row in logged_prompt_fragments]
        if logged_prompt_fragment_hashes != current_fragment_hashes:
            continue

        # Check system fragments
        logged_system_fragments = list(
            db.query(
                """
            SELECT f.hash FROM fragments f
            JOIN system_fragments sf ON f.id = sf.fragment_id
            WHERE sf.response_id = ? ORDER BY f.hash
            """,
                [response_id],
            )
        )
        logged_system_fragment_hashes = [row["hash"] for row in logged_system_fragments]
        if logged_system_fragment_hashes != current_system_fragment_hashes:
            continue

        # Check attachments
        logged_attachments = list(
            db.query(
                """
            SELECT a.id FROM attachments a
            JOIN prompt_attachments pa ON a.id = pa.attachment_id
            WHERE pa.response_id = ? ORDER BY a.id
            """,
                [response_id],
            )
        )
        logged_attachment_ids = [row["id"] for row in logged_attachments]
        if logged_attachment_ids != current_attachment_ids:
            continue

        # If all checks pass, this is our match
        full_response_row = db["responses"].get(response_id)
        if full_response_row:
            return RawResponse(db, full_response_row)

    return None


class EvalException(BaseException):
    pass


class CompletedResult(TypedDict):
    provider_id: str
    prompt_id: str
    test: str
    prompt: str
    result: Optional[str]
    error: Optional[str]
    duration_ms: Optional[float]
    passed: Optional[bool]
    expected: Optional[str]  # Store the expected value for display
    timestamp: str  # ISO format timestamp when test was run


class Color:
    # ANSI escape codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @classmethod
    def colored(cls, text, color):
        return f"{color}{text}{cls.END}"

    @classmethod
    def disable_if_needed(cls):
        if (
            os.getenv("NO_COLOR")
            or not hasattr(sys.stdout, "isatty")
            or not sys.stdout.isatty()
        ):
            for attr in dir(cls):
                if (
                    not attr.startswith("_")
                    and attr != "colored"
                    and attr != "disable_if_needed"
                ):
                    setattr(cls, attr, "")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def parse_boolean(text: str) -> bool:
    """Parse the output of an LLM call to a boolean."""
    text_lower = text.lower().strip()

    # Handle JSON responses with category field
    if "{" in text and '"category"' in text:
        try:
            data = json.loads(text)
            return bool(data.get("category", False))
        except json.JSONDecodeError:
            pass

    # Handle simple true/false responses
    if text_lower in ["true", "false"]:
        return text_lower == "true"

    # Handle responses with "true" or "false" in quotes
    if '"true"' in text_lower or "'true'" in text_lower:
        return True
    if '"false"' in text_lower or "'false'" in text_lower:
        return False

    # Original logic for TRUE/FALSE patterns
    TRUE_VAL = "TRUE"
    FALSE_VAL = "FALSE"
    regexp = rf"\b({TRUE_VAL}|{FALSE_VAL})\b"

    truthy = {
        val.upper()
        for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
    }
    if TRUE_VAL.upper() in truthy and FALSE_VAL.upper() not in truthy:
        return True
    elif FALSE_VAL.upper() in truthy and TRUE_VAL.upper() not in truthy:
        return False

    # If we can't parse it, return the original text and let the assertion handle it
    return text


def strip_think_tags(text: str) -> str:
    """Remove all content between <think> and </think> tags."""
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def extract_items_array(text: str) -> str:
    """Extract array from JSON, handling both top-level arrays and {items: []} structure."""
    try:
        data = json.loads(text)
        # If it's already an array, return as-is
        if isinstance(data, list):
            return json.dumps(data)
        # If it's an object with 'items' key, extract the items array
        elif isinstance(data, dict) and 'items' in data:
            return json.dumps(data['items'])
        # Otherwise return original
        else:
            return text
    except json.JSONDecodeError:
        # If it's not valid JSON, return original
        return text


def normalize_json(text: str) -> str:
    """Normalize JSON formatting by parsing and re-serializing with consistent formatting."""
    try:
        data = json.loads(text)
        # Re-serialize with consistent formatting: sorted keys, no extra whitespace
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    except json.JSONDecodeError:
        # If it's not valid JSON, return original
        return text


def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def apply_transforms(text: str, transforms: List[str]) -> Any:
    """Apply a list of transform functions to text."""
    result = text
    for transform_name in transforms:
        if transform_name == "strip_think_tags":
            result = strip_think_tags(result)
        elif transform_name == "parse_boolean":
            result = parse_boolean(result)
        elif transform_name == "extract_items_array":
            result = extract_items_array(result)
        elif transform_name == "normalize_json":
            result = normalize_json(result)
        elif transform_name == "to_uppercase":
            result = to_uppercase(result)
        else:
            logger.warning(f"Unknown transform: {transform_name}")
    return result


def cli():
    parser = ArgumentParser()
    parser.add_argument("eval_config", help="Eval config file (yaml)")

    # Test limiting option
    parser.add_argument(
        "--max-per-provider", type=int, help="Run at most N tests per provider"
    )

    # Provider filtering option
    parser.add_argument(
        "--provider", type=str, help="Run tests only for the specified provider ID"
    )

    # Debug mode for command tests
    parser.add_argument(
        "--debug-tempdir", action="store_true", help="Pause after command tests to inspect temp directory"
    )

    return parser.parse_args()


def check_assertion(actual, expected):
    """Simple assertion checker - detects patterns in expected and runs appropriate check."""
    # Convert expected to string if it's not already
    expected_str = str(expected)
    
    if expected_str.startswith("levenshtein(") and "):" in expected_str:
        # levenshtein(N):VALUE
        parts = expected_str.split("):", 1)
        distance = int(parts[0].replace("levenshtein(", ""))
        expected_value = parts[1]
        actual_distance = levenshtein_distance(str(actual).strip().upper(), expected_value.upper())
        return actual_distance <= distance
    
    elif expected_str.startswith("icontains:"):
        # icontains: VALUE (case-insensitive contains)
        expected_value = expected_str.replace("icontains:", "").strip()
        return expected_value.lower() in str(actual).lower()
    
    elif expected_str.startswith("icontains-any:"):
        # icontains-any: VALUE1, VALUE2 (case-insensitive contains any)
        expected_values = expected_str.replace("icontains-any:", "").split(',')
        for expected_value in expected_values:
            if expected_value.lower().strip() in str(actual).lower():
                return True
        return False
    
    elif expected_str.startswith("contains:"):
        # contains: VALUE (case-sensitive contains)
        expected_value = expected_str.replace("contains:", "").strip()
        return expected_value in str(actual)
    
    else:
        # Default: exact match with boolean conversion
        if isinstance(actual, bool) and isinstance(expected, str):
            if expected.lower() in ["true", "false"]:
                expected = expected.lower() == "true"
        elif isinstance(actual, str) and isinstance(expected, bool):
            if actual.lower() in ["true", "false"]:
                actual = actual.lower() == "true"
        return actual == expected


def load_csv_tests(csv_path: Path) -> List[Dict[str, Any]]:
    """Load tests from a CSV file, handling assertion columns."""
    tests = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Separate assertion columns from regular variables
            test_vars = {}
            expected_value = None

            for key, value in row.items():
                if key == "__expected":
                    test_vars["__expected"] = value
                    expected_value = value
                elif not key.startswith("__"):
                    test_vars[key] = value

            # Create test
            test = {
                "id": f"test-{i+1}",
                "vars": test_vars,
                "expected": expected_value,
            }
            tests.append(test)
    return tests


def add_prompt_ids(prompts: List[Any]) -> List[Dict[str, Any]]:
    """Process prompts to ensure they have IDs."""
    processed_prompts = []
    for i, prompt in enumerate(prompts):
        if isinstance(prompt, dict):
            # Prompt already has structure (id, text, etc.)
            processed_prompts.append(prompt)
        else:
            # Prompt is just a string, create structure
            processed_prompts.append({"id": f"prompt-{i+1}", "text": prompt})
    return processed_prompts


def load_config(eval_config):
    with open(eval_config) as f:
        config = yaml.safe_load(f)

    config_dir = Path(eval_config).parent

    # Handle CSV test files
    if isinstance(config.get("tests"), str) and config["tests"].startswith("file://"):
        csv_file = config["tests"].replace("file://", "")
        csv_path = config_dir / csv_file

        if csv_path.exists():
            config["tests"] = load_csv_tests(csv_path)
        else:
            raise EvalException(f"Tests file not found: {csv_path}")
    else:
        # Add IDs to inline tests and handle file:// references in expected values
        tests = config.get("tests", [])
        for i, test in enumerate(tests):
            if "id" not in test:
                test["id"] = f"test-{i+1}"
            
            # Handle file:// references in expected values
            if "expected" in test:
                expected_value = test["expected"]
                if isinstance(expected_value, str) and expected_value.startswith("file://"):
                    file_path = expected_value.replace("file://", "")
                    expected_file_path = config_dir / file_path
                    if expected_file_path.exists():
                        with open(expected_file_path) as f:
                            expected_value = f.read().strip()
                        # Apply same normalization to expected values as we do to results
                        expected_value = normalize_json(expected_value)
                        test["expected"] = expected_value

    # Process prompts to ensure they have IDs
    config["prompts"] = add_prompt_ids(config.get("prompts", []))

    return config


def run_command(command, temp_dir):
    print(Color.colored(f"Running: {command}", Color.BLUE))
    logger.debug(f"Running: {command}")

    process = subprocess.Popen(
        command,
        shell=True,
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        # Combine stderr into stdout
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    stdout_lines = []

    command_bin = command.split(" ")[0]
    for line in process.stdout:
        print(Color.colored(f"{command_bin}: {line}", Color.GRAY), end="")
        logger.debug(f"{command_bin}: {line}")
        stdout_lines.append(line)

    process.wait()

    full_output = "".join(stdout_lines)

    logger.debug(f"Return code: {process.returncode}")
    if process.returncode != 0:
        print(f"❌ Command failed: {command}")
        raise EvalException(f"Error running {command}\nOutput: {full_output}")
    else:
        print(f"✅ Completed: {command}")

    return full_output


def run_llm_test(
    provider: Dict[str, Any], test: Dict[str, Any], prompt: str, config_dir: Path
) -> tuple[str, Optional[float], bool]:
    """Run an LLM-based test using the llm library with caching. Returns (result, duration_ms, was_cached)."""

    model_name = provider.get("model")
    model_options = {
        k: v
        for k, v in provider.items()
        if k not in ["id", "model", "label", "transforms", "options"]
    }

    # Handle options field if present
    if "options" in provider:
        model_options.update(provider["options"])

    # Get test vars - all test configuration is now under vars
    test_vars = test.get("vars", {})
    
    # Process attachments
    attachments = []
    for attachment_path in test_vars.get("attachments", []):
        if isinstance(attachment_path, str):
            # Convert relative path to absolute path relative to config directory
            abs_path = config_dir / attachment_path
            attachments.append(llm.Attachment(path=str(abs_path)))
        elif isinstance(attachment_path, dict) and "path" in attachment_path:
            abs_path = config_dir / attachment_path["path"]
            attachments.append(llm.Attachment(path=str(abs_path)))

    # Process schema
    schema = None
    schema_config = test_vars.get("schema")
    if schema_config:
        if isinstance(schema_config, dict) and schema_config.get("syntax") == "dsl":
            schema = llm.schema_dsl(
                schema_config["content"], multi=schema_config.get("multi", False)
            )
        elif isinstance(schema_config, dict):
            schema = schema_config

    # Render prompt with variables (exclude non-template vars like attachments, schema)
    template = Template(prompt)
    template_vars = {k: v for k, v in test_vars.items() 
                    if k not in ["attachments", "schema", "transforms"]}
    rendered_prompt = template.render(**template_vars)

    # Initialize cache database
    db_main = sqlite_utils.Database(db_path_main)
    migrate(db_main)

    # Cache lookup phase
    cache_start = time.time()
    raw_response = None
    for db_path in db_paths_search:
        db = sqlite_utils.Database(db_path)
        try:
            raw_response = find_cached_response(
                db,
                prompt_text=rendered_prompt,
                model_id_or_alias=model_name,
                schema=schema,
                attachments=attachments,
                **model_options,
            )
            if raw_response:
                # If we found the response
                if db_path != db_path_main:
                    # If we found it in a db that is not the main db, replicate it
                    # to the main db
                    replication_start = time.time()
                    raw_response.to_response().log_to_db(db_main)
                    replication_duration = time.time() - replication_start
                    logger.info(f"Cache replication took: {replication_duration:.3f}s")
                break
        except Exception as e:
            logger.debug(f"Cache lookup error for {db_path}: {e}")
            continue

    cache_duration = time.time() - cache_start
    logger.info(f"Cache lookup took: {cache_duration:.3f}s")

    was_cached = raw_response is not None

    if not raw_response:
        # LLM call phase - no cache hit
        model = llm.get_model(model_name)
        logger.info(f"Using model: {model_name}")

        output = model.prompt(
            rendered_prompt, schema=schema, attachments=attachments, **model_options
        )

        # Gather text so that timing is accurate
        output.text()

        # DB write phase
        output.log_to_db(db_main)

        raw_response = output
        logger.info("Fresh LLM call completed")
    else:
        logger.info("Cache hit - skipped LLM call")

    # Get duration from database if available
    duration_ms = None
    if hasattr(raw_response, "response_row") and raw_response.response_row:
        duration_ms = raw_response.response_row.get("duration_ms")
    elif hasattr(raw_response, "_response") and hasattr(
        raw_response._response, "duration_ms"
    ):
        duration_ms = raw_response._response.duration_ms

    # Get text and apply transforms
    result_text = raw_response.text().strip()
    # Apply transforms from test config first, then provider config
    test_transforms = test.get("vars", {}).get("transforms", [])
    transforms = test_transforms + provider.get("transforms", [])
    if transforms:
        result_text = apply_transforms(result_text, transforms)

    logger.info(
        f"LLM test completed, duration: {duration_ms}ms"
        if duration_ms
        else "LLM test completed"
    )

    return result_text, duration_ms, was_cached


def run_command_test(
    provider: Dict[str, Any], test: Dict[str, Any], prompt: str, config_dir: Path, server_process: Optional[subprocess.Popen] = None, debug_tempdir: bool = False
) -> str:
    """Run a command-based test."""
    # Render prompt with test variables
    test_vars = test.get("vars", {})
    template_vars = {k: v for k, v in test_vars.items()
                    if k not in ["attachments", "schema", "transforms"]}
    rendered_prompt = Template(prompt).render(**template_vars)

    # Setup log file
    log_dir = config_dir / "results"
    os.makedirs(log_dir, exist_ok=True)
    # Slugify IDs for safe filenames
    provider_slug = re.sub(r'[^\w\-]', '_', provider['id'])
    test_slug = re.sub(r'[^\w\-]', '_', test['id'])
    log_file = log_dir / f"{provider_slug}_{test_slug}.log"

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug(f"Running test in temp dir {temp_dir}")
        print(Color.colored(f"Temp dir: {temp_dir}", Color.GRAY))

        all_output = []

        # Setup command from provider
        setup_command = provider.get("setup_command")
        if setup_command:
            setup_rendered = Template(setup_command).render(**template_vars)
            all_output.append(f"=== Setup Command ===\n{setup_rendered}\n\n")
            output = run_command(setup_rendered, temp_dir)
            all_output.append(f"{output}\n\n")

        # Eval command from provider
        eval_command_raw = Template(provider.get("command"))
        eval_command = eval_command_raw.render(prompt=rendered_prompt, **template_vars)
        all_output.append(f"=== Eval Command ===\n{eval_command}\n\n")
        output = run_command(eval_command, temp_dir)
        all_output.append(f"{output}\n\n")

        # Result command from provider
        result_command = provider.get("result_command")
        result_rendered = Template(result_command).render(**template_vars)
        all_output.append(f"=== Result Command ===\n{result_rendered}\n\n")
        result = run_command(result_rendered, temp_dir)
        all_output.append(f"{result}\n\n")

        # Write all output to log file
        with open(log_file, "w") as f:
            f.write("".join(all_output))
        print(Color.colored(f"Command log written to {log_file}", Color.BLUE))

        if debug_tempdir:
            print(Color.colored(f"\n🔍 Debug mode: Paused for inspection", Color.YELLOW + Color.BOLD))
            print(Color.colored(f"Temp directory: {temp_dir}", Color.YELLOW))
            print(Color.colored("Press Enter to continue and clean up temp dir...", Color.YELLOW))
            input()

        return result


def run_assertions(result: Any, expected: str) -> Optional[bool]:
    """Run assertion against the test result."""
    if not expected:
        logger.warning("Test has no assertions - manual review required")
        return None
    
    try:
        passed = check_assertion(result, expected)
        if not passed:
            logger.warning(f"Assertion failed: expected '{expected}', got '{result}'")
        return passed
    except Exception as e:
        logger.warning(f"Assertion failed with error: {e}")
        return False


def run_test(
    provider: Dict[str, Any], test: Dict[str, Any], prompt: str, config_dir: Path, debug_tempdir: bool = False
) -> tuple[Any, bool, float, bool]:
    """Run a single test and return (result, passed, duration_ms, was_cached)."""
    start_time = time.time()
    print(
        Color.colored(
            f"Running test {test['id']} with {provider['id']}", Color.BLUE + Color.BOLD
        )
    )
    logger.debug(f"Running test {test['id']} with {provider['id']}")

    # Determine test type based on provider configuration
    was_cached = False
    if "model" in provider:
        # LLM provider
        result, llm_duration_ms, was_cached = run_llm_test(
            provider, test, prompt, config_dir
        )
        # Use LLM duration if available, otherwise fall back to wall-clock time
        duration_ms = (
            llm_duration_ms
            if llm_duration_ms is not None
            else (time.time() - start_time) * 1000
        )
    elif "command" in provider:
        # Command provider
        result = run_command_test(provider, test, prompt, config_dir, debug_tempdir=debug_tempdir)
        duration_ms = (time.time() - start_time) * 1000
    else:
        raise EvalException(
            f"Unknown provider type for {provider.get('id', 'unknown')}"
        )

    # Run assertions
    expected = test.get("expected")
    passed = run_assertions(result, expected)

    return result, passed, duration_ms, was_cached


def aggregate_results(results: List[CompletedResult]) -> List[Dict[str, Any]]:
    """Aggregate results by provider_id to show summary statistics."""
    from collections import defaultdict

    # Group by provider_id and prompt_id
    grouped = defaultdict(list)
    for result in results:
        key = (result["provider_id"], result["prompt_id"])
        grouped[key].append(result)

    aggregated = []
    for (provider_id, prompt_id), provider_results in grouped.items():
        total_tests = len(provider_results)
        passed_tests = sum(1 for r in provider_results if r["passed"] is True)
        failed_tests = sum(1 for r in provider_results if r["passed"] is False)
        no_assertions_tests = sum(1 for r in provider_results if r["passed"] is None)

        # Calculate timing stats (only for successful runs)
        durations = [
            r["duration_ms"] for r in provider_results if r["duration_ms"] is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else None
        min_duration = min(durations) if durations else None
        max_duration = max(durations) if durations else None

        # Count errors
        error_count = sum(1 for r in provider_results if r["error"] is not None)

        aggregated.append(
            {
                "provider_id": provider_id,
                "prompt_id": prompt_id,
                "total_tests": total_tests,
                "assertions": total_tests - no_assertions_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "no_assertions": no_assertions_tests,
                "pass_rate": (
                    round(passed_tests / (passed_tests + failed_tests) * 100, 1)
                    if (passed_tests + failed_tests) > 0
                    else 0.0
                ),
                "errors": error_count,
                "avg_duration_ms": (
                    round(avg_duration, 1) if avg_duration is not None else None
                ),
                "min_duration_ms": (
                    round(min_duration, 1) if min_duration is not None else None
                ),
                "max_duration_ms": (
                    round(max_duration, 1) if max_duration is not None else None
                ),
            }
        )

    return aggregated


def write_results(results: List[CompletedResult], out_file: str, format: str = "csv"):
    """Write results to file in CSV or JSON format."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if format == "json":
        with open(out_file.replace(".csv", ".json"), "w") as f:
            json.dump(results, f, indent=2)
        print(
            Color.colored(
                f"Results written to {out_file.replace('.csv', '.json')}", Color.BLUE
            )
        )
    else:
        # Write detailed results
        # Dynamic fieldnames: start with core fields, then add any additional fields from results
        core_fieldnames = [
            "provider_id",
            "prompt_id",
            "test",
            "prompt",
            "result",
            "error",
            "duration_ms",
            "passed",
            "expected",
            "timestamp",
        ]
        additional_fieldnames = set()
        for result in results:
            additional_fieldnames.update(
                k for k in result.keys() if k not in core_fieldnames
            )
        fieldnames = core_fieldnames + sorted(additional_fieldnames)

        with open(out_file, "w") as o:
            writer = csv.DictWriter(o, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(Color.colored(f"Results written to {out_file}", Color.BLUE))

        # Write aggregated results
        aggregated = aggregate_results(results)
        agg_file = out_file.replace("results.csv", "aggregate.csv")
        agg_fieldnames = [
            "provider_id",
            "prompt_id",
            "total_tests",
            "assertions",
            "passed",
            "failed",
            "no_assertions",
            "pass_rate",
            "errors",
            "avg_duration_ms",
            "min_duration_ms",
            "max_duration_ms",
        ]
        with open(agg_file, "w") as o:
            writer = csv.DictWriter(o, fieldnames=agg_fieldnames)
            writer.writeheader()
            writer.writerows(aggregated)
        print(Color.colored(f"Summary written to {agg_file}", Color.BLUE))


def main():
    Color.disable_if_needed()
    args = cli()
    config = load_config(args.eval_config)
    config_dir = Path(args.eval_config).parent

    logger.info(f"Using LLM cache at: {logs_db_path()}")
    logger.debug(f"All models: {llm.get_models()}")

    providers = config["providers"]
    tests = config["tests"]
    prompts = config["prompts"]

    # Filter providers if specified
    if args.provider:
        providers = [p for p in providers if p["id"] == args.provider]
        if not providers:
            print(
                Color.colored(
                    f"Error: Provider '{args.provider}' not found in config", Color.RED
                )
            )
            available_providers = [p["id"] for p in config["providers"]]
            print(
                Color.colored(
                    f"Available providers: {', '.join(available_providers)}",
                    Color.YELLOW,
                )
            )
            return
        print(Color.colored(f"Filtering to provider: {args.provider}", Color.YELLOW))



    # Generate all combinations
    all_combinations = [(p, pr, t) for p in providers for t in tests for pr in prompts]

    # Apply limiting
    if args.max_per_provider:
        # Limit tests per provider
        limited_combinations = []
        for provider in providers:
            provider_tests = [
                (p, pr, t) for p, pr, t in all_combinations if p["id"] == provider["id"]
            ]
            limited_combinations.extend(provider_tests[: args.max_per_provider * len(prompts)])
        total_tests = len(limited_combinations)
        total_combinations = len(all_combinations)
        print(
            Color.colored(
                f"Running max {args.max_per_provider} tests per provider ({total_tests} total, reduced from {total_combinations})",
                Color.YELLOW,
            )
        )
    else:
        limited_combinations = all_combinations
        total_tests = len(limited_combinations)

    results: List[CompletedResult] = []
    test_count = 0

    # Track servers by provider ID
    provider_servers = {}

    try:
        for provider, prompt, test in limited_combinations:
            test_count += 1
            print(Color.colored(f"Running {test_count} of {total_tests}", Color.BLUE))

            # Start server if needed for this provider
            provider_id = provider["id"]
            if provider_id not in provider_servers and "server_command" in provider:
                server_cmd = provider["server_command"]
                print(Color.colored(f"Starting server: {server_cmd}", Color.BLUE))
                logger.info(f"Starting server for {provider_id}: {server_cmd}")
                server_process = subprocess.Popen(
                    server_cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                provider_servers[provider_id] = server_process
                # Give server time to start
                time.sleep(2)
                print(Color.colored(f"Server started (PID {server_process.pid})", Color.GREEN))

            test_vars = test.get("vars", {})
            from datetime import datetime, timezone

            result: CompletedResult = {
                "provider_id": provider["id"],
                "prompt_id": prompt["id"],
                "test": test["id"],
                "prompt": prompt["text"],
                "result": None,
                "error": None,
                "duration_ms": None,
                "passed": None,
                "expected": test.get("expected", test_vars.get("__expected", "")),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Add all test variables to the result
            result.update(test_vars)

            try:
                test_result, passed, duration_ms, was_cached = run_test(
                    provider, test, prompt["text"], config_dir, args.debug_tempdir
                )
                result["result"] = str(test_result)
                result["passed"] = passed
                result["duration_ms"] = duration_ms

                if passed is True:
                    status = "✅ PASSED"
                    color = Color.GREEN
                elif passed is False:
                    status = "❌ FAILED"
                    color = Color.YELLOW
                else:  # passed is None
                    status = "❓ NO ASSERTIONS"
                    color = Color.BLUE
                cache_indicator = " (cached)" if was_cached else ""
                print(
                    Color.colored(
                        f"{status} - Completed {test_count} of {total_tests} ({duration_ms:.1f}ms{cache_indicator})",
                        color,
                    )
                )

            except EvalException as e:
                print(Color.colored(f"❌ Test case failed", Color.YELLOW))
                logger.warning(f"Test case failed: {e}")
                result["error"] = str(e)
                result["passed"] = False
            except Exception as e:
                error_msg = str(e).split("\n")[0][:100]  # First line, max 100 chars
                print(Color.colored(f"❌ Unexpected error: {error_msg}", Color.RED))
                logger.error(f"Unexpected error: {e}")
                result["error"] = str(e)
                result["passed"] = False

            results.append(result)
    finally:
        # Kill all servers
        for provider_id, server_process in provider_servers.items():
            print(Color.colored(f"Stopping server for {provider_id} (PID {server_process.pid})", Color.BLUE))
            logger.info(f"Stopping server for {provider_id}")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(Color.colored(f"Force killing server (PID {server_process.pid})", Color.YELLOW))
                server_process.kill()

    # Write results
    output_file = config_dir / "results" / "results.csv"
    write_results(results, str(output_file))

    # Print summary
    passed_count = sum(1 for r in results if r["passed"] is True)
    failed_count = sum(1 for r in results if r["passed"] is False)
    no_assertions_count = sum(1 for r in results if r["passed"] is None)
    total_count = len(results)

    summary_parts = [f"{passed_count}/{total_count} tests passed"]
    if failed_count > 0:
        summary_parts.append(f"{failed_count} failed")
    if no_assertions_count > 0:
        summary_parts.append(f"{no_assertions_count} have no assertions")

    summary = f"\nSummary: {', '.join(summary_parts)}"
    print(
        Color.colored(
            summary,
            (
                Color.GREEN
                if passed_count == total_count and no_assertions_count == 0
                else Color.YELLOW
            ),
        )
    )


if __name__ == "__main__":
    main()
