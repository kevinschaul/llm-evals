import os
import csv
import re
import yaml
import json
import hashlib
import functools
import importlib.util
import time as import_time
from typing import Any, Callable, Dict, TypeVar, cast

import llm
import click
import pandas as pd
from pydantic_evals import Dataset, Case
from pydantic_evals.dataset import set_eval_attribute
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.evaluators import EqualsExpected

T = TypeVar("T")
# Base directory for cache; set to config file dir when running CLI
CACHE_BASE_DIR: str | None = None


class JsonEncoder(json.JSONEncoder):
    """Extended JSON encoder to handle more Python types."""

    def default(self, o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        if isinstance(o, (set, frozenset)):
            return list(o)
        return str(o)


def hash_arg(arg) -> str:
    """Create a hash string for a single argument."""
    if isinstance(arg, (int, float, bool, type(None))):
        # Simple types are converted to string and hashed
        arg_str = str(arg)
    elif isinstance(arg, str):
        # For strings, create a fixed hash for the article
        arg_str = str(arg)
    elif isinstance(arg, (list, tuple)):
        # For sequences, hash each element and join
        arg_str = "[" + ",".join(hash_arg(x) for x in arg) + "]"
    elif isinstance(arg, dict):
        # For dictionaries, hash each key-value pair and join
        arg_str = (
            "{" + ",".join(f"{k}:{hash_arg(v)}" for k, v in sorted(arg.items())) + "}"
        )
    else:
        # For complex objects, use their string representation
        arg_str = str(arg)

    # Create a hash for the string representation
    return hashlib.md5(arg_str.encode()).hexdigest()[:8]


def cache(cache_dir: str = None):
    """
    A decorator that caches function results in files, one file per set of arguments.
    Files are named as: func_name-hash(arg1)-hash(arg2)...

    Args:
        cache_dir: Directory to store cache files (default: None, will use CONFIG_CACHE_DIR)

    Returns:
        Decorated function that uses file-based caching
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Use the provided cache_dir, or fall back to CONFIG_CACHE_DIR if set
            effective_cache_dir = cache_dir or CONFIG_CACHE_DIR

            # Determine the effective cache directory, relative to config if set
            if os.path.isabs(effective_cache_dir):
                final_cache_dir = effective_cache_dir
            else:
                base = CACHE_BASE_DIR or os.getcwd()
                final_cache_dir = os.path.join(base, effective_cache_dir)
            # Ensure the cache directory exists
            os.makedirs(final_cache_dir, exist_ok=True)
            # Generate a unique filename based on function name and arguments
            func_name = func.__name__
            arg_hashes = [hash_arg(arg) for arg in args]

            # Add keyword arguments as well, sorted by key
            for key, value in sorted(kwargs.items()):
                arg_hashes.append(f"{key}-{hash_arg(value)}")

            # Create the cache file path with hyphens between components
            filename_parts = [func_name] + arg_hashes
            cache_file = os.path.join(
                final_cache_dir, "-".join(filename_parts) + ".json"
            )
            # Uncomment for debugging cache issues
            # print(f"Cache file: {cache_file}")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)
                        # with open(
                        #     os.path.join(
                        #         "new-cache", "-".join(filename_parts) + ".json"
                        #     ),
                        #     "w",
                        # ) as f:
                        #     json.dump(cached_data, f, cls=JsonEncoder)
                        # Return just the result part of the cached data
                        return cast(T, cached_data["result"])
                except (json.JSONDecodeError, KeyError) as e:
                    # Handle corrupted cache file or missing result key
                    os.remove(cache_file)

            # Cache miss or corrupted cache, call the function
            result = func(*args, **kwargs)

            # Save both arguments and result to cache
            try:
                cache_data = {
                    "args": args,
                    "kwargs": kwargs,
                    "result": result,
                    "cached_at": import_time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(cache_file, "w") as f:
                    json.dump(cache_data, f, cls=JsonEncoder)
            except Exception as e:
                print(f"Warning: failed to cache result: {e}")

            return result

        return wrapper

    return decorator


# Directory for caching LLM responses
CONFIG_CACHE_DIR = "./cached-llm-responses/"


# Fact Checker
# openai/gpt-4o-mini
# Is this article ...
# dae45be3


@cache()
def run_llm(model_name: str, prompt: str) -> str:
    model = llm.get_model(model_name)
    response = model.prompt(prompt)
    return response.text().strip()


def expand_dict_columns(df: pd.DataFrame, columns_to_expand: list[str]) -> pd.DataFrame:
    """
    Expand dictionary columns in a DataFrame into individual columns with dotted notation.

    Parameters:
    - df: pandas DataFrame
    - columns_to_expand: list of column names containing dictionaries to expand

    Returns:
    - DataFrame with expanded columns and original dictionary columns removed
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Track the original column order
    original_columns = list(result_df.columns)
    columns_to_drop = []

    # Process each dictionary column to expand it
    for col in columns_to_expand:
        if col not in result_df.columns:
            continue

        # Find the position of this column in the original list
        col_index = original_columns.index(col)

        # Check if the column contains dictionaries
        if result_df[col].apply(lambda x: isinstance(x, dict)).any():
            # Mark this column for dropping
            columns_to_drop.append(col)

            # Get all unique keys from the dictionaries
            all_keys = set()
            for d in result_df[col].dropna():
                if isinstance(d, dict):
                    all_keys.update(d.keys())

            # For each key, create a new column
            new_columns = []
            for key in sorted(all_keys):
                new_col_name = f"{col}.{key}"
                result_df[new_col_name] = result_df[col].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )
                new_columns.append(new_col_name)

            # Update the column order list - replace the original column with the new columns
            original_columns = (
                original_columns[:col_index]
                + new_columns
                + original_columns[col_index + 1 :]
            )

    # Drop the original dictionary columns
    result_df = result_df.drop(columns=columns_to_drop)

    # Reorder the columns to maintain original positioning with expanded columns
    final_columns = [col for col in original_columns if col in result_df.columns]
    return result_df[final_columns]


def report_to_df(report: EvaluationReport) -> pd.DataFrame:
    records = []
    for case in report.cases:
        case_data = {
            "input": case.inputs,
            "metadata": case.metadata,
            "attributes": case.attributes,
            "expected_output": case.expected_output,
            "output": case.output,
            # TODO handle scores, labels, metrics
        }

        # Calculate assertions pass rate
        assertions_total = len(case.assertions)
        assertions_passed = sum(1 for a in case.assertions.values() if a.value)
        case_data["assertions_passed_rate"] = (
            assertions_passed / assertions_total if assertions_total > 0 else 1.0
        )

        # Add individual assertion columns
        for assertion_name, assertion_obj in case.assertions.items():
            case_data[f"assertion.{assertion_name}"] = assertion_obj.value

        records.append(case_data)

    df = pd.DataFrame(records)
    df = expand_dict_columns(df, ["input", "attributes"])
    return df


def calculate_aggregates(df):
    # Group by both model and prompt
    result = df.groupby(["attributes.model", "attributes.prompt"]).agg(
        count=("assertion.EqualsExpected", "count"),
        is_correct=("assertion.EqualsExpected", lambda x: x.sum()),
    )

    result["share_correct"] = result["is_correct"] / result["count"]
    return result


def run_eval(models, prompts, dataset, transform_output=None):
    n_cases = len(dataset.cases) * len(models) * len(prompts)
    print(f"Starting to evaluate {n_cases}")
    global i
    i = 1

    reports = []
    for model_name in models:
        for prompt in prompts:

            async def _run_llm(input: Dict) -> str:
                set_eval_attribute("model", model_name)
                set_eval_attribute("prompt", prompt)

                global i
                print(f"Evaluating {i} of {n_cases}")
                i += 1
                prompt_filled = prompt.format(**input)
                result = "[ERROR]"
                try:
                    result = run_llm(model_name, prompt_filled)
                    if transform_output:
                        result = transform_output(result)
                except Exception as e:
                    print(e)
                    pass
                print(model_name, result)
                return result

            report = dataset.evaluate_sync(_run_llm, name=model_name)
            reports.append(report)

    report_dfs = [report_to_df(report) for report in reports]
    results = pd.concat(report_dfs, axis=0, ignore_index=True)
    aggregate_df = calculate_aggregates(results)
    return (results, aggregate_df)


def load_function_from_file(file_path, function_name):
    """
    Dynamically load a function from a Python file.

    Args:
        file_path: Path to the Python file
        function_name: Name of the function to load

    Returns:
        The loaded function
    """
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise ValueError(f"Function '{function_name}' not found in {file_path}")

    return getattr(module, function_name)


def parse_boolean(text: str) -> bool:
    """Parse the output of an LLM call to a boolean.

    Args:
        text: output of a language model
    Returns:
        boolean
    """
    TRUE_VAL = "YES"
    FALSE_VAL = "NO"
    regexp = rf"\b({TRUE_VAL}|{FALSE_VAL})\b"
    truthy = {
        val.upper()
        for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
    }

    if TRUE_VAL.upper() in truthy:
        if FALSE_VAL.upper() in truthy:
            raise ValueError(
                f"Ambiguous response. Both {TRUE_VAL} and {FALSE_VAL} "
                f"in received: {text}."
            )
        return True
    elif FALSE_VAL.upper() in truthy:
        if TRUE_VAL.upper() in truthy:
            raise ValueError(
                f"Ambiguous response. Both {TRUE_VAL} and {FALSE_VAL} "
                f"in received: {text}."
            )
        return False

    raise ValueError(
        f"BooleanOutputParser expected output value to include either "
        f"{TRUE_VAL} or {FALSE_VAL}. Received {text}."
    )


# Built-in transform functions that can be selected by name
TRANSFORM_FUNCTIONS = {
    "parse_boolean": parse_boolean,
    # Add more built-in functions here as needed
}


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_models_from_csv(csv_path):
    """Load models from a CSV file.

    Expected format:
    model
    openai/gpt-4.1
    claude-3.7-sonnet
    ...
    """
    models = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        # Skip header row if it exists
        try:
            header = next(reader)
            # Check if the first row looks like a header
            if len(header) == 1 and header[0].lower() == "model":
                pass  # Skip header
            else:
                # If not a header, add it as a model
                models.append(header[0].strip())
        except StopIteration:
            pass  # Empty file

        # Read the rest of the models
        for row in reader:
            if row and row[0].strip():  # Check for non-empty rows
                models.append(row[0].strip())

    return models


def load_prompts_from_csv(csv_path):
    """Load prompts from a CSV file.

    Expected format:
    prompt
    "Prompt template 1 with {placeholder}"
    "Prompt template 2 with {placeholder}"
    ...
    """
    prompts = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        # Skip header row if it exists
        try:
            header = next(reader)
            # Check if the first row looks like a header
            if len(header) == 1 and header[0].lower() == "prompt":
                pass  # Skip header
            else:
                # If not a header, add it as a prompt
                prompts.append(header[0].replace("\\n", "\n"))
        except StopIteration:
            pass  # Empty file

        # Read the rest of the prompts
        for row in reader:
            if row:  # Check for non-empty rows
                prompts.append(row[0].replace("\\n", "\n"))

    return prompts


def load_models(models_input):
    """Load models from a file or list."""
    if isinstance(models_input, list):
        return models_input
    elif isinstance(models_input, str):
        return load_models_from_csv(models_input)
    return []


def load_prompts(prompts_input):
    """Load prompts from a file or list."""
    if isinstance(prompts_input, list):
        return prompts_input
    elif isinstance(prompts_input, str) and os.path.isfile(prompts_input):
        return load_prompts_from_csv(prompts_input)
    return []


def get_transform_function(transform_spec, default_func_name="transform_output"):
    """Get a transform function based on specification."""
    if not transform_spec:
        return None

    # If it's already a function, return it
    if callable(transform_spec):
        return transform_spec

    # If it's a built-in function name
    if isinstance(transform_spec, str):
        if transform_spec in TRANSFORM_FUNCTIONS:
            return TRANSFORM_FUNCTIONS[transform_spec]
        elif os.path.isfile(transform_spec):
            return load_function_from_file(transform_spec, default_func_name)

    # If it's a dictionary with file and function name
    elif isinstance(transform_spec, dict):
        file_path = transform_spec.get("file")
        func_name = transform_spec.get("function", default_func_name)
        if file_path and os.path.isfile(file_path):
            return load_function_from_file(file_path, func_name)

    raise ValueError(f"Invalid transform function specification: {transform_spec}")


@click.group()
def cli():
    """LLM Evals - A CLI tool for running and managing LLM evaluations.

    Examples:
      Create a new evaluation:
        uv run cli/cli.py create my-new-eval

      Run an evaluation:
        uv run cli/cli.py run src/evals/my-eval/llm-evals-config.yaml

      View results in browser:
        uv run cli/cli.py view
    """
    pass


@cli.command()
@click.argument("eval_name")
def create(eval_name):
    """Create a new evaluation with the given name."""
    # Create directory structure
    eval_dir = os.path.join("src/evals", eval_name)
    if os.path.exists(eval_dir):
        click.echo(f"Error: Evaluation '{eval_name}' already exists in {eval_dir}")
        return

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "cached-llm-responses"), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, "results"), exist_ok=True)

    # Create template files
    create_config_template(eval_dir, eval_name)
    create_template_files(eval_dir, eval_name)

    click.echo(f"Created new evaluation '{eval_name}' in {eval_dir}")
    click.echo(f"Next steps:")
    click.echo(f"1. Add your models to {eval_dir}/models.csv")
    click.echo(f"2. Add your prompts to {eval_dir}/prompts.csv")
    click.echo(f"3. Add your test cases to {eval_dir}/tests.csv")
    click.echo(f"4. Run: uv run cli/cli.py run {eval_dir}/llm-evals-config.yaml")


@cli.command()
def view():
    """Launch the web UI to view evaluation results."""
    click.echo("Launching web UI to view evaluation results...")
    os.system("npm run dev")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def run(config_path):
    """Run an evaluation with settings from a configuration file."""
    # Load configuration
    config_data = load_config(config_path)

    # Resolve file paths in config relative to the config file directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    # Set base directory for cache relative to config
    global CACHE_BASE_DIR, CONFIG_CACHE_DIR
    CACHE_BASE_DIR = config_dir

    # Use cache_dir from config if available
    if "cache_dir" in config_data:
        CONFIG_CACHE_DIR = config_data["cache_dir"]

    # Resolve relative paths in config
    _resolve_relative_paths(config_data, config_dir)

    # Extract configuration values
    models = config_data.get("models")
    prompts = config_data.get("prompts")
    dataset = config_data.get("tests")
    transform_func = config_data.get("transform_func")
    output_dir = config_data.get("output_dir", "./results")
    results_file = config_data.get("results_file", "results.csv")
    aggregate_file = config_data.get("aggregate_file", "aggregate.csv")

    # Validate required parameters
    if not models:
        raise click.UsageError("Models must be specified in config file")
    if not prompts:
        raise click.UsageError("Prompts must be specified in config file")
    if not dataset:
        raise click.UsageError("Dataset must be specified in config file")

    # Process models and prompts
    model_list = load_models(models)
    prompt_list = load_prompts(prompts)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    test_dataset = _load_dataset(dataset)

    # Get transform function if specified
    transform_output = (
        get_transform_function(transform_func) if transform_func else None
    )

    # Run the evaluation
    click.echo(
        f"Running evaluation with {len(model_list)} models and {len(prompt_list)} prompts..."
    )
    results, aggregate_df = run_eval(
        model_list, prompt_list, test_dataset, transform_output=transform_output
    )

    # Save results
    results_path = os.path.join(output_dir, results_file)
    aggregate_path = os.path.join(output_dir, aggregate_file)
    results.to_csv(results_path, index=False)
    aggregate_df.to_csv(aggregate_path)

    click.echo(f"Evaluation complete. Results saved to:")
    click.echo(f"  - Detailed results: {results_path}")
    click.echo(f"  - Aggregate results: {aggregate_path}")


def _resolve_relative_paths(config_data, config_dir):
    """Resolve relative paths in config to absolute paths."""
    # Resolve models, prompts, and tests file paths
    for key in ("models", "prompts", "tests"):
        val = config_data.get(key)
        if (
            isinstance(val, str)
            and not os.path.isabs(val)
            and not re.match(r"^[a-z]+://", val)
        ):
            config_data[key] = os.path.join(config_dir, val)

    # Resolve transform_func path if specified
    tf = config_data.get("transform_func")
    if isinstance(tf, str) and not os.path.isabs(tf) and tf not in TRANSFORM_FUNCTIONS:
        config_data["transform_func"] = os.path.join(config_dir, tf)
    elif isinstance(tf, dict) and "file" in tf:
        fp = tf["file"]
        if (
            isinstance(fp, str)
            and not os.path.isabs(fp)
            and not re.match(r"^[a-z]+://", fp)
        ):
            tf["file"] = os.path.join(config_dir, fp)
        config_data["transform_func"] = tf

    # Resolve output_dir relative to config directory
    out_dir = config_data.get("output_dir")
    if isinstance(out_dir, str) and not os.path.isabs(out_dir):
        config_data["output_dir"] = os.path.join(config_dir, out_dir)


def _load_dataset(dataset):
    """Load dataset from csv file."""
    df = pd.read_csv(dataset)
    records = df.to_dict(orient="records")
    cases = []
    for rec in records:
        name = rec.get("name")
        expected = rec.get("expected_output")
        # Inputs are all other fields
        inputs = {k: v for k, v in rec.items() if k not in ("name", "expected_output")}
        cases.append(Case(name=name, inputs=inputs, expected_output=expected))
    return Dataset(cases=cases, evaluators=[EqualsExpected()])


def create_config_template(eval_dir, eval_name):
    """Create a template config file for the evaluation."""
    config_content = f"""# Configuration for {eval_name} evaluation
name: {eval_name}
description: "Description of the {eval_name} evaluation"

# Input files
models: models.csv
prompts: prompts.csv
tests: tests.csv

# Output settings
cache_dir: cached-llm-responses
output_dir: results
results_file: results.csv
aggregate_file: aggregate.csv

# Optional: Custom transformation function
# transform_func: parse_boolean  # Built-in function to parse YES/NO responses
"""
    config_path = os.path.join(eval_dir, "llm-evals-config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)


def create_template_files(eval_dir, eval_name):
    """Create template files for models, prompts, and tests."""
    # Create models.csv template
    models_content = """model
# Add your models here, one per line
claude-3.5-sonnet
openai/gpt-4o-mini
gemini-1.5-flash"""
    with open(os.path.join(eval_dir, "models.csv"), "w") as f:
        f.write(models_content)

    # Create prompts.csv template
    prompts_content = """prompt
"Your prompt here with {placeholder} for variables from test cases"
# Add more prompts as needed, one per line"""
    with open(os.path.join(eval_dir, "prompts.csv"), "w") as f:
        f.write(prompts_content)

    # Create tests.csv template
    tests_content = """name,input_text,expected_output
"test1","Sample input text","Expected output"
"test2","Another sample input","Another expected output"
# Add your test cases here"""
    with open(os.path.join(eval_dir, "tests.csv"), "w") as f:
        f.write(tests_content)

    # Create a simple README
    readme_content = f"""# {eval_name} Evaluation

## Description
[Add description of this evaluation here]

## Data
- models.csv: List of models to test
- prompts.csv: Prompt templates to use
- tests.csv: Test cases with expected outputs

## Running
Run with: `uv run cli/cli.py run {os.path.join("src/evals", eval_name, "llm-evals-config.yaml")}`
"""
    with open(os.path.join(eval_dir, "index.md"), "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    cli()
