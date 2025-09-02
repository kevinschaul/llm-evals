#!/usr/bin/env python

from argparse import ArgumentParser
import csv
import logging
import os
import sys
import subprocess
from typing import List, Optional, TypedDict
from jinja2 import Template
import tempfile
import yaml

"""
Run the LLM eval suite specified in the passed-in config file.

Focused on command evals for now (`codex`, `claude`, etc.)

TODO:
- P0
    - validate config
    - count up number of eval permutations to be run
    - write results into directory of config file
    - cache result somewhere
- P1
    - add timing
    - store version of command tool?
- P2
    - add type=llm for replacing promptfoo in regular evals?
"""

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="eval.log",
    filemode="w",
)
logger = logging.getLogger(__name__)


class EvalException(BaseException):
    pass


class CompletedResult(TypedDict):
    provider_id: str
    test: str
    prompt: str
    result: Optional[str]
    error: Optional[str]


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


def cli():
    parser = ArgumentParser()
    parser.add_argument("eval_config", help="Eval config file (yaml)")
    args = parser.parse_args()
    return args


def load_config(eval_config):
    with open(eval_config) as f:
        config = yaml.safe_load(f)
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


def run_test(provider, test, prompt):
    print(Color.colored(f"Running test {test['id']}", Color.BLUE + Color.BOLD))
    logger.debug(f"Running test {test['id']}")
    if test.get("type") == "command":
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Running test in temp dir {temp_dir}")

            setup_command = test.get("setup_command")
            if setup_command:
                run_command(setup_command, temp_dir)

            eval_command_raw = Template(provider.get("command"))
            eval_command = eval_command_raw.render(prompt=prompt)
            run_command(eval_command, temp_dir)

            result_command = test.get("result_command")
            result = run_command(result_command, temp_dir)
            return result
    else:
        raise EvalException("Unknown test type")


def write_results(results, out_file):
    fieldnames = ["provider_id", "test", "prompt", "result", "error"]
    with open(out_file, "w") as o:
        writer = csv.DictWriter(o, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(Color.colored(f"Results written to {out_file}", Color.BLUE))


def main():
    Color.disable_if_needed()
    args = cli()
    config = load_config(args.eval_config)

    results: List[CompletedResult] = []
    for provider in config["providers"]:
        for test in config["tests"]:
            print(Color.colored(f"Running X of Y", Color.BLUE))
            result: CompletedResult = {
                "provider_id": provider["id"],
                "test": test["id"],
                "prompt": test["prompt"],
                "result": None,
                "error": None,
            }
            try:
                test_result = run_test(provider, test, test["prompt"])
                result["result"] = str(test_result)
                print(Color.colored(f"Completed X of Y", Color.BLUE))
            except EvalException as e:
                print(Color.colored(f"Test case failed", Color.YELLOW))
                logger.warning(f"Test case failed: {e}")
                result["error"] = str(e)
            results.append(result)

    write_results(results, "eval_results.csv")


if __name__ == "__main__":
    main()
