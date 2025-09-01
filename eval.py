#!/usr/bin/env python

from argparse import ArgumentParser
import csv
import logging
import os
import subprocess
from typing import List, Optional, TypedDict
from jinja2 import Template
import tempfile
import yaml

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cli():
    parser = ArgumentParser()
    parser.add_argument("eval_slug", help="Eval slug")
    args = parser.parse_args()
    return args


class EvalException(BaseException):
    pass


def load_config(eval_slug):
    with open(os.path.join("src", "evals", eval_slug, "eval.yaml")) as f:
        config = yaml.safe_load(f)
        return config


def run_command(command, temp_dir):
    logger.debug(f"Running command {command}")
    result = subprocess.run(
        command, shell=True, cwd=temp_dir, check=False, capture_output=True
    )
    logger.debug(f"Command returncode: {result.returncode}")
    logger.debug(f"Command stderr: {result.stderr}")
    logger.debug(f"Command stdout: {result.stdout}")

    if result.returncode != 0:
        raise EvalException(
            f"Error running {command}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def run_test(provider, test, prompt):
    if test.get("type") == "command":
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.debug(f"Running test in temp dir {temp_dir}")

            setup_command = test.get("setup")
            if setup_command:
                run_command(setup_command, temp_dir)

            eval_command_raw = Template(provider.get("command"))
            eval_command = eval_command_raw.render(prompt=prompt)
            run_command(eval_command, temp_dir)

            result_command = test.get("result")
            result = run_command(result_command, temp_dir)
            return result.stdout
    else:
        raise EvalException("Unknown test type")


class CompletedResult(TypedDict):
    provider_id: str
    test: str
    prompt: str
    result: Optional[str]
    error: Optional[str]


def write_results(results, out_file):
    fieldnames = ["provider_id", "test", "prompt", "result", "error"]
    with open(out_file, "w") as o:
        writer = csv.DictWriter(o, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results written to {out_file}")


def main():
    args = cli()
    config = load_config(args.eval_slug)
    # TODO validate config

    results: List[CompletedResult] = []
    for provider in config["providers"]:
        for test in config["tests"]:
            for prompt in config["prompts"]:
                result: CompletedResult = {
                    "provider_id": provider["id"],
                    "test": test["id"],
                    "prompt": prompt,
                    "result": None,
                    "error": None,
                }
                try:
                    test_result = run_test(provider, test, prompt)
                    result["result"] = str(test_result)
                except EvalException as e:
                    logger.warning(f"Test case failed: {e}")
                    result["error"] = str(e)
                results.append(result)

    write_results(results, "eval_results.csv")


if __name__ == "__main__":
    main()
