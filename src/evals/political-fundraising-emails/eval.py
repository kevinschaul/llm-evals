import sys
import os
import textwrap
from typing import Any
from pydantic_evals import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import eval_utils


def main():
    models = [
        # "gemini-2.0-flash",
        "llama-3.2-3b-4bit",
    ]
    prompts = [
        textwrap.dedent(
            """
            Given this political fundraising email, respond with the name of the committee in the disclaimer that begins with Paid for by but does not include Paid for by, the committee address or the treasurer name. If no committee is present, return "None". Do not include any other text, no yapping.
            name: {name}
            email: {email}
            subject: {subject}
            body: {body}
            """
        )
    ]
    dataset = Dataset[Any, Any, Any].from_file("./tests.json")
    results, aggregate_df = eval_utils.run_eval(models, prompts, dataset)
    results.to_csv(f"./results/results.csv", index=False)
    aggregate_df.to_csv(f"./results/aggregate.csv")


if __name__ == "__main__":
    main()
