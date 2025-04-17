import sys
import os
import textwrap
import re
from typing import Any
from pydantic_evals import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import eval_utils


def parse_boolean(text: str) -> bool:
    """Parse the output of an LLM call to a boolean.

    https://python.langchain.com/api_reference/_modules/langchain/output_parsers/boolean.html#BooleanOutputParser

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


def main():
    models = [
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "openai/o4-mini-2025-04-16",
        "claude-3.7-sonnet",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
        "gemini-1.5-flash-latest",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "llama-3.2-3b-4bit",
    ]
    prompts = [
        textwrap.dedent(
            """
            Is this article primarily about a new action or announcement by the Trump administration/White House/Executive? Respond simply with "yes" or "no" .
            {headline}
            {content}
            """
        ),
        # textwrap.dedent(
        #     """
        #     Does this describe a specific, new, official action or formal announcement already taken/made by the Trump administration? Answer only "yes" or "no".
        #     {headline}
        #     {content}
        #     """
        # ),
    ]
    dataset = Dataset[Any, Any, Any].from_file("./tests.json")
    results, aggregate_df = eval_utils.run_eval(
        models, prompts, dataset, transform_output=parse_boolean
    )
    results.to_csv(f"./results/results.csv", index=False)
    aggregate_df.to_csv(f"./results/aggregate.csv")


if __name__ == "__main__":
    main()
