"""
A provider for promptfoo that calls the llm library

Provide the model to use with the "model" option

Docs:
- https://www.promptfoo.dev/docs/providers/python/
- https://llm.datasette.io/en/stable/index.html
"""

import re
import llm
from llm.cli import logs_db_path
from llm.utils import sqlite_utils


def parse_boolean(text: str) -> bool:
    """Parse the output of an LLM call to a boolean.

    https://python.langchain.com/api_reference/_modules/langchain/output_parsers/boolean.html#BooleanOutputParser

    Args:
        text: output of a language model

    Returns:
        boolean
    """

    TRUE_VAL = "TRUE"
    FALSE_VAL = "FALSE"

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


def find_cached_response(db, prompt, system, model):
    """Search the llm db for this exact query, and return it if it already exists"""

    RESPONSE_SQL = """
    select * from responses
    where responses.model = :model
    and responses.prompt = :prompt
    and responses.system is :system
    order by datetime_utc desc
    limit 1;
    """
    rows = list(
        db.query(RESPONSE_SQL, {"prompt": prompt, "system": system, "model": model})
    )
    if len(rows):
        return llm.Response.from_row(db, rows[0])


def call_api(prompt, options, context):
    config = options.get("config", {})
    model_name = config.get("model")

    transform_func = config.get("transform_func")
    transform = lambda x: x
    if transform_func:
        # TODO Load the real func
        transform = parse_boolean

    db = sqlite_utils.Database(logs_db_path())
    output = find_cached_response(db, prompt, None, model_name)
    if not output:
        model = llm.get_model(model_name)
        output = model.prompt(prompt)
        output.log_to_db(db)

    try:
        parsed = transform(output.text().strip())
    except ValueError:
        parsed = None

    result = {
        "output": parsed,
    }

    return result


# Just for testing this script
if __name__ == "__main__":
    response = call_api(
        "Tell me a dog joke", {"config": {"model": "openai/gpt-4o-mini"}}, None
    )
    print(response)
