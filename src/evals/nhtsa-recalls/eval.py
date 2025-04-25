import re
import sys
import os
import textwrap
from typing import Any
from pydantic_evals import Dataset
import json

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
    try:
        res = json.loads(text)
        if res["result"] in [True, False]:
            return res["result"]
        else:
            raise ValueError(
                f"BooleanOutputParser expected JSON response 'result' value "
                f"to be a boolean. Received {text}."
            )
    except json.JSONDecodeError:
        raise ValueError(
            f"BooleanOutputParser expected output value to be valid JSON "
            f"Received {text}."
        )
    except KeyError:
        raise ValueError(
            f"BooleanOutputParser expected JSON response to have key 'result'"
            f"Received {text}."
        )


def main():
    models = [
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo"
    ]
    prompts = [
        textwrap.dedent(
            """
            Determine whether this NHTSA investigation summary is relevant to the Washington Post. Return `{{"result": true}}` or `{{"result": false}}`
            content: {content}
            """
        ),
        textwrap.dedent(
            """
            You are a helpful newspaper reporter who is very knowledgeable about US motor vehicle regulation and the NHTSA recall process, as well as
            the sorts of things that the Washington Post writes about. 

            Read the provided NHTSA investigation summary and return a JSON response of `{{"result": true}}` if the recall is newsworthy to the Washington Post or  `{{"result": false}}` if it is not.

            As a reminder, the Washington Post is very interested in automated vehicles, automation and advanced driver assistance systems, electric vehicles and Teslas,
            as well as any defect that lead to multiple deaths or serious injuries. The Post is not interested in recalls, component failures (mechanical, electronic or hydraulic), 
            or fuel leaks that haven't led to deaths or injuries, so you should return  `{{"result": false}}` for those.

            {content}
            """
        ),
        textwrap.dedent(
            """
            You are a helpful newspaper reporter who is very knowledgeable about US motor vehicle regulation and the NHTSA recall process, as well as
            the sorts of things that the Washington Post writes about. 

            Read the provided NHTSA investigation summary and return a JSON response of `{{"result": true}}` if the recall is newsworthy to the Washington Post or  `{{"result": false}}` if it is not.

            As a reminder, the Washington Post is very interested in automated vehicles, automation and advanced driver assistance systems, major failures of electric vehicles,
            as well as any defect that lead to multiple deaths or serious injuries. The Post is not interested in recalls, component failures (mechanical, electronic or hydraulic), 
            or fuel leaks that haven't led to deaths or injuries, so you should return  `{{"result": false}}` for those.

            {content}
            """
        ),
    ]
    dataset = Dataset[Any, Any, Any].from_file("./tests.json")
    results, aggregate_df = eval_utils.run_eval(
        models, prompts, dataset, transform_output=parse_boolean
    )
    os.makedirs("./results", exist_ok=True)
    results.to_csv(f"./results/results.csv", index=False)
    aggregate_df.to_csv(f"./results/aggregate.csv")


if __name__ == "__main__":
    main()
