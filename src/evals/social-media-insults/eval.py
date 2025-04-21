import re
import sys
import os
import textwrap
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
        "openai/gpt-4o-mini",
    ]
    prompts = [
        textwrap.dedent(
            """
            Determine whether this social media post from Donald Trump contains a personal insult to Kamala Harris's intelligence. Simply return "yes" or "no".
            content: {content}
            """
        ),
        textwrap.dedent(
            """
            You are an expert in political rhetoric around American elections.

            Categorize the following social media posts by whether or not they include a personal insult about Kamala Harris's intelligence, and simply return "yes" or "no".

            Review your analysis carefully before returning a response.
            Go slowly and use context clues.

            Here are some examples to help you:

            Input: "Kamala Harris doesn’t have the mental capacity to do a REAL Debate against me, scheduled for September 4th in Pennsylvania. She’s afraid to do it because there is no way she can justify her Corrupt and Open Borders, the Environmental Destruction of our Country, the Afghanistan Embarrassment, Runaway."
            Response: "yes"

            Input: "For the past four years as Border Czar, Kamala Harris... has imported an ARMY of illegal alien gang members and migrant criminals from the dungeons of the third-world... from prisons and jails, insane"
            Response: "no"

            Input: "savage Venezuelan Gang “Tren de Aragua” are plotting to conduct ambush attacks on police and law enforcement in the United States—all while Harris and Biden sit in the White House and try to figure out who is dumber.On Sunday, in a separate incident, a Venezuelan illegal alien criminal"
            Response: "yes"

            Now, categorize this post:
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
