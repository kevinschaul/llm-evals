import json
import re

def transform_output(text: str) -> bool:
    """Parse the output of an LLM call to a boolean.
    https://python.langchain.com/api_reference/_modules/langchain/output_parsers/boolean.html#BooleanOutputParser
    Args:
        text: output of a language model
    Returns:
        boolean
    """
    try:
        # Sometimes the result includes backticks to format it as code in markdown, so lets remove those
        clean_text = re.sub(r'```json\s*|\s*```', '', text)

        res = json.loads(clean_text)
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
