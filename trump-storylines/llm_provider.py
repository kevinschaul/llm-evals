"""
A provider for promptfoo that calls the llm library

Provide the model to use with the "model" option

Docs:
- https://www.promptfoo.dev/docs/providers/python/
- https://llm.datasette.io/en/stable/index.html
"""

from typing import Any, Dict, List, Optional, Union
import re
import llm


class ProviderOptions:
    id: Optional[str]
    config: Optional[Dict[str, Any]]


class CallApiContextParams:
    vars: Dict[str, str]


class TokenUsage:
    total: int
    prompt: int
    completion: int


class ProviderResponse:
    output: Optional[Union[str, Dict[str, Any]]]
    error: Optional[str]
    tokenUsage: Optional[TokenUsage]
    cost: Optional[float]
    cached: Optional[bool]
    logProbs: Optional[List[float]]


class ProviderEmbeddingResponse:
    embedding: List[float]
    tokenUsage: Optional[TokenUsage]
    cached: Optional[bool]


class ProviderClassificationResponse:
    classification: Dict[str, Any]
    tokenUsage: Optional[TokenUsage]
    cached: Optional[bool]


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


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> ProviderResponse:
    config = options.get("config", {})

    model = llm.get_model(config.get("model"))
    output = model.prompt(prompt)

    try:
        parsed = parse_boolean(output.text())
    except ValueError:
        parsed = None

    result = {
        "output": parsed,
    }

    return result
