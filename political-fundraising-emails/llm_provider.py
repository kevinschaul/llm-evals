"""
A provider for promptfoo that calls the llm library

Provide the model to use with the "model" option

Docs:
- https://www.promptfoo.dev/docs/providers/python/
- https://llm.datasette.io/en/stable/index.html
"""

from typing import Any, Dict, List, Optional, Union
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


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> ProviderResponse:
    config = options.get("config", {})

    model = llm.get_model(config.get("model"))
    output = model.prompt(prompt)

    result = {
        "output": output.text().strip(),
    }

    return result
