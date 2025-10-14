"""
Custom Inspect model provider that wraps the llm library.
This allows Inspect to use the existing llm cache during migration.
"""
from typing import Any
import llm
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatCompletionChoice,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolInfo, ToolChoice


class LLMProviderAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config)
        self.model_name = model_name
        self.model_args = model_args

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a response using the llm library."""

        # Convert Inspect messages to llm format
        system_prompt = None
        user_prompts = []

        for msg in input:
            if isinstance(msg, ChatMessageSystem):
                system_prompt = msg.content
            elif isinstance(msg, ChatMessageUser):
                user_prompts.append(msg.content)
            elif isinstance(msg, ChatMessageAssistant):
                # llm library doesn't support multi-turn conversations in the same way
                # For migration purposes, we'll focus on single-turn prompts
                pass
            elif isinstance(msg, ChatMessageTool):
                # Skip tool messages for now
                pass

        # Combine user prompts
        prompt_text = "\n".join(str(p) for p in user_prompts)

        # Get the llm model
        # The llm library uses format like "gpt-4o-mini" or "openai/gpt-4o-mini"
        # Inspect passes us the model name after stripping the provider prefix
        # So we receive either "gpt-4o-mini" or "openai/gpt-4o-mini"
        model = llm.get_model(self.model_name)

        # Build model options from config
        model_options = {}
        if config.temperature is not None:
            model_options["temperature"] = config.temperature
        if config.top_p is not None:
            model_options["top_p"] = config.top_p
        if config.max_tokens is not None:
            model_options["max_tokens"] = config.max_tokens

        # Add any additional model args
        model_options.update(self.model_args)

        # Call the model (this will use llm's cache if available)
        response = model.prompt(
            prompt_text,
            system=system_prompt,
            **model_options
        )

        # Get the response text
        response_text = response.text()

        # Create Inspect ModelOutput
        output = ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=response_text),
                    stop_reason="stop"
                )
            ]
        )

        return output


@modelapi(name="llm")
def llm_provider():
    """Register the llm provider for Inspect."""
    return LLMProviderAPI
