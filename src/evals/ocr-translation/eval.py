"""
OCR and translation task evaluation using inspect_ai
"""

from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ChatMessageUser,
    ContentText,
    ContentImage,
)
from inspect_ai.scorer import includes
from inspect_ai.solver import generate


@task
def ocr_translation():
    """OCR and translate a floor plan document"""

    eval_dir = Path(__file__).parent
    jpg_path = eval_dir / "cadastral-floor-plan.jpg"
    prompt_text = "Translate this. Include both original language and English."

    targets = [
        "STATE AGENCY FOR LAND RELATIONS",
        "Locativă",
        "Chișinău",
        "Pentru Primărie",
    ]

    sample_input: list[ChatMessage] = [
        ChatMessageUser(
            content=[
                ContentImage(image=str(jpg_path)),
                ContentText(text=prompt_text),
            ],
        )
    ]

    samples = [Sample(input=sample_input, target=t) for t in targets]

    return Task(
        dataset=MemoryDataset(samples),
        solver=[generate(cache=True)],
        scorer=includes(),
        config=GenerateConfig(temperature=0.0),
    )
