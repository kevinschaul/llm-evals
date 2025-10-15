"""
Extract FEMA incidents evaluation using inspect_ai
"""
import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, ChatMessageUser, ContentText, ContentImage
from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import generate


@scorer(metrics=[mean()])
def json_equal():
    """Score by comparing JSON output to expected JSON"""
    async def score(state, target: Target):
        try:
            # Extract response text
            response = state.output.completion

            # Try to extract JSON from response (handle markdown code blocks)
            if '```' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                if start != -1 and end > 0:
                    response = response[start:end]

            # Parse both as JSON
            output_json = json.loads(response)
            expected_json = json.loads(target.text)

            # Handle {"items": [...]} wrapper
            if isinstance(output_json, dict) and 'items' in output_json:
                output_json = output_json['items']

            # Compare as JSON (order-independent)
            match = output_json == expected_json

            return Score(
                value=1.0 if match else 0.0,
                explanation=f"JSON match: {match}"
            )
        except (json.JSONDecodeError, Exception) as e:
            return Score(value=0.0, explanation=f"Error: {e}")

    return score


@task
def extract_fema_incidents():
    """Extract FEMA incidents from documents"""

    # Load expected results
    expected_file = Path(__file__).parent / "expected.json"
    with open(expected_file, 'r') as f:
        expected_data = json.load(f)

    expected_json = json.dumps(expected_data)

    # Get file paths
    eval_dir = Path(__file__).parent
    jpg_path = eval_dir / "fema-daily-operation-brief-p9.jpg"
    pdf_path = eval_dir / "fema-daily-operation-brief.pdf"

    # Create prompt
    prompt_text = """Extract all declaration requests from the provided document.

Return ONLY a valid JSON array (starting with [ and ending with ]) with no additional text or explanation.

Each object in the array should have these fields based on the following schema:
state_or_tribe_or_territory
incident_description
incident_type
IA bool
PA bool
HM bool
requested str: YYYY-MM-DD, current year is 2025

Example format:
[{"state_or_tribe_or_territory": "KY", "incident_description": "Severe Winter Storms", "incident_type": "DR", "IA": false, "PA": true, "HM": true, "requested": "2025-01-30"}]"""

    # Create samples
    samples = [
        Sample(
            input=[
                ChatMessageUser(content=[
                    ContentImage(image=str(jpg_path)),
                    ContentText(text=prompt_text)
                ])
            ],
            target=expected_json,
            id="jpg"
        ),
        Sample(
            input=[
                ChatMessageUser(content=[
                    ContentImage(image=str(pdf_path)),
                    ContentText(text=prompt_text)
                ])
            ],
            target=expected_json,
            id="pdf"
        )
    ]

    return Task(
        dataset=MemoryDataset(samples),
        solver=generate(cache=True),
        scorer=json_equal(),
        config=GenerateConfig(temperature=0.0)
    )

