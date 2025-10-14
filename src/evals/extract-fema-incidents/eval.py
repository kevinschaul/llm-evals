"""
Extract FEMA incidents evaluation using inspect_ai
"""
import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, Model, ChatMessageUser, ContentText, ContentImage
from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import generate, system_message

def create_fema_dataset() -> MemoryDataset:
    """Create dataset for FEMA incident extraction"""
    # Load expected results
    expected_file = Path(__file__).parent / "expected.json"
    with open(expected_file, 'r') as f:
        expected_data = json.load(f)

    # Create samples for each document
    schema_text = """state_or_tribe_or_territory
incident_description
incident_type
IA bool
PA bool
HM bool
requested str: YYYY-MM-DD, current year is 2025"""

    # JSON-serialize expected_data for storage in Sample.target
    expected_json = json.dumps(expected_data)

    # Get file paths
    eval_dir = Path(__file__).parent
    jpg_path = eval_dir / "fema-daily-operation-brief-p9.jpg"
    pdf_path = eval_dir / "fema-daily-operation-brief.pdf"

    # Create samples with image/PDF attachments
    prompt_text = f"""Extract all declaration requests from the provided document.

Return ONLY a valid JSON array (starting with [ and ending with ]) with no additional text or explanation.

Each object in the array should have these fields based on the following schema:
{schema_text}

Example format:
[{{"state_or_tribe_or_territory": "KY", "incident_description": "Severe Winter Storms", "incident_type": "DR", "IA": false, "PA": true, "HM": true, "requested": "2025-01-30"}}]"""

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
        # Note: PDF support requires different handling - skipping for now
        # Sample(
        #     input=[
        #         ChatMessageUser(content=[
        #             ContentImage(image=str(pdf_path)),
        #             ContentText(text=prompt_text)
        #         ])
        #     ],
        #     target=expected_json,
        #     id="pdf"
        # )
    ]

    return MemoryDataset(samples)

def calculate_extraction_accuracy(predicted: list, expected: list) -> float:
    """Calculate accuracy for extracted FEMA incidents"""
    if not expected:
        return 1.0 if not predicted else 0.0

    if not predicted:
        return 0.0

    # Score based on how many expected items were correctly extracted
    correct_extractions = 0
    total_expected = len(expected)

    for expected_item in expected:
        # Look for a matching prediction
        for predicted_item in predicted:
            if (predicted_item.get('state_or_tribe_or_territory') == expected_item.get('state_or_tribe_or_territory') and
                predicted_item.get('incident_description') == expected_item.get('incident_description') and
                predicted_item.get('incident_type') == expected_item.get('incident_type') and
                predicted_item.get('IA') == expected_item.get('IA') and
                predicted_item.get('PA') == expected_item.get('PA') and
                predicted_item.get('HM') == expected_item.get('HM') and
                predicted_item.get('requested') == expected_item.get('requested')):
                correct_extractions += 1
                break

    return correct_extractions / total_expected

@scorer(metrics=[mean()])
def fema_extraction_scorer():
    """Score FEMA incident extraction results"""
    async def score(state, target):
        try:
            # Parse JSON response
            response = state.output.message.content

            # Extract text from content (handle both string and list formats)
            if isinstance(response, list):
                # Content is a list of content blocks, extract text from first block
                response_text = response[0].text if hasattr(response[0], 'text') else str(response[0])
            else:
                response_text = response

            # Try to extract JSON from response
            if isinstance(response_text, str):
                # Look for JSON array in the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx > 0:
                    json_str = response_text[start_idx:end_idx]
                    predicted = json.loads(json_str)
                else:
                    predicted = json.loads(response_text)

                # Ensure predicted is a list
                if not isinstance(predicted, list):
                    predicted = [predicted] if predicted else []
            else:
                predicted = []

            # Parse expected from JSON string
            # target is a Target object, get the actual text value
            target_text = target.text if hasattr(target, 'text') else str(target)
            if isinstance(target_text, str):
                expected = json.loads(target_text)
            else:
                expected = target_text

            # Calculate extraction accuracy
            accuracy = calculate_extraction_accuracy(predicted, expected)

            return Score(
                value=accuracy,
                explanation=f"Extracted {len(predicted) if predicted else 0} items, expected {len(expected) if expected else 0}, accuracy: {accuracy:.2f}"
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return Score(
                value=0.0,
                explanation=f"Failed to parse extraction results: {e}"
            )

    return score

@task
def extract_fema_incidents():
    """Extract FEMA incidents task"""

    return Task(
        dataset=create_fema_dataset(),
        solver=[
            generate(cache=True)
        ],
        scorer=fema_extraction_scorer(),
        config=GenerateConfig(temperature=0.0)
    )