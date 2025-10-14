"""
Extract FEMA incidents evaluation using inspect_ai
"""
import json
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, Model
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

    samples = [
        Sample(
            input=f"Document: fema-daily-operation-brief-p9.jpg\nSchema: {schema_text}",
            target=expected_data
        ),
        Sample(
            input=f"Document: fema-daily-operation-brief.pdf\nSchema: {schema_text}",
            target=expected_data
        )
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

            # Try to extract JSON from response
            if isinstance(response, str):
                # Look for JSON array in the response
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    predicted = json.loads(json_str)
                else:
                    predicted = json.loads(response)
            else:
                predicted = response

            expected = target

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

    prompt = """Extract all declaration requests from the provided FEMA document(s).

For each declaration request, extract the following information in JSON format:
- state_or_tribe_or_territory: The state, tribe, or territory name
- incident_description: Description of the incident
- incident_type: Type of incident (DR, EM, etc.)
- IA: Individual Assistance (boolean)
- PA: Public Assistance (boolean)
- HM: Hazard Mitigation (boolean)
- requested: Date requested in YYYY-MM-DD format (current year is 2025)

Return the results as a JSON array of objects. For example:
[
  {
    "state_or_tribe_or_territory": "KY",
    "incident_description": "Severe Winter Storms and Snowstorms",
    "incident_type": "DR",
    "IA": false,
    "PA": true,
    "HM": true,
    "requested": "2025-01-30"
  }
]

{input}"""

    return Task(
        dataset=create_fema_dataset(),
        solver=[
            system_message(prompt),
            generate(cache=True)
        ],
        scorer=fema_extraction_scorer(),
        config=GenerateConfig(temperature=0.0)
    )