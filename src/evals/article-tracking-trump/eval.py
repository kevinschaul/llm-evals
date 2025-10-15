"""
Article tracking Trump storylines evaluation using inspect_ai
"""
import csv
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message

def read_csv_dataset(csv_path: str) -> MemoryDataset:
    """Read CSV test data and convert to inspect_ai Dataset format"""
    samples = []
    csv_file = Path(__file__).parent / csv_path

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine headline and content into a single input string
            input_text = f"{row['headline']}\n{row['content']}"
            samples.append(Sample(
                input=input_text,
                target=row['__expected']
            ))

    return MemoryDataset(samples)

@task
def article_tracking_trump():
    """Article tracking Trump storylines task"""

    prompt = """Is this article primarily about a new action or announcement by the Trump administration/White House/Executive? Respond simply with "true" or "false".
{input}"""

    return Task(
        dataset=read_csv_dataset("tests.csv"),
        solver=[
            system_message(prompt),
            generate(cache=True)
        ],
        scorer=includes(),
        config=GenerateConfig(temperature=0.0)
    )
