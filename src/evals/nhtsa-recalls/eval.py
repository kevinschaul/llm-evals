"""
NHTSA recalls data extraction evaluation using inspect_ai
"""
import json
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
            samples.append(Sample(
                input=row['content'],
                target=row['__expected']
            ))

    return MemoryDataset(samples)

# Using built-in includes() scorer

@task
def nhtsa_recalls():
    """NHTSA recalls data extraction task"""

    prompt = """You are a helpful newspaper reporter who is very knowledgeable about US motor vehicle regulation and the NHTSA recall process, as well as the sorts of things that the Washington Post writes about.

Read the provided NHTSA investigation summary and return a JSON response of `{"result": true}` if the recall is newsworthy to the Washington Post or  `{"result": false}` if it is not.

As a reminder, the Washington Post is very interested in automated vehicles, automation and advanced driver assistance systems, electric vehicles and Teslas, as well as any defect that lead to multiple deaths or serious injuries. The Post is not interested in recalls, component failures (mechanical, electronic or hydraulic), or fuel leaks that haven't led to deaths or injuries, so you should return `{"result": false}` for those.

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