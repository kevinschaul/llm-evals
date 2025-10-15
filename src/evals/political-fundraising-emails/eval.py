"""
Political fundraising emails evaluation using inspect_ai
"""
import csv
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate, system_message

def read_csv_dataset(csv_path: str) -> MemoryDataset:
    """Read CSV test data and convert to inspect_ai Dataset format"""
    samples = []
    csv_file = Path(__file__).parent / csv_path

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Combine all fields into a single input string
            input_text = f"name: {row['name']}\nemail: {row['email']}\nsubject: {row['subject']}\nbody: {row['body']}"
            samples.append(Sample(
                input=input_text,
                target=row['__expected']
            ))

    return MemoryDataset(samples)

@task
def political_fundraising_emails():
    """Political fundraising emails committee extraction task"""

    prompt = """Given this political fundraising email, respond with the name of the committee in the disclaimer that begins with Paid for by but does not include Paid for by, the committee address or the treasurer name. If no committee is present, return "None". Do not include any other text, no yapping.
{input}"""

    return Task(
        dataset=read_csv_dataset("tests.csv"),
        solver=[
            system_message(prompt),
            generate(cache=True)
        ],
        scorer=match(),
        config=GenerateConfig(temperature=0.0)
    )
