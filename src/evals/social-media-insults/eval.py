"""
Social media insults classification evaluation using inspect_ai
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
def social_media_insults():
    """Social media insults classification task"""

    prompt = """You are an expert in political rhetoric around American elections.
Categorize the following social media posts by whether or not they include a personal insult about Kamala Harris's intelligence, and return
the results in JSON format, as follows:
{"category": true} or {"category": false}
Review your analysis carefully before returning a response.
Go slowly and use context clues.
Here are some examples to help you:
Input: "Kamala Harris doesn't have the mental capacity to do a REAL Debate against me, scheduled for September 4th in Pennsylvania. She's afraid to do it because there is no way she can justify her Corrupt and Open Borders, the Environmental Destruction of our Country, the Afghanistan Embarrassment, Runaway."
Response: {"category": true}
Input: "For the past four years as Border Czar, Kamala Harris... has imported an ARMY of illegal alien gang members and migrant criminals from the dungeons of the third-world... from prisons and jails, insane"
Response: {"category": false}
Input: "savage Venezuelan Gang "Tren de Aragua" are plotting to conduct ambush attacks on police and land enforcement in the United States—all while Harris and Biden sit in the White House and try to figure out who is dumber.On Sunday, in a separate incident, a Venezuelan illegal alien criminal"
Response: {"category": true}
Now, categorize this post: {input}"""

    return Task(
        dataset=read_csv_dataset("tests.csv"),
        solver=[
            system_message(prompt),
            generate(cache=True)
        ],
        scorer=includes(),
        config=GenerateConfig(temperature=0.1)
    )