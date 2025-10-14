"""
Article tracking Trump storylines - Categories evaluation using inspect_ai
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
def article_tracking_categories():
    """Article tracking Trump storylines - Categories task"""

    prompt = """Put this article into one of the following categories. Use the most specific category that makes sense. If you are unsure, respond Other.

The allowed categories are:
Abortion, AI, Associated Press, Birthright citizenship, CFPB, Climate, Crypto, D.C., Debt ceiling, DEI in schools, DEI in the federal government, Department of Education, Deportations, DOGE, End birthright citizenship, Eric Adams case, Expanding executive power, Federal funding, Federal grant pause, Federal office space, Federal worker buyout, Federal workers return to office, Federal workforce reductions, FEMA, Firing government watchdogs, Food safety, Foreign aid, Foreign policy, Gaza, Government shutdown, Guantánamo Bay, Health agencies' communications pause, Health care, Immigration policy, Interest rates, Iran, Islamic State, Israel, Jan. 6, Kennedy Center, NIH, NYC congestion toll, Other, Panama Canal, Pardons, Reclassifying federal jobs, Sanctions, Senior government officials ousted, Taiwan, Targeting political enemies, Targeting protesters, Tariffs, The media, TikTok ban, Transgender policy, U.S.-Mexico border, USAID, USPS, War in Ukraine, World Health Organization

Respond ONLY with the category.

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
