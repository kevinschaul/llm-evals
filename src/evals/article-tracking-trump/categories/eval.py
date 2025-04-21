import sys
import os
import textwrap
from typing import Any
from pydantic_evals import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import eval_utils


categories = [
    "Abortion",
    "AI",
    "Associated Press",
    "Birthright citizenship",
    "CFPB",
    "Climate",
    "Crypto",
    "D.C.",
    "Debt ceiling",
    "DEI in schools",
    "DEI in the federal government",
    "Department of Education",
    "Deportations",
    "DOGE",
    "End birthright citizenship",
    "Eric Adams case",
    "Expanding executive power",
    "Federal funding",
    "Federal grant pause",
    "Federal office space",
    "Federal worker buyout",
    "Federal workers return to office",
    "Federal workforce reductions",
    "FEMA",
    "Firing government watchdogs",
    "Food safety",
    "Foreign aid",
    "Foreign policy",
    "Gaza",
    "Government shutdown",
    "Guantánamo Bay",
    "Health agencies' communications pause",
    "Health care",
    "Immigration policy",
    "Interest rates",
    "Iran",
    "Islamic State",
    "Israel",
    "Jan. 6",
    "Kennedy Center",
    "NIH",
    "NYC congestion toll",
    "Other",
    "Panama Canal",
    "Pardons",
    "Reclassifying federal jobs",
    "Sanctions",
    "Senior government officials ousted",
    "Taiwan",
    "Targeting political enemies",
    "Targeting protesters",
    "Tariffs",
    "The media",
    "TikTok ban",
    "Transgender policy",
    "U.S.-Mexico border",
    "USAID",
    "USPS",
    "War in Ukraine",
    "World Health Organization",
]
categories_str = ", ".join(categories)


def main():
    models = [
        "gemini-1.5-flash-latest",
    ]
    prompts = [
        textwrap.dedent(
            """
            Put this article into one of the following categories. Use the most specific category that makes sense. If you are unsure, respond Other.

            The allowed categories are:
            """
            + categories_str
            + """

            Respond ONLY with the category.

            {headline}
            {content}
            """
        ),
    ]
    dataset = Dataset[Any, Any, Any].from_file("./tests.json")
    results, aggregate_df = eval_utils.run_eval(models, prompts, dataset)
    os.makedirs("./results", exist_ok=True)
    results.to_csv(f"./results/results.csv", index=False)
    aggregate_df.to_csv(f"./results/aggregate.csv")


if __name__ == "__main__":
    main()
