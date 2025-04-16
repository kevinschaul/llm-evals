from io import StringIO
from typing import Any, Dict
from pydantic_evals import Dataset
import llm
from pydantic_evals.dataset import set_eval_attribute
from pydantic_evals.reporting import EvaluationReport
from rich.console import Console
from cachier import cachier
import pandas as pd


@cachier(cache_dir="./cached-llm-responses", separate_files=True)
def run_llm(model_name: str, prompt: str) -> str:
    model = llm.get_model(model_name)
    response = model.prompt(prompt)
    return response.text()


def expand_dict_columns(df: pd.DataFrame, columns_to_expand: list[str]) -> pd.DataFrame:
    """
    Expand dictionary columns in a DataFrame into individual columns with dotted notation.

    Parameters:
    - df: pandas DataFrame
    - columns_to_expand: list of column names containing dictionaries to expand

    Returns:
    - DataFrame with expanded columns and original dictionary columns removed
    """
    # Make a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Track the original column order
    original_columns = list(result_df.columns)
    columns_to_drop = []

    # Process each dictionary column to expand it
    for col in columns_to_expand:
        if col not in result_df.columns:
            continue

        # Find the position of this column in the original list
        col_index = original_columns.index(col)

        # Check if the column contains dictionaries
        if result_df[col].apply(lambda x: isinstance(x, dict)).any():
            # Mark this column for dropping
            columns_to_drop.append(col)

            # Get all unique keys from the dictionaries
            all_keys = set()
            for d in result_df[col].dropna():
                if isinstance(d, dict):
                    all_keys.update(d.keys())

            # For each key, create a new column
            new_columns = []
            for key in sorted(all_keys):
                new_col_name = f"{col}.{key}"
                result_df[new_col_name] = result_df[col].apply(
                    lambda x: x.get(key) if isinstance(x, dict) else None
                )
                new_columns.append(new_col_name)

            # Update the column order list - replace the original column with the new columns
            original_columns = (
                original_columns[:col_index]
                + new_columns
                + original_columns[col_index + 1 :]
            )

    # Drop the original dictionary columns
    result_df = result_df.drop(columns=columns_to_drop)

    # Reorder the columns to maintain original positioning with expanded columns
    final_columns = [col for col in original_columns if col in result_df.columns]
    return result_df[final_columns]


def report_to_df(report: EvaluationReport) -> pd.DataFrame:
    records = []
    for case in report.cases:
        case_data = {
            "input": case.inputs,
            "metadata": case.metadata,
            "attributes": case.attributes,
            "expected_output": case.expected_output,
            "output": case.output,
            # TODO handle scores, labels, metrics
        }

        # Calculate assertions pass rate
        assertions_total = len(case.assertions)
        assertions_passed = sum(1 for a in case.assertions.values() if a.value)
        case_data["assertions_passed_rate"] = (
            assertions_passed / assertions_total if assertions_total > 0 else 1.0
        )

        # Add individual assertion columns
        for assertion_name, assertion_obj in case.assertions.items():
            case_data[f"assertion.{assertion_name}"] = assertion_obj.value

        records.append(case_data)

    df = pd.DataFrame(records)
    df = expand_dict_columns(df, ["input", "attributes"])
    return df


def calculate_aggregates(df):
    result = df.groupby("attributes.model").agg(
        total_count=("assertion.EqualsExpected", "count"),
        equals_expected_true=("assertion.EqualsExpected", lambda x: x.sum()),
    )

    # Calculate the rate
    result["equals_expected_rate"] = (
        result["equals_expected_true"] / result["total_count"]
    )

    return result


def run_eval(models, prompts, dataset):
    n_cases = len(dataset.cases) * len(models) * len(prompts)
    print(f"Starting to evaluate {n_cases}")
    global i
    i = 1

    reports = []
    for model_name in models:
        for prompt in prompts:
            async def _run_llm(input: Dict) -> str:
                set_eval_attribute('model', model_name)
                set_eval_attribute('prompt', prompt)

                global i
                print(f"Evaluating {i} of {n_cases}")
                i += 1
                prompt_filled = prompt.format(**input)
                result = run_llm(model_name, prompt_filled)
                return result

            report = dataset.evaluate_sync(_run_llm, name=model_name)
            reports.append(report)

            table = report.console_table(
                include_input=True, include_output=True, include_expected_output=True
            )
            with open(f"results/{model_name}.txt", "w", encoding="utf-8") as file:
                io_file = StringIO()
                Console(file=io_file).print(table)
                file.write(io_file.getvalue())
                print(f"Report written to results/{model_name}.txt")

        if len(reports) >= 2:
            table = reports[0].console_table(
                baseline=reports[1], include_output=True, include_expected_output=True
            )
            with open(f"results/diff.txt", "w", encoding="utf-8") as file:
                io_file = StringIO()
                Console(file=io_file).print(table)
                file.write(io_file.getvalue())
                print(f"Report written to results/diff.txt")

    report_dfs = [report_to_df(report) for report in reports]
    results = pd.concat(report_dfs, axis=0, ignore_index=True)
    aggregate_df = calculate_aggregates(results)
    return (results, aggregate_df)

def main():
    models = [
        "gemini-2.0-flash",
        "llama-3.2-3b-4bit",
    ]
    prompts = [
        """
        Given this political fundraising email, respond with the name of the committee in the disclaimer that begins with Paid for by but does not include Paid for by, the committee address or the treasurer name. If no committee is present, return "None". Do not include any other text, no yapping.
        name: {name}
        email: {email}
        subject: {subject}
        body: {body}
        """
    ]
    dataset = Dataset[Any, Any, Any].from_file("tests.json")
    results, aggregate_df = run_eval(models, prompts, dataset)
    results.to_csv(f"./results/results.csv", index=False)
    aggregate_df.to_csv(f"./results/aggregate.csv")

if __name__ == "__main__":
    main()
