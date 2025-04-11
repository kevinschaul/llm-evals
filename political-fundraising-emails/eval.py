from io import StringIO
from typing import Any, Dict
from pydantic_evals import Dataset
import llm
from rich.console import Console
from cachier import cachier


@cachier(cache_dir="./cached-llm-responses", separate_files=True)
def run_llm(model_name: str, prompt: str) -> str:
    model = llm.get_model(model_name)
    response = model.prompt(prompt)
    return response.text()


def main():
    models = [
        "gemini-2.0-flash",
        "llama-3.2-3b-4bit",
    ]
    dataset = Dataset[Any, Any, Any].from_file("tests.json")

    reports = []
    for model_name in models:

        async def _run_llm(input: Dict) -> str:
            prompt = f"""
            Given this political fundraising email, respond with the name of the committee in the disclaimer that begins with Paid for by but does not include Paid for by, the committee address or the treasurer name. If no committee is present, return "None". Do not include any other text, no yapping.
            name: {input['name']}
            email: {input['email']}
            subject: {input['subject']}
            body: {input['body']}
            """
            result = run_llm(model_name, prompt)
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

    table = reports[0].console_table(
        baseline=reports[1], include_output=True, include_expected_output=True
    )
    with open(f"results/diff.txt", "w", encoding="utf-8") as file:
        io_file = StringIO()
        Console(file=io_file).print(table)
        file.write(io_file.getvalue())


if __name__ == "__main__":
    main()
