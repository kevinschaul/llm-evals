from collections import defaultdict
import csv
import os
import re
import sys


def aggregate(input_file):
    model_prompt_results = defaultdict(lambda: {"passes": 0, "total": 0})
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            for column in reader.fieldnames:
                if "[" in column and "]" in column:
                    # Extract model name
                    model = re.search(r"\[(.*?)\]", column).group(1)

                    prompt = column.replace(f"[{model}]", "", 1).strip()
                    prompt = prompt.split("\\n")[0].split("{{")[0].strip()

                    # Use model-prompt as the key
                    key = f"{model} - {prompt}"

                    # Count passes and total
                    if column in row and row[column]:
                        model_prompt_results[key]["total"] += 1
                        if "[PASS]" in row[column]:
                            model_prompt_results[key]["passes"] += 1

    output_file = os.path.join(os.path.dirname(input_file) or ".", "aggregate.csv")
    with open(output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "prompt", "count", "n_correct", "share_correct"])

        for key, stats in model_prompt_results.items():
            # Split the key back into model and prompt
            model, prompt = key.split(" - ", 1)
            total = stats["total"]
            passes = stats["passes"]
            share_correct = f"{passes / total:.2f}" if total > 0 else "0.00"

            writer.writerow([model, prompt, total, passes, share_correct])

    print(f"Aggregates written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_csv.py <input_csv_file>")
        sys.exit(1)

    aggregate(sys.argv[1])
