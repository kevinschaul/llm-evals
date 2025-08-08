#!/usr/bin/env python
import csv
import glob


def find_error_models():
    csv_files = glob.glob("src/evals/**/results.csv", recursive=True)

    for csv_file in csv_files:
        eval_name = csv_file.replace("src/evals/", "").replace("/results.csv", "")

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Find columns that start with "["
            model_cols = []
            for i, col in enumerate(header):
                if col.strip().startswith("["):
                    model_name = col.split("]")[0][1:]
                    model_cols.append((i, model_name))

            # Check each row for errors
            error_models = set()
            for row in reader:
                for col_idx, model_name in model_cols:
                    if col_idx < len(row) and "[ERROR]" in row[col_idx]:
                        error_models.add(model_name)

            for model in sorted(error_models):
                print(f"  {eval_name}: {model}")


if __name__ == "__main__":
    print("Models with errors:")
    find_error_models()
