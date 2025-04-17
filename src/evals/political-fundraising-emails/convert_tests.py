import csv
import json
import sys
import random


def convert_csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific headers.

    Args:
        csv_file_path: Path to the input CSV file
        json_file_path: Path to the output JSON file
    """
    data = []

    try:
        with open(csv_file_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)

            # Validate headers
            expected_headers = [
                "committee",
                "name",
                "email",
                "subject",
                "date",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "domain",
                "body",
                "party",
                "disclaimer",
            ]

            # Check if all expected headers are present
            actual_headers = csv_reader.fieldnames
            if not actual_headers:
                raise ValueError("CSV file has no headers")

            missing_headers = [h for h in expected_headers if h not in actual_headers]
            if missing_headers:
                raise ValueError(
                    f"Missing expected headers: {', '.join(missing_headers)}"
                )

            # Process each row
            row_count = 0
            for row in csv_reader:
                row_count += 1

                # Create vars object (all columns except committee)
                vars_obj = {}
                for key, value in row.items():
                    if key != "committee":
                        vars_obj[key] = value

                # Create JSON structure
                json_entry = {
                    "inputs": vars_obj,
                    "expected_output": row["committee"],
                    "name": f"Test #{row_count} - {row.get('name', 'Unknown')}",
                }
                data.append(json_entry)

        sample = random.sample(data, 100)

        # Write to JSON file
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json_file.write(
                json.dumps({"cases": sample, "evaluators": ["EqualsExpected"]}, indent=2)
            )

        print(f"Successfully converted {csv_file_path} to {json_file_path}")
        print(f"Processed {len(data)} records")

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found")
        return False
    except ValueError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python csv_to_json.py <input_csv_file> <output_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    success = convert_csv_to_json(input_file, output_file)
    sys.exit(0 if success else 1)
