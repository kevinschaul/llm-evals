#!/usr/bin/env python3
import json
import sys

def convert_jsonl_to_json(input_file, output_file):
    """
    Convert JSONL file from:
    {"vars": {"headline": XX, "content": XX}, "assert": [{"type": "equals", "value": true}]}
    
    To a single JSON array of:
    {"inputs": {"headline": XX, "content": XX}, "expected_output": true}
    """
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            try:
                # Parse the original JSON
                original = json.loads(line.strip())
                
                # Extract the required fields
                headline = original['vars']['headline']
                content = original['vars']['content']
                expected_output = original['assert'][0]['value']
                
                # Create the new format
                new_format = {
                    "inputs": {
                        "headline": headline,
                        "content": content
                    },
                    "expected_output": expected_output
                }
                
                # Add to results array
                results.append(new_format)
                
            except json.JSONDecodeError:
                print(f"Error: Line {line_num} is not valid JSON. Skipping.")
            except KeyError as e:
                print(f"Error: Line {line_num} is missing key {e}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}. Skipping.")
    
    # Write all results as a single JSON array
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python converter.py input.jsonl output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Converting {input_file} to {output_file}...")
    convert_jsonl_to_json(input_file, output_file)
    print("Conversion complete!")
