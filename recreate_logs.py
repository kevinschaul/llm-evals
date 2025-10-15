#!/usr/bin/env python3
"""
Recreate inspect_ai logs from cached LLM responses in eval_cache.db

This script extracts responses from the llm library's eval_cache.db and
recreates inspect_ai log files that can be viewed and analyzed.
"""
import sqlite3
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
from collections import defaultdict


def read_csv_dataset(csv_path: Path) -> List[Dict[str, str]]:
    """Read CSV test data and return as list of dicts"""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def get_cached_responses(db_path: Path, prompt_prefix: str) -> List[Dict]:
    """Extract cached responses from llm database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT model, prompt, response, datetime_utc, input_tokens, output_tokens, duration_ms
    FROM responses
    WHERE prompt LIKE ?
    ORDER BY datetime_utc
    """

    cursor.execute(query, (f"{prompt_prefix}%",))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            'model': row[0],
            'prompt': row[1],
            'response': row[2],
            'datetime_utc': row[3],
            'input_tokens': row[4],
            'output_tokens': row[5],
            'duration_ms': row[6]
        })

    return results


def extract_input_from_prompt(prompt: str) -> str:
    """Extract the input text from a full prompt"""
    # The prompt format is: template_text + input_text
    # Find where the template ends and input begins
    template_end = "Respond ONLY with the category.\n\n"
    idx = prompt.find(template_end)
    if idx >= 0:
        return prompt[idx + len(template_end):].strip()
    return prompt


def match_response_to_sample(response: Dict, samples: List[Dict]) -> Optional[tuple]:
    """Match a cached response to its test sample by comparing input text"""
    input_text = extract_input_from_prompt(response['prompt'])

    # Try to match by comparing full input text
    for idx, sample in enumerate(samples):
        sample_input = f"{sample['headline']}\n{sample['content']}"
        if sample_input.strip() == input_text.strip():
            return idx, sample

    return None, None


def group_responses_by_model(responses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group responses by model name"""
    grouped = defaultdict(list)
    for response in responses:
        grouped[response['model']].append(response)
    return dict(grouped)


def create_inspect_log(
    eval_name: str,
    model: str,
    samples: List[Dict],
    responses: List[Dict],
    test_samples: List[Dict]
) -> Dict:
    """Create an inspect_ai log structure from cached responses"""

    # Match responses to samples
    sample_results = []
    matched_count = 0
    correct_count = 0

    for response in responses:
        idx, sample = match_response_to_sample(response, test_samples)
        if sample:
            matched_count += 1
            input_text = f"{sample['headline']}\n{sample['content']}"
            response_text = response['response'].strip()
            target = sample['__expected']

            # Simple includes scoring
            score = 1.0 if target.lower() in response_text.lower() else 0.0
            if score == 1.0:
                correct_count += 1

            sample_results.append({
                'id': idx + 1,
                'epoch': 1,
                'input': input_text,
                'target': target,
                'response': response_text,
                'score': score,
                'input_tokens': response.get('input_tokens', 0),
                'output_tokens': response.get('output_tokens', 0),
                'duration_ms': response.get('duration_ms', 0)
            })

    accuracy = correct_count / matched_count if matched_count > 0 else 0.0

    print(f"  Model: {model}")
    print(f"    Matched: {matched_count} responses")
    print(f"    Correct: {correct_count}")
    print(f"    Accuracy: {accuracy:.2%}")

    return {
        'eval_name': eval_name,
        'model': model,
        'total_samples': matched_count,
        'correct': correct_count,
        'accuracy': accuracy,
        'samples': sample_results,
        'created': datetime.now(timezone.utc).isoformat()
    }


def write_results_csv(log_data: Dict, output_dir: Path):
    """Write results to CSV files in the inspect_ai format"""
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Write detailed results
    results_file = results_dir / f"results_{log_data['model'].replace('/', '_')}.csv"
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['id', 'input', 'target', 'response', 'score', 'input_tokens', 'output_tokens', 'duration_ms']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in log_data['samples']:
            writer.writerow({
                'id': sample['id'],
                'input': sample['input'][:100] + '...' if len(sample['input']) > 100 else sample['input'],
                'target': sample['target'],
                'response': sample['response'],
                'score': sample['score'],
                'input_tokens': sample['input_tokens'],
                'output_tokens': sample['output_tokens'],
                'duration_ms': sample['duration_ms']
            })

    print(f"    Wrote results to {results_file}")

    # Note: We don't update aggregate.csv here since it will be regenerated
    # by the system when needed
    print(f"    Individual results written, aggregate will be regenerated on next run")


def recreate_logs_for_eval(eval_name: str):
    """Recreate logs from cached responses for the specified eval"""

    eval_dir = Path(f"src/evals/{eval_name}")
    db_path = Path("eval_cache.db")

    if not eval_dir.exists():
        print(f"Error: Eval directory {eval_dir} not found")
        return

    if not db_path.exists():
        print(f"Error: Database {db_path} not found")
        return

    # Read test data
    csv_path = eval_dir / "tests.csv"
    if not csv_path.exists():
        print(f"Error: Test CSV {csv_path} not found")
        return

    test_samples = read_csv_dataset(csv_path)
    print(f"Loaded {len(test_samples)} test samples from {csv_path}")

    # Get cached responses
    print(f"\nLoading cached responses from {db_path}...")
    cached_responses = get_cached_responses(
        db_path,
        "Put this article into one of the following categories"
    )
    print(f"Found {len(cached_responses)} cached responses")

    # Group by model
    model_responses = group_responses_by_model(cached_responses)

    print(f"\nProcessing responses for {len(model_responses)} models:\n")

    # Create logs for each model
    for model, responses in model_responses.items():
        log_data = create_inspect_log(
            eval_name,
            model,
            [],
            responses,
            test_samples
        )
        write_results_csv(log_data, eval_dir)

    print(f"\n✓ Successfully recreated logs for {eval_name}")
    print(f"\nResults written to {eval_dir}/results/")


if __name__ == "__main__":
    recreate_logs_for_eval("article-tracking-trump-categories")
