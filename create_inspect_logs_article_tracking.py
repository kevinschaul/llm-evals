#!/usr/bin/env python3
"""
Create inspect_ai .eval log files from cached LLM responses in eval_cache.db

This script generates proper inspect_ai log files that can be used for
caching when re-running evaluations.
"""
import sqlite3
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
from collections import defaultdict
import hashlib

from inspect_ai.log import EvalLog, write_eval_log
from inspect_ai.log._log import (
    EvalConfig, EvalDataset, EvalMetric, EvalSpec, EvalResults,
    EvalRevision, EvalSample, EvalScore, EvalStats
)
from inspect_ai.model import (
    ChatMessageAssistant, ChatMessageSystem, ChatMessageUser,
    ModelOutput, ModelUsage, ChatCompletionChoice, GenerateConfig,
    ContentText
)
from inspect_ai.scorer import Score


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
            'input_tokens': row[4] or 0,
            'output_tokens': row[5] or 0,
            'duration_ms': row[6] or 0
        })

    return results


def extract_input_from_prompt(prompt: str) -> str:
    """Extract the input text from a full prompt"""
    # For article-tracking-trump, the template ends with the question
    template_end = 'Respond simply with "true" or "false".\n'
    idx = prompt.find(template_end)
    if idx >= 0:
        return prompt[idx + len(template_end):].strip()
    return prompt


def match_response_to_sample(response: Dict, samples: List[Dict]) -> Optional[tuple]:
    """Match a cached response to its test sample by comparing input text"""
    input_text = extract_input_from_prompt(response['prompt'])

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


def create_eval_sample(
    sample_id: int,
    test_sample: Dict,
    response: Dict,
    system_prompt: str
) -> EvalSample:
    """Create an EvalSample from test data and cached response"""
    input_text = f"{test_sample['headline']}\n{test_sample['content']}"
    target = test_sample['__expected']
    response_text = response['response'].strip()

    # Score using includes logic
    score_value = 1.0 if target.lower() in response_text.lower() else 0.0

    # Create messages
    messages = [
        ChatMessageSystem(content=system_prompt, source=None),
        ChatMessageUser(content=input_text, source="input")
    ]

    # Create output with ContentText list (matching real inspect_ai format)
    output = ModelOutput(
        model=response['model'],
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[
                        ContentText(
                            type='text',
                            text=response_text
                        )
                    ],
                    source="generate"
                ),
                stop_reason="stop"
            )
        ],
        completion=response_text,
        usage=ModelUsage(
            input_tokens=response['input_tokens'],
            output_tokens=response['output_tokens'],
            total_tokens=response['input_tokens'] + response['output_tokens']
        )
    )

    # Create score
    scores = {
        "includes": Score(
            value=score_value,
            explanation=f"Expected: {target}, Got: {response_text}"
        )
    }

    return EvalSample(
        id=sample_id,
        epoch=1,
        input=input_text,
        target=target,
        messages=messages,
        output=output,
        scores=scores,
        metadata={
            "headline": test_sample['headline'],
            "expected": target
        }
    )


def create_eval_log(
    eval_name: str,
    task_file: str,
    model: str,
    responses: List[Dict],
    test_samples: List[Dict],
    system_prompt: str
) -> EvalLog:
    """Create a complete EvalLog from cached responses"""

    # Match responses to samples and create EvalSamples
    eval_samples = []
    for response in responses:
        idx, sample = match_response_to_sample(response, test_samples)
        if sample:
            eval_sample = create_eval_sample(
                sample_id=idx + 1,
                test_sample=sample,
                response=response,
                system_prompt=system_prompt
            )
            eval_samples.append(eval_sample)

    # Sort by sample ID
    eval_samples.sort(key=lambda s: s.id)

    # Calculate stats
    scores = [s.scores.get("includes", Score(value=0.0)).value
              for s in eval_samples]
    accuracy = sum(scores) / len(scores) if scores else 0.0

    # Generate task ID
    task_id = hashlib.sha256(f"{eval_name}_{model}".encode()).hexdigest()[:20]
    eval_id = hashlib.sha256(f"{eval_name}_{model}_{datetime.now().isoformat()}".encode()).hexdigest()[:20]

    # Create EvalLog
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            task=eval_name,
            task_id=task_id,
            task_version=0,
            task_file=task_file,
            task_display_name=eval_name,
            task_registry_name=eval_name,
            eval_id=eval_id,
            run_id=eval_id,
            created=datetime.now(timezone.utc).isoformat(),
            dataset=EvalDataset(
                samples=len(eval_samples),
                sample_ids=list(range(1, len(eval_samples) + 1))
            ),
            model=model,
            config=EvalConfig(
                epochs=1,
                fail_on_error=False,
                log_samples=True
            ),
            revision=EvalRevision(
                type="git",
                origin="reconstructed",
                commit="from-cache"
            ),
            packages={"inspect_ai": "0.3.136"}
        ),
        results=EvalResults(
            scores=[
                EvalScore(
                    name="includes",
                    scorer="includes",
                    value=accuracy,
                    metrics={
                        "accuracy": EvalMetric(name="accuracy", value=accuracy)
                    },
                    scored_samples=len(eval_samples),
                    unscored_samples=0
                )
            ],
            total_samples=len(eval_samples)
        ),
        stats=EvalStats(
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat()
        ),
        samples=eval_samples
    )

    return log


def create_logs_for_eval(eval_name: str, eval_dir: Path):
    """Create inspect_ai log files from cached responses"""

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

    # System prompt for article-tracking-trump
    system_prompt = """Is this article primarily about a new action or announcement by the Trump administration/White House/Executive? Respond simply with "true" or "false".
{input}"""

    # Get cached responses
    print(f"\nLoading cached responses from {db_path}...")
    cached_responses = get_cached_responses(
        db_path,
        "Is this article primarily about a new action or announcement by the Trump administration"
    )
    print(f"Found {len(cached_responses)} cached responses")

    # Group by model
    model_responses = group_responses_by_model(cached_responses)
    print(f"\nProcessing responses for {len(model_responses)} models:\n")

    # Create logs for each model
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    task_file = str(eval_dir / "eval.py")

    for model, responses in model_responses.items():
        print(f"  Model: {model}")
        print(f"    Creating log with {len(responses)} responses...")

        eval_log = create_eval_log(
            eval_name=eval_name,
            task_file=task_file,
            model=model,
            responses=responses,
            test_samples=test_samples,
            system_prompt=system_prompt
        )

        # Generate log filename
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-00-00")
        model_slug = model.replace("/", "-")
        task_id = eval_log.eval.task_id
        log_filename = f"{timestamp}_{eval_name}_{model_slug}_{task_id}.eval"
        log_path = logs_dir / log_filename

        # Write log
        write_eval_log(eval_log, str(log_path))
        print(f"    ✓ Wrote log to {log_path}")

        # Print stats
        if eval_log.results and eval_log.results.scores:
            accuracy = eval_log.results.scores[0].metrics.get("accuracy", EvalMetric(name="accuracy", value=0.0)).value
            print(f"    Accuracy: {accuracy:.2%}")

    print(f"\n✓ Successfully created inspect_ai logs for {eval_name}")
    print(f"\nLog files written to {logs_dir}/")


if __name__ == "__main__":
    eval_name = "article-tracking-trump"
    eval_dir = Path(f"src/evals/{eval_name}")
    create_logs_for_eval(eval_name, eval_dir)
