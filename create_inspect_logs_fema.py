#!/usr/bin/env python3
"""
Create inspect_ai .eval log files for extract-fema-incidents from cached LLM responses
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict
from collections import defaultdict
import hashlib

from inspect_ai.log import EvalLog, write_eval_log
from inspect_ai.log._log import (
    EvalConfig, EvalDataset, EvalMetric, EvalSpec, EvalResults,
    EvalRevision, EvalSample, EvalScore, EvalStats
)
from inspect_ai.model import (
    ChatMessageAssistant, ChatMessageUser,
    ModelOutput, ModelUsage, ChatCompletionChoice, GenerateConfig,
    ContentText
)
from inspect_ai.scorer import Score


def get_cached_responses(db_path: Path, attachment_path: str) -> List[Dict]:
    """Extract cached responses from database for specific attachment"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get responses for specific attachment (JPG or PDF)
    query = """
    SELECT r.model, r.prompt, r.response, r.datetime_utc, r.input_tokens, r.output_tokens, r.duration_ms
    FROM responses r
    JOIN prompt_attachments pa ON r.id = pa.response_id
    JOIN attachments a ON pa.attachment_id = a.id
    WHERE r.prompt LIKE '%Extract all declaration requests%'
    AND a.path = ?
    ORDER BY r.model, r.datetime_utc DESC
    """

    cursor.execute(query, (attachment_path,))
    rows = cursor.fetchall()
    conn.close()

    # Group by model and take most recent per model
    model_responses = {}
    for row in rows:
        model = row[0]
        if model not in model_responses:
            model_responses[model] = {
                'model': row[0],
                'prompt': row[1],
                'response': row[2],
                'datetime_utc': row[3],
                'input_tokens': row[4] or 0,
                'output_tokens': row[5] or 0,
                'duration_ms': row[6] or 0
            }

    return list(model_responses.values())


def group_responses_by_model(responses: List[Dict]) -> Dict[str, List[Dict]]:
    """Group responses by model name"""
    grouped = defaultdict(list)
    for response in responses:
        grouped[response['model']].append(response)
    return dict(grouped)


def create_eval_sample(
    sample_id: str,
    response: Dict,
    expected_json: str
) -> EvalSample:
    """Create an EvalSample from cached response"""

    # Extract input from prompt (the part after the instructions)
    prompt = response['prompt']
    # For FEMA eval, the input is just a description since it's an image/PDF task
    input_text = "Extract all declaration requests (from cached response)"

    response_text = response['response'].strip()

    # Apply solver transformations to response text
    response_text_transformed = response_text

    # Solver 1: extract_json_from_markdown
    if '```' in response_text_transformed:
        start = response_text_transformed.find('[')
        end = response_text_transformed.rfind(']') + 1
        if start != -1 and end > 0:
            response_text_transformed = response_text_transformed[start:end]

    # Solver 2: unwrap_items
    try:
        parsed = json.loads(response_text_transformed)
        if isinstance(parsed, dict) and 'items' in parsed:
            response_text_transformed = json.dumps(parsed['items'])
    except (json.JSONDecodeError, Exception):
        pass

    # Score the transformed response
    try:
        response_json = json.loads(response_text_transformed)
        expected_json_obj = json.loads(expected_json)
        score_value = 1.0 if response_json == expected_json_obj else 0.0
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        score_value = 0.0

    # Create messages
    messages = [
        ChatMessageUser(content=input_text, source="input")
    ]

    # Create output (use transformed text for completion)
    output = ModelOutput(
        model=response['model'],
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[ContentText(type='text', text=response_text_transformed)],
                    source="generate"
                ),
                stop_reason="stop"
            )
        ],
        completion=response_text_transformed,
        usage=ModelUsage(
            input_tokens=response['input_tokens'],
            output_tokens=response['output_tokens'],
            total_tokens=response['input_tokens'] + response['output_tokens']
        )
    )

    # Create score
    scores = {
        "fema_extraction": Score(
            value=score_value,
            explanation=f"Matched {int(score_value * len(json.loads(expected_json)))} of {len(json.loads(expected_json))} incidents"
        )
    }

    return EvalSample(
        id=sample_id,
        epoch=1,
        input=input_text,
        target=expected_json,  # Store full JSON - DO NOT TRUNCATE
        messages=messages,
        output=output,
        scores=scores,
        metadata={}
    )


def create_eval_log(
    eval_name: str,
    task_file: str,
    model: str,
    jpg_response: Dict,
    pdf_response: Dict,
    expected_json: str
) -> EvalLog:
    """Create a complete EvalLog from cached responses"""

    # Create samples for both JPG and PDF
    eval_samples = []

    # JPG sample
    if jpg_response:
        jpg_sample = create_eval_sample(
            sample_id="jpg",
            response=jpg_response,
            expected_json=expected_json
        )
        eval_samples.append(jpg_sample)

    # PDF sample
    if pdf_response:
        pdf_sample = create_eval_sample(
            sample_id="pdf",
            response=pdf_response,
            expected_json=expected_json
        )
        eval_samples.append(pdf_sample)

    # Calculate stats
    scores = [s.scores.get("fema_extraction", Score(value=0.0)).value
              for s in eval_samples]
    accuracy = sum(scores) / len(scores) if scores else 0.0

    # Generate IDs
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
                sample_ids=[s.id for s in eval_samples]
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
                    name="fema_extraction",
                    scorer="fema_extraction",
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


def main():
    eval_name = "extract-fema-incidents"
    eval_dir = Path(f"src/evals/{eval_name}")

    db_path = Path("eval_cache.db")

    if not eval_dir.exists():
        print(f"Error: Eval directory {eval_dir} not found")
        return

    if not db_path.exists():
        print(f"Error: Database {db_path} not found")
        return

    # Load expected JSON
    expected_file = eval_dir / "expected.json"
    with open(expected_file, 'r') as f:
        expected_json = f.read().strip()

    print(f"Loaded expected JSON ({len(expected_json)} chars)")

    # Get cached responses for JPG and PDF separately
    print(f"\nLoading cached responses from {db_path}...")

    jpg_path = "src/evals/extract-fema-incidents/fema-daily-operation-brief-p9.jpg"
    pdf_path = "src/evals/extract-fema-incidents/fema-daily-operation-brief.pdf"

    jpg_responses = get_cached_responses(db_path, jpg_path)
    pdf_responses = get_cached_responses(db_path, pdf_path)

    print(f"Found {len(jpg_responses)} JPG responses")
    print(f"Found {len(pdf_responses)} PDF responses")

    # Combine models from both
    all_models = set([r['model'] for r in jpg_responses] + [r['model'] for r in pdf_responses])
    print(f"\nProcessing responses for {len(all_models)} models:\n")

    # Create logs for each model
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    task_file = str(eval_dir / "eval.py")

    for model in sorted(all_models):
        print(f"  Model: {model}")

        # Get responses for this model
        jpg_resp = next((r for r in jpg_responses if r['model'] == model), None)
        pdf_resp = next((r for r in pdf_responses if r['model'] == model), None)

        print(f"    JPG: {'✓' if jpg_resp else '✗'}  PDF: {'✓' if pdf_resp else '✗'}")

        eval_log = create_eval_log(
            eval_name=eval_name,
            task_file=task_file,
            model=model,
            jpg_response=jpg_resp,
            pdf_response=pdf_resp,
            expected_json=expected_json
        )

        # Verify target is not truncated
        if eval_log.samples:
            target_len = len(eval_log.samples[0].target)
            print(f"    Target length: {target_len} chars")
            if target_len < len(expected_json):
                print(f"    ⚠️  WARNING: Target was truncated!")

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
    main()
