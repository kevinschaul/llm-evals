"""
A promptfoo provider to call the llm library via HTTP

Docs:
- https://www.promptfoo.dev/docs/providers/python/
- https://llm.datasette.io/en/stable/index.html
"""

import re
from typing import Optional, Union, Dict, List, Any
import json
from pydantic import ValidationError
import llm
from llm.cli import logs_db_path, migrate
from llm.utils import make_schema_id, sqlite_utils
from flask import Flask, request, jsonify
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

db_path_main = "raw_responses.db"
# db_paths_search = [db_path_main, logs_db_path()]
db_paths_search = [db_path_main]


def parse_boolean(text: str) -> bool:
    """Parse the output of an LLM call to a boolean.

    https://python.langchain.com/api_reference/_modules/langchain/output_parsers/boolean.html#BooleanOutputParser

    Args:
        text: output of a language model

    Returns:
        boolean
    """

    TRUE_VAL = "TRUE"
    FALSE_VAL = "FALSE"

    regexp = rf"\b({TRUE_VAL}|{FALSE_VAL})\b"

    truthy = {
        val.upper()
        for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
    }
    if TRUE_VAL.upper() in truthy:
        if FALSE_VAL.upper() in truthy:
            raise ValueError(
                f"Ambiguous response. Both {TRUE_VAL} and {FALSE_VAL} "
                f"in received: {text}."
            )
        return True
    elif FALSE_VAL.upper() in truthy:
        if TRUE_VAL.upper() in truthy:
            raise ValueError(
                f"Ambiguous response. Both {TRUE_VAL} and {FALSE_VAL} "
                f"in received: {text}."
            )
        return False
    raise ValueError(
        f"BooleanOutputParser expected output value to include either "
        f"{TRUE_VAL} or {FALSE_VAL}. Received {text}."
    )


def to_uppercase(text: str) -> str:
    return text.upper()


def strip_think_tags(text):
    """
    Remove all content between <think> and </think> tags, including the tags themselves.
    """
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


class RawResponse:
    def __init__(self, db, response_row) -> None:
        self.db = db
        self.response_row = response_row

    def to_response(self):
        return llm.Response.from_row(self.db, self.response_row)

    def text(self):
        return self.response_row["response"]


def find_cached_response(
    db: sqlite_utils.Database,
    prompt_text: str,
    model_id_or_alias: str,
    system_prompt_text: Optional[str] = None,
    schema: Optional[Union[Dict[str, Any], type[llm.models.BaseModel]]] = None,
    fragments: Optional[List[str]] = None,
    system_fragments: Optional[List[str]] = None,
    attachments: Optional[List[llm.Attachment]] = None,
    **options: Any,
) -> Optional[RawResponse]:
    """
    Searches the LLM database for an exact cached query and returns the response row if found.

    This function attempts to match based on the core components of a prompt:
    - The main prompt text.
    - The model ID (resolved from alias if necessary).
    - The system prompt text.
    - Model-specific options.
    - The JSON schema used (if any).
    - Fragments and system_fragments (by their content hash).
    - Attachments (by their content hash or URL).

    Args:
        db: An active sqlite_utils.Database connection to the LLM logs.db.
        prompt_text: The main text of the prompt.
        model_id_or_alias: The ID or alias of the model used.
        system_prompt_text: The system prompt text, if any.
        schema: The JSON schema (as a dict) or Pydantic model class, if any.
        fragments: A list of fragment strings used in the prompt.
        system_fragments: A list of fragment strings used in the system prompt.
        attachments: A list of llm.Attachment objects.
        **options: Keyword arguments representing model-specific options.

    Returns:
        A RawResponse object if an exact match is found, otherwise None.
    """
    try:
        model_obj = llm.get_model(model_id_or_alias)
    except llm.UnknownModelError:
        # Model not found, so no cache entry can exist for it.
        return None

    canonical_model_id = model_obj.model_id

    # 1. Canonicalize options
    # Pydantic model `model_obj.Options` will validate and apply defaults
    try:
        prompt_options_obj = model_obj.Options(**options)
    except ValidationError:
        # Invalid options for this model, so it couldn't have been logged like this.
        return None

    # Prepare options_json as it would be stored in the database
    options_dict_for_db = {
        k: v for k, v in dict(prompt_options_obj).items() if v is not None
    }
    # An empty options dict is stored as '{}'
    options_json_to_match = json.dumps(options_dict_for_db or {}, sort_keys=True)

    # 2. Canonicalize schema
    schema_id_to_match: Optional[str] = None
    if schema:
        if not isinstance(schema, dict) and issubclass(schema, llm.models.BaseModel):
            current_schema_dict = schema.model_json_schema()
        else:
            current_schema_dict = schema
        schema_id_to_match, _ = make_schema_id(current_schema_dict)

    where_clauses = []
    params = {}

    where_clauses.append("model = :model_id")
    params["model_id"] = canonical_model_id

    where_clauses.append("prompt = :prompt_text")
    params["prompt_text"] = prompt_text

    if system_prompt_text is None:
        where_clauses.append("system IS NULL")
    else:
        where_clauses.append("system = :system_prompt_text")
        params["system_prompt_text"] = system_prompt_text

    where_clauses.append("options_json = :options_json")
    params["options_json"] = options_json_to_match

    if schema_id_to_match is None:
        where_clauses.append("schema_id IS NULL")
    else:
        where_clauses.append("schema_id = :schema_id")
        params["schema_id"] = schema_id_to_match

    candidate_responses = list(
        db["responses"].rows_where(
            where=" AND ".join(where_clauses),
            where_args=params,
            select="id",
            order_by="id DESC",
        )
    )

    if not candidate_responses:
        return None

    # Convert list of dicts to list of IDs
    candidate_response_ids = [row["id"] for row in candidate_responses]

    # 4. Canonicalize input fragments and attachments for comparison
    current_fragment_hashes = sorted([llm.Fragment(f).id() for f in (fragments or [])])
    current_system_fragment_hashes = sorted(
        [llm.Fragment(f).id() for f in (system_fragments or [])]
    )
    current_attachment_ids = sorted([att.id() for att in (attachments or [])])

    for response_id in candidate_response_ids:
        # Check fragments
        logged_prompt_fragments = list(
            db.query(
                """
            SELECT f.hash FROM fragments f
            JOIN prompt_fragments pf ON f.id = pf.fragment_id
            WHERE pf.response_id = ? ORDER BY f.hash
            """,
                [response_id],
            )
        )
        logged_prompt_fragment_hashes = [row["hash"] for row in logged_prompt_fragments]
        if logged_prompt_fragment_hashes != current_fragment_hashes:
            continue

        # Check system fragments
        logged_system_fragments = list(
            db.query(
                """
            SELECT f.hash FROM fragments f
            JOIN system_fragments sf ON f.id = sf.fragment_id
            WHERE sf.response_id = ? ORDER BY f.hash
            """,
                [response_id],
            )
        )
        logged_system_fragment_hashes = [row["hash"] for row in logged_system_fragments]
        if logged_system_fragment_hashes != current_system_fragment_hashes:
            continue

        # Check attachments
        logged_attachments = list(
            db.query(
                """
            SELECT a.id FROM attachments a
            JOIN prompt_attachments pa ON a.id = pa.attachment_id
            WHERE pa.response_id = ? ORDER BY a.id
            """,
                [response_id],
            )
        )
        logged_attachment_ids = [row["id"] for row in logged_attachments]
        if logged_attachment_ids != current_attachment_ids:
            continue

        # If all checks pass, this is our match
        full_response_row = db["responses"].get(response_id)
        if full_response_row:
            return RawResponse(db, full_response_row)

    return None


def call_llm(prompt, options, attachments=[], schema={}):
    start_time = time.time()
    
    config = options.get("config", {})
    model_name = config.get("model")
    model_options = config.get("options", {})

    # Setup phase
    setup_start = time.time()
    transform_funcs_raw = config.get("transform_funcs")
    if not transform_funcs_raw:
        transform_funcs_raw = [config.get("transform_func", "nop")]

    transform_funcs = []
    for transform_func in transform_funcs_raw:
        if transform_func == "nop":
            # no-op
            continue
        if transform_func == "parse_boolean":
            transform_funcs.append(parse_boolean)
        elif transform_func == "to_uppercase":
            transform_funcs.append(to_uppercase)
        elif transform_func == "strip_think_tags":
            transform_funcs.append(strip_think_tags)
        else:
            raise ValueError(f"Unknown transform_func: {transform_func}")

    schema_raw = schema
    if schema_raw:
        if schema_raw["syntax"] == "dsl":
            schema = llm.schema_dsl(schema_raw["content"], multi=schema_raw["multi"])
        else:
            schema = schema_raw["content"]

    attachments_raw = attachments
    attachments = [llm.Attachment(**a) for a in attachments_raw]
    
    setup_duration = time.time() - setup_start
    logger.info(f"Setup phase took: {setup_duration:.3f}s")

    db_main = sqlite_utils.Database(db_path_main)

    # Cache lookup phase
    cache_start = time.time()
    raw_response = None
    for db_path in db_paths_search:
        db = sqlite_utils.Database(db_path)
        raw_response = find_cached_response(
            db,
            prompt_text=prompt,
            model_id_or_alias=model_name,
            schema=schema,
            attachments=attachments,
            **model_options,
        )
        if raw_response:
            # If we found the response
            if db_path != db_path_main:
                # If we found it in a db that is not the main db, replicate it
                # to the main db
                replication_start = time.time()
                raw_response.to_response().log_to_db(db_main)
                replication_duration = time.time() - replication_start
                logger.info(f"Cache replication took: {replication_duration:.3f}s")
            break
    
    cache_duration = time.time() - cache_start
    logger.info(f"Cache lookup took: {cache_duration:.3f}s")

    if not raw_response:
        # LLM call phase
        llm_start = time.time()
        model = llm.get_model(model_name)
        logger.info(f"Loaded model {model}")
        output = model.prompt(
            prompt, schema=schema, attachments=attachments, **model_options
        )

        # Gather text so that timing is accurate
        output.text()
        llm_duration = time.time() - llm_start
        logger.info(f"LLM API call took: {llm_duration:.3f}s")
        
        # DB write phase
        db_write_start = time.time()
        output.log_to_db(db_main)
        db_write_duration = time.time() - db_write_start
        logger.info(f"DB write took: {db_write_duration:.3f}s")
        
        raw_response = output
    else:
        logger.info("Cache hit - skipped LLM call")

    # Transform phase
    transform_start = time.time()
    try:
        parsed = raw_response.text().strip()
        for func in transform_funcs:
            parsed = func(parsed)
    except ValueError:
        parsed = None
    transform_duration = time.time() - transform_start
    logger.info(f"Transform phase took: {transform_duration:.3f}s")

    total_duration = time.time() - start_time
    logger.info(f"Total call_llm took: {total_duration:.3f}s")

    result = {
        "output": parsed,
    }

    return result


@app.route("/eval", methods=["POST"])
def evaluate():
    start_time = time.time()
    try:
        data = request.json
        logger.info(f"Request received: {data}")
        # logger.info(f"Available models: {llm.get_models()}")

        # Get the raw result
        call_start = time.time()
        response = call_llm(
            data["prompt"],
            data["options"],
            data.get("attachments", []),
            data.get("schema", {}),
        )
        call_duration = time.time() - call_start
        logger.info(f"call_llm took: {call_duration:.3f}s")

        # Ensure we return a proper JSON response
        output = response["output"]
        logger.info(f"Raw output: {output}")
        
        total_duration = time.time() - start_time
        logger.info(f"Total /eval request took: {total_duration:.3f}s")
        
        if isinstance(output, bool):
            return "true" if output else "false"
        else:
            # For strings and other types, return as-is (Flask will handle)
            return output

    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    migrate(sqlite_utils.Database(db_path_main))
    logger.info("Starting fast evaluation server on port 4242")
    app.run(host="127.0.0.1", port=4242, threaded=True, debug=False)
