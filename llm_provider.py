"""
A provider for promptfoo that calls the llm library

Provide the model to use with the "model" option

Docs:
- https://www.promptfoo.dev/docs/providers/python/
- https://llm.datasette.io/en/stable/index.html
"""

import re
from typing import Optional, Union, Dict, List, Any
import json
from pydantic import ValidationError
import llm
from llm.cli import logs_db_path
from llm.utils import make_schema_id, sqlite_utils


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


def find_cached_response(
    db: sqlite_utils.Database,
    prompt_text: Optional[str],
    model_id_or_alias: str,
    system_prompt_text: Optional[str] = None,
    schema: Optional[Union[Dict[str, Any], type[llm.models.BaseModel]]] = None,
    fragments: Optional[List[str]] = None,
    system_fragments: Optional[List[str]] = None,
    attachments: Optional[List[llm.Attachment]] = None,
    # For simplicity, tools and tool_results are not included in this basic cache key.
    # A full "exact query" match for tools would be more complex.
    **options: Any,
) -> Optional[llm.Response]:
    """
    Searches the LLM database for an exact cached query and returns the Response object if found.

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
        An llm.Response object if an exact match is found, otherwise None.
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

    # 3. Build the WHERE clause and parameters for the SQL query
    where_clauses: List[str] = []
    sql_params: Dict[str, Any] = {}

    where_clauses.append("model = :model_id")
    sql_params["model_id"] = canonical_model_id

    if prompt_text is None:
        where_clauses.append("prompt IS NULL")
    else:
        where_clauses.append("prompt = :prompt_text")
        sql_params["prompt_text"] = prompt_text

    if system_prompt_text is None:
        where_clauses.append("system IS NULL")
    else:
        where_clauses.append("system = :system_prompt_text")
        sql_params["system_prompt_text"] = system_prompt_text

    # `options_json` is stored as '{}' for empty options by log_to_db
    where_clauses.append("options_json = :options_json")
    sql_params["options_json"] = options_json_to_match

    if schema_id_to_match is None:
        where_clauses.append("schema_id IS NULL")
    else:
        where_clauses.append("schema_id = :schema_id")
        sql_params["schema_id"] = schema_id_to_match

    # Construct the main query for the `responses` table
    # We select `id` first to find potential candidates
    sql_query = f"""
        SELECT id FROM responses
        WHERE {' AND '.join(where_clauses)}
        ORDER BY id DESC
    """

    # Find candidate response IDs
    candidate_response_ids = [row["id"] for row in db.query(sql_query, sql_params)]

    if not candidate_response_ids:
        return None

    # 4. Filter candidates by matching fragments, system_fragments, and attachments
    # This requires fetching their details and comparing.

    # Canonicalize input fragments and attachments for comparison
    current_fragment_hashes = sorted([llm.Fragment(f).id() for f in (fragments or [])])
    current_system_fragment_hashes = sorted(
        [llm.Fragment(f).id() for f in (system_fragments or [])]
    )
    current_attachment_ids = sorted([att.id() for att in (attachments or [])])

    for response_id in candidate_response_ids:
        # Check fragments
        logged_prompt_fragments_sql = """
            SELECT f.hash FROM fragments f
            JOIN prompt_fragments pf ON f.id = pf.fragment_id
            WHERE pf.response_id = ? ORDER BY f.hash
        """
        logged_prompt_fragment_hashes = [
            row["hash"] for row in db.query(logged_prompt_fragments_sql, [response_id])
        ]
        if logged_prompt_fragment_hashes != current_fragment_hashes:
            continue

        # Check system fragments
        logged_system_fragments_sql = """
            SELECT f.hash FROM fragments f
            JOIN system_fragments sf ON f.id = sf.fragment_id
            WHERE sf.response_id = ? ORDER BY f.hash
        """
        logged_system_fragment_hashes = [
            row["hash"] for row in db.query(logged_system_fragments_sql, [response_id])
        ]
        if logged_system_fragment_hashes != current_system_fragment_hashes:
            continue

        # Check attachments
        logged_attachments_sql = """
            SELECT a.id FROM attachments a
            JOIN prompt_attachments pa ON a.id = pa.attachment_id
            WHERE pa.response_id = ? ORDER BY a.id
        """
        logged_attachment_ids = [
            row["id"] for row in db.query(logged_attachments_sql, [response_id])
        ]
        if logged_attachment_ids != current_attachment_ids:
            continue

        # If all checks pass, this is our match. Fetch the full row and reconstruct.
        full_response_row = db["responses"].get(response_id)
        if full_response_row:
            # llm.Response.from_row will handle reconstructing the full object,
            # including its fragments, attachments, etc., from the database.
            return llm.Response.from_row(db, full_response_row)

    return None


def call_api(prompt, options, context):
    config = options.get("config", {})
    model_name = config.get("model")

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

    schema = None
    schema_raw = context.get("vars", {}).get("schema")
    if schema_raw:
        if schema_raw["syntax"] == "dsl":
            schema = llm.schema_dsl(schema_raw["content"], multi=schema_raw["multi"])
        else:
            schema = schema_raw["content"]

    attachments_raw = context.get("vars", {}).get("attachments", [])
    attachments = [llm.Attachment(**a) for a in attachments_raw]

    db = sqlite_utils.Database(logs_db_path())
    output = find_cached_response(
        db,
        prompt_text=prompt,
        model_id_or_alias=model_name,
        schema=schema,
        attachments=attachments,
    )
    if not output:
        model = llm.get_model(model_name)
        output = model.prompt(prompt, schema=schema, attachments=attachments)
        output.log_to_db(db)

    try:
        parsed = output.text().strip()
        for func in transform_funcs:
            parsed = func(parsed)
    except ValueError:
        parsed = None

    result = {
        "output": parsed,
    }

    return result


# Just for testing this script
if __name__ == "__main__":
    response = call_api(
        "Tell me a dog joke", {"config": {"model": "openai/gpt-4o-mini"}}, None
    )
    print(response)
