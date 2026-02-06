from __future__ import annotations

import json
import os
import sys
import math
import time
from functools import lru_cache
import re
import operator
from typing import Any, Annotated, Literal, TypedDict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import bigquery
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.memory import MemorySaver as GraphMemorySaver
except ImportError:  # pragma: no cover - fallback for older langgraph
    from langgraph.checkpoint.memory import InMemorySaver as GraphMemorySaver

from .config import load_config
from .utils import log_event
from .ripple_report import build_ripple_report


app = FastAPI(title="Metadata RAG API", version="1.0")


class MatchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class MatchTable(BaseModel):
    asset_id: str
    dataset_id: str
    table_name: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float


class MatchColumn(BaseModel):
    asset_id: str
    dataset_id: str
    table_name: str
    column_name: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float


class LineageRequest(BaseModel):
    table_name: str | None = None
    dataset_id: str | None = None
    query: str | None = None


class LineageEdgeResponse(BaseModel):
    source_dataset: str | None
    source_table: str | None
    source_column: str | None
    target_dataset: str | None
    target_table: str | None
    target_column: str | None
    relationship_type: str | None
    transformation_logic: str | None
    confidence_score: float | None
    discovery_method: str | None
    metadata: dict | None


class LineageResponse(BaseModel):
    status: str
    table_name: str | None = None
    dataset_id: str | None = None
    options: list[dict] = []
    edges: list[LineageEdgeResponse] = []


class TableRef(BaseModel):
    dataset_id: str
    table_name: str


class SQLGenerationTrace(BaseModel):
    candidate_tables: list[str] = []
    schema_column_counts: dict[str, int] = {}
    parse_error: bool = False
    sql_empty: bool = False
    response_snippet: str | None = None
    attempts: int = 1


class GenerateSQLRequest(BaseModel):
    user_query: str = Field(..., min_length=1)
    tables: list[TableRef] = []


class GenerateSQLResponse(BaseModel):
    sql: str
    notes: str | None = None
    tables_used: list[str] = []
    query_validation: dict[str, Any] | None = None
    sql_generation_trace: SQLGenerationTrace | None = None


class ExecuteSQLRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    max_rows: int = Field(200, ge=1, le=1000)
    dry_run: bool = False


class ExecuteSQLResponse(BaseModel):
    job_id: str | None
    total_rows: int | None
    rows: list[dict[str, Any]]


class ValidateQueryRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    budget_usd: float = Field(5.0, gt=0)
    warn_threshold_pct: float = Field(80.0, ge=0, le=100)
    project_id: str | None = None


class ValidateQueryResponse(BaseModel):
    status: str
    recommendation: str
    approved: bool
    is_suboptimal: bool
    cost: dict[str, Any]
    efficiency: dict[str, Any]
    issues: list[str] = []
    suggestions: list[str] = []


class ClassifyIntentRequest(BaseModel):
    query: str = Field(..., min_length=1)


class ClassifyIntentResponse(BaseModel):
    intent: str
    confidence: float
    entities: dict[str, Any]
    rationale: str | None = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    confidence: float | None = None
    search_results: list[dict[str, Any]] = []
    lineage: dict[str, Any] | None = None
    sql: str | None = None
    tables_used: list[str] = []
    query_validation: dict[str, Any] | None = None
    table_validation: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    impact_assessment: dict[str, Any] | None = None
    sql_generation_trace: SQLGenerationTrace | None = None
    needs_selection: bool = False


class TableSchemaRequest(BaseModel):
    dataset_id: str
    table_name: str


class TableSchemaResponse(BaseModel):
    dataset_id: str
    table_name: str
    columns: list[dict[str, Any]]


class RippleReportRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    table_name: str = Field(..., min_length=1)
    column_name: str = Field(..., min_length=1)
    max_hops: int = Field(3, ge=1, le=6)
    include_query_patterns: bool = False
    include_prompt_templates: bool = False


class RippleReportResponse(BaseModel):
    status: str
    target: dict[str, Any]
    executive_summary: dict[str, Any]
    upstream_analysis: dict[str, Any]
    downstream_analysis: dict[str, Any]
    visualization: dict[str, Any]
    recommendations: dict[str, Any]
    cycles: list[list[str]] = []


class ValidateTableRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    table_name: str = Field(..., min_length=1)
    layer: str = Field("silver", min_length=1)
    budget_usd: float = Field(5.0, gt=0)
    auto_detect_keys: bool = True
    auto_detect_grain: bool = True
    include_llm_summary: bool = True


class ValidateTableResponse(BaseModel):
    status: str
    validation_result: dict[str, Any]
    llm_summary: dict[str, Any] | None = None
    cli_output: str | None = None


class ValidateTableV2Request(BaseModel):
    table: str = Field(..., min_length=1)
    layer: Literal["bronze", "silver", "gold"] = "silver"
    project_id: str | None = None
    budget_usd: float = Field(5.0, gt=0)
    auto_detect_keys: bool = True
    auto_detect_grain: bool = True
    include_llm_summary: bool = True


CONFIG = load_config()
QUERYABLE_DATASETS = {"silver_layer", "gold_layer"}


def _is_queryable_dataset(dataset_id: str | None) -> bool:
    return bool(dataset_id) and dataset_id in QUERYABLE_DATASETS


def _get_bq_client() -> bigquery.Client:
    return bigquery.Client(project=CONFIG.project_id)


def _get_genai_client() -> genai.Client:
    return genai.Client(vertexai=True, project=CONFIG.project_id, location=CONFIG.vertex_location)


def _get_table_row_count(dataset_id: str, table_name: str) -> int | None:
    try:
        client = _get_bq_client()
        table = client.get_table(f"{CONFIG.project_id}.{dataset_id}.{table_name}")
        return int(table.num_rows)
    except Exception:
        return None


def _get_table_schema(dataset_id: str, table_name: str, limit: int = 80) -> list[dict[str, Any]]:
    try:
        client = _get_bq_client()
        table = client.get_table(f"{CONFIG.project_id}.{dataset_id}.{table_name}")
        fields: list[dict[str, Any]] = []
        for field in list(table.schema)[:limit]:
            fields.append(
                {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description,
                }
            )
        return fields
    except Exception:
        return []


def _extract_response_text(response: object) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        if parts:
            return "".join(part.text for part in parts if hasattr(part, "text"))
    return ""


@lru_cache
def _load_data_valid_ai() -> None:
    try:
        import data_valid_ai  # noqa: F401
        return None
    except ModuleNotFoundError:
        pass
    extra_path = os.getenv("DATA_VALID_AI_PATH", "/Users/gugha/data_validation_ai/data-valid-ai/src")
    if extra_path and extra_path not in sys.path:
        sys.path.append(extra_path)
    try:
        import data_valid_ai  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "data-valid-ai package not found. "
            "Set DATA_VALID_AI_PATH or install the package."
        ) from exc


def _get_data_valid_tools() -> tuple[Any, Any, Any]:
    _load_data_valid_ai()
    from data_valid_ai.tools.cost_estimation import estimate_cost, check_query_efficiency
    from data_valid_ai.agents.orchestrator import OrchestratorAgent

    return estimate_cost, check_query_efficiency, OrchestratorAgent


def _render_data_valid_cli_output(result: Any) -> str | None:
    """Render data_valid_ai CLI report text for API responses."""
    try:
        _load_data_valid_ai()
        import data_valid_ai.cli as data_valid_cli
        from rich.console import Console
    except Exception:
        return None

    display_fn = getattr(data_valid_cli, "_display_validation_result", None)
    if not callable(display_fn):
        return None

    original_console = getattr(data_valid_cli, "console", None)
    capture_console = Console(record=True, force_terminal=False, width=160)
    try:
        data_valid_cli.console = capture_console
        display_fn(result)
        text = capture_console.export_text(clear=False).strip()
        return text or None
    except Exception:
        return None
    finally:
        if original_console is not None:
            data_valid_cli.console = original_console


def _resolve_validation_project(project_id: str | None) -> str | None:
    if project_id and project_id.strip():
        return project_id.strip()
    env_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
    if env_project:
        return env_project
    return CONFIG.project_id or None


def _parse_validation_table(table: str, project_id: str | None) -> tuple[str, str, str]:
    cleaned = table.strip().strip("`")
    parts = [part.strip() for part in cleaned.split(".")]
    if len(parts) == 3:
        proj, dataset, tbl = parts
    elif len(parts) == 2:
        proj = _resolve_validation_project(project_id)
        dataset, tbl = parts
    else:
        raise ValueError("Table must be in format project.dataset.table or dataset.table")
    if not proj:
        raise ValueError("No project specified. Use project_id or set GOOGLE_CLOUD_PROJECT")
    if not dataset or not tbl:
        raise ValueError("Table must be in format project.dataset.table or dataset.table")
    return proj, dataset, tbl


def _run_table_validation(
    dataset_id: str,
    table_name: str,
    layer: str,
    budget_usd: float,
    auto_detect_keys: bool,
    auto_detect_grain: bool,
    include_llm_summary: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    # Preserved for API compatibility; data_valid_ai.cli.validate flow does not consume these options.
    _ = (budget_usd, auto_detect_keys, auto_detect_grain)
    _, _, OrchestratorAgent = _get_data_valid_tools()
    agent = OrchestratorAgent(
        project=CONFIG.project_id,
        location=CONFIG.vertex_location,
        model_name=CONFIG.gemini_model,
        use_llm=include_llm_summary,
    )
    result = agent.validate_table(
        dataset=dataset_id,
        table=table_name,
        layer=layer,
    )
    cli_output = _render_data_valid_cli_output(result)
    payload = (
        result.model_dump()
        if hasattr(result, "model_dump")
        else (result if isinstance(result, dict) else {})
    )
    if not payload or not payload.get("table"):
        payload = {
            "table": f"{CONFIG.project_id}.{dataset_id}.{table_name}",
            "layer": layer,
            "status": "error",
            "rules_executed": 0,
            "rules_passed": 0,
            "rules_failed": 0,
            "rules_skipped": 0,
            "violations": [],
            "metadata": {"error": "validation_result_empty"},
        }
        _attach_partitioning_metadata(payload)
        return payload, {
            "summary": "Validation returned no results. Check data_valid_ai rules and permissions.",
            "risks": ["No validation rules executed."],
            "recommendations": ["Verify rule configuration for the target layer.", "Check BigQuery permissions."],
            "next_steps": ["Re-run validation after confirming rules load successfully."],
        }, cli_output

    _attach_partitioning_metadata(payload)

    llm_summary = None
    if include_llm_summary:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            raw_summary = metadata.get("llm_summary")
            if isinstance(raw_summary, dict):
                llm_summary = raw_summary
            elif isinstance(raw_summary, str) and raw_summary.strip():
                llm_summary = {
                    "summary": raw_summary.strip(),
                    "risks": [],
                    "recommendations": [],
                    "next_steps": [],
                }

    return payload, llm_summary, cli_output


def _run_table_validation_v2(
    table: str,
    layer: str,
    project_id: str | None,
    budget_usd: float,
    auto_detect_keys: bool,
    auto_detect_grain: bool,
    include_llm_summary: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    resolved_project, dataset_id, table_name = _parse_validation_table(table, project_id)
    _, _, OrchestratorAgent = _get_data_valid_tools()
    agent = OrchestratorAgent(
        project=resolved_project,
        location=CONFIG.vertex_location,
        model_name=CONFIG.gemini_model,
        use_llm=include_llm_summary,
    )
    result = agent.validate_table(
        dataset=dataset_id,
        table=table_name,
        layer=layer,
        budget_usd=budget_usd,
        auto_detect_keys=auto_detect_keys,
        auto_detect_grain=auto_detect_grain,
    )
    cli_output = _render_data_valid_cli_output(result)
    payload = (
        result.model_dump()
        if hasattr(result, "model_dump")
        else (result if isinstance(result, dict) else {})
    )
    if not payload or not payload.get("table"):
        payload = {
            "table": f"{resolved_project}.{dataset_id}.{table_name}",
            "layer": layer,
            "status": "error",
            "rules_executed": 0,
            "rules_passed": 0,
            "rules_failed": 0,
            "rules_skipped": 0,
            "violations": [],
            "metadata": {"error": "validation_result_empty"},
        }
        _attach_partitioning_metadata(payload)
        return payload, {
            "summary": "Validation returned no results. Check data_valid_ai rules and permissions.",
            "risks": ["No validation rules executed."],
            "recommendations": ["Verify rule configuration for the target layer.", "Check BigQuery permissions."],
            "next_steps": ["Re-run validation after confirming rules load successfully."],
        }, cli_output

    _attach_partitioning_metadata(payload)

    llm_summary = None
    if include_llm_summary:
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            raw_summary = metadata.get("llm_summary")
            if isinstance(raw_summary, dict):
                llm_summary = raw_summary
            elif isinstance(raw_summary, str) and raw_summary.strip():
                llm_summary = {
                    "summary": raw_summary.strip(),
                    "risks": [],
                    "recommendations": [],
                    "next_steps": [],
                }

    return payload, llm_summary, cli_output


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "can",
    "for",
    "from",
    "get",
    "give",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "list",
    "me",
    "of",
    "on",
    "or",
    "please",
    "show",
    "table",
    "the",
    "to",
    "which",
    "with",
    "find",
    "generate",
    "query",
    "sql",
    "need",
    "want",
    "where",
}

_SYNONYMS = {
    "item": ["line", "lineitem", "line_item", "detail"],
    "items": ["line", "lineitem", "line_item", "detail", "item"],
    "detail": ["line", "lineitem", "line_item"],
    "details": ["detail", "line", "lineitem", "line_item"],
    "order": ["salesorder", "sales_order"],
    "orders": ["order", "salesorders", "sales_orders"],
}


def _tokenize_base(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens = [tok for tok in tokens if len(tok) >= 2 and tok not in _STOPWORDS]
    return tokens[:12] if tokens else []


def _tokenize(text: str) -> list[str]:
    base = _tokenize_base(text)
    if not base:
        return []
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        if tok and tok not in seen:
            seen.add(tok)
            ordered.append(tok)

    for tok in base:
        _add(tok)
        if tok.endswith("ies") and len(tok) > 4:
            _add(tok[:-3] + "y")
        elif tok.endswith("s") and len(tok) > 3:
            _add(tok[:-1])
        for syn in _SYNONYMS.get(tok, []):
            _add(syn)

    return ordered[:20]


def _is_greeting(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return True
    greetings = {
        "hi",
        "hello",
        "hey",
        "yo",
        "hola",
        "sup",
        "help",
        "capabilities",
        "what can you do",
        "what can you do?",
        "how can you help",
        "how can you help?",
    }
    if normalized in greetings:
        return True
    if len(normalized.split()) <= 2 and any(word in greetings for word in normalized.split()):
        return True
    return False


def _is_likely_sql(text: str) -> bool:
    return bool(re.search(r"\bselect\b[\s\S]+\bfrom\b", text, re.IGNORECASE))


def _extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    quoted = re.search(r"([\"'])(.*?(?:select|with).*?)\1", text, re.IGNORECASE | re.DOTALL)
    if quoted:
        return quoted.group(2).strip()
    match = re.search(r"(?is)(?:^|\\n)\\s*(with\\b.*|select\\b.*)", text)
    if match:
        sql = match.group(1).strip()
        sql = sql.rstrip("`\"'")
        return sql
    return text.strip() if _is_likely_sql(text) else ""


def _extract_impact_target(text: str, entities: dict[str, Any] | None) -> dict[str, str] | None:
    if not text:
        return None

    cleaned = re.sub(r"[`'\"]", "", text)
    dataset_id = None
    table_name = None
    column_name = None

    four_part = re.search(
        r"\b([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b",
        cleaned,
    )
    if four_part:
        dataset_id = four_part.group(2)
        table_name = four_part.group(3)
        column_name = four_part.group(4)
    else:
        triple = re.search(
            r"\b([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b",
            cleaned,
        )
        if triple:
            if "-" in triple.group(1):
                dataset_id = triple.group(2)
                table_name = triple.group(3)
            else:
                dataset_id = triple.group(1)
                table_name = triple.group(2)
                column_name = triple.group(3)

    if not dataset_id or not table_name:
        table_match = re.search(
            r"\btable\b\s*(?:name\s*)?(?:is\s*)?(?::|=)?\s*"
            r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b",
            cleaned,
            re.IGNORECASE,
        )
        if table_match:
            dataset_id = table_match.group(1)
            table_name = table_match.group(2)

    if not dataset_id or not table_name:
        double = re.search(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b(?!\.)", cleaned)
        if double:
            dataset_id = double.group(1)
            table_name = double.group(2)

    if not column_name:
        stopwords = {
            "change",
            "changing",
            "column",
            "columns",
            "field",
            "fields",
            "attribute",
            "attributes",
            "data",
            "type",
            "datatype",
            "from",
            "to",
            "string",
            "integer",
            "int",
            "bigint",
            "smallint",
            "float",
            "double",
            "decimal",
            "numeric",
            "boolean",
            "bool",
        }
        patterns = [
            r"\bcolumn\s+([a-zA-Z0-9_]+)\b",
            r"\bfield\s+([a-zA-Z0-9_]+)\b",
            r"\battribute\s+([a-zA-Z0-9_]+)\b",
            r"\b([a-zA-Z0-9_]+)\s+column\b",
            r"\b([a-zA-Z0-9_]+)\s+field\b",
            r"\b([a-zA-Z0-9_]+)\s+attribute\b",
            r"\b(?:data\s*type|datatype|type)\s+of\s+([a-zA-Z0-9_]+)\b",
            r"\b([a-zA-Z0-9_]+)\s+(?:data\s*type|datatype|type)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1)
            if candidate and candidate.lower() not in stopwords:
                column_name = candidate
                break

    entities = entities or {}
    columns = entities.get("columns") if isinstance(entities.get("columns"), list) else []
    if not dataset_id and entities.get("dataset_id"):
        dataset_id = entities["dataset_id"]
    if not table_name and entities.get("table_name"):
        table_name = entities["table_name"]
    if not column_name and columns:
        column_name = columns[0]

    if dataset_id and table_name and column_name:
        return {
            "dataset_id": dataset_id,
            "table_name": table_name,
            "column_name": column_name,
        }
    return None


def _map_ripple_to_assessment(report: dict[str, Any]) -> dict[str, Any]:
    severity = report.get("executive_summary", {}).get("severity_distribution", {})
    tsunami = severity.get("Tsunami", 0)
    wave = severity.get("Wave", 0)
    impact_level = "HIGH" if tsunami > 0 else "MEDIUM" if wave > 0 else "LOW"
    upstream_count = len(report.get("upstream_analysis", {}).get("sources", []) or [])
    downstream_count = len(report.get("downstream_analysis", {}).get("impacted_tables", []) or [])
    recs = report.get("recommendations", {}) or {}

    impact_reasoning = [
        report.get("executive_summary", {}).get("summary"),
        f"Risk score: {report.get('executive_summary', {}).get('risk_score', 'unknown')}",
        f"Upstream sources: {upstream_count}",
        f"Downstream tables: {downstream_count}",
    ]

    return {
        "impact_level": impact_level,
        "impact_reasoning": [item for item in impact_reasoning if item],
        "downstream_tables": report.get("downstream_analysis", {}).get("impacted_tables", []),
        "has_upstream": upstream_count > 0,
        "column": (
            f"{report.get('target', {}).get('dataset_id')}."
            f"{report.get('target', {}).get('table_name')}."
            f"{report.get('target', {}).get('column_name')}"
        ),
        "table": (
            f"{report.get('target', {}).get('dataset_id')}."
            f"{report.get('target', {}).get('table_name')}"
        ),
        "recommendation": report.get("executive_summary", {}).get("recommended_action", "review"),
        "recommendation_reasoning": [
            *list(recs.get("breaking_change_warnings") or []),
            *list(recs.get("migration_steps") or []),
        ],
        "actions": [
            *list(recs.get("testing_checklist") or []),
            *list(recs.get("rollback_procedures") or []),
        ],
        "ripple_summary": report.get("executive_summary", {}),
    }


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(val * val for val in vec))


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _embed_query_uncached(text: str) -> list[float]:
    client = _get_genai_client()
    response = client.models.embed_content(
        model=CONFIG.embedding_model,
        contents=[text],
        config=types.EmbedContentConfig(task_type=CONFIG.embedding_task_type),
    )
    return list(response.embeddings[0].values)


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    return tuple(_embed_query_uncached(text))


def _cache_bucket(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        return 0
    return int(time.time() // ttl_seconds)


def _dataset_priority_sql() -> str:
    return (
        "CASE dataset_id WHEN 'gold_layer' THEN 3 "
        "WHEN 'silver_layer' THEN 2 "
        "WHEN 'bronze_layer_mssql' THEN 1 ELSE 0 END"
    )


def _query_matching_columns_uncached(query: str, top_k: int) -> list[MatchColumn]:
    embedding = list(_embed_query_cached(query))
    norm_q = _norm(embedding)
    if norm_q == 0:
        return []
    base_tokens = _tokenize_base(query)
    tokens = _tokenize(query)
    token_count = len(base_tokens) if base_tokens else len(tokens)
    candidate_limit = CONFIG.search_candidate_limit if tokens else 0
    client = _get_bq_client()
    dataset_priority = _dataset_priority_sql()

    base_sql = f"""
    WITH base AS (
      SELECT
        e.column_doc_id,
        e.asset_id,
        e.column_name,
        e.embedding,
        d.doc_text,
        a.dataset_id,
        a.asset_name
      FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.column_embeddings` e
      JOIN `{CONFIG.project_id}.{CONFIG.metadata_dataset}.column_documents` d
        ON e.column_doc_id = d.column_doc_id
      JOIN `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets` a
        ON e.asset_id = a.asset_id
    )
    """

    filtered_sql = """
    , prefilter AS (
      SELECT
        asset_id,
        column_name,
        dataset_id,
        asset_name,
        doc_text,
        embedding,
        (
          SELECT SUM(
            CASE
              WHEN STRPOS(LOWER(column_name), token) > 0 THEN 1 ELSE 0 END
          )
          FROM UNNEST(@tokens) token
        ) AS column_hits,
        (
          SELECT SUM(
            CASE
              WHEN STRPOS(LOWER(asset_name), token) > 0 THEN 1 ELSE 0 END
          )
          FROM UNNEST(@tokens) token
        ) AS asset_hits,
        (
          SELECT SUM(
            CASE
              WHEN STRPOS(LOWER(doc_text), token) > 0 THEN 1 ELSE 0 END
          )
          FROM UNNEST(@tokens) token
        ) AS doc_hits
      FROM base
    ),
    filtered AS (
      SELECT *
      FROM prefilter
      ORDER BY (column_hits * 2 + asset_hits + doc_hits) DESC
      LIMIT @candidate_limit
    )
    """

    no_token_sql = """
    , prefilter AS (
      SELECT
        *,
        0 AS column_hits,
        0 AS asset_hits,
        0 AS doc_hits
      FROM base
    ),
    filtered AS (SELECT * FROM prefilter)
    """
    sql = base_sql + (filtered_sql if candidate_limit > 0 else no_token_sql) + f"""
    , scored AS (
      SELECT
        asset_id,
        column_name,
        dataset_id,
        asset_name,
        (
          SELECT SUM(v * q)
          FROM UNNEST(embedding) v WITH OFFSET idx
          JOIN UNNEST(@query_embedding) q WITH OFFSET qidx
            ON idx = qidx
        ) AS dot,
        SQRT((SELECT SUM(v * v) FROM UNNEST(embedding) v)) AS norm_e,
        (column_hits * 2 + asset_hits + doc_hits) AS keyword_hits
      FROM filtered
    )
    SELECT
      asset_id,
      dataset_id,
      asset_name AS table_name,
      column_name,
      SAFE_DIVIDE(dot, norm_e * @norm_q) AS semantic_score,
      SAFE_DIVIDE(keyword_hits, GREATEST(1, @token_count * 4)) AS keyword_score,
      (0.7 * SAFE_DIVIDE(dot, norm_e * @norm_q)
       + 0.3 * SAFE_DIVIDE(keyword_hits, GREATEST(1, @token_count * 4))) AS hybrid_score
    FROM scored
    ORDER BY hybrid_score DESC, {dataset_priority} DESC
    LIMIT @limit
    """

    params = [
        bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embedding),
        bigquery.ScalarQueryParameter("norm_q", "FLOAT64", norm_q),
        bigquery.ArrayQueryParameter("tokens", "STRING", tokens),
        bigquery.ScalarQueryParameter("limit", "INT64", top_k),
        bigquery.ScalarQueryParameter("token_count", "INT64", max(token_count, 1)),
    ]
    if candidate_limit > 0:
        params.append(bigquery.ScalarQueryParameter("candidate_limit", "INT64", candidate_limit))

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = client.query(sql, job_config=job_config).result()
    results = [
        MatchColumn(
            asset_id=row["asset_id"],
            dataset_id=row["dataset_id"],
            table_name=row["table_name"],
            column_name=row["column_name"],
            semantic_score=row["semantic_score"] or 0.0,
            keyword_score=row["keyword_score"] or 0.0,
            hybrid_score=row["hybrid_score"] or 0.0,
        )
        for row in rows
    ]
    return results


def _query_matching_tables_uncached(query: str, top_k: int) -> list[MatchTable]:
    embedding = list(_embed_query_cached(query))
    norm_q = _norm(embedding)
    if norm_q == 0:
        return []
    base_tokens = _tokenize_base(query)
    tokens = _tokenize(query)
    token_count = len(base_tokens) if base_tokens else len(tokens)
    candidate_limit = CONFIG.search_candidate_limit if tokens else 0
    client = _get_bq_client()
    dataset_priority = _dataset_priority_sql()

    base_sql = f"""
    WITH base AS (
      SELECT
        e.doc_id,
        e.asset_id,
        e.embedding,
        d.doc_text,
        a.dataset_id,
        a.asset_name
      FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.embeddings` e
      JOIN `{CONFIG.project_id}.{CONFIG.metadata_dataset}.documents` d
        ON e.doc_id = d.doc_id
      JOIN `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets` a
        ON e.asset_id = a.asset_id
    )
    """

    filtered_sql = """
    , prefilter AS (
      SELECT
        asset_id,
        dataset_id,
        asset_name,
        doc_text,
        embedding,
        (
          SELECT SUM(
            CASE
              WHEN STRPOS(LOWER(asset_name), token) > 0 THEN 1 ELSE 0 END
          )
          FROM UNNEST(@tokens) token
        ) AS asset_hits,
        (
          SELECT SUM(
            CASE
              WHEN STRPOS(LOWER(doc_text), token) > 0 THEN 1 ELSE 0 END
          )
          FROM UNNEST(@tokens) token
        ) AS doc_hits
      FROM base
    ),
    filtered AS (
      SELECT *
      FROM prefilter
      ORDER BY (asset_hits * 2 + doc_hits) DESC
      LIMIT @candidate_limit
    )
    """

    no_token_sql = """
    , prefilter AS (
      SELECT
        *,
        0 AS asset_hits,
        0 AS doc_hits
      FROM base
    ),
    filtered AS (SELECT * FROM prefilter)
    """

    sql = base_sql + (filtered_sql if candidate_limit > 0 else no_token_sql) + f"""
    , scored AS (
      SELECT
        asset_id,
        dataset_id,
        asset_name,
        (
          SELECT SUM(v * q)
          FROM UNNEST(embedding) v WITH OFFSET idx
          JOIN UNNEST(@query_embedding) q WITH OFFSET qidx
            ON idx = qidx
        ) AS dot,
        SQRT((SELECT SUM(v * v) FROM UNNEST(embedding) v)) AS norm_e,
        (asset_hits * 2 + doc_hits) AS keyword_hits
      FROM filtered
    )
    SELECT
      asset_id,
      dataset_id,
      asset_name AS table_name,
      SAFE_DIVIDE(dot, norm_e * @norm_q) AS semantic_score,
      SAFE_DIVIDE(keyword_hits, GREATEST(1, @token_count * 3)) AS keyword_score,
      (0.7 * SAFE_DIVIDE(dot, norm_e * @norm_q)
       + 0.3 * SAFE_DIVIDE(keyword_hits, GREATEST(1, @token_count * 3))) AS hybrid_score
    FROM scored
    ORDER BY hybrid_score DESC, {dataset_priority} DESC
    LIMIT @limit
    """

    params = [
        bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", embedding),
        bigquery.ScalarQueryParameter("norm_q", "FLOAT64", norm_q),
        bigquery.ArrayQueryParameter("tokens", "STRING", tokens),
        bigquery.ScalarQueryParameter("limit", "INT64", top_k),
        bigquery.ScalarQueryParameter("token_count", "INT64", max(token_count, 1)),
    ]
    if candidate_limit > 0:
        params.append(bigquery.ScalarQueryParameter("candidate_limit", "INT64", candidate_limit))

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = client.query(sql, job_config=job_config).result()
    results = [
        MatchTable(
            asset_id=row["asset_id"],
            dataset_id=row["dataset_id"],
            table_name=row["table_name"],
            semantic_score=row["semantic_score"] or 0.0,
            keyword_score=row["keyword_score"] or 0.0,
            hybrid_score=row["hybrid_score"] or 0.0,
        )
        for row in rows
    ]
    return results


@lru_cache(maxsize=256)
def _query_matching_tables_cached(query: str, top_k: int, bucket: int) -> tuple[MatchTable, ...]:
    return tuple(_query_matching_tables_uncached(query, top_k))


@lru_cache(maxsize=256)
def _query_matching_columns_cached(query: str, top_k: int, bucket: int) -> tuple[MatchColumn, ...]:
    return tuple(_query_matching_columns_uncached(query, top_k))


def _extract_table_ref(text: str | None) -> tuple[str | None, str | None, str | None]:
    if not text:
        return None, None, None
    cleaned = text.replace("`", " ")
    triple = re.search(r"([a-zA-Z0-9_-]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", cleaned)
    if triple:
        return triple.group(1), triple.group(2), triple.group(3)
    double = re.search(r"([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)", cleaned)
    if double:
        return None, double.group(1), double.group(2)
    return None, None, None


def _resolve_explicit_table(text: str) -> TableRef | None:
    project_id, dataset_id, table_name = _extract_table_ref(text)
    if not dataset_id or not table_name:
        return None
    if project_id and project_id != CONFIG.project_id:
        return None
    client = _get_bq_client()
    sql = f"""
    SELECT dataset_id, asset_name
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets`
    WHERE dataset_id = @dataset_id
      AND asset_name = @table_name
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", dataset_id),
            bigquery.ScalarQueryParameter("table_name", "STRING", table_name),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    if not rows:
        return None
    row = rows[0]
    return TableRef(dataset_id=row["dataset_id"], table_name=row["asset_name"])


def _hint_tables_from_query(
    query: str,
    limit_per: int = 2,
    allowed_datasets: set[str] | None = None,
) -> list[TableRef]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    priority_terms: list[str] = []
    if "category" in tokens:
        priority_terms.extend(["category", "subcategory"])
    if "customer" in tokens:
        priority_terms.append("customer")
    if "product" in tokens:
        priority_terms.append("product")
    if "sales" in tokens:
        priority_terms.append("sales")
    if "order" in tokens:
        priority_terms.append("order")

    hint_terms = priority_terms + [token for token in tokens if token not in priority_terms]
    seen: set[str] = set()
    candidates: list[TableRef] = []
    client = _get_bq_client()
    dataset_priority = _dataset_priority_sql()
    for term in hint_terms:
        if term in seen or len(term) < 3:
            continue
        seen.add(term)
        dataset_filter = "AND dataset_id IN UNNEST(@dataset_ids)" if allowed_datasets else ""
        sql = f"""
        SELECT dataset_id, asset_name
        FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets`
        WHERE LOWER(asset_name) LIKE CONCAT('%', @search, '%')
        {dataset_filter}
        ORDER BY {dataset_priority} DESC
        LIMIT @limit
        """
        params = [
            bigquery.ScalarQueryParameter("search", "STRING", term),
            bigquery.ScalarQueryParameter("limit", "INT64", limit_per),
        ]
        if allowed_datasets:
            params.append(bigquery.ArrayQueryParameter("dataset_ids", "STRING", list(allowed_datasets)))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = client.query(sql, job_config=job_config).result()
        for row in rows:
            candidates.append(TableRef(dataset_id=row["dataset_id"], table_name=row["asset_name"]))
    return candidates


def _resolve_table_candidates(table_name: str | None, query: str | None) -> list[dict[str, Any]]:
    if not table_name and not query:
        return []
    client = _get_bq_client()
    raw_search = table_name or query or ""
    _, dataset_hint, table_hint = _extract_table_ref(raw_search)
    search = (table_hint or raw_search).lower()
    dataset_filter = "AND dataset_id = @dataset_id" if dataset_hint else ""
    sql = f"""
    SELECT asset_id, dataset_id, asset_name
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets`
    WHERE LOWER(asset_name) LIKE CONCAT('%', @search, '%')
    {dataset_filter}
    ORDER BY CASE dataset_id WHEN 'gold_layer' THEN 3 WHEN 'silver_layer' THEN 2 WHEN 'bronze_layer_mssql' THEN 1 ELSE 0 END DESC
    LIMIT 20
    """
    params = [bigquery.ScalarQueryParameter("search", "STRING", search)]
    if dataset_hint:
        params.append(bigquery.ScalarQueryParameter("dataset_id", "STRING", dataset_hint))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = client.query(sql, job_config=job_config).result()
    return [
        {"asset_id": row["asset_id"], "dataset_id": row["dataset_id"], "table_name": row["asset_name"]}
        for row in rows
    ]


def _fetch_lineage_edges(table_name: str, dataset_id: str | None) -> list[LineageEdgeResponse]:
    client = _get_bq_client()
    sql = f"""
    SELECT *
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.lineage_edges`
    WHERE (source_table = @table_name OR target_table = @table_name)
    {"AND (source_dataset = @dataset_id OR target_dataset = @dataset_id)" if dataset_id else ""}
    """
    params = [bigquery.ScalarQueryParameter("table_name", "STRING", table_name)]
    if dataset_id:
        params.append(bigquery.ScalarQueryParameter("dataset_id", "STRING", dataset_id))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = client.query(sql, job_config=job_config).result()
    edges: list[LineageEdgeResponse] = []
    for row in rows:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}
        edges.append(
            LineageEdgeResponse(
                source_dataset=row.get("source_dataset"),
                source_table=row.get("source_table"),
                source_column=row.get("source_column"),
                target_dataset=row.get("target_dataset"),
                target_table=row.get("target_table"),
                target_column=row.get("target_column"),
                relationship_type=row.get("relationship_type"),
                transformation_logic=row.get("transformation_logic"),
                confidence_score=row.get("confidence_score"),
                discovery_method=row.get("discovery_method"),
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return edges


def _fetch_table_schema(asset_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not asset_ids:
        return {}
    client = _get_bq_client()
    sql = f"""
    SELECT asset_id, column_name, data_type, is_nullable, column_description
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.columns`
    WHERE asset_id IN UNNEST(@asset_ids)
    ORDER BY asset_id, ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)]
    )
    rows = client.query(sql, job_config=job_config).result()
    schema: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        schema.setdefault(row["asset_id"], []).append(
            {
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "is_nullable": row["is_nullable"],
                "description": row["column_description"] or "",
            }
        )
    missing_assets = [asset_id for asset_id in asset_ids if asset_id not in schema]
    if missing_assets:
        datasets: dict[str, list[str]] = {}
        for asset_id in missing_assets:
            parts = asset_id.split(".")
            if len(parts) != 3 or parts[0] != CONFIG.project_id:
                continue
            datasets.setdefault(parts[1], []).append(parts[2])
        for dataset_id, table_names in datasets.items():
            sql = f"""
            SELECT table_name, column_name, data_type, is_nullable, ordinal_position
            FROM `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name IN UNNEST(@table_names)
            ORDER BY table_name, ordinal_position
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("table_names", "STRING", table_names)]
            )
            rows = client.query(sql, job_config=job_config).result()
            for row in rows:
                asset_id = f"{CONFIG.project_id}.{dataset_id}.{row['table_name']}"
                schema.setdefault(asset_id, []).append(
                    {
                        "column_name": row["column_name"],
                        "data_type": row["data_type"],
                        "is_nullable": row["is_nullable"],
                        "description": "",
                    }
                )
    return schema


def _fetch_table_schema_single(dataset_id: str, table_name: str) -> list[dict[str, Any]]:
    client = _get_bq_client()
    asset_id = f"{CONFIG.project_id}.{dataset_id}.{table_name}"
    sql = f"""
    SELECT column_name, data_type, is_nullable, column_description
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.columns`
    WHERE asset_id = @asset_id
    ORDER BY ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("asset_id", "STRING", asset_id)]
    )
    rows = client.query(sql, job_config=job_config).result()
    results = [
        {
            "name": row["column_name"],
            "data_type": row["data_type"],
            "is_nullable": row["is_nullable"],
            "description": row["column_description"] or "",
        }
        for row in rows
    ]
    if results:
        return results
    sql = f"""
    SELECT column_name, data_type, is_nullable, ordinal_position
    FROM `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @table_name
    ORDER BY ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("table_name", "STRING", table_name)]
    )
    rows = client.query(sql, job_config=job_config).result()
    return [
        {
            "name": row["column_name"],
            "data_type": row["data_type"],
            "is_nullable": row["is_nullable"],
            "description": "",
        }
        for row in rows
    ]


def _normalize_asset_id(asset_id: str | None) -> str | None:
    if not asset_id:
        return None
    parts = asset_id.split(".")
    if len(parts) == 3:
        return asset_id
    if len(parts) == 2:
        return f"{CONFIG.project_id}.{parts[0]}.{parts[1]}"
    return None


def _normalize_clustering_fields(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.replace("\"", "").replace("'", "")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return ", ".join(parts) if parts else text


def _format_partitioning_from_options(
    partitioning_type: Any,
    partitioning_field: Any,
) -> str:
    part_type = str(partitioning_type).strip() if partitioning_type is not None else ""
    part_field = str(partitioning_field).strip() if partitioning_field is not None else ""
    parts: list[str] = []
    if part_type:
        parts.append(f"type={part_type}")
    if part_field:
        parts.append(f"field={part_field}")
    return ",".join(parts)


def _extract_partitioning_fields(partitioning: str) -> dict[str, Any]:
    if not partitioning:
        return {}
    raw = partitioning.strip()
    if not raw:
        return {}
    if ":" in raw:
        _, raw = raw.split(":", 1)
    info: dict[str, Any] = {}
    for part in raw.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if key == "type":
            info["partitioning_type"] = value
        elif key == "field":
            info["partitioning_field"] = value
        elif key == "require_partition_filter":
            info["require_partition_filter"] = value
        elif key == "expiration_ms":
            info["partition_expiration_ms"] = value
    return info


def _fetch_table_partitioning(asset_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not asset_ids:
        return {}
    try:
        client = _get_bq_client()
    except Exception:
        return {}
    info: dict[str, dict[str, Any]] = {}
    try:
        sql = f"""
        SELECT asset_id, partitioning, clustering
        FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets`
        WHERE asset_id IN UNNEST(@asset_ids)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)]
        )
        rows = client.query(sql, job_config=job_config).result()
        for row in rows:
            partitioning = str(row.get("partitioning") or "").strip()
            clustering = _normalize_clustering_fields(row.get("clustering"))
            entry: dict[str, Any] = {
                "partitioning": partitioning,
                "clustering": clustering,
            }
            entry.update(_extract_partitioning_fields(partitioning))
            info[row["asset_id"]] = entry
    except Exception:
        info = {}

    missing_assets = [asset_id for asset_id in asset_ids if asset_id not in info]
    if not missing_assets:
        return info

    datasets: dict[str, list[str]] = {}
    for asset_id in missing_assets:
        parts = asset_id.split(".")
        if len(parts) != 3 or parts[0] != CONFIG.project_id:
            continue
        datasets.setdefault(parts[1], []).append(parts[2])

    for dataset_id, table_names in datasets.items():
        try:
            sql = f"""
            SELECT
              table_name,
              MAX(IF(option_name = 'partitioning_type', option_value, NULL)) AS partitioning_type,
              MAX(IF(option_name = 'partitioning_field', option_value, NULL)) AS partitioning_field,
              MAX(IF(option_name = 'clustering_fields', option_value, NULL)) AS clustering_fields
            FROM `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLE_OPTIONS`
            WHERE table_name IN UNNEST(@table_names)
            GROUP BY table_name
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ArrayQueryParameter("table_names", "STRING", table_names)]
            )
            rows = client.query(sql, job_config=job_config).result()
            for row in rows:
                partitioning_type = row.get("partitioning_type")
                partitioning_field = row.get("partitioning_field")
                partitioning = _format_partitioning_from_options(partitioning_type, partitioning_field)
                clustering = _normalize_clustering_fields(row.get("clustering_fields"))
                info[f"{CONFIG.project_id}.{dataset_id}.{row['table_name']}"] = {
                    "partitioning": partitioning,
                    "partitioning_type": partitioning_type,
                    "partitioning_field": partitioning_field,
                    "clustering": clustering,
                }
        except Exception:
            continue

    return info


def _partitioning_summary(info: dict[str, Any] | None) -> str:
    if not info:
        return "none"
    raw = info.get("partitioning")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    parts: list[str] = []
    part_type = info.get("partitioning_type")
    part_field = info.get("partitioning_field")
    if part_type:
        parts.append(f"type={part_type}")
    if part_field:
        parts.append(f"field={part_field}")
    return ", ".join(parts) if parts else "none"


def _clustering_summary(info: dict[str, Any] | None) -> str:
    if not info:
        return "none"
    raw = info.get("clustering")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "none"


def _attach_partitioning_metadata(payload: dict[str, Any]) -> None:
    if not payload:
        return
    asset_id = _normalize_asset_id(payload.get("table"))
    if not asset_id:
        return
    info = _fetch_table_partitioning([asset_id]).get(asset_id, {})
    if not info:
        return
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if info.get("partitioning"):
        metadata["partitioning"] = info.get("partitioning")
    if info.get("partitioning_type"):
        metadata.setdefault("partitioning_type", info.get("partitioning_type"))
    if info.get("partitioning_field"):
        metadata.setdefault("partitioning_field", info.get("partitioning_field"))
    if info.get("require_partition_filter"):
        metadata.setdefault("require_partition_filter", info.get("require_partition_filter"))
    if info.get("partition_expiration_ms"):
        metadata.setdefault("partition_expiration_ms", info.get("partition_expiration_ms"))
    if info.get("clustering"):
        metadata["clustering"] = info.get("clustering")
    payload["metadata"] = metadata


@lru_cache(maxsize=1024)
def _table_exists_in_catalog(dataset_id: str, table_name: str) -> bool:
    client = _get_bq_client()
    sql = f"""
    SELECT 1
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.assets`
    WHERE dataset_id = @dataset_id
      AND asset_name = @table_name
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("dataset_id", "STRING", dataset_id),
            bigquery.ScalarQueryParameter("table_name", "STRING", table_name),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    if rows:
        return True
    try:
        client.get_table(f"{CONFIG.project_id}.{dataset_id}.{table_name}")
        return True
    except Exception:
        return False


def _is_conversion_request(text: str) -> bool:
    normalized = text.lower()
    keywords = ("convert", "conversion", "currency", "exchange rate", "fx", "usd", "cad", "eur", "gbp", "jpy")
    return any(k in normalized for k in keywords)


@lru_cache(maxsize=256)
def _fetch_fk_relationships_for_dataset(dataset_id: str, table_names_key: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    if not table_names_key:
        return tuple()
    client = _get_bq_client()
    sql = f"""
    SELECT
      fk.table_name AS fk_table,
      fk.column_name AS fk_column,
      fk.constraint_name AS constraint_name,
      ccu.table_name AS pk_table,
      ccu.column_name AS pk_column
    FROM `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` fk
    JOIN `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc
      ON tc.table_name = fk.table_name
     AND tc.constraint_name = fk.constraint_name
    JOIN `{CONFIG.project_id}.{dataset_id}.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE` ccu
      ON ccu.constraint_name = fk.constraint_name
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND (
        fk.table_name IN UNNEST(@table_names)
        OR ccu.table_name IN UNNEST(@table_names)
      )
    ORDER BY fk.table_name, fk.constraint_name, fk.ordinal_position
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("table_names", "STRING", list(table_names_key))]
    )
    rows = client.query(sql, job_config=job_config).result()
    rels: list[dict[str, str]] = []
    for row in rows:
        rels.append(
            {
                "dataset_id": dataset_id,
                "fk_table": row["fk_table"],
                "fk_column": row["fk_column"],
                "pk_table": row["pk_table"],
                "pk_column": row["pk_column"],
                "constraint_name": row["constraint_name"],
            }
        )
    return tuple(rels)


def _collect_fk_relationships(tables: list[TableRef]) -> list[dict[str, str]]:
    if not tables:
        return []
    by_dataset: dict[str, set[str]] = {}
    for t in tables:
        by_dataset.setdefault(t.dataset_id, set()).add(t.table_name)

    relationships: list[dict[str, str]] = []
    for dataset_id, names in by_dataset.items():
        try:
            rows = _fetch_fk_relationships_for_dataset(dataset_id, tuple(sorted(names)))
            relationships.extend(list(rows))
        except Exception:
            continue
    return relationships


def _augment_sql_tables(tables: list[TableRef], max_tables: int = 6) -> list[TableRef]:
    augmented: list[TableRef] = []
    seen: set[tuple[str, str]] = set()
    for t in tables:
        key = (t.dataset_id, t.table_name)
        if key in seen:
            continue
        augmented.append(t)
        seen.add(key)
    if len(augmented) >= max_tables:
        return augmented

    while len(augmented) < max_tables:
        try:
            relationships = _collect_fk_relationships(augmented)
        except Exception:
            return augmented
        added = False
        for rel in relationships:
            for candidate_table in (rel["fk_table"], rel["pk_table"]):
                key = (rel["dataset_id"], candidate_table)
                if key in seen:
                    continue
                if not _table_exists_in_catalog(rel["dataset_id"], candidate_table):
                    continue
                augmented.append(TableRef(dataset_id=rel["dataset_id"], table_name=candidate_table))
                seen.add(key)
                added = True
                if len(augmented) >= max_tables:
                    return augmented
        if not added:
            break
    return augmented


def _column_exists(dataset_id: str, table_name: str, column_name: str) -> bool:
    client = _get_bq_client()
    asset_id = f"{CONFIG.project_id}.{dataset_id}.{table_name}"
    sql = f"""
    SELECT 1
    FROM `{CONFIG.project_id}.{CONFIG.metadata_dataset}.columns`
    WHERE asset_id = @asset_id AND column_name = @column_name
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("asset_id", "STRING", asset_id),
            bigquery.ScalarQueryParameter("column_name", "STRING", column_name),
        ]
    )
    rows = list(client.query(sql, job_config=job_config).result())
    return bool(rows)


def _build_sql_context(
    user_query: str,
    tables: list[TableRef],
    schema: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    asset_ids = [f"{CONFIG.project_id}.{t.dataset_id}.{t.table_name}" for t in tables]
    if schema is None:
        schema = _fetch_table_schema(asset_ids)
    partitioning_info = _fetch_table_partitioning(asset_ids)
    table_blocks = []
    for asset_id in asset_ids:
        columns = schema.get(asset_id, [])
        info = partitioning_info.get(asset_id)
        partitioning = _partitioning_summary(info)
        clustering = _clustering_summary(info)
        col_lines = [
            f"- {col['column_name']} ({col['data_type']}) {('NULLABLE' if col['is_nullable'] else 'REQUIRED')} {col['description']}"
            for col in columns
        ]
        table_blocks.append(
            f"Table: {asset_id}\n"
            f"Partitioning: {partitioning}\n"
            f"Clustering: {clustering}\n"
            + "\n".join(col_lines)
        )
    schema_text = "\n\n".join(table_blocks) if table_blocks else "No schema provided."
    fk_relationships = _collect_fk_relationships(tables)
    fk_lines = []
    for rel in fk_relationships:
        fk_lines.append(
            f"- {CONFIG.project_id}.{rel['dataset_id']}.{rel['fk_table']}.{rel['fk_column']} "
            f"-> {CONFIG.project_id}.{rel['dataset_id']}.{rel['pk_table']}.{rel['pk_column']} "
            f"({rel['constraint_name']})"
        )
    fk_text = "\n".join(fk_lines) if fk_lines else "- none found for currently provided tables"

    conversion_guidance = ""
    if _is_conversion_request(user_query):
        conversion_guidance = (
            "For conversion requests, never hardcode multipliers. "
            "Use conversion/rate columns from joined tables when available; otherwise explain missing inputs in notes.\n"
        )

    context = (
        f"{conversion_guidance}"
        f"User request: {user_query}\n\n"
        f"Declared FK relationships among provided tables:\n{fk_text}\n\n"
        f"Available tables and columns:\n{schema_text}\n"
    )
    return context, schema


def _build_sql_prompt(user_query: str, tables: list[TableRef]) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    context, schema = _build_sql_context(user_query, tables)
    prompt = (
        "You are a BigQuery SQL assistant. Use ONLY the provided tables. "
        "Return STRICT JSON with schema: {\"sql\": string, \"notes\": string, \"tables_used\": [string]} . "
        "If unsure, produce best-effort SQL and describe assumptions in notes. "
        "Only return an empty sql if the request cannot be mapped to the provided tables.\n"
        "When joins are needed, prefer declared foreign-key relationships over guessed joins.\n"
        "Use partitioning/clustering info to improve efficiency (partition filters, clustering-friendly predicates) "
        "when it does not change the business intent.\n"
        "Do not fabricate dimension attributes as literals when those attributes exist in available tables. "
        "If the user requests fields like code/name/description, source them from columns via joins.\n"
        f"{context}"
    )
    return prompt, schema


def _build_sql_revision_prompt(
    user_query: str,
    tables: list[TableRef],
    sql: str,
    validation: dict[str, Any],
    schema: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    context, schema = _build_sql_context(user_query, tables, schema=schema)
    issues = validation.get("issues") if isinstance(validation, dict) else []
    suggestions = validation.get("suggestions") if isinstance(validation, dict) else []
    cost = validation.get("cost") if isinstance(validation, dict) else {}
    if not isinstance(issues, list):
        issues = []
    if not isinstance(suggestions, list):
        suggestions = []
    if not isinstance(cost, dict):
        cost = {}

    issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "- none"
    suggestions_text = "\n".join(f"- {item}" for item in suggestions) if suggestions else "- none"

    cost_lines: list[str] = []
    estimated_cost = cost.get("estimated_cost_usd")
    if isinstance(estimated_cost, (int, float)):
        cost_lines.append(f"Estimated cost USD: {estimated_cost:.4f}")
    budget_usd = cost.get("budget_usd")
    if isinstance(budget_usd, (int, float)):
        cost_lines.append(f"Budget USD: {budget_usd:.2f}")
    bytes_gb = cost.get("bytes_processed_gb")
    if isinstance(bytes_gb, (int, float)):
        cost_lines.append(f"Bytes processed GB: {bytes_gb:.2f}")
    recommendation = cost.get("recommendation")
    if isinstance(recommendation, str) and recommendation:
        cost_lines.append(f"Cost recommendation: {recommendation}")
    cost_text = "\n".join(f"- {line}" for line in cost_lines) if cost_lines else "- unknown"

    prompt = (
        "You are a BigQuery SQL assistant. Revise the SQL to address the validation issues. "
        "Use ONLY the provided tables and columns. "
        "Return STRICT JSON with schema: {\"sql\": string, \"notes\": string, \"tables_used\": [string]} . "
        "If you are unsure, output `unknown` rather than guessing. If unsure, return sql as an empty string and notes as \"unknown\". "
        "Keep the business intent the same and avoid introducing new tables or columns. "
        "Use partitioning/clustering info to improve efficiency (partition filters, clustering-friendly predicates) "
        "when it does not change the business intent.\n"
        "If no changes are needed, return the original SQL unchanged.\n"
        f"Original SQL:\n```sql\n{sql}\n```\n\n"
        f"Validation issues:\n{issues_text}\n\n"
        f"Suggestions:\n{suggestions_text}\n\n"
        f"Cost summary:\n{cost_text}\n\n"
        f"{context}"
    )
    return prompt, schema


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _generate_sql_raw(prompt: str, max_output_tokens: int = 8192) -> str:
    client = _get_genai_client()
    response = client.models.generate_content(
        model=CONFIG.gemini_pro_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=max_output_tokens,
        ),
    )
    return _extract_response_text(response)


def _parse_sql_payload(text: str) -> tuple[dict, bool]:
    def _extract_sql(raw: str) -> str:
        if not raw:
            return ""
        fence = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
        if fence:
            return fence.group(1).strip()
        match = re.search(r"(?is)(?:^|\\n)\\s*(with\\b.*|select\\b.*)", raw)
        if not match:
            return ""
        sql = match.group(1).strip()
        if "```" in sql:
            sql = sql.split("```", 1)[0].strip()
        if ";" in sql:
            sql = sql.split(";", 1)[0].strip() + ";"
        return sql

    def _unescape_sql(value: str) -> str:
        sql = value.replace("\\\\", "\\")
        sql = sql.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        sql = sql.replace('\\"', '"')
        return sql

    payload: dict[str, Any] = {}
    parse_error = False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parse_error = True
        else:
            parse_error = True

    if not payload or not str(payload.get("sql", "")).strip():
        sql_match = re.search(r'"sql"\s*:\s*"((?:\\\\.|[^"])*)"', text, re.DOTALL)
        if not sql_match:
            sql_match = re.search(r'"sql"\s*:\s*"(?P<sql>.*)$', text, re.DOTALL)
        if sql_match:
            raw_sql = sql_match.group(1)
            raw_sql = raw_sql.rstrip().rstrip("}").rstrip()
            raw_sql = raw_sql.rstrip('",')
            extracted = _unescape_sql(raw_sql)
            if extracted.strip():
                payload = payload or {"notes": "Recovered SQL from incomplete JSON", "tables_used": []}
                payload["sql"] = extracted.strip()
                parse_error = True

    extracted_sql = _extract_sql(text)
    if extracted_sql and (not payload or not str(payload.get("sql", "")).strip()):
        payload = payload or {"notes": "Parsed from non-JSON response", "tables_used": []}
        payload["sql"] = extracted_sql
        parse_error = True

    return payload, parse_error


def _generate_sql_payload(prompt: str) -> tuple[dict, bool, str, int]:
    text = _generate_sql_raw(prompt, max_output_tokens=8192)
    payload, parse_error = _parse_sql_payload(text)
    attempts = 1
    if parse_error or not str(payload.get("sql", "")).strip():
        strict_prompt = (
            prompt
            + "\nReturn ONLY the JSON object. No markdown, no prose, no code fences. "
            + "Ensure the JSON matches the schema exactly.\n"
        )
        retry_text = _generate_sql_raw(strict_prompt, max_output_tokens=8192)
        retry_payload, retry_error = _parse_sql_payload(retry_text)
        attempts = 2
        if str(retry_payload.get("sql", "")).strip() or not retry_error:
            payload = retry_payload
            parse_error = retry_error
            text = retry_text
    return payload, parse_error, text, attempts


def _generate_sql(prompt: str) -> dict:
    payload, _parse_error, _text, _attempts = _generate_sql_payload(prompt)
    return payload


def _should_revise_sql(validation: dict[str, Any] | None) -> bool:
    if not validation:
        return False
    recommendation = str(validation.get("recommendation", "")).strip().lower()
    return recommendation == "needs review"


def _maybe_revise_sql(
    user_query: str,
    tables: list[TableRef],
    sql: str,
    validation: dict[str, Any] | None,
    schema: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    if not sql or not _should_revise_sql(validation):
        return sql, validation, None

    prompt, _schema = _build_sql_revision_prompt(
        user_query=user_query,
        tables=tables,
        sql=sql,
        validation=validation or {},
        schema=schema,
    )
    payload, _parse_error, _raw_text, _attempts = _generate_sql_payload(prompt)
    revised_sql = str(payload.get("sql", "")).strip()
    if not revised_sql or revised_sql == sql:
        return sql, validation, None

    try:
        revised_validation = _validate_query_internal(revised_sql)
    except Exception as exc:
        log_event("validate_query_failed", error=str(exc), stage="revision")
        return sql, validation, None

    return revised_sql, revised_validation, payload


def _build_intent_prompt(query: str) -> str:
    schema = {
        "intent": "string",
        "confidence": 0.0,
        "entities": {
            "table_name": "string",
            "dataset_id": "string",
            "columns": ["string"],
        },
        "rationale": "string",
    }
    return (
        "You are an intent classifier for a data catalog assistant. "
        "Return STRICT JSON matching this schema:\\n"
        f"{json.dumps(schema)}\\n"
        "Allowed intents: chat, get_matching_columns, get_matching_tables, get_lineage, ripple_report, generate_sql, execute_sql, validate_query, validate_table. "
        "If the query is a greeting, help request, or unclear, return intent as chat with confidence <= 0.4.\\n\\n"
        f"User query: {query}\\n"
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _classify_intent(prompt: str) -> dict:
    client = _get_genai_client()
    response = client.models.generate_content(
        model=CONFIG.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=256,
        ),
    )
    text = _extract_response_text(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _build_chat_prompt(query: str) -> str:
    schema = {"reply": "string"}
    return (
        "You are QueryMind, a data catalog assistant. "
        "Return STRICT JSON matching this schema:\n"
        f"{json.dumps(schema)}\n"
        "If the user greets you or asks for help, provide a concise capability list "
        "as bullet points and include 2 example queries. "
        "If unsure, output 'unknown'.\n\n"
        f"User message: {query}\n"
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _chat_response(prompt: str) -> dict:
    client = _get_genai_client()
    response = client.models.generate_content(
        model=CONFIG.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _assess_sql(sql: str) -> None:
    stripped = sql.strip().lower()
    if not stripped:
        raise HTTPException(status_code=400, detail="SQL is empty")
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise HTTPException(status_code=400, detail="Only SELECT/WITH queries are allowed")
    if ";" in stripped:
        raise HTTPException(status_code=400, detail="Multiple statements are not allowed")


def _validate_query_internal(
    sql: str,
    budget_usd: float = 5.0,
    warn_threshold_pct: float = 80.0,
    project_id: str | None = None,
) -> dict[str, Any]:
    _assess_sql(sql)
    estimate_cost, check_query_efficiency, _ = _get_data_valid_tools()
    project = project_id or CONFIG.project_id

    cost = estimate_cost(
        query=sql,
        project=project,
        budget_usd=budget_usd,
        warn_threshold_pct=warn_threshold_pct,
    )
    efficiency = check_query_efficiency(sql)
    issues: list[str] = []
    suggestions: list[str] = []

    if cost.recommendation == "REJECT":
        issues.append(
            f"Estimated cost ${cost.estimated_cost_usd:.4f} exceeds budget ${budget_usd:.2f}"
        )
    elif cost.recommendation == "WARN":
        issues.append(
            f"Estimated cost ${cost.estimated_cost_usd:.4f} is approaching budget ${budget_usd:.2f}"
        )

    issues.extend(efficiency.get("issues", []))
    suggestions.extend(efficiency.get("suggestions", []))

    status = "ok"
    if cost.recommendation == "REJECT":
        status = "fail"
    elif cost.recommendation == "WARN" or issues:
        status = "warn"

    recommendation = cost.recommendation
    if issues:
        recommendation = "needs review"

    cost_payload = {
        "bytes_processed": cost.bytes_processed,
        "bytes_processed_gb": cost.bytes_processed_gb,
        "estimated_cost_usd": cost.estimated_cost_usd,
        "budget_usd": cost.budget_usd,
        "within_budget": cost.within_budget,
        "referenced_tables": cost.referenced_tables,
        "schema_fields": cost.schema_fields,
        "recommendation": cost.recommendation,
    }

    return {
        "status": status,
        "recommendation": recommendation,
        "approved": recommendation == "APPROVE",
        "is_suboptimal": bool(issues),
        "cost": cost_payload,
        "efficiency": efficiency,
        "issues": issues,
        "suggestions": suggestions,
    }


def _build_table_validation_prompt(result: dict[str, Any]) -> str:
    schema = {
        "summary": "string",
        "risks": ["string"],
        "recommendations": [
            {
                "rule": "string",
                "recommendation": "string",
                "sql": "string",
            }
        ],
        "next_steps": ["string"],
    }
    violations = result.get("violations") or []
    condensed_violations = []
    for v in violations[:10]:
        condensed_violations.append(
            {
                "rule": v.get("rule_name"),
                "severity": v.get("severity"),
                "column": v.get("column"),
                "message": v.get("message"),
                "row_count": v.get("row_count"),
                "remediation_suggestion": v.get("remediation_suggestion"),
                "remediation_sql": v.get("remediation_sql"),
            }
        )
    metadata = result.get("metadata") or {}
    context = {
        "table": result.get("table"),
        "layer": result.get("layer"),
        "status": result.get("status"),
        "rules_executed": result.get("rules_executed"),
        "rules_failed": result.get("rules_failed"),
        "rules_passed": result.get("rules_passed"),
        "table_row_count": metadata.get("table_row_count"),
        "table_schema": metadata.get("table_schema"),
        "partitioning": metadata.get("partitioning"),
        "partitioning_type": metadata.get("partitioning_type"),
        "partitioning_field": metadata.get("partitioning_field"),
        "require_partition_filter": metadata.get("require_partition_filter"),
        "clustering": metadata.get("clustering"),
        "violations": condensed_violations,
        "grain_analysis": metadata.get("grain_analysis"),
    }
    return (
        "You are a data quality analyst. Return STRICT JSON matching this schema:\n"
        f"{json.dumps(schema)}\n"
        "If you are unsure, output 'unknown'. Provide detailed, actionable insights.\n"
        "Use table_row_count as the authoritative row count; do not assume the table is empty "
        "just because individual violations show row_count 0.\n"
        "Summary should be a detailed paragraph. Risks should list ROOT CAUSES and include one item per applicable failed rule "
        "prefixed with the rule name (e.g., 'Partition Key Validation: ...'). "
        "Ignore rules that are not applicable for this table (e.g., rules requiring columns that do not exist). "
        "Recommendations must be an array of objects, one per applicable failed rule, in the same order as the applicable violations. "
        "For each recommendation object: "
        "rule must equal the rule_name, recommendation should describe the fix, and sql should include a concrete SQL "
        "statement using actual column names from table_schema when possible. "
        "If remediation_sql is provided in a violation, include it verbatim as sql. "
        "If a SQL fix is not applicable, set sql to 'unknown'. "
        "If column names are missing, use clearly-marked placeholders (e.g., required_column) rather than guessing. "
        "Next_steps should list PRIORITY order (what to fix first).\n\n"
        f"Validation result: {json.dumps(context)}\n"
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(2))
def _generate_table_validation_summary(prompt: str) -> dict[str, Any]:
    client = _get_genai_client()
    response = client.models.generate_content(
        model=CONFIG.gemini_pro_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=256),
        ),
    )
    text = _extract_response_text(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


class ChatState(TypedDict):
    messages: Annotated[list[dict[str, str]], operator.add]
    intent: str
    confidence: float
    entities: dict[str, Any]
    response: dict[str, Any]


def _map_table_results(rows: list[MatchTable]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        results.append(
            {
                "table_name": row.table_name,
                "column_name": None,
                "dataset_id": row.dataset_id,
                "full_name": f"{CONFIG.project_id}.{row.dataset_id}.{row.table_name}",
                "similarity": row.hybrid_score or row.semantic_score or 0,
            }
        )
    return results


def _map_column_results(rows: list[MatchColumn]) -> list[dict[str, Any]]:
    results = []
    for row in rows:
        results.append(
            {
                "table_name": row.table_name,
                "column_name": row.column_name,
                "dataset_id": row.dataset_id,
                "full_name": f"{CONFIG.project_id}.{row.dataset_id}.{row.table_name}",
                "similarity": row.hybrid_score or row.semantic_score or 0,
            }
        )
    return results


def _map_lineage_edges(edges: list[LineageEdgeResponse]) -> list[dict[str, Any]]:
    return [
        {
            "project_id": CONFIG.project_id,
            "dataset_id": edge.source_dataset,
            "from_table": edge.source_table,
            "to_project_id": CONFIG.project_id,
            "to_dataset_id": edge.target_dataset,
            "to_table": edge.target_table,
        }
        for edge in edges
    ]


def _run_sql(sql: str, max_rows: int = 100) -> dict[str, Any]:
    _assess_sql(sql)
    client = _get_bq_client()
    job_config = bigquery.QueryJobConfig(dry_run=False, use_query_cache=True)
    job = client.query(sql, job_config=job_config)
    rows = job.result(max_results=max_rows)
    data = [dict(row.items()) for row in rows]
    columns = list(data[0].keys()) if data else []
    return {
        "job_id": job.job_id,
        "total_rows": rows.total_rows,
        "rows": data,
        "columns": columns,
    }


def _classify_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    normalized = user_text.lower()
    if ("validate" in normalized or "assess" in normalized) and (
        "sql" in normalized or "query" in normalized
    ):
        return {"intent": "validate_query", "confidence": 0.85, "entities": {}}
    if "validate table" in normalized or "table validation" in normalized or "data quality" in normalized:
        return {"intent": "validate_table", "confidence": 0.85, "entities": {}}
    if (
        "impact" in normalized
        or "ripple" in normalized
        or "schema change" in normalized
        or ("change" in normalized and "column" in normalized)
    ):
        return {"intent": "ripple_report", "confidence": 0.8, "entities": {}}
    if "lineage" in normalized or "upstream" in normalized or "downstream" in normalized:
        return {"intent": "get_lineage", "confidence": 0.8, "entities": {}}
    if _is_likely_sql(user_text) or "execute" in normalized or "run sql" in normalized:
        return {"intent": "execute_sql", "confidence": 0.8, "entities": {}}
    if ("generate" in normalized or "write" in normalized or "build" in normalized) and (
        "sql" in normalized or "query" in normalized
    ):
        return {"intent": "generate_sql", "confidence": 0.8, "entities": {}}
    if _is_greeting(user_text):
        return {"intent": "chat", "confidence": 0.9, "entities": {}}
    prompt = _build_intent_prompt(user_text)
    payload = _classify_intent(prompt)
    intent = str(payload.get("intent", "chat")).strip()
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    if confidence < 0.5:
        intent = "chat"
    return {"intent": intent or "chat", "confidence": confidence, "entities": entities}


def _chat_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    prompt = _build_chat_prompt(user_text)
    payload = _chat_response(prompt)
    reply = str(payload.get("reply", "")).strip()
    if not reply or reply.lower() == "unknown":
        reply = (
            "I can help you search tables/columns, show lineage, generate SQL, execute SQL, "
            "and run impact assessments. Try:\n"
            "- \"Find tables with revenue\"\n"
            "- \"Show lineage for revenue_summary\""
        )
    response = {
        "reply": reply,
        "intent": "chat",
        "confidence": state.get("confidence"),
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _validate_query_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    sql = _extract_sql_from_text(user_text)
    if not sql:
        reply = "Please provide the SQL query to validate."
        response = {"reply": reply, "intent": "validate_query", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    try:
        validation = _validate_query_internal(sql)
    except Exception as exc:
        reply = f"Failed to validate query: {exc}"
        response = {
            "reply": reply,
            "intent": "validate_query",
            "confidence": state.get("confidence"),
            "query_validation": None,
        }
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    recommendation = validation.get("recommendation") if isinstance(validation, dict) else None
    reply = f"Validation complete: {recommendation}." if recommendation else "Validation complete."
    response = {
        "reply": reply,
        "intent": "validate_query",
        "confidence": state.get("confidence"),
        "query_validation": validation,
        "sql": sql,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _resolve_validation_target(text: str) -> tuple[str | None, str | None]:
    project_id, dataset_id, table_name = _extract_table_ref(text)
    if dataset_id and table_name:
        return dataset_id, table_name
    candidates = _resolve_table_candidates(None, text)
    if len(candidates) == 1:
        return candidates[0]["dataset_id"], candidates[0]["table_name"]
    return None, None


def _validate_table_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    dataset_id, table_name = _resolve_validation_target(user_text)
    if not dataset_id or not table_name:
        reply = "Please provide a dataset and table for validation (e.g. silver_layer.orders)."
        response = {"reply": reply, "intent": "validate_table", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    try:
        payload, llm_summary, cli_output = _run_table_validation(
            dataset_id=dataset_id,
            table_name=table_name,
            layer="gold" if "gold" in dataset_id else "bronze" if "bronze" in dataset_id else "silver",
            budget_usd=5.0,
            auto_detect_keys=True,
            auto_detect_grain=True,
            include_llm_summary=True,
        )
    except Exception as exc:
        reply = f"Failed to validate table: {exc}"
        response = {
            "reply": reply,
            "intent": "validate_table",
            "confidence": state.get("confidence"),
            "table_validation": None,
        }
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    reply = f"Data quality results for {payload.get('table')}."
    response = {
        "reply": reply,
        "intent": "validate_table",
        "confidence": state.get("confidence"),
        "table_validation": {
            "status": "ok",
            "validation_result": payload,
            "llm_summary": llm_summary,
            "cli_output": cli_output,
        },
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _tables_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    results = list(_query_matching_tables_cached(user_text, 10, _cache_bucket(300)))
    mapped = _map_table_results(results)
    count = min(len(mapped), 5)
    reply = "No Matches found" if count == 0 else f"Below are top {count} tables matched"
    response = {
        "reply": reply,
        "intent": "get_matching_tables",
        "confidence": state.get("confidence"),
        "search_results": mapped,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _columns_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    results = list(_query_matching_columns_cached(user_text, 10, _cache_bucket(300)))
    mapped = _map_column_results(results)
    count = min(len(mapped), 5)
    reply = "No Matches found" if count == 0 else f"Below are top {count} tables matched with relevant columns"
    response = {
        "reply": reply,
        "intent": "get_matching_columns",
        "confidence": state.get("confidence"),
        "search_results": mapped,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _lineage_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    entities = state.get("entities") or {}
    candidates = _resolve_table_candidates(entities.get("table_name"), user_text)
    if not candidates:
        reply = "No lineage found."
        response = {"reply": reply, "intent": "get_lineage", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}

    if entities.get("dataset_id"):
        candidates = [c for c in candidates if c["dataset_id"] == entities.get("dataset_id")] or candidates

    if entities.get("table_name"):
        exact = [c for c in candidates if c["table_name"].lower() == entities["table_name"].lower()]
        if len(exact) == 1:
            table = exact[0]
            edges = _fetch_lineage_edges(table["table_name"], table["dataset_id"])
            mapped_edges = _map_lineage_edges(edges)
            reply = f"Lineage for {table['table_name']}."
            response = {
                "reply": reply,
                "intent": "get_lineage",
                "confidence": state.get("confidence"),
                "lineage": {"edges": mapped_edges, "table": table["table_name"]},
            }
            return {"response": response, "messages": [{"role": "assistant", "content": reply}]}

    if len(candidates) > 1:
        mapped = _map_table_results(
            [
                MatchTable(
                    asset_id=opt["asset_id"],
                    dataset_id=opt["dataset_id"],
                    table_name=opt["table_name"],
                    semantic_score=1.0,
                    keyword_score=1.0,
                    hybrid_score=1.0,
                )
                for opt in candidates
            ]
        )
        reply = "Which table did you mean? Select one of the options."
        response = {
            "reply": reply,
            "intent": "get_lineage",
            "confidence": state.get("confidence"),
            "search_results": mapped,
            "needs_selection": True,
        }
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}

    table = candidates[0]
    edges = _fetch_lineage_edges(table["table_name"], table["dataset_id"])
    mapped_edges = _map_lineage_edges(edges)
    reply = f"Lineage for {table['table_name']}."
    response = {
        "reply": reply,
        "intent": "get_lineage",
        "confidence": state.get("confidence"),
        "lineage": {"edges": mapped_edges, "table": table["table_name"]},
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _generate_sql_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    explicit_table = _resolve_explicit_table(user_text)
    table_results: list[MatchTable] = []
    if explicit_table:
        if not _is_queryable_dataset(explicit_table.dataset_id):
            reply = (
                f"SQL generation is restricted to {', '.join(sorted(QUERYABLE_DATASETS))} datasets. "
                f"{explicit_table.dataset_id} is not queryable in this view."
            )
            response = {"reply": reply, "intent": "generate_sql", "confidence": state.get("confidence")}
            return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
        top_tables = [explicit_table]
    else:
        table_results = list(_query_matching_tables_cached(user_text, 8, _cache_bucket(300)))
        table_results = [row for row in table_results if _is_queryable_dataset(row.dataset_id)]
        top_tables: list[TableRef] = []
        if table_results:
            top_tables.append(TableRef(dataset_id=table_results[0].dataset_id, table_name=table_results[0].table_name))
        for hint in _hint_tables_from_query(user_text, allowed_datasets=QUERYABLE_DATASETS):
            if hint.table_name == (top_tables[0].table_name if top_tables else None) and hint.dataset_id == (top_tables[0].dataset_id if top_tables else None):
                continue
            if any(t.dataset_id == hint.dataset_id and t.table_name == hint.table_name for t in top_tables):
                continue
            top_tables.append(hint)
            if len(top_tables) >= 4:
                break
        if len(top_tables) < 3:
            for row in table_results[1:]:
                ref = TableRef(dataset_id=row.dataset_id, table_name=row.table_name)
                if any(t.dataset_id == ref.dataset_id and t.table_name == ref.table_name for t in top_tables):
                    continue
                top_tables.append(ref)
                if len(top_tables) >= 3:
                    break
    if not top_tables:
        reply = "No relevant tables found to generate SQL."
        response = {"reply": reply, "intent": "generate_sql", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    top_tables = _augment_sql_tables(top_tables)
    prompt, schema = _build_sql_prompt(user_text, top_tables)
    candidate_asset_ids = [f"{CONFIG.project_id}.{t.dataset_id}.{t.table_name}" for t in top_tables]
    schema_column_counts = {asset_id: len(schema.get(asset_id, [])) for asset_id in candidate_asset_ids}
    if schema_column_counts and all(count == 0 for count in schema_column_counts.values()):
        reply = "No schema available for the candidate tables. Run schema enrichment and retry."
        response = {
            "reply": reply,
            "intent": "generate_sql",
            "confidence": state.get("confidence"),
            "sql_generation_trace": SQLGenerationTrace(
                candidate_tables=list(schema_column_counts.keys()),
                schema_column_counts=schema_column_counts,
                parse_error=False,
                sql_empty=True,
            ),
        }
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    payload, parse_error, raw_text, attempts = _generate_sql_payload(prompt)
    sql = str(payload.get("sql", "")).strip()
    tables_for_prompt = top_tables
    if not sql and not explicit_table and table_results:
        extended_tables: list[TableRef] = list(top_tables)
        for row in table_results:
            ref = TableRef(dataset_id=row.dataset_id, table_name=row.table_name)
            if any(t.dataset_id == ref.dataset_id and t.table_name == ref.table_name for t in extended_tables):
                continue
            extended_tables.append(ref)
            if len(extended_tables) >= 6:
                break
        extended_tables = _augment_sql_tables(extended_tables)
        prompt, schema = _build_sql_prompt(user_text, extended_tables)
        payload, parse_error, raw_text, attempts = _generate_sql_payload(prompt)
        sql = str(payload.get("sql", "")).strip()
        tables_for_prompt = extended_tables
    tables_used = payload.get("tables_used") or [f"{t.dataset_id}.{t.table_name}" for t in tables_for_prompt]
    reply = "Generated SQL based on the most relevant tables." if sql else "Unable to generate SQL for that request."
    sql_generation_trace = None
    if not sql:
        candidate_tables = [f"{CONFIG.project_id}.{t.dataset_id}.{t.table_name}" for t in tables_for_prompt]
        schema_column_counts = {asset_id: len(schema.get(asset_id, [])) for asset_id in candidate_tables}
        snippet = raw_text.strip().replace("\n", " ")
        response_snippet = snippet[:400] if snippet else None
        sql_generation_trace = SQLGenerationTrace(
            candidate_tables=candidate_tables,
            schema_column_counts=schema_column_counts,
            parse_error=parse_error,
            sql_empty=True,
            response_snippet=response_snippet,
            attempts=attempts,
        )
    query_validation = None
    if sql:
        try:
            query_validation = _validate_query_internal(sql)
        except Exception as exc:
            log_event("validate_query_failed", error=str(exc))
    if sql:
        revised_sql, revised_validation, revision_payload = _maybe_revise_sql(
            user_query=user_text,
            tables=tables_for_prompt,
            sql=sql,
            validation=query_validation,
            schema=schema,
        )
        if revised_sql != sql:
            sql = revised_sql
            query_validation = revised_validation
            revised_tables_used = revision_payload.get("tables_used") if isinstance(revision_payload, dict) else None
            if isinstance(revised_tables_used, list) and revised_tables_used:
                tables_used = revised_tables_used
    response = {
        "reply": reply,
        "intent": "generate_sql",
        "confidence": state.get("confidence"),
        "sql": sql or None,
        "tables_used": tables_used,
        "query_validation": query_validation,
        "sql_generation_trace": sql_generation_trace,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _execute_sql_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    sql = user_text if _is_likely_sql(user_text) else ""
    tables_used: list[str] = []
    if not sql:
        explicit_table = _resolve_explicit_table(user_text)
        if explicit_table:
            if not _is_queryable_dataset(explicit_table.dataset_id):
                reply = (
                    f"SQL execution is restricted to {', '.join(sorted(QUERYABLE_DATASETS))} datasets. "
                    f"{explicit_table.dataset_id} is not queryable in this view."
                )
                response = {"reply": reply, "intent": "execute_sql", "confidence": state.get("confidence")}
                return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
            top_tables = [explicit_table]
        else:
            table_results = list(_query_matching_tables_cached(user_text, 6, _cache_bucket(300)))
            table_results = [row for row in table_results if _is_queryable_dataset(row.dataset_id)]
            top_tables: list[TableRef] = []
            if table_results:
                top_tables.append(TableRef(dataset_id=table_results[0].dataset_id, table_name=table_results[0].table_name))
            for hint in _hint_tables_from_query(user_text, allowed_datasets=QUERYABLE_DATASETS):
                if any(t.dataset_id == hint.dataset_id and t.table_name == hint.table_name for t in top_tables):
                    continue
                top_tables.append(hint)
                if len(top_tables) >= 4:
                    break
            if len(top_tables) < 3:
                for row in table_results[1:]:
                    ref = TableRef(dataset_id=row.dataset_id, table_name=row.table_name)
                    if any(t.dataset_id == ref.dataset_id and t.table_name == ref.table_name for t in top_tables):
                        continue
                    top_tables.append(ref)
                    if len(top_tables) >= 3:
                        break
        if not top_tables:
            reply = "No relevant tables found to execute."
            response = {"reply": reply, "intent": "execute_sql", "confidence": state.get("confidence")}
            return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
        top_tables = _augment_sql_tables(top_tables)
        prompt, _schema = _build_sql_prompt(user_text, top_tables)
        payload = _generate_sql(prompt)
        sql = str(payload.get("sql", "")).strip()
        tables_used = payload.get("tables_used") or [f"{t.dataset_id}.{t.table_name}" for t in top_tables]
    if not sql:
        reply = "Could not generate SQL to execute."
        response = {"reply": reply, "intent": "execute_sql", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}

    query_validation = None
    try:
        query_validation = _validate_query_internal(sql)
    except Exception as exc:
        log_event("validate_query_failed", error=str(exc))

    result = _run_sql(sql, max_rows=100)
    reply = "Executed SQL." if _is_likely_sql(user_text) else "Generated and executed SQL."
    response = {
        "reply": reply,
        "intent": "execute_sql",
        "confidence": state.get("confidence"),
        "sql": sql,
        "tables_used": tables_used,
        "query_validation": query_validation,
        "execution_result": result,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _ripple_node(state: ChatState) -> dict[str, Any]:
    user_text = state["messages"][-1]["content"]
    target = _extract_impact_target(user_text, state.get("entities"))
    if not target:
        reply = "Please provide a fully qualified column like `dataset.table.column` for impact assessment."
        response = {"reply": reply, "intent": "ripple_report", "confidence": state.get("confidence")}
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    if not _column_exists(target["dataset_id"], target["table_name"], target["column_name"]):
        reply = (
            f"Column not found in metadata: {target['dataset_id']}."
            f"{target['table_name']}.{target['column_name']}."
        )
        response = {
            "reply": reply,
            "intent": "ripple_report",
            "confidence": state.get("confidence"),
            "impact_assessment": {"error": reply},
        }
        return {"response": response, "messages": [{"role": "assistant", "content": reply}]}
    report = build_ripple_report(
        _get_bq_client(),
        CONFIG,
        dataset_id=target["dataset_id"],
        table_name=target["table_name"],
        column_name=target["column_name"],
        max_hops=3,
    )
    assessment = _map_ripple_to_assessment(report)
    reply = report.get("executive_summary", {}).get("summary") or (
        f"Impact assessment for {target['dataset_id']}.{target['table_name']}.{target['column_name']}."
    )
    response = {
        "reply": reply,
        "intent": "ripple_report",
        "confidence": state.get("confidence"),
        "impact_assessment": assessment,
    }
    return {"response": response, "messages": [{"role": "assistant", "content": reply}]}


def _route_intent(state: ChatState) -> Literal[
    "chat",
    "get_matching_tables",
    "get_matching_columns",
    "get_lineage",
    "generate_sql",
    "execute_sql",
    "validate_query",
    "validate_table",
    "ripple_report",
]:
    return state.get("intent", "chat")


_graph_builder = StateGraph(ChatState)
_graph_builder.add_node("classify", _classify_node)
_graph_builder.add_node("chat", _chat_node)
_graph_builder.add_node("get_matching_tables", _tables_node)
_graph_builder.add_node("get_matching_columns", _columns_node)
_graph_builder.add_node("get_lineage", _lineage_node)
_graph_builder.add_node("generate_sql", _generate_sql_node)
_graph_builder.add_node("execute_sql", _execute_sql_node)
_graph_builder.add_node("validate_query", _validate_query_node)
_graph_builder.add_node("validate_table", _validate_table_node)
_graph_builder.add_node("ripple_report", _ripple_node)
_graph_builder.add_edge(START, "classify")
_graph_builder.add_conditional_edges(
    "classify",
    _route_intent,
    {
        "chat": "chat",
        "get_matching_tables": "get_matching_tables",
        "get_matching_columns": "get_matching_columns",
        "get_lineage": "get_lineage",
        "generate_sql": "generate_sql",
        "execute_sql": "execute_sql",
        "validate_query": "validate_query",
        "validate_table": "validate_table",
        "ripple_report": "ripple_report",
    },
)
for node_name in (
    "chat",
    "get_matching_tables",
    "get_matching_columns",
    "get_lineage",
    "generate_sql",
    "execute_sql",
    "validate_query",
    "validate_table",
    "ripple_report",
):
    _graph_builder.add_edge(node_name, END)

CHAT_GRAPH = _graph_builder.compile(checkpointer=GraphMemorySaver())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/get_matching_columns")
def get_matching_columns(request: MatchRequest) -> list[MatchColumn]:
    log_event("api_match_columns", query=request.query, top_k=request.top_k)
    results = _query_matching_columns_cached(request.query, request.top_k, _cache_bucket(300))
    return list(results)


@app.post("/get_matching_tables")
def get_matching_tables(request: MatchRequest) -> list[MatchTable]:
    log_event("api_match_tables", query=request.query, top_k=request.top_k)
    results = _query_matching_tables_cached(request.query, request.top_k, _cache_bucket(300))
    return list(results)


@app.post("/get_lineage")
def get_lineage(request: LineageRequest) -> LineageResponse:
    candidates = _resolve_table_candidates(request.table_name, request.query)
    if not candidates:
        return LineageResponse(status="not_found")

    if request.dataset_id:
        candidates = [c for c in candidates if c["dataset_id"] == request.dataset_id] or candidates

    if request.table_name:
        exact = [c for c in candidates if c["table_name"].lower() == request.table_name.lower()]
        if len(exact) == 1:
            table = exact[0]
            edges = _fetch_lineage_edges(table["table_name"], table["dataset_id"])
            return LineageResponse(
                status="ok",
                table_name=table["table_name"],
                dataset_id=table["dataset_id"],
                edges=edges,
            )

    if len(candidates) > 1:
        return LineageResponse(status="needs_selection", options=candidates)

    table = candidates[0]
    edges = _fetch_lineage_edges(table["table_name"], table["dataset_id"])
    return LineageResponse(
        status="ok",
        table_name=table["table_name"],
        dataset_id=table["dataset_id"],
        edges=edges,
    )


@app.post("/generate_sql")
def generate_sql(request: GenerateSQLRequest) -> GenerateSQLResponse:
    tables_for_prompt = _augment_sql_tables(request.tables)
    prompt, schema = _build_sql_prompt(request.user_query, tables_for_prompt)
    candidate_asset_ids = [f"{CONFIG.project_id}.{t.dataset_id}.{t.table_name}" for t in tables_for_prompt]
    schema_column_counts = {asset_id: len(schema.get(asset_id, [])) for asset_id in candidate_asset_ids}
    if schema_column_counts and all(count == 0 for count in schema_column_counts.values()):
        return GenerateSQLResponse(
            sql="",
            notes="No schema available for candidate tables. Run schema enrichment and retry.",
            tables_used=[f"{t.dataset_id}.{t.table_name}" for t in tables_for_prompt],
            query_validation=None,
            sql_generation_trace=SQLGenerationTrace(
                candidate_tables=list(schema_column_counts.keys()),
                schema_column_counts=schema_column_counts,
                parse_error=False,
                sql_empty=True,
            ),
        )
    payload, parse_error, raw_text, attempts = _generate_sql_payload(prompt)
    sql = str(payload.get("sql", "")).strip()
    notes = payload.get("notes")
    tables_used = payload.get("tables_used") or []
    if not isinstance(tables_used, list):
        tables_used = []
    sql_generation_trace = None
    if not sql:
        candidate_tables = [f"{CONFIG.project_id}.{t.dataset_id}.{t.table_name}" for t in tables_for_prompt]
        schema_column_counts = {asset_id: len(schema.get(asset_id, [])) for asset_id in candidate_tables}
        snippet = raw_text.strip().replace("\n", " ")
        response_snippet = snippet[:400] if snippet else None
        sql_generation_trace = SQLGenerationTrace(
            candidate_tables=candidate_tables,
            schema_column_counts=schema_column_counts,
            parse_error=parse_error,
            sql_empty=True,
            response_snippet=response_snippet,
            attempts=attempts,
        )
    query_validation = None
    if sql:
        try:
            query_validation = _validate_query_internal(sql)
        except Exception as exc:
            log_event("validate_query_failed", error=str(exc))
    if sql:
        revised_sql, revised_validation, revision_payload = _maybe_revise_sql(
            user_query=request.user_query,
            tables=tables_for_prompt,
            sql=sql,
            validation=query_validation,
            schema=schema,
        )
        if revised_sql != sql:
            sql = revised_sql
            query_validation = revised_validation
            revision_notes = revision_payload.get("notes") if isinstance(revision_payload, dict) else None
            if isinstance(revision_notes, str) and revision_notes:
                notes = revision_notes
            revised_tables_used = revision_payload.get("tables_used") if isinstance(revision_payload, dict) else None
            if isinstance(revised_tables_used, list) and revised_tables_used:
                tables_used = revised_tables_used
    return GenerateSQLResponse(
        sql=sql,
        notes=notes,
        tables_used=tables_used,
        query_validation=query_validation,
        sql_generation_trace=sql_generation_trace,
    )


@app.post("/execute_sql")
def execute_sql(request: ExecuteSQLRequest) -> ExecuteSQLResponse:
    _assess_sql(request.sql)
    client = _get_bq_client()
    job_config = bigquery.QueryJobConfig(dry_run=request.dry_run, use_query_cache=not request.dry_run)
    job = client.query(request.sql, job_config=job_config)
    if request.dry_run:
        return ExecuteSQLResponse(job_id=job.job_id, total_rows=0, rows=[])
    rows = job.result(max_results=request.max_rows)
    data = [dict(row.items()) for row in rows]
    total_rows = rows.total_rows
    return ExecuteSQLResponse(job_id=job.job_id, total_rows=total_rows, rows=data)


@app.post("/validate_query")
def validate_query(request: ValidateQueryRequest) -> ValidateQueryResponse:
    try:
        result = _validate_query_internal(
            request.sql,
            budget_usd=request.budget_usd,
            warn_threshold_pct=request.warn_threshold_pct,
            project_id=request.project_id,
        )
        return ValidateQueryResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/classify_intent")
def classify_intent(request: ClassifyIntentRequest) -> ClassifyIntentResponse:
    if _is_greeting(request.query):
        return ClassifyIntentResponse(
            intent="chat",
            confidence=0.9,
            entities={},
            rationale="Greeting or help request detected",
        )
    prompt = _build_intent_prompt(request.query)
    payload = _classify_intent(prompt)
    intent = str(payload.get("intent", "get_matching_tables")).strip()
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    entities = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    rationale = payload.get("rationale")
    if confidence < 0.5:
        intent = "chat"
    return ClassifyIntentResponse(
        intent=intent or "get_matching_tables",
        confidence=confidence,
        entities=entities,
        rationale=rationale if isinstance(rationale, str) else None,
    )


@app.post("/get_table_schema")
def get_table_schema(request: TableSchemaRequest) -> TableSchemaResponse:
    columns = _fetch_table_schema_single(request.dataset_id, request.table_name)
    if not columns:
        raise HTTPException(status_code=404, detail="Table schema not found")
    return TableSchemaResponse(
        dataset_id=request.dataset_id,
        table_name=request.table_name,
        columns=columns,
    )


@app.post("/validate_table")
def validate_table(request: ValidateTableRequest) -> ValidateTableResponse:
    try:
        payload, llm_summary, cli_output = _run_table_validation(
            dataset_id=request.dataset_id,
            table_name=request.table_name,
            layer=request.layer,
            budget_usd=request.budget_usd,
            auto_detect_keys=request.auto_detect_keys,
            auto_detect_grain=request.auto_detect_grain,
            include_llm_summary=request.include_llm_summary,
        )
        return ValidateTableResponse(
            status="ok",
            validation_result=payload,
            llm_summary=llm_summary,
            cli_output=cli_output,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/validate_table_v2")
def validate_table_v2(request: ValidateTableV2Request) -> ValidateTableResponse:
    try:
        payload, llm_summary, cli_output = _run_table_validation_v2(
            table=request.table,
            layer=request.layer,
            project_id=request.project_id,
            budget_usd=request.budget_usd,
            auto_detect_keys=request.auto_detect_keys,
            auto_detect_grain=request.auto_detect_grain,
            include_llm_summary=request.include_llm_summary,
        )
        return ValidateTableResponse(
            status="ok",
            validation_result=payload,
            llm_summary=llm_summary,
            cli_output=cli_output,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or "default"
    state = {"messages": [{"role": "user", "content": request.query}]}
    response = None
    try:
        result = CHAT_GRAPH.invoke(state, {"configurable": {"thread_id": session_id}})
        response = result.get("response") if isinstance(result, dict) else None
    except Exception as exc:
        log_event("chat_graph_failed", error=str(exc))

    if not isinstance(response, dict):
        normalized = request.query.lower()
        if (
            "impact" in normalized
            or "ripple" in normalized
            or "schema change" in normalized
            or ("change" in normalized and "column" in normalized)
        ):
            fallback = _ripple_node(
                {
                    "messages": [{"role": "user", "content": request.query}],
                    "intent": "ripple_report",
                    "confidence": 0.0,
                    "entities": {},
                    "response": {},
                }
            )
            response = fallback.get("response")
        if not isinstance(response, dict):
            response = {
                "reply": (
                    "I can help you search tables/columns, show lineage, generate SQL, "
                    "execute SQL, and run impact assessments."
                ),
                "intent": "chat",
            }
    return ChatResponse(**response)


@app.post("/ripple_report")
def ripple_report(request: RippleReportRequest) -> RippleReportResponse:
    if not _column_exists(request.dataset_id, request.table_name, request.column_name):
        raise HTTPException(status_code=404, detail="Column not found in metadata")
    report = build_ripple_report(
        _get_bq_client(),
        CONFIG,
        dataset_id=request.dataset_id,
        table_name=request.table_name,
        column_name=request.column_name,
        max_hops=request.max_hops,
        include_query_patterns=request.include_query_patterns,
        include_prompt_templates=request.include_prompt_templates,
    )
    return RippleReportResponse(**report)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
