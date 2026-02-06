from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable

from google import genai
from google.genai import types
from google.cloud import bigquery
from tenacity import retry, stop_after_attempt, wait_exponential

from .bq_extract import ColumnRecord, compute_column_hash
from .smart_load import merge_columns
from .utils import json_dumps, log_event, utc_now


@dataclass(frozen=True)
class AssetContext:
    asset_id: str
    table_description: str
    concepts: list[str]
    grain: str
    notes: str


def _safe_json_parse(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _extract_response_text(response: object) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        if parts:
            return "".join(part.text for part in parts if hasattr(part, "text"))
    return ""


def _parse_policy_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(tag) for tag in raw if tag is not None]
    if isinstance(raw, str):
        if not raw.strip():
            return []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(value, list):
            return [str(tag) for tag in value if tag is not None]
    return []


def _build_column_prompt(
    asset_id: str,
    columns: Iterable[ColumnRecord],
    missing_columns: list[str],
    context: AssetContext | None,
) -> str:
    schema = {"columns": [{"column_name": "string", "description": "string"}]}
    column_lines = []
    for col in columns:
        desc = col.column_description or "unknown"
        column_lines.append(f"- {col.column_name} | {col.data_type} | {desc}")
    context_lines = []
    if context:
        context_lines = [
            f"Table description: {context.table_description or 'unknown'}",
            f"Concepts: {', '.join(context.concepts) if context.concepts else 'unknown'}",
            f"Grain: {context.grain or 'unknown'}",
            f"Notes: {context.notes or 'unknown'}",
        ]
    missing_list = ", ".join(missing_columns) if missing_columns else "none"
    return (
        "You are a data catalog assistant. Output STRICT JSON matching this schema:\n"
        f"{json_dumps(schema)}\n"
        "Rules: Only include columns listed under Missing Columns. "
        "If you are unsure, output 'unknown' rather than guessing. "
        "Do not include sensitive data samples. Keep descriptions concise (<= 120 chars).\n\n"
        f"Table: {asset_id}\n"
        + ("\n".join(context_lines) + "\n" if context_lines else "")
        + f"Missing Columns: {missing_list}\n"
        f"Columns:\n{chr(10).join(column_lines)}\n"
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _call_gemini(
    project_id: str,
    location: str,
    model: str,
    prompt: str,
) -> dict[str, Any]:
    client = genai.Client(vertexai=True, project=project_id, location=location)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=1024,
        ),
    )
    text = _extract_response_text(response)
    return _safe_json_parse(text)


def _normalize_descriptions(
    payload: dict[str, Any],
    missing_columns: list[str],
) -> dict[str, str]:
    missing = {col.lower(): col for col in missing_columns}
    results: dict[str, str] = {}
    for item in payload.get("columns", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        col_raw = str(item.get("column_name", "")).strip()
        desc = str(item.get("description", "")).strip()
        if not col_raw or not desc:
            continue
        key = missing.get(col_raw.lower())
        if not key:
            continue
        results[key] = desc
    for col in missing_columns:
        results.setdefault(col, "unknown")
    return results


def _fetch_assets_with_missing_columns(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
) -> list[str]:
    sql = f"""
    SELECT asset_id
    FROM `{project_id}.{metadata_dataset}.columns`
    WHERE SPLIT(asset_id, '.')[OFFSET(1)] IN ('silver_layer', 'gold_layer')
    GROUP BY asset_id
    HAVING COUNTIF(IFNULL(TRIM(column_description), '') = '') > 0
    """
    return [row["asset_id"] for row in client.query(sql).result()]


def _fetch_columns_for_assets(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> tuple[dict[str, list[ColumnRecord]], dict[str, list[str]]]:
    if not asset_ids:
        return {}, {}
    sql = f"""
    SELECT asset_id, column_name, data_type, is_nullable, ordinal_position, column_description, policy_tags
    FROM `{project_id}.{metadata_dataset}.columns`
    WHERE asset_id IN UNNEST(@asset_ids)
    ORDER BY asset_id, ordinal_position
    """
    params = [bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)]
    rows = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    columns_by_asset: dict[str, list[ColumnRecord]] = {}
    missing_by_asset: dict[str, list[str]] = {}
    for row in rows:
        policy_tags = _parse_policy_tags(row.get("policy_tags"))
        col = ColumnRecord(
            asset_id=row["asset_id"],
            column_name=row["column_name"],
            data_type=row.get("data_type") or "STRING",
            is_nullable=bool(row.get("is_nullable")),
            ordinal_position=int(row.get("ordinal_position") or 0),
            column_description=row.get("column_description") or "",
            policy_tags=policy_tags,
            column_hash="",
        )
        col = ColumnRecord(**{**col.__dict__, "column_hash": compute_column_hash(col)})
        columns_by_asset.setdefault(row["asset_id"], []).append(col)
        if not (row.get("column_description") or "").strip():
            missing_by_asset.setdefault(row["asset_id"], []).append(row["column_name"])
    return columns_by_asset, missing_by_asset


def _fetch_asset_context(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[str, AssetContext]:
    if not asset_ids:
        return {}
    params = [bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)]
    assets_sql = f"""
    SELECT asset_id, table_description
    FROM `{project_id}.{metadata_dataset}.assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    enriched_sql = f"""
    SELECT asset_id, concepts, grain, notes
    FROM `{project_id}.{metadata_dataset}.enriched_assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    assets = {
        row["asset_id"]: row.get("table_description") or ""
        for row in client.query(assets_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    }
    enriched = {
        row["asset_id"]: {
            "concepts": row.get("concepts") or [],
            "grain": row.get("grain") or "",
            "notes": row.get("notes") or "",
        }
        for row in client.query(enriched_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    }
    contexts: dict[str, AssetContext] = {}
    for asset_id in asset_ids:
        info = enriched.get(asset_id, {})
        contexts[asset_id] = AssetContext(
            asset_id=asset_id,
            table_description=assets.get(asset_id, ""),
            concepts=info.get("concepts") or [],
            grain=info.get("grain") or "",
            notes=info.get("notes") or "",
        )
    return contexts


def enrich_column_descriptions(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    vertex_location: str,
    gemini_model: str,
    max_workers: int,
) -> int:
    asset_ids = _fetch_assets_with_missing_columns(client, project_id, metadata_dataset)
    if not asset_ids:
        log_event("column_description_enrich_skipped", reason="no_missing_columns")
        return 0

    columns_by_asset, missing_by_asset = _fetch_columns_for_assets(
        client, project_id, metadata_dataset, asset_ids
    )
    contexts = _fetch_asset_context(client, project_id, metadata_dataset, asset_ids)

    rows: list[dict[str, Any]] = []
    now = utc_now()

    def _enrich_one(asset_id: str) -> tuple[str, dict[str, str]]:
        columns = columns_by_asset.get(asset_id, [])
        missing = missing_by_asset.get(asset_id, [])
        if not columns or not missing:
            return asset_id, {}
        prompt = _build_column_prompt(asset_id, columns, missing, contexts.get(asset_id))
        payload = _call_gemini(project_id, vertex_location, gemini_model, prompt)
        return asset_id, _normalize_descriptions(payload, missing)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_enrich_one, asset_id): asset_id for asset_id in asset_ids}
        for future in as_completed(futures):
            asset_id = futures[future]
            try:
                _, descs = future.result()
            except Exception as exc:
                log_event("column_description_enrich_failed", asset_id=asset_id, error=str(exc))
                continue
            if not descs:
                continue
            columns = {col.column_name: col for col in columns_by_asset.get(asset_id, [])}
            for col_name, desc in descs.items():
                existing = columns.get(col_name)
                if not existing:
                    continue
                updated = ColumnRecord(
                    asset_id=existing.asset_id,
                    column_name=existing.column_name,
                    data_type=existing.data_type,
                    is_nullable=existing.is_nullable,
                    ordinal_position=existing.ordinal_position,
                    column_description=desc,
                    policy_tags=existing.policy_tags,
                    column_hash="",
                )
                updated = ColumnRecord(
                    **{**updated.__dict__, "column_hash": compute_column_hash(updated)}
                )
                rows.append(
                    {
                        "asset_id": updated.asset_id,
                        "column_name": updated.column_name,
                        "data_type": updated.data_type,
                        "is_nullable": updated.is_nullable,
                        "ordinal_position": updated.ordinal_position,
                        "column_description": updated.column_description,
                        "policy_tags": json_dumps(updated.policy_tags or []),
                        "column_hash": updated.column_hash,
                        "updated_at": now,
                    }
                )

    merge_columns(client, project_id, metadata_dataset, rows)
    log_event(
        "column_description_enriched",
        assets=len(asset_ids),
        columns=len(rows),
    )
    return len(rows)
