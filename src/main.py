from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any, Iterable

from google.cloud import bigquery

from .bq_extract import (
    AssetRecord,
    ColumnRecord,
    compute_column_hash,
    ensure_metadata_dataset,
    extract_all_assets_columns,
    extract_lineage_edges,
)
from .column_doc_builder import build_column_document
from .column_desc_enrich import enrich_column_descriptions
from .config import load_config
from .doc_builder import build_table_document
from .embed import (
    embed_documents,
    fetch_column_docs_to_embed,
    fetch_table_docs_to_embed,
)
from .enrich_gemini import enrich_assets
from .smart_load import (
    merge_assets,
    merge_column_documents,
    merge_column_embeddings,
    merge_columns,
    merge_documents,
    merge_embeddings,
    merge_enriched_assets,
    merge_lineage_edges,
)
from .utils import json_dumps, log_event, utc_now


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _run_query(
    client: bigquery.Client,
    query: str,
    params: list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter],
) -> list[dict[str, Any]]:
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = client.query(query, job_config=job_config).result()
    return [dict(row.items()) for row in rows]


def fetch_existing_assets(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not asset_ids:
        return {}
    results: dict[str, dict[str, Any]] = {}
    query = f"""
    SELECT asset_id, schema_hash, table_meta_hash, doc_hash
    FROM `{project_id}.{metadata_dataset}.assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    for batch in _chunked(asset_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", batch)],
        )
        for row in rows:
            results[row["asset_id"]] = row
    return results


def fetch_existing_columns(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[tuple[str, str], str]:
    if not asset_ids:
        return {}
    results: dict[tuple[str, str], str] = {}
    query = f"""
    SELECT asset_id, column_name, column_hash
    FROM `{project_id}.{metadata_dataset}.columns`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    for batch in _chunked(asset_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", batch)],
        )
        for row in rows:
            results[(row["asset_id"], row["column_name"])] = row["column_hash"]
    return results


def fetch_existing_column_descriptions(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[tuple[str, str], str]:
    if not asset_ids:
        return {}
    results: dict[tuple[str, str], str] = {}
    query = f"""
    SELECT asset_id, column_name, column_description
    FROM `{project_id}.{metadata_dataset}.columns`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    for batch in _chunked(asset_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", batch)],
        )
        for row in rows:
            desc = row.get("column_description") or ""
            if desc.strip():
                results[(row["asset_id"], row["column_name"])] = desc
    return results


def fetch_existing_docs(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    doc_ids: list[str],
) -> dict[str, str]:
    if not doc_ids:
        return {}
    results: dict[str, str] = {}
    query = f"""
    SELECT doc_id, doc_hash
    FROM `{project_id}.{metadata_dataset}.documents`
    WHERE doc_id IN UNNEST(@doc_ids)
    """
    for batch in _chunked(doc_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("doc_ids", "STRING", batch)],
        )
        for row in rows:
            results[row["doc_id"]] = row["doc_hash"]
    return results


def fetch_existing_column_docs(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[str, str]:
    if not asset_ids:
        return {}
    existing: dict[str, str] = {}
    query = f"""
    SELECT column_doc_id, doc_hash
    FROM `{project_id}.{metadata_dataset}.column_documents`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    for batch in _chunked(asset_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", batch)],
        )
        for row in rows:
            existing[row["column_doc_id"]] = row["doc_hash"]
    return existing


def fetch_existing_enriched(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    asset_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not asset_ids:
        return {}
    results: dict[str, dict[str, Any]] = {}
    query = f"""
    SELECT asset_id,
           enrichment_version,
           concepts,
           grain,
           synonyms,
           join_hints,
           pii_flags,
           notes,
           enriched_hash
    FROM `{project_id}.{metadata_dataset}.enriched_assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    for batch in _chunked(asset_ids, 1000):
        rows = _run_query(
            client,
            query,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", batch)],
        )
        for row in rows:
            results[row["asset_id"]] = row
    return results


def ensure_tables(client: bigquery.Client, project_id: str, metadata_dataset: str, location: str) -> None:
    ensure_metadata_dataset(client, project_id, metadata_dataset, location)
    with open("sql/create_metadata_tables.sql", "r", encoding="utf-8") as handle:
        sql = handle.read()
    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]
    for stmt in statements:
        client.query(stmt).result()


def _build_column_map(columns: list[ColumnRecord]) -> dict[str, list[ColumnRecord]]:
    by_asset: dict[str, list[ColumnRecord]] = defaultdict(list)
    for col in columns:
        by_asset[col.asset_id].append(col)
    return by_asset


def _doc_enrichment_payload(enrichment: dict | None) -> dict | None:
    if not enrichment:
        return None
    return {
        "enrichment_version": enrichment.get("enrichment_version"),
        "concepts": enrichment.get("concepts"),
        "grain": enrichment.get("grain"),
        "synonyms": enrichment.get("synonyms"),
        "join_hints": enrichment.get("join_hints"),
        "pii_flags": enrichment.get("pii_flags"),
        "notes": enrichment.get("notes"),
        "enriched_hash": enrichment.get("enriched_hash"),
    }


def _log_stage(stage: str, started: float, **fields: Any) -> None:
    log_event(
        "stage_completed",
        stage=stage,
        duration_s=round(time.monotonic() - started, 3),
        **fields,
    )


def _split_asset_id(asset_id: str | None) -> tuple[str | None, str | None]:
    if not asset_id:
        return None, None
    parts = asset_id.split(".")
    if len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def run_embed(client: bigquery.Client, config) -> None:
    table_docs = fetch_table_docs_to_embed(
        client, config.project_id, config.metadata_dataset, config.embedding_model
    )
    log_event("table_docs_to_embed", count=len(table_docs))
    table_embeddings = embed_documents(
        table_docs,
        config.project_id,
        config.vertex_location,
        config.embedding_model,
        config.embedding_task_type,
        config.embedding_dim,
        config.max_embed_workers,
        config.embed_batch_size,
    )
    table_rows = [
        {
            "doc_id": row["doc_id"],
            "asset_id": row["asset_id"],
            "embedding": row["embedding"],
            "embedding_model": row["embedding_model"],
            "embedding_dim": row["embedding_dim"],
            "doc_hash": row["doc_hash"],
            "updated_at": row["updated_at"],
        }
        for row in table_embeddings
    ]
    merge_embeddings(client, config.project_id, config.metadata_dataset, table_rows)

    column_docs = fetch_column_docs_to_embed(
        client, config.project_id, config.metadata_dataset, config.embedding_model
    )
    log_event("column_docs_to_embed", count=len(column_docs))
    column_embeddings = embed_documents(
        column_docs,
        config.project_id,
        config.vertex_location,
        config.embedding_model,
        config.embedding_task_type,
        config.embedding_dim,
        config.max_embed_workers,
        config.embed_batch_size,
    )
    column_rows = [
        {
            "column_doc_id": row["doc_id"],
            "asset_id": row["asset_id"],
            "column_name": row.get("column_name"),
            "embedding": row["embedding"],
            "embedding_model": row["embedding_model"],
            "embedding_dim": row["embedding_dim"],
            "doc_hash": row["doc_hash"],
            "updated_at": row["updated_at"],
        }
        for row in column_embeddings
    ]
    merge_column_embeddings(client, config.project_id, config.metadata_dataset, column_rows)


def run_pipeline(client: bigquery.Client, config, include_embeddings: bool) -> None:
    stage_start = time.monotonic()
    assets, columns = extract_all_assets_columns(
        client,
        config.project_id,
        config.datasets,
        config.ignore_datasets,
        config.bq_location,
        config.max_bq_workers,
    )
    _log_stage("extract", stage_start, assets=len(assets), columns=len(columns))

    stage_start = time.monotonic()
    lineage_edges = extract_lineage_edges(client, config.project_id, config.datasets)
    _log_stage("lineage_extract", stage_start, edges=len(lineage_edges))

    asset_ids = [asset.asset_id for asset in assets]
    existing_assets = fetch_existing_assets(client, config.project_id, config.metadata_dataset, asset_ids)
    existing_columns = fetch_existing_columns(client, config.project_id, config.metadata_dataset, asset_ids)
    existing_docs = fetch_existing_docs(client, config.project_id, config.metadata_dataset, asset_ids)
    existing_column_docs = fetch_existing_column_docs(
        client, config.project_id, config.metadata_dataset, asset_ids
    )
    existing_enriched = fetch_existing_enriched(
        client, config.project_id, config.metadata_dataset, asset_ids
    )
    existing_descriptions = fetch_existing_column_descriptions(
        client, config.project_id, config.metadata_dataset, asset_ids
    )
    description_backfilled_assets: set[str] = set()
    if existing_descriptions:
        updated_columns: list[ColumnRecord] = []
        for col in columns:
            if col.column_description:
                updated_columns.append(col)
                continue
            existing_desc = existing_descriptions.get((col.asset_id, col.column_name))
            if not existing_desc:
                updated_columns.append(col)
                continue
            updated = ColumnRecord(
                asset_id=col.asset_id,
                column_name=col.column_name,
                data_type=col.data_type,
                is_nullable=col.is_nullable,
                ordinal_position=col.ordinal_position,
                column_description=existing_desc,
                policy_tags=col.policy_tags,
                column_hash="",
            )
            updated = ColumnRecord(**{**updated.__dict__, "column_hash": compute_column_hash(updated)})
            updated_columns.append(updated)
            description_backfilled_assets.add(col.asset_id)
        columns = updated_columns

    columns_by_asset = _build_column_map(columns)

    asset_changes: set[str] = set()
    for asset in assets:
        existing = existing_assets.get(asset.asset_id)
        if not existing:
            asset_changes.add(asset.asset_id)
            continue
        if (
            existing.get("schema_hash") != asset.schema_hash
            or existing.get("table_meta_hash") != asset.table_meta_hash
        ):
            asset_changes.add(asset.asset_id)

    enrich_targets = [
        asset
        for asset in assets
        if asset.dataset_id in {"silver_layer", "gold_layer"}
        and (
            asset.asset_id in asset_changes
            or asset.asset_id not in existing_enriched
            or existing_enriched.get(asset.asset_id, {}).get("enrichment_version")
            != config.enrichment_version
        )
    ]

    stage_start = time.monotonic()
    enrichment_results = enrich_assets(
        config.project_id,
        config.vertex_location,
        config.gemini_model,
        config.enrichment_version,
        enrich_targets,
        columns_by_asset,
        max_workers=config.max_enrich_workers,
    )
    _log_stage("enrich", stage_start, assets=len(enrichment_results))

    enrichment_changed: set[str] = set()
    for asset in enrich_targets:
        existing = existing_enriched.get(asset.asset_id)
        new_hash = enrichment_results.get(asset.asset_id, {}).get("enriched_hash")
        if not existing or existing.get("enriched_hash") != new_hash:
            enrichment_changed.add(asset.asset_id)

    docs_needed: set[str] = set()
    for asset in assets:
        if asset.asset_id in asset_changes or asset.asset_id in enrichment_changed:
            docs_needed.add(asset.asset_id)
            continue
        if asset.asset_id in description_backfilled_assets:
            docs_needed.add(asset.asset_id)
            continue
        if asset.asset_id not in existing_docs:
            docs_needed.add(asset.asset_id)

    stage_start = time.monotonic()
    doc_rows: list[dict[str, Any]] = []
    doc_hash_map: dict[str, str] = {k: v for k, v in existing_docs.items()}

    for asset in assets:
        if asset.asset_id not in docs_needed:
            continue
        enrichment = _doc_enrichment_payload(
            enrichment_results.get(asset.asset_id) or existing_enriched.get(asset.asset_id)
        )
        doc_text, doc_hash = build_table_document(
            asset,
            columns_by_asset.get(asset.asset_id, []),
            enrichment,
            config.doc_column_limit,
        )
        doc_rows.append(
            {
                "doc_id": asset.asset_id,
                "asset_id": asset.asset_id,
                "doc_text": doc_text,
                "doc_hash": doc_hash,
                "updated_at": utc_now(),
            }
        )
        doc_hash_map[asset.asset_id] = doc_hash

    column_doc_rows: list[dict[str, Any]] = []
    for asset in assets:
        rebuild_columns = asset.asset_id in asset_changes or asset.asset_id in enrichment_changed
        asset_enrichment = _doc_enrichment_payload(
            enrichment_results.get(asset.asset_id) or existing_enriched.get(asset.asset_id)
        )
        for col in columns_by_asset.get(asset.asset_id, []):
            column_doc_id = f"{asset.asset_id}:{col.column_name}"
            existing_hash = existing_columns.get((col.asset_id, col.column_name))
            column_changed = existing_hash != col.column_hash
            existing_doc_hash = existing_column_docs.get(column_doc_id)
            needs_doc = rebuild_columns or column_changed or existing_doc_hash is None
            doc_text, doc_hash = build_column_document(asset, col, asset_enrichment)
            if not needs_doc and existing_doc_hash == doc_hash:
                continue
            column_doc_rows.append(
                {
                    "column_doc_id": column_doc_id,
                    "asset_id": col.asset_id,
                    "column_name": col.column_name,
                    "doc_text": doc_text,
                    "doc_hash": doc_hash,
                    "updated_at": utc_now(),
                }
            )
    _log_stage(
        "build_docs",
        stage_start,
        table_docs=len(doc_rows),
        column_docs=len(column_doc_rows),
    )

    asset_rows: list[dict[str, Any]] = []
    now = utc_now()
    for asset in assets:
        asset_rows.append(
            {
                "asset_id": asset.asset_id,
                "project_id": asset.project_id,
                "dataset_id": asset.dataset_id,
                "asset_name": asset.asset_name,
                "asset_type": asset.asset_type,
                "location": asset.location,
                "table_description": asset.table_description,
                "labels": json_dumps(asset.labels or {}),
                "partitioning": asset.partitioning,
                "clustering": asset.clustering,
                "created_time": asset.created_time,
                "last_modified_time": asset.last_modified_time,
                "table_meta_hash": asset.table_meta_hash,
                "schema_hash": asset.schema_hash,
                "doc_hash": doc_hash_map.get(asset.asset_id),
                "updated_at": now,
            }
        )

    column_rows: list[dict[str, Any]] = []
    for col in columns:
        column_rows.append(
            {
                "asset_id": col.asset_id,
                "column_name": col.column_name,
                "data_type": col.data_type,
                "is_nullable": col.is_nullable,
                "ordinal_position": col.ordinal_position,
                "column_description": col.column_description,
                "policy_tags": json_dumps(col.policy_tags or []),
                "column_hash": col.column_hash,
                "updated_at": now,
            }
        )

    enrichment_rows = list(enrichment_results.values())

    lineage_rows: list[dict[str, Any]] = []
    now = utc_now()
    for edge in lineage_edges:
        source_dataset, source_table = _split_asset_id(edge.source_asset_id)
        target_dataset, target_table = _split_asset_id(edge.target_asset_id)
        metadata = {
            "constraint_name": edge.constraint_name,
            "constraint_schema": edge.constraint_schema,
            "constraint_type": edge.constraint_type,
            "is_enforced": edge.is_enforced,
        }
        lineage_rows.append(
            {
                "edge_id": edge.edge_id,
                "source_dataset": source_dataset,
                "source_table": source_table,
                "source_column": edge.source_column,
                "target_dataset": target_dataset,
                "target_table": target_table,
                "target_column": edge.target_column,
                "relationship_type": "FOREIGN_KEY",
                "transformation_logic": "NOT_ENFORCED_CONSTRAINT",
                "confidence_score": 1.0,
                "discovery_method": "bigquery_constraints",
                "impact_weight": 1,
                "metadata": json_dumps(metadata),
                "created_at": now,
                "last_verified": now,
            }
        )

    stage_start = time.monotonic()
    merge_assets(client, config.project_id, config.metadata_dataset, asset_rows)
    merge_columns(client, config.project_id, config.metadata_dataset, column_rows)
    merge_documents(client, config.project_id, config.metadata_dataset, doc_rows)
    merge_column_documents(client, config.project_id, config.metadata_dataset, column_doc_rows)
    merge_enriched_assets(client, config.project_id, config.metadata_dataset, enrichment_rows)
    merge_lineage_edges(client, config.project_id, config.metadata_dataset, lineage_rows)
    _log_stage(
        "merge",
        stage_start,
        assets=len(asset_rows),
        columns=len(column_rows),
        docs=len(doc_rows),
        column_docs=len(column_doc_rows),
        enriched=len(enrichment_rows),
        lineage=len(lineage_rows),
    )

    if include_embeddings:
        stage_start = time.monotonic()
        run_embed(client, config)
        _log_stage("embed", stage_start)


def run_enrich_only(client: bigquery.Client, config) -> None:
    stage_start = time.monotonic()
    assets, columns = extract_all_assets_columns(
        client,
        config.project_id,
        config.datasets,
        config.ignore_datasets,
        config.bq_location,
        config.max_bq_workers,
    )
    _log_stage("extract", stage_start, assets=len(assets), columns=len(columns))
    asset_ids = [asset.asset_id for asset in assets]
    existing_assets = fetch_existing_assets(client, config.project_id, config.metadata_dataset, asset_ids)
    existing_enriched = fetch_existing_enriched(client, config.project_id, config.metadata_dataset, asset_ids)
    existing_docs = fetch_existing_docs(client, config.project_id, config.metadata_dataset, asset_ids)

    columns_by_asset = _build_column_map(columns)

    asset_changes: set[str] = set()
    for asset in assets:
        existing = existing_assets.get(asset.asset_id)
        if not existing:
            asset_changes.add(asset.asset_id)
            continue
        if (
            existing.get("schema_hash") != asset.schema_hash
            or existing.get("table_meta_hash") != asset.table_meta_hash
        ):
            asset_changes.add(asset.asset_id)

    enrich_targets = [
        asset
        for asset in assets
        if asset.dataset_id in {"silver_layer", "gold_layer"}
        and (
            asset.asset_id in asset_changes
            or asset.asset_id not in existing_enriched
            or existing_enriched.get(asset.asset_id, {}).get("enrichment_version")
            != config.enrichment_version
        )
    ]

    stage_start = time.monotonic()
    enrichment_results = enrich_assets(
        config.project_id,
        config.vertex_location,
        config.gemini_model,
        config.enrichment_version,
        enrich_targets,
        columns_by_asset,
        max_workers=config.max_enrich_workers,
    )
    _log_stage("enrich", stage_start, assets=len(enrichment_results))

    enrichment_changed: set[str] = set()
    for asset in enrich_targets:
        existing = existing_enriched.get(asset.asset_id)
        new_hash = enrichment_results.get(asset.asset_id, {}).get("enriched_hash")
        if not existing or existing.get("enriched_hash") != new_hash:
            enrichment_changed.add(asset.asset_id)

    stage_start = time.monotonic()
    doc_rows: list[dict[str, Any]] = []
    doc_hash_map: dict[str, str] = {k: v for k, v in existing_docs.items()}

    for asset in assets:
        if asset.asset_id not in enrichment_changed:
            continue
        enrichment = _doc_enrichment_payload(enrichment_results.get(asset.asset_id))
        doc_text, doc_hash = build_table_document(
            asset,
            columns_by_asset.get(asset.asset_id, []),
            enrichment,
            config.doc_column_limit,
        )
        doc_rows.append(
            {
                "doc_id": asset.asset_id,
                "asset_id": asset.asset_id,
                "doc_text": doc_text,
                "doc_hash": doc_hash,
                "updated_at": utc_now(),
            }
        )
        doc_hash_map[asset.asset_id] = doc_hash
    _log_stage("build_docs", stage_start, table_docs=len(doc_rows))

    asset_rows: list[dict[str, Any]] = []
    now = utc_now()
    for asset in assets:
        if asset.asset_id not in enrichment_changed:
            continue
        asset_rows.append(
            {
                "asset_id": asset.asset_id,
                "project_id": asset.project_id,
                "dataset_id": asset.dataset_id,
                "asset_name": asset.asset_name,
                "asset_type": asset.asset_type,
                "location": asset.location,
                "table_description": asset.table_description,
                "labels": json_dumps(asset.labels or {}),
                "partitioning": asset.partitioning,
                "clustering": asset.clustering,
                "created_time": asset.created_time,
                "last_modified_time": asset.last_modified_time,
                "table_meta_hash": asset.table_meta_hash,
                "schema_hash": asset.schema_hash,
                "doc_hash": doc_hash_map.get(asset.asset_id),
                "updated_at": now,
            }
        )

    stage_start = time.monotonic()
    merge_enriched_assets(client, config.project_id, config.metadata_dataset, list(enrichment_results.values()))
    merge_documents(client, config.project_id, config.metadata_dataset, doc_rows)
    merge_assets(client, config.project_id, config.metadata_dataset, asset_rows)
    _log_stage(
        "merge",
        stage_start,
        assets=len(asset_rows),
        docs=len(doc_rows),
        enriched=len(enrichment_results),
    )

    stage_start = time.monotonic()
    enriched_cols = enrich_column_descriptions(
        client,
        config.project_id,
        config.metadata_dataset,
        config.vertex_location,
        config.gemini_model,
        config.max_enrich_workers,
    )
    _log_stage("column_descriptions", stage_start, columns=enriched_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata RAG pipeline")
    parser.add_argument("--mode", required=True, choices=["backfill", "delta", "enrich", "embed"])
    args = parser.parse_args()

    config = load_config()
    client = bigquery.Client(project=config.project_id)

    ensure_tables(client, config.project_id, config.metadata_dataset, config.bq_location)

    if args.mode == "embed":
        run_embed(client, config)
        return

    if args.mode == "enrich":
        run_enrich_only(client, config)
        return

    include_embeddings = True
    run_pipeline(client, config, include_embeddings)


if __name__ == "__main__":
    main()
