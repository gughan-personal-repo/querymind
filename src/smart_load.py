from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from google.cloud import bigquery

from .utils import log_event


def _create_staging_table(client: bigquery.Client, target_table: str) -> str:
    staging_table = f"{target_table}_staging_{uuid.uuid4().hex}"
    client.query(
        f"CREATE TABLE `{staging_table}` AS SELECT * FROM `{target_table}` WHERE 1=0"
    ).result()
    return staging_table


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _load_rows(client: bigquery.Client, table_id: str, rows: list[dict[str, Any]]) -> None:
    prepared = [{key: _serialize_value(val) for key, val in row.items()} for row in rows]
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    load_job = client.load_table_from_json(prepared, table_id, job_config=job_config)
    load_job.result()


def _drop_table(client: bigquery.Client, table_id: str) -> None:
    client.delete_table(table_id, not_found_ok=True)


def merge_assets(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.assets"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.asset_id = S.asset_id
        WHEN MATCHED AND (
            T.schema_hash != S.schema_hash OR
            T.table_meta_hash != S.table_meta_hash OR
            IFNULL(T.doc_hash, '') != IFNULL(S.doc_hash, '') OR
            IFNULL(T.table_description, '') != IFNULL(S.table_description, '') OR
            IFNULL(T.partitioning, '') != IFNULL(S.partitioning, '') OR
            IFNULL(T.clustering, '') != IFNULL(S.clustering, '') OR
            IFNULL(TO_JSON_STRING(T.labels), '') != IFNULL(TO_JSON_STRING(PARSE_JSON(S.labels)), '')
        ) THEN
          UPDATE SET
            project_id = S.project_id,
            dataset_id = S.dataset_id,
            asset_name = S.asset_name,
            asset_type = S.asset_type,
            location = S.location,
            table_description = S.table_description,
            labels = PARSE_JSON(S.labels),
            partitioning = S.partitioning,
            clustering = S.clustering,
            created_time = S.created_time,
            last_modified_time = S.last_modified_time,
            table_meta_hash = S.table_meta_hash,
            schema_hash = S.schema_hash,
            doc_hash = S.doc_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (
            asset_id,
            project_id,
            dataset_id,
            asset_name,
            asset_type,
            location,
            table_description,
            labels,
            partitioning,
            clustering,
            created_time,
            last_modified_time,
            table_meta_hash,
            schema_hash,
            doc_hash,
            updated_at
          )
          VALUES (
            S.asset_id,
            S.project_id,
            S.dataset_id,
            S.asset_name,
            S.asset_type,
            S.location,
            S.table_description,
            PARSE_JSON(S.labels),
            S.partitioning,
            S.clustering,
            S.created_time,
            S.last_modified_time,
            S.table_meta_hash,
            S.schema_hash,
            S.doc_hash,
            S.updated_at
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_assets_completed", rows=len(rows))


def merge_columns(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.columns"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.asset_id = S.asset_id AND T.column_name = S.column_name
        WHEN MATCHED AND (
            T.column_hash != S.column_hash OR
            IFNULL(T.column_description, '') != IFNULL(S.column_description, '') OR
            IFNULL(T.policy_tags, '') != IFNULL(S.policy_tags, '') OR
            T.data_type != S.data_type OR
            T.is_nullable != S.is_nullable OR
            T.ordinal_position != S.ordinal_position
        ) THEN
          UPDATE SET
            data_type = S.data_type,
            is_nullable = S.is_nullable,
            ordinal_position = S.ordinal_position,
            column_description = S.column_description,
            policy_tags = S.policy_tags,
            column_hash = S.column_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (
            asset_id,
            column_name,
            data_type,
            is_nullable,
            ordinal_position,
            column_description,
            policy_tags,
            column_hash,
            updated_at
          )
          VALUES (
            S.asset_id,
            S.column_name,
            S.data_type,
            S.is_nullable,
            S.ordinal_position,
            S.column_description,
            S.policy_tags,
            S.column_hash,
            S.updated_at
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_columns_completed", rows=len(rows))


def merge_documents(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.documents"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.doc_id = S.doc_id
        WHEN MATCHED AND (T.doc_hash != S.doc_hash) THEN
          UPDATE SET
            doc_text = S.doc_text,
            doc_hash = S.doc_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (doc_id, asset_id, doc_text, doc_hash, updated_at)
          VALUES (S.doc_id, S.asset_id, S.doc_text, S.doc_hash, S.updated_at)
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_documents_completed", rows=len(rows))


def merge_column_documents(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.column_documents"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.column_doc_id = S.column_doc_id
        WHEN MATCHED AND (T.doc_hash != S.doc_hash) THEN
          UPDATE SET
            doc_text = S.doc_text,
            doc_hash = S.doc_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (column_doc_id, asset_id, column_name, doc_text, doc_hash, updated_at)
          VALUES (S.column_doc_id, S.asset_id, S.column_name, S.doc_text, S.doc_hash, S.updated_at)
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_column_documents_completed", rows=len(rows))


def merge_enriched_assets(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.enriched_assets"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING (
          SELECT
            asset_id,
            enrichment_version,
            concepts,
            grain,
            (SELECT ARRAY(SELECT AS STRUCT s.term, s.maps_to FROM UNNEST(synonyms) AS s)) AS synonyms,
            (
              SELECT ARRAY(
                SELECT AS STRUCT j.other_asset_id, j.keys, j.confidence, j.evidence
                FROM UNNEST(join_hints) AS j
              )
            ) AS join_hints,
            pii_flags,
            notes,
            enriched_hash,
            updated_at
          FROM `{staging}`
        ) S
        ON T.asset_id = S.asset_id
        WHEN MATCHED AND (T.enriched_hash != S.enriched_hash OR T.enrichment_version != S.enrichment_version) THEN
          UPDATE SET
            enrichment_version = S.enrichment_version,
            concepts = S.concepts,
            grain = S.grain,
            synonyms = S.synonyms,
            join_hints = S.join_hints,
            pii_flags = S.pii_flags,
            notes = S.notes,
            enriched_hash = S.enriched_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (
            asset_id,
            enrichment_version,
            concepts,
            grain,
            synonyms,
            join_hints,
            pii_flags,
            notes,
            enriched_hash,
            updated_at
          )
          VALUES (
            S.asset_id,
            S.enrichment_version,
            S.concepts,
            S.grain,
            S.synonyms,
            S.join_hints,
            S.pii_flags,
            S.notes,
            S.enriched_hash,
            S.updated_at
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_enriched_assets_completed", rows=len(rows))


def merge_embeddings(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.embeddings"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.doc_id = S.doc_id
        WHEN MATCHED AND (T.doc_hash != S.doc_hash OR T.embedding_model != S.embedding_model) THEN
          UPDATE SET
            asset_id = S.asset_id,
            embedding = S.embedding,
            embedding_model = S.embedding_model,
            embedding_dim = S.embedding_dim,
            doc_hash = S.doc_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (
            doc_id,
            asset_id,
            embedding,
            embedding_model,
            embedding_dim,
            doc_hash,
            updated_at
          )
          VALUES (
            S.doc_id,
            S.asset_id,
            S.embedding,
            S.embedding_model,
            S.embedding_dim,
            S.doc_hash,
            S.updated_at
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_embeddings_completed", rows=len(rows))


def merge_column_embeddings(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.column_embeddings"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.column_doc_id = S.column_doc_id
        WHEN MATCHED AND (T.doc_hash != S.doc_hash OR T.embedding_model != S.embedding_model) THEN
          UPDATE SET
            asset_id = S.asset_id,
            column_name = S.column_name,
            embedding = S.embedding,
            embedding_model = S.embedding_model,
            embedding_dim = S.embedding_dim,
            doc_hash = S.doc_hash,
            updated_at = S.updated_at
        WHEN NOT MATCHED THEN
          INSERT (
            column_doc_id,
            asset_id,
            column_name,
            embedding,
            embedding_model,
            embedding_dim,
            doc_hash,
            updated_at
          )
          VALUES (
            S.column_doc_id,
            S.asset_id,
            S.column_name,
            S.embedding,
            S.embedding_model,
            S.embedding_dim,
            S.doc_hash,
            S.updated_at
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_column_embeddings_completed", rows=len(rows))


def merge_lineage_edges(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    target = f"{project_id}.{metadata_dataset}.lineage_edges"
    staging = _create_staging_table(client, target)
    try:
        _load_rows(client, staging, rows)
        sql = f"""
        MERGE `{target}` T
        USING `{staging}` S
        ON T.edge_id = S.edge_id
        WHEN MATCHED AND (
            IFNULL(T.source_dataset, '') != IFNULL(S.source_dataset, '') OR
            IFNULL(T.source_table, '') != IFNULL(S.source_table, '') OR
            IFNULL(T.source_column, '') != IFNULL(S.source_column, '') OR
            IFNULL(T.target_dataset, '') != IFNULL(S.target_dataset, '') OR
            IFNULL(T.target_table, '') != IFNULL(S.target_table, '') OR
            IFNULL(T.target_column, '') != IFNULL(S.target_column, '') OR
            IFNULL(T.relationship_type, '') != IFNULL(S.relationship_type, '') OR
            IFNULL(T.transformation_logic, '') != IFNULL(S.transformation_logic, '') OR
            IFNULL(T.confidence_score, 0) != IFNULL(S.confidence_score, 0) OR
            IFNULL(T.discovery_method, '') != IFNULL(S.discovery_method, '') OR
            IFNULL(T.impact_weight, 0) != IFNULL(S.impact_weight, 0) OR
            IFNULL(TO_JSON_STRING(T.metadata), '') != IFNULL(TO_JSON_STRING(PARSE_JSON(S.metadata)), '')
        ) THEN
          UPDATE SET
            source_dataset = S.source_dataset,
            source_table = S.source_table,
            source_column = S.source_column,
            target_dataset = S.target_dataset,
            target_table = S.target_table,
            target_column = S.target_column,
            relationship_type = S.relationship_type,
            transformation_logic = S.transformation_logic,
            confidence_score = S.confidence_score,
            discovery_method = S.discovery_method,
            impact_weight = S.impact_weight,
            metadata = PARSE_JSON(S.metadata),
            last_verified = S.last_verified
        WHEN NOT MATCHED THEN
          INSERT (
            edge_id,
            source_dataset,
            source_table,
            source_column,
            target_dataset,
            target_table,
            target_column,
            relationship_type,
            transformation_logic,
            confidence_score,
            discovery_method,
            impact_weight,
            metadata,
            created_at,
            last_verified
          )
          VALUES (
            S.edge_id,
            S.source_dataset,
            S.source_table,
            S.source_column,
            S.target_dataset,
            S.target_table,
            S.target_column,
            S.relationship_type,
            S.transformation_logic,
            S.confidence_score,
            S.discovery_method,
            S.impact_weight,
            PARSE_JSON(S.metadata),
            S.created_at,
            S.last_verified
          )
        """
        client.query(sql).result()
    finally:
        _drop_table(client, staging)

    log_event("merge_lineage_edges_completed", rows=len(rows))
