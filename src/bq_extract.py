from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Tuple

from google.cloud import bigquery

from .utils import hash_text, json_dumps, log_event


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    project_id: str
    dataset_id: str
    asset_name: str
    asset_type: str
    location: str
    table_description: str
    labels: dict
    partitioning: str
    clustering: str
    created_time: datetime | None
    last_modified_time: datetime | None
    table_meta_hash: str
    schema_hash: str
    doc_hash: str | None
    view_query: str | None


@dataclass(frozen=True)
class ColumnRecord:
    asset_id: str
    column_name: str
    data_type: str
    is_nullable: bool
    ordinal_position: int
    column_description: str
    policy_tags: list[str]
    column_hash: str


@dataclass(frozen=True)
class LineageEdge:
    edge_id: str
    source_asset_id: str
    source_column: str | None
    target_asset_id: str | None
    target_column: str | None
    constraint_name: str
    constraint_schema: str
    constraint_type: str
    is_enforced: bool | None


def ensure_metadata_dataset(client: bigquery.Client, project_id: str, dataset_id: str, location: str) -> None:
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset, exists_ok=True)
        log_event("metadata_dataset_created", dataset=f"{project_id}.{dataset_id}")


def format_partitioning(table: bigquery.Table) -> str:
    if table.time_partitioning:
        tp = table.time_partitioning
        parts = [f"type={tp.type_}"]
        if tp.field:
            parts.append(f"field={tp.field}")
        if tp.expiration_ms:
            parts.append(f"expiration_ms={tp.expiration_ms}")
        if tp.require_partition_filter:
            parts.append("require_partition_filter=true")
        return "time:" + ",".join(parts)
    if table.range_partitioning:
        rp = table.range_partitioning
        range_part = rp.range_
        return (
            "range:field="
            f"{rp.field},start={range_part.start},end={range_part.end},interval={range_part.interval}"
        )
    return ""


def map_asset_type(table_type: str) -> str | None:
    if table_type in {"TABLE", "VIEW"}:
        return table_type
    if table_type == "MATERIALIZED_VIEW":
        return "VIEW"
    if table_type == "EXTERNAL":
        return "TABLE"
    return None


def flatten_schema(
    fields: Iterable[bigquery.SchemaField],
) -> list[Tuple[str, bigquery.SchemaField]]:
    ordered: list[Tuple[str, bigquery.SchemaField]] = []

    def visit(field: bigquery.SchemaField, prefix: str) -> None:
        name = f"{prefix}.{field.name}" if prefix else field.name
        ordered.append((name, field))
        for sub in field.fields or []:
            visit(sub, name)

    for field in fields:
        visit(field, "")
    return ordered


def compute_schema_hash(columns: list[ColumnRecord]) -> str:
    parts = [
        f"{col.column_name}|{col.data_type}|{col.is_nullable}|{col.column_description or ''}"
        for col in columns
    ]
    return hash_text("\n".join(parts))


def compute_table_meta_hash(
    description: str,
    labels: dict,
    partitioning: str,
    clustering: str,
) -> str:
    payload = {
        "description": description or "",
        "labels": labels or {},
        "partitioning": partitioning or "",
        "clustering": clustering or "",
    }
    return hash_text(json_dumps(payload))


def compute_column_hash(col: ColumnRecord) -> str:
    payload = {
        "column_name": col.column_name,
        "data_type": col.data_type,
        "is_nullable": col.is_nullable,
        "description": col.column_description or "",
        "policy_tags": sorted(col.policy_tags or []),
    }
    return hash_text(json_dumps(payload))


def _extract_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    location: str,
    table_item: bigquery.TableListItem,
) -> tuple[AssetRecord | None, list[ColumnRecord]]:
    table = client.get_table(table_item)
    asset_type = map_asset_type(table.table_type)
    if not asset_type:
        return None, []
    asset_id = f"{project_id}.{dataset_id}.{table.table_id}"
    labels = table.labels or {}
    partitioning = format_partitioning(table)
    clustering = ",".join(table.clustering_fields or [])
    description = table.description or ""
    view_query = table.view_query if asset_type == "VIEW" else None

    flattened = flatten_schema(table.schema)
    col_records: list[ColumnRecord] = []
    ordinal = 1
    for column_path, field in flattened:
        policy_tags = []
        if field.policy_tags and getattr(field.policy_tags, "names", None):
            policy_tags = list(field.policy_tags.names)
        col = ColumnRecord(
            asset_id=asset_id,
            column_name=column_path,
            data_type=field.field_type,
            is_nullable=(field.mode != "REQUIRED"),
            ordinal_position=ordinal,
            column_description=field.description or "",
            policy_tags=policy_tags,
            column_hash="",
        )
        col = ColumnRecord(
            **{**col.__dict__, "column_hash": compute_column_hash(col)}
        )
        col_records.append(col)
        ordinal += 1

    schema_hash = compute_schema_hash(col_records)
    table_meta_hash = compute_table_meta_hash(description, labels, partitioning, clustering)

    asset = AssetRecord(
        asset_id=asset_id,
        project_id=project_id,
        dataset_id=dataset_id,
        asset_name=table.table_id,
        asset_type=asset_type,
        location=location,
        table_description=description,
        labels=labels,
        partitioning=partitioning,
        clustering=clustering,
        created_time=table.created,
        last_modified_time=table.modified,
        table_meta_hash=table_meta_hash,
        schema_hash=schema_hash,
        doc_hash=None,
        view_query=view_query,
    )
    return asset, col_records


def extract_dataset_assets_columns(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    location: str,
    max_workers: int,
) -> tuple[list[AssetRecord], list[ColumnRecord]]:
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    tables = list(client.list_tables(dataset_ref))
    assets: list[AssetRecord] = []
    columns: list[ColumnRecord] = []

    log_event("dataset_tables_listed", dataset=dataset_id, table_count=len(tables))

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(tables)))) as executor:
        future_map = {
            executor.submit(
                _extract_table,
                client,
                project_id,
                dataset_id,
                location,
                table_item,
            ): table_item.table_id
            for table_item in tables
        }
        for future in as_completed(future_map):
            try:
                asset, col_records = future.result()
                if asset:
                    assets.append(asset)
                if col_records:
                    columns.extend(col_records)
            except Exception as exc:
                log_event("table_extract_failed", dataset=dataset_id, error=str(exc))

    log_event(
        "dataset_assets_extracted",
        dataset=dataset_id,
        assets=len(assets),
        columns=len(columns),
    )
    return assets, columns


def extract_all_assets_columns(
    client: bigquery.Client,
    project_id: str,
    dataset_ids: list[str],
    ignore_datasets: set[str],
    location: str,
    max_workers: int,
) -> tuple[list[AssetRecord], list[ColumnRecord]]:
    assets: list[AssetRecord] = []
    columns: list[ColumnRecord] = []
    targets = [dataset_id for dataset_id in dataset_ids if dataset_id not in ignore_datasets]

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(targets)))) as executor:
        future_map = {
            executor.submit(
                extract_dataset_assets_columns,
                client,
                project_id,
                dataset_id,
                location,
                max_workers,
            ): dataset_id
            for dataset_id in targets
        }
        for future in as_completed(future_map):
            dataset_id = future_map[future]
            try:
                dataset_assets, dataset_columns = future.result()
                assets.extend(dataset_assets)
                columns.extend(dataset_columns)
            except Exception as exc:
                log_event("dataset_extract_failed", dataset=dataset_id, error=str(exc))

    log_event(
        "all_assets_extracted",
        datasets=targets,
        assets=len(assets),
        columns=len(columns),
    )
    return assets, columns


def _union_constraints(project_id: str, dataset_ids: list[str], view: str, where: str | None = None) -> str:
    parts = []
    for dataset_id in dataset_ids:
        clause = f"SELECT * FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.{view}`"
        if where:
            clause += f" WHERE {where}"
        parts.append(clause)
    return " UNION ALL ".join(parts) if parts else "SELECT NULL WHERE FALSE"


def extract_lineage_edges(
    client: bigquery.Client,
    project_id: str,
    dataset_ids: list[str],
) -> list[LineageEdge]:
    if not dataset_ids:
        return []

    fk_constraints = _union_constraints(
        project_id, dataset_ids, "TABLE_CONSTRAINTS", "constraint_type = 'FOREIGN KEY'"
    )
    fk_source_raw = _union_constraints(project_id, dataset_ids, "KEY_COLUMN_USAGE")
    fk_target_raw = _union_constraints(project_id, dataset_ids, "CONSTRAINT_COLUMN_USAGE")

    query = f"""
    WITH fk_constraints AS (
      {fk_constraints}
    ),
    fk_source_raw AS (
      {fk_source_raw}
    ),
    fk_target_raw AS (
      {fk_target_raw}
    ),
    fk_source AS (
      SELECT
        constraint_name,
        constraint_schema,
        table_catalog,
        table_schema,
        table_name,
        column_name,
        ROW_NUMBER() OVER (
          PARTITION BY constraint_name, constraint_schema, table_schema, table_name
          ORDER BY ordinal_position
        ) AS rn
      FROM fk_source_raw
    ),
    fk_target AS (
      SELECT
        constraint_name,
        constraint_schema,
        table_catalog,
        table_schema,
        table_name,
        column_name,
        ROW_NUMBER() OVER (
          PARTITION BY constraint_name, constraint_schema, table_schema, table_name
          ORDER BY column_name
        ) AS rn
      FROM fk_target_raw
    )
    SELECT
      CONCAT(fk.table_catalog, '.', fk.table_schema, '.', fk.table_name) AS source_asset_id,
      fs.column_name AS source_column,
      CONCAT(ft.table_catalog, '.', ft.table_schema, '.', ft.table_name) AS target_asset_id,
      ft.column_name AS target_column,
      fk.constraint_name,
      fk.constraint_schema,
      fk.constraint_type,
      fk.enforced
    FROM fk_constraints fk
    LEFT JOIN fk_source fs
      ON fk.constraint_name = fs.constraint_name
     AND fk.constraint_schema = fs.constraint_schema
     AND fk.table_schema = fs.table_schema
     AND fk.table_name = fs.table_name
    LEFT JOIN fk_target ft
      ON fk.constraint_name = ft.constraint_name
     AND fk.constraint_schema = ft.constraint_schema
     AND fs.rn = ft.rn
    ORDER BY source_asset_id, source_column
    """

    rows = client.query(query).result()
    edges: list[LineageEdge] = []
    for row in rows:
        source_asset_id = row.get("source_asset_id")
        source_column = row.get("source_column")
        target_asset_id = row.get("target_asset_id")
        target_column = row.get("target_column")
        constraint_name = row.get("constraint_name")
        constraint_schema = row.get("constraint_schema")
        constraint_type = row.get("constraint_type")
        enforced_raw = row.get("enforced")
        is_enforced = None
        if isinstance(enforced_raw, str):
            is_enforced = enforced_raw.upper() in {"YES", "TRUE"}
        elif enforced_raw is not None:
            is_enforced = bool(enforced_raw)

        edge_key = "|".join(
            [
                source_asset_id or "",
                source_column or "",
                target_asset_id or "",
                target_column or "",
                constraint_name or "",
            ]
        )
        edge_id = hash_text(edge_key)
        edges.append(
            LineageEdge(
                edge_id=edge_id,
                source_asset_id=source_asset_id,
                source_column=source_column,
                target_asset_id=target_asset_id,
                target_column=target_column,
                constraint_name=constraint_name,
                constraint_schema=constraint_schema,
                constraint_type=constraint_type,
                is_enforced=is_enforced,
            )
        )

    log_event("lineage_edges_extracted", edges=len(edges))
    return edges
