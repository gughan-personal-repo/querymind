from __future__ import annotations

import json
from typing import Iterable

from .bq_extract import AssetRecord, ColumnRecord
from .utils import hash_text, json_dumps


MAX_VIEW_SQL_CHARS = 2000


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def build_table_document(
    asset: AssetRecord,
    columns: Iterable[ColumnRecord],
    enrichment: dict | None,
    column_limit: int,
) -> tuple[str, str]:
    ordered_columns = sorted(columns, key=lambda col: col.ordinal_position)
    lines: list[str] = []
    lines.append(f"asset_id: {asset.asset_id}")
    lines.append(f"asset_type: {asset.asset_type}")
    lines.append(f"location: {asset.location}")
    lines.append(f"description: {asset.table_description or 'unknown'}")
    labels = asset.labels or {}
    lines.append(f"labels: {json_dumps(labels)}")
    lines.append(f"partitioning: {asset.partitioning or 'none'}")
    lines.append(f"clustering: {asset.clustering or 'none'}")
    lines.append("columns:")

    shown = ordered_columns[:column_limit]
    for col in shown:
        nullable = "NULLABLE" if col.is_nullable else "REQUIRED"
        desc = col.column_description or "unknown"
        lines.append(f"- {col.column_name} | {col.data_type} | {nullable} | {desc}")

    remaining = len(ordered_columns) - len(shown)
    if remaining > 0:
        lines.append(f"... {remaining} more columns not shown")

    if asset.asset_type == "VIEW" and asset.view_query:
        lines.append("view_sql:")
        lines.append(_truncate(asset.view_query, MAX_VIEW_SQL_CHARS))

    if enrichment:
        lines.append("enrichment:")
        lines.append(json_dumps(enrichment))

    doc_text = "\n".join(lines)
    doc_hash = hash_text(doc_text)
    return doc_text, doc_hash
