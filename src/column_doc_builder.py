from __future__ import annotations

from .bq_extract import AssetRecord, ColumnRecord
from .utils import hash_text, json_dumps


def build_column_document(
    asset: AssetRecord,
    column: ColumnRecord,
    enrichment: dict | None,
) -> tuple[str, str]:
    lines: list[str] = []
    lines.append(f"asset_id: {asset.asset_id}")
    lines.append(f"dataset: {asset.dataset_id}")
    lines.append(f"table: {asset.asset_name}")
    lines.append(f"table_description: {asset.table_description or 'unknown'}")
    lines.append(f"column: {column.column_name}")
    lines.append(f"data_type: {column.data_type}")
    lines.append(f"nullable: {column.is_nullable}")
    lines.append(f"column_description: {column.column_description or 'unknown'}")

    if enrichment:
        concepts = enrichment.get("concepts") if isinstance(enrichment, dict) else None
        lines.append(f"parent_concepts: {json_dumps(concepts or ['unknown'])}")
        grain = enrichment.get("grain") if isinstance(enrichment, dict) else None
        lines.append(f"parent_grain: {grain or 'unknown'}")

    doc_text = "\n".join(lines)
    doc_hash = hash_text(doc_text)
    return doc_text, doc_hash
