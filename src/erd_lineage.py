from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader
from google.cloud import bigquery

from .config import PipelineConfig
from .smart_load import merge_lineage_edges
from .utils import hash_text, json_dumps, utc_now, log_event


@dataclass(frozen=True)
class ERDTable:
    name: str
    pk: set[str]
    fk: set[str]


TABLE_LINE = re.compile(r"^[A-Z][A-Za-z0-9_]+$")
PK_PREFIX = re.compile(r"^(PK|PK,|PK\s|PK,FK|PK,FK\d+)")
MODIFIED_PREFIX = re.compile(r"^ModifiedDate([A-Z][A-Za-z0-9_]+)$")
CONSTRAINT_PREFIX = re.compile(
    r"^(?P<prefix>(?:PK|FK\d+|U\d+)(?:,(?:PK|FK\d+|U\d+))*)\s*(?P<rest>.+)$"
)


def _extract_text(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return [line.strip() for line in text.splitlines() if line.strip()]


def _table_name_at(lines: list[str], idx: int) -> str | None:
    line = lines[idx]
    modified = MODIFIED_PREFIX.match(line)
    if modified:
        return modified.group(1)
    if not TABLE_LINE.match(line):
        return None
    if idx + 1 < len(lines) and PK_PREFIX.match(lines[idx + 1]):
        return line
    return None


def _parse_column_line(line: str) -> tuple[str, str] | None:
    match = CONSTRAINT_PREFIX.match(line)
    if not match:
        return None
    prefix = match.group("prefix")
    rest = match.group("rest").strip()
    if not rest:
        return None
    column = rest.split()[0]
    return prefix, column


def parse_erd_tables(lines: list[str]) -> dict[str, ERDTable]:
    tables: dict[str, ERDTable] = {}
    current: ERDTable | None = None

    for idx, line in enumerate(lines):
        name = _table_name_at(lines, idx)
        if name:
            current = tables.get(name) or ERDTable(name=name, pk=set(), fk=set())
            tables[name] = current
            continue
        if not current:
            continue
        parsed = _parse_column_line(line)
        if not parsed:
            continue
        prefix, column = parsed
        column_norm = column.lower()
        if "PK" in prefix:
            current.pk.add(column_norm)
        if "FK" in prefix:
            current.fk.add(column_norm)
    return tables


def _resolve_fk_target(
    fk_column: str,
    pk_map: dict[str, set[str]],
    allowed_tables: set[str],
) -> tuple[str | None, str | None, float]:
    exact_candidates: list[tuple[str, str]] = []
    suffix_candidates: list[tuple[str, str]] = []
    for table, pks in pk_map.items():
        if table not in allowed_tables:
            continue
        for pk in pks:
            if fk_column == pk:
                exact_candidates.append((table, pk))
            elif fk_column.endswith(pk):
                suffix_candidates.append((table, pk))

    candidates = exact_candidates or suffix_candidates
    if not candidates:
        return None, None, 0.0
    if len(candidates) == 1:
        table, pk = candidates[0]
        return table, pk, 0.9 if exact_candidates else 0.7

    base = fk_column[:-2] if fk_column.lower().endswith("id") else fk_column
    scored: list[tuple[int, str, str]] = []
    for table, pk in candidates:
        score = 0
        if table.lower() == base.lower():
            score += 3
        if table.lower().startswith(base.lower()):
            score += 2
        if base.lower() in table.lower():
            score += 1
        scored.append((score, table, pk))
    scored.sort(reverse=True)
    return scored[0][1], scored[0][2], 0.6 if scored[0][0] else 0.5


def _fetch_sales_tables(
    client: bigquery.Client, config: PipelineConfig, dataset_id: str, prefix: str
) -> set[str]:
    sql = f"""
    SELECT asset_name
    FROM `{config.project_id}.{config.metadata_dataset}.assets`
    WHERE dataset_id = @dataset_id AND asset_name LIKE @prefix
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("dataset_id", "STRING", dataset_id),
                bigquery.ScalarQueryParameter("prefix", "STRING", f"{prefix}%"),
            ]
        ),
    )
    return {row["asset_name"] for row in job.result()}


def _fetch_columns(
    client: bigquery.Client,
    config: PipelineConfig,
    dataset_id: str,
    tables: Iterable[str],
) -> dict[str, set[str]]:
    asset_ids = [f"{config.project_id}.{dataset_id}.{t}" for t in tables]
    if not asset_ids:
        return {}
    sql = f"""
    SELECT asset_id, column_name
    FROM `{config.project_id}.{config.metadata_dataset}.columns`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)]
        ),
    )
    result: dict[str, set[str]] = {}
    for row in job.result():
        result.setdefault(row["asset_id"], set()).add(row["column_name"])
    return result


def update_sales_lineage_from_erd(
    pdf_path: Path,
    config: PipelineConfig,
    dataset_id: str = "silver_layer",
    table_prefix: str = "sales__",
) -> None:
    lines = _extract_text(pdf_path)
    tables = parse_erd_tables(lines)

    client = bigquery.Client(project=config.project_id)
    sales_tables = _fetch_sales_tables(client, config, dataset_id, table_prefix)
    if not sales_tables:
        raise ValueError(f"No tables found with prefix {table_prefix} in {dataset_id}.")

    allowed_erd_tables = {
        name for name in tables if f"{table_prefix}{name.lower()}" in sales_tables
    }
    pk_map = {name: tables[name].pk for name in allowed_erd_tables}

    column_map = _fetch_columns(client, config, dataset_id, sales_tables)

    now = utc_now()
    rows: list[dict] = []
    edges_added = 0

    for src_table in sorted(allowed_erd_tables):
        src_bq_table = f"{table_prefix}{src_table.lower()}"
        src_asset_id = f"{config.project_id}.{dataset_id}.{src_bq_table}"
        src_columns = column_map.get(src_asset_id, set())
        for fk_col in sorted(tables[src_table].fk):
            target_table, target_pk, confidence = _resolve_fk_target(
                fk_col, pk_map, allowed_erd_tables
            )
            if not target_table or not target_pk:
                continue
            tgt_bq_table = f"{table_prefix}{target_table.lower()}"
            tgt_asset_id = f"{config.project_id}.{dataset_id}.{tgt_bq_table}"
            tgt_columns = column_map.get(tgt_asset_id, set())
            if fk_col not in src_columns or target_pk not in tgt_columns:
                continue

            edge_key = "|".join(
                [dataset_id, src_bq_table, fk_col, dataset_id, tgt_bq_table, fk_col, "ERD_FK"]
            )
            edge_id = hash_text(edge_key)
            rows.append(
                {
                    "edge_id": edge_id,
                    "source_dataset": dataset_id,
                    "source_table": src_bq_table,
                    "source_column": fk_col,
                    "target_dataset": dataset_id,
                    "target_table": tgt_bq_table,
                    "target_column": target_pk,
                    "relationship_type": "FOREIGN_KEY",
                    "transformation_logic": "ERD_FK",
                    "confidence_score": confidence,
                    "discovery_method": "erd_pdf",
                    "impact_weight": 1,
                    "metadata": json_dumps(
                        {
                            "source": pdf_path.name,
                            "heuristic": "pk_name_match",
                            "er_table": src_table,
                            "er_target_table": target_table,
                        }
                    ),
                    "created_at": now,
                    "last_verified": now,
                }
            )
            edges_added += 1

    if rows:
        merge_lineage_edges(client, config.project_id, config.metadata_dataset, rows)
    log_event("erd_lineage_merged", edges=edges_added, dataset=dataset_id, prefix=table_prefix)
