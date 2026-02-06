from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

from google import genai
from google.cloud import bigquery
from google.genai import types
from google.api_core.exceptions import BadRequest, NotFound

from .config import PipelineConfig
from .utils import json_dumps, log_event


@dataclass(frozen=True)
class ColumnNode:
    dataset_id: str
    table_name: str
    column_name: str
    node_type_override: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset_id, self.table_name, self.column_name)

    @property
    def node_id(self) -> str:
        return f"{self.dataset_id}.{self.table_name}.{self.column_name}"

    @property
    def node_type(self) -> str:
        if self.node_type_override:
            return self.node_type_override
        return "table" if self.column_name == "*" else "column"


@dataclass(frozen=True)
class Edge:
    source: ColumnNode
    target: ColumnNode
    relationship_type: str
    impact_weight: int
    discovery_method: str | None
    transformation_logic: str | None
    metadata: dict[str, Any] | None
    bidirectional: bool = False

    @property
    def edge_id(self) -> str:
        key = (
            self.source.node_id,
            self.target.node_id,
            self.relationship_type,
            self.discovery_method or "",
        )
        return "|".join(key)


def _get_genai_client(config: PipelineConfig) -> genai.Client:
    return genai.Client(vertexai=True, project=config.project_id, location=config.vertex_location)


def _safe_query(
    client: bigquery.Client,
    sql: str,
    params: list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter],
    *,
    missing_table_ok: bool = False,
) -> list[dict[str, Any]]:
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        rows = client.query(sql, job_config=job_config).result()
        return [dict(row.items()) for row in rows]
    except (NotFound, BadRequest) as exc:
        if missing_table_ok and "Not found" in str(exc):
            log_event("ripple_optional_table_missing", error=str(exc))
            return []
        raise


def _node_from_row(dataset_id: str | None, table_name: str | None, column_name: str | None) -> ColumnNode | None:
    if not dataset_id or not table_name:
        return None
    column = column_name or "*"
    return ColumnNode(dataset_id=dataset_id, table_name=table_name, column_name=column)


def _edge_from_row(row: dict[str, Any]) -> Edge | None:
    source = _node_from_row(row.get("source_dataset"), row.get("source_table"), row.get("source_column"))
    target = _node_from_row(row.get("target_dataset"), row.get("target_table"), row.get("target_column"))
    if not source or not target:
        return None
    relationship_type = row.get("relationship_type") or "UNKNOWN"
    impact_weight = int(row.get("impact_weight") or 1)
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw": metadata}
    return Edge(
        source=source,
        target=target,
        relationship_type=relationship_type,
        impact_weight=impact_weight,
        discovery_method=row.get("discovery_method"),
        transformation_logic=row.get("transformation_logic"),
        metadata=metadata if isinstance(metadata, dict) else None,
        bidirectional=relationship_type == "FOREIGN_KEY",
    )


def _fetch_lineage_edges(
    client: bigquery.Client,
    config: PipelineConfig,
    table_keys: list[str],
) -> list[Edge]:
    if not table_keys:
        return []
    sql = f"""
    SELECT
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
      metadata
    FROM `{config.project_id}.{config.metadata_dataset}.lineage_edges`
    WHERE CONCAT(source_dataset, '.', source_table) IN UNNEST(@table_keys)
       OR CONCAT(target_dataset, '.', target_table) IN UNNEST(@table_keys)
    """
    rows = _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("table_keys", "STRING", table_keys)],
    )
    edges: list[Edge] = []
    for row in rows:
        edge = _edge_from_row(row)
        if edge:
            edges.append(edge)
    return edges


def _fetch_join_hint_edges(
    client: bigquery.Client,
    config: PipelineConfig,
    asset_ids: list[str],
) -> list[Edge]:
    if not asset_ids:
        return []
    sql = f"""
    SELECT asset_id, join_hints
    FROM `{config.project_id}.{config.metadata_dataset}.enriched_assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    rows = _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)],
        missing_table_ok=True,
    )
    edges: list[Edge] = []
    for row in rows:
        asset_id = row.get("asset_id")
        if not asset_id:
            continue
        parts = asset_id.split(".")
        if len(parts) < 3:
            continue
        dataset_id, table_name = parts[1], parts[2]
        join_hints = row.get("join_hints") or []
        if not isinstance(join_hints, list):
            continue
        for hint in join_hints:
            if not isinstance(hint, dict):
                continue
            other_asset_id = hint.get("other_asset_id")
            keys = hint.get("keys") or []
            if not other_asset_id or not isinstance(keys, list):
                continue
            other_parts = str(other_asset_id).split(".")
            if len(other_parts) < 3:
                continue
            other_dataset, other_table = other_parts[1], other_parts[2]
            for key in keys:
                if not key:
                    continue
                source = ColumnNode(dataset_id=dataset_id, table_name=table_name, column_name=key)
                target = ColumnNode(
                    dataset_id=other_dataset,
                    table_name=other_table,
                    column_name=key,
                )
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        relationship_type="JOIN_HINT",
                        impact_weight=1,
                        discovery_method="gemini_enrichment",
                        transformation_logic=None,
                        metadata={"evidence": hint.get("evidence", "unknown")},
                        bidirectional=True,
                    )
                )
    return edges


def _fetch_query_pattern_edges(
    client: bigquery.Client,
    config: PipelineConfig,
    column_ids: list[str],
) -> tuple[list[Edge], list[dict[str, Any]]]:
    if not column_ids:
        return [], []
    sql = f"""
    SELECT
      pattern_id,
      dataset_id,
      table_name,
      column_name,
      query_text,
      co_columns,
      frequency,
      last_seen
    FROM `{config.project_id}.{config.metadata_dataset}.query_patterns`
    WHERE CONCAT(dataset_id, '.', table_name, '.', column_name) IN UNNEST(@column_ids)
    """
    rows = _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("column_ids", "STRING", column_ids)],
        missing_table_ok=True,
    )
    edges: list[Edge] = []
    patterns: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = row.get("dataset_id")
        table_name = row.get("table_name")
        column_name = row.get("column_name")
        if not (dataset_id and table_name and column_name):
            continue
        source = ColumnNode(dataset_id=dataset_id, table_name=table_name, column_name=column_name)
        co_columns = row.get("co_columns") or []
        if isinstance(co_columns, list):
            for co in co_columns:
                if not co or not isinstance(co, str):
                    continue
                parts = co.split(".")
                if len(parts) < 3:
                    continue
                co_dataset, co_table, co_column = parts[0], parts[1], ".".join(parts[2:])
                target = ColumnNode(dataset_id=co_dataset, table_name=co_table, column_name=co_column)
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        relationship_type="QUERY_PATTERN",
                        impact_weight=1,
                        discovery_method="query_patterns",
                        transformation_logic=None,
                        metadata={"pattern_id": row.get("pattern_id")},
                        bidirectional=True,
                    )
                )
        patterns.append(
            {
                "pattern_id": row.get("pattern_id"),
                "dataset_id": dataset_id,
                "table_name": table_name,
                "column_name": column_name,
                "query_text": row.get("query_text"),
                "frequency": row.get("frequency") or 0,
                "last_seen": row.get("last_seen"),
            }
        )
    return edges, patterns


def _fetch_prompt_templates(
    client: bigquery.Client,
    config: PipelineConfig,
    column_ids: list[str],
) -> list[dict[str, Any]]:
    if not column_ids:
        return []
    sql = f"""
    SELECT template_id, template_name, template_type, version, updated_at
    FROM `{config.project_id}.{config.metadata_dataset}.prompt_templates`
    WHERE EXISTS (
      SELECT 1
      FROM UNNEST(used_columns) AS col
      WHERE col IN UNNEST(@column_ids)
    )
    """
    return _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("column_ids", "STRING", column_ids)],
        missing_table_ok=True,
    )


def _build_graph(
    edges: Iterable[Edge],
) -> tuple[dict[tuple[str, str, str], ColumnNode], dict[tuple[str, str, str], list[Edge]], dict[tuple[str, str, str], list[Edge]]]:
    nodes: dict[tuple[str, str, str], ColumnNode] = {}
    adjacency: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    reverse: dict[tuple[str, str, str], list[Edge]] = defaultdict(list)
    seen_edges: set[str] = set()

    for edge in edges:
        if edge.edge_id in seen_edges:
            continue
        seen_edges.add(edge.edge_id)
        nodes[edge.source.key] = edge.source
        nodes[edge.target.key] = edge.target
        adjacency[edge.source.key].append(edge)
        reverse[edge.target.key].append(edge)
        if edge.bidirectional:
            adjacency[edge.target.key].append(
                Edge(
                    source=edge.target,
                    target=edge.source,
                    relationship_type=edge.relationship_type,
                    impact_weight=edge.impact_weight,
                    discovery_method=edge.discovery_method,
                    transformation_logic=edge.transformation_logic,
                    metadata=edge.metadata,
                    bidirectional=True,
                )
            )
            reverse[edge.source.key].append(
                Edge(
                    source=edge.target,
                    target=edge.source,
                    relationship_type=edge.relationship_type,
                    impact_weight=edge.impact_weight,
                    discovery_method=edge.discovery_method,
                    transformation_logic=edge.transformation_logic,
                    metadata=edge.metadata,
                    bidirectional=True,
                )
            )
    return nodes, adjacency, reverse


def _bfs(
    start: ColumnNode,
    adjacency: dict[tuple[str, str, str], list[Edge]],
    max_hops: int,
    *,
    reverse: bool = False,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    visited: dict[tuple[str, str, str], dict[str, Any]] = {start.key: {"hop": 0, "via": None}}
    queue: deque[tuple[ColumnNode, int]] = deque([(start, 0)])
    while queue:
        node, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for edge in adjacency.get(node.key, []):
            next_node = edge.source if reverse else edge.target
            if next_node.key in visited:
                continue
            visited[next_node.key] = {"hop": hop + 1, "via": edge}
            queue.append((next_node, hop + 1))
    return visited


def _bfs_tables(
    start: str,
    adjacency: dict[str, set[str]],
    max_hops: int,
) -> dict[str, dict[str, Any]]:
    visited: dict[str, dict[str, Any]] = {start: {"hop": 0}}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, hop = queue.popleft()
        if hop >= max_hops:
            continue
        for next_node in adjacency.get(node, set()):
            if next_node in visited:
                continue
            visited[next_node] = {"hop": hop + 1}
            queue.append((next_node, hop + 1))
    return visited


def _build_table_graph(edges: Iterable[Edge]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source_table = f"{edge.source.dataset_id}.{edge.source.table_name}"
        target_table = f"{edge.target.dataset_id}.{edge.target.table_name}"
        if source_table == target_table:
            continue
        adjacency[source_table].add(target_table)
        reverse[target_table].add(source_table)
        if edge.bidirectional:
            adjacency[target_table].add(source_table)
            reverse[source_table].add(target_table)
    return adjacency, reverse


def _is_trivial_fk_cycle(
    cycle_nodes: list[tuple[str, str, str]],
    adjacency: dict[tuple[str, str, str], list[Edge]],
) -> bool:
    if len(cycle_nodes) != 3:
        return False
    if len(set(cycle_nodes)) != 2:
        return False
    for source_key, target_key in zip(cycle_nodes, cycle_nodes[1:]):
        if not any(
            edge.target.key == target_key
            and edge.relationship_type == "FOREIGN_KEY"
            and edge.bidirectional
            for edge in adjacency.get(source_key, [])
        ):
            return False
    return True


def _detect_cycles(
    nodes: set[tuple[str, str, str]],
    adjacency: dict[tuple[str, str, str], list[Edge]],
    limit: int = 5,
) -> list[list[str]]:
    cycles: list[list[str]] = []
    state: dict[tuple[str, str, str], int] = {}
    path: list[tuple[str, str, str]] = []
    seen_cycle_keys: set[str] = set()

    def dfs(node_key: tuple[str, str, str]) -> None:
        if len(cycles) >= limit:
            return
        state[node_key] = 1
        path.append(node_key)
        for edge in adjacency.get(node_key, []):
            target_key = edge.target.key
            if target_key not in nodes:
                continue
            if state.get(target_key, 0) == 0:
                dfs(target_key)
            elif state.get(target_key) == 1:
                if target_key in path:
                    idx = path.index(target_key)
                    cycle_nodes = path[idx:] + [target_key]
                    if _is_trivial_fk_cycle(cycle_nodes, adjacency):
                        continue
                    cycle_key = "->".join(f"{n[0]}.{n[1]}.{n[2]}" for n in cycle_nodes)
                    if cycle_key not in seen_cycle_keys:
                        seen_cycle_keys.add(cycle_key)
                        cycles.append([f"{n[0]}.{n[1]}.{n[2]}" for n in cycle_nodes])
                        if len(cycles) >= limit:
                            break
        state[node_key] = 2
        path.pop()

    for node_key in nodes:
        if state.get(node_key, 0) == 0:
            dfs(node_key)
        if len(cycles) >= limit:
            break
    return cycles


def _classify_severity(hop: int, relationship_type: str, critical: bool) -> str:
    score = 0
    if hop == 1:
        score += 2
    elif hop == 2:
        score += 1
    if relationship_type in {"TRANSFORMATION", "EMBEDDING", "QUERY_PATTERN"}:
        score += 1
    if critical:
        score += 1
    if score >= 3:
        return "Tsunami"
    if score >= 2:
        return "Wave"
    return "Ripple"


def _impact_risk_score(
    ripple: int,
    wave: int,
    tsunami: int,
    cycles: bool,
    embedding_count: int,
) -> int:
    raw = ripple * 4 + wave * 12 + tsunami * 25
    if cycles:
        raw += 10
    if embedding_count > 0:
        raw += 8
    return min(100, raw)


def _fetch_column_metadata(
    client: bigquery.Client,
    config: PipelineConfig,
    column_nodes: Iterable[ColumnNode],
) -> dict[tuple[str, str], dict[str, Any]]:
    column_ids: list[str] = []
    for node in column_nodes:
        if node.node_type != "column":
            continue
        asset_id = f"{config.project_id}.{node.dataset_id}.{node.table_name}"
        column_ids.append(f"{asset_id}:{node.column_name}")
    if not column_ids:
        return {}
    sql = f"""
    SELECT asset_id, column_name, data_type, is_nullable, column_description
    FROM `{config.project_id}.{config.metadata_dataset}.columns`
    WHERE CONCAT(asset_id, ':', column_name) IN UNNEST(@column_ids)
    """
    rows = _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("column_ids", "STRING", column_ids)],
    )
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        metadata[(row["asset_id"], row["column_name"])] = {
            "data_type": row.get("data_type"),
            "is_nullable": row.get("is_nullable"),
            "description": row.get("column_description") or "",
        }
    return metadata


def _fetch_asset_metadata(
    client: bigquery.Client,
    config: PipelineConfig,
    asset_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    asset_ids = list(set(asset_ids))
    if not asset_ids:
        return {}
    sql = f"""
    SELECT asset_id, dataset_id, asset_name, asset_type, table_description
    FROM `{config.project_id}.{config.metadata_dataset}.assets`
    WHERE asset_id IN UNNEST(@asset_ids)
    """
    rows = _safe_query(
        client,
        sql,
        [bigquery.ArrayQueryParameter("asset_ids", "STRING", asset_ids)],
    )
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata[row["asset_id"]] = {
            "dataset_id": row.get("dataset_id"),
            "table_name": row.get("asset_name"),
            "asset_type": row.get("asset_type"),
            "description": row.get("table_description") or "",
        }
    return metadata


def _fetch_embedding_impacts(
    client: bigquery.Client,
    config: PipelineConfig,
    column_nodes: Iterable[ColumnNode],
) -> dict[str, Any]:
    column_ids: list[str] = []
    table_ids: set[str] = set()
    for node in column_nodes:
        asset_id = f"{config.project_id}.{node.dataset_id}.{node.table_name}"
        table_ids.add(asset_id)
        if node.node_type == "column":
            column_ids.append(f"{asset_id}:{node.column_name}")
    column_embeddings: list[dict[str, Any]] = []
    table_embeddings: list[dict[str, Any]] = []
    if column_ids:
        sql = f"""
        SELECT column_doc_id, asset_id, column_name, embedding_model, updated_at
        FROM `{config.project_id}.{config.metadata_dataset}.column_embeddings`
        WHERE CONCAT(asset_id, ':', column_name) IN UNNEST(@column_ids)
        """
        column_embeddings = _safe_query(
            client,
            sql,
            [bigquery.ArrayQueryParameter("column_ids", "STRING", column_ids)],
            missing_table_ok=True,
        )
    if table_ids:
        sql = f"""
        SELECT doc_id, asset_id, embedding_model, updated_at
        FROM `{config.project_id}.{config.metadata_dataset}.embeddings`
        WHERE asset_id IN UNNEST(@asset_ids)
        """
        table_embeddings = _safe_query(
            client,
            sql,
            [bigquery.ArrayQueryParameter("asset_ids", "STRING", list(table_ids))],
            missing_table_ok=True,
        )
    return {"column_embeddings": column_embeddings, "table_embeddings": table_embeddings}


def _build_llm_prompt(payload: dict[str, Any]) -> str:
    schema = {
        "executive_summary": {
            "summary": "string",
            "recommended_action": "proceed|review|block|unknown",
        },
        "recommendations": {
            "breaking_change_warnings": ["string"],
            "migration_steps": ["string"],
            "testing_checklist": ["string"],
            "rollback_procedures": ["string"],
        },
    }
    return (
        "You are a data platform reviewer. Return STRICT JSON matching this schema:\n"
        f"{json_dumps(schema)}\n"
        "Rules: If you are unsure, output 'unknown' rather than guessing. "
        "Use the provided metrics exactly for counts and risk score. "
        "Use column descriptions from the context when forming the summary or recommendations.\n\n"
        f"Context:\n{json_dumps(payload)}\n"
    )


def _extract_response_text(response: object) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        if parts:
            return "".join(part.text for part in parts if hasattr(part, "text"))
    return ""


def _call_llm_summary(
    config: PipelineConfig,
    payload: dict[str, Any],
) -> dict[str, Any]:
    client = _get_genai_client(config)
    prompt = _build_llm_prompt(payload)
    response = client.models.generate_content(
        model=config.gemini_pro_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=2048,
        ),
    )
    text = _extract_response_text(response)
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


def build_ripple_report(
    client: bigquery.Client,
    config: PipelineConfig,
    *,
    dataset_id: str,
    table_name: str,
    column_name: str,
    max_hops: int = 3,
    include_query_patterns: bool = False,
    include_prompt_templates: bool = False,
) -> dict[str, Any]:
    target = ColumnNode(dataset_id=dataset_id, table_name=table_name, column_name=column_name)

    visited_edges: list[Edge] = []
    visited_nodes: dict[tuple[str, str, str], ColumnNode] = {target.key: target}
    frontier_tables: set[str] = {f"{dataset_id}.{table_name}"}
    visited_tables: set[str] = set(frontier_tables)

    for _hop in range(max_hops):
        if not frontier_tables:
            break
        table_keys = sorted(frontier_tables)
        asset_ids = sorted({f"{config.project_id}.{table_key}" for table_key in table_keys})

        lineage_edges = _fetch_lineage_edges(client, config, table_keys)
        join_edges = _fetch_join_hint_edges(client, config, asset_ids)
        candidate_edges = lineage_edges + join_edges

        next_frontier_tables: set[str] = set()
        for edge in candidate_edges:
            visited_edges.append(edge)
            for node in (edge.source, edge.target):
                if node.key not in visited_nodes:
                    visited_nodes[node.key] = node
            source_table = f"{edge.source.dataset_id}.{edge.source.table_name}"
            target_table = f"{edge.target.dataset_id}.{edge.target.table_name}"
            if source_table in frontier_tables and target_table not in visited_tables:
                next_frontier_tables.add(target_table)
            if target_table in frontier_tables and source_table not in visited_tables:
                next_frontier_tables.add(source_table)
        frontier_tables = next_frontier_tables
        visited_tables.update(frontier_tables)

    column_ids = [
        f"{n.dataset_id}.{n.table_name}.{n.column_name}"
        for n in visited_nodes.values()
        if n.node_type == "column"
    ]
    query_edges: list[Edge] = []
    query_patterns: list[dict[str, Any]] = []
    if include_query_patterns and column_ids:
        query_edges, query_patterns = _fetch_query_pattern_edges(client, config, column_ids)
        visited_edges.extend(query_edges)
        for edge in query_edges:
            visited_nodes[edge.source.key] = edge.source
            visited_nodes[edge.target.key] = edge.target

    nodes, adjacency, reverse = _build_graph(visited_edges)
    if target.key not in nodes:
        nodes[target.key] = target

    downstream = _bfs(target, adjacency, max_hops)
    upstream = _bfs(target, reverse, max_hops, reverse=True)

    downstream.pop(target.key, None)
    upstream.pop(target.key, None)

    impacted_nodes = {target.key} | set(downstream.keys()) | set(upstream.keys())
    cycles = _detect_cycles(impacted_nodes, adjacency)

    table_adjacency, table_reverse = _build_table_graph(visited_edges)
    target_table_key = f"{dataset_id}.{table_name}"
    table_downstream = _bfs_tables(target_table_key, table_adjacency, max_hops)
    table_upstream = _bfs_tables(target_table_key, table_reverse, max_hops)

    column_metadata = _fetch_column_metadata(client, config, nodes.values())
    asset_ids = {f"{config.project_id}.{n.dataset_id}.{n.table_name}" for n in nodes.values()}
    asset_metadata = _fetch_asset_metadata(client, config, asset_ids)

    def node_asset_id(node: ColumnNode) -> str:
        return f"{config.project_id}.{node.dataset_id}.{node.table_name}"

    def node_meta(node: ColumnNode) -> dict[str, Any]:
        asset_id = node_asset_id(node)
        return column_metadata.get((asset_id, node.column_name), {})

    severity_counts = {"Ripple": 0, "Wave": 0, "Tsunami": 0}
    impacted_columns: list[dict[str, Any]] = []

    for direction, visited in (("downstream", downstream), ("upstream", upstream)):
        for node_key, info in visited.items():
            node = nodes.get(node_key)
            if not node or node.node_type != "column":
                continue
            via_edge = info.get("via")
            relationship_type = via_edge.relationship_type if via_edge else "UNKNOWN"
            critical = node.dataset_id == "gold_layer"
            severity = _classify_severity(info["hop"], relationship_type, critical)
            severity_counts[severity] += 1
            meta = node_meta(node)
            impacted_columns.append(
                {
                    "dataset_id": node.dataset_id,
                    "table_name": node.table_name,
                    "column_name": node.column_name,
                    "hop": info["hop"],
                    "relationship_type": relationship_type,
                    "direction": direction,
                    "severity": severity,
                    "data_type": meta.get("data_type"),
                    "is_nullable": meta.get("is_nullable"),
                    "description": meta.get("description") or "unknown",
                }
            )

    total_impacted = len(impacted_columns)
    embedding_impacts = _fetch_embedding_impacts(client, config, nodes.values())
    risk_score = _impact_risk_score(
        severity_counts["Ripple"],
        severity_counts["Wave"],
        severity_counts["Tsunami"],
        cycles=bool(cycles),
        embedding_count=len(embedding_impacts.get("column_embeddings", [])),
    )

    recommended_action = "proceed"
    if risk_score >= 70 or severity_counts["Tsunami"] > 0:
        recommended_action = "block"
    elif risk_score >= 40 or severity_counts["Wave"] > 0:
        recommended_action = "review"

    upstream_sources: list[dict[str, Any]] = []
    for node_key, info in upstream.items():
        node = nodes.get(node_key)
        if not node or node.node_type != "column":
            continue
        meta = node_meta(node)
        contribution = round(1 / max(1, info["hop"]), 3)
        upstream_sources.append(
            {
                "dataset_id": node.dataset_id,
                "table_name": node.table_name,
                "column_name": node.column_name,
                "hop": info["hop"],
                "relationship_type": info["via"].relationship_type if info.get("via") else "UNKNOWN",
                "contribution": contribution,
                "data_type": meta.get("data_type"),
                "is_nullable": meta.get("is_nullable"),
                "description": meta.get("description") or "unknown",
            }
        )

    upstream_sources.sort(key=lambda item: (item.get("hop", 0), item.get("table_name", "")))

    nullable_upstream = [src for src in upstream_sources if src.get("is_nullable")]
    data_quality_implications = []
    if nullable_upstream:
        data_quality_implications.append(
            f"{len(nullable_upstream)} upstream columns are nullable; review null-handling."
        )
    if not data_quality_implications:
        data_quality_implications.append("unknown")

    potential_risks: list[str] = []
    for info in impacted_columns:
        if info["direction"] != "downstream":
            continue
        hop = info["hop"]
        if hop != 1:
            continue
        # compare types for direct downstream edges
        target_asset_id = f"{config.project_id}.{info['dataset_id']}.{info['table_name']}"
        target_type = column_metadata.get((target_asset_id, info["column_name"]), {}).get("data_type")
        via_edge = None
        if target_key := (info["dataset_id"], info["table_name"], info["column_name"]):
            via_edge = downstream.get(target_key, {}).get("via")
        if not via_edge:
            continue
        source = via_edge.source
        source_asset_id = f"{config.project_id}.{source.dataset_id}.{source.table_name}"
        source_type = column_metadata.get((source_asset_id, source.column_name), {}).get("data_type")
        if not source_type or not target_type or source_type == target_type:
            continue
        if source_type in {"FLOAT64", "NUMERIC", "BIGNUMERIC"} and target_type in {"INT64", "INTEGER"}:
            potential_risks.append(
                f"Potential truncation: {source.node_id} ({source_type}) -> {info['dataset_id']}.{info['table_name']}.{info['column_name']} ({target_type})."
            )
        if source_type in {"TIMESTAMP", "DATETIME"} and target_type == "DATE":
            potential_risks.append(
                f"Potential data loss: {source.node_id} ({source_type}) -> {info['dataset_id']}.{info['table_name']}.{info['column_name']} ({target_type})."
            )
        if source_type in {"STRING", "BYTES"} and target_type not in {"STRING", "BYTES"}:
            potential_risks.append(
                f"Potential parsing risk: {source.node_id} ({source_type}) -> {info['dataset_id']}.{info['table_name']}.{info['column_name']} ({target_type})."
            )

    if not potential_risks:
        potential_risks.append("unknown")

    downstream_tables: dict[str, dict[str, Any]] = {}
    for node_key in downstream.keys():
        node = nodes.get(node_key)
        if not node:
            continue
        asset_id = node_asset_id(node)
        meta = asset_metadata.get(asset_id, {})
        entry = downstream_tables.setdefault(
            asset_id,
            {
                "dataset_id": node.dataset_id,
                "table_name": node.table_name,
                "asset_type": meta.get("asset_type"),
                "impacted_columns": 0,
            },
        )
        if node.node_type == "column":
            entry["impacted_columns"] += 1

    related_tables = {**table_downstream, **table_upstream}
    for table_key in related_tables.keys():
        if table_key == target_table_key:
            continue
        dataset_part, table_part = table_key.split(".", 1)
        asset_id = f"{config.project_id}.{table_key}"
        meta = asset_metadata.get(asset_id, {})
        downstream_tables.setdefault(
            asset_id,
            {
                "dataset_id": dataset_part,
                "table_name": table_part,
                "asset_type": meta.get("asset_type"),
                "impacted_columns": 0,
            },
        )

    impacted_tables = sorted(
        downstream_tables.values(),
        key=lambda item: (item.get("impacted_columns", 0), item.get("table_name", "")),
        reverse=True,
    )

    prompt_templates: list[dict[str, Any]] = []
    if include_prompt_templates and column_ids:
        prompt_templates = _fetch_prompt_templates(client, config, column_ids)

    heatmap = []
    for hop in range(1, max_hops + 1):
        heatmap.append(
            {
                "hop": hop,
                "upstream_count": sum(
                    1
                    for key, item in upstream.items()
                    if item["hop"] == hop and nodes.get(key) and nodes[key].node_type == "column"
                ),
                "downstream_count": sum(
                    1
                    for key, item in downstream.items()
                    if item["hop"] == hop and nodes.get(key) and nodes[key].node_type == "column"
                ),
            }
        )

    flow_paths: list[dict[str, Any]] = []
    for node_key in list(downstream.keys())[:10]:
        if node_key not in nodes:
            continue
        path = [nodes[node_key].node_id]
        current_key = node_key
        while current_key != target.key:
            via_edge = downstream.get(current_key, {}).get("via")
            if not via_edge:
                break
            parent_key = via_edge.source.key
            path.append(via_edge.source.node_id)
            current_key = parent_key
            if len(path) > max_hops + 2:
                break
        flow_paths.append(
            {
                "target": nodes[node_key].node_id,
                "path": list(reversed(path)),
            }
        )

    embedding_edges: list[Edge] = []
    for emb in embedding_impacts.get("column_embeddings", []):
        asset_id = emb.get("asset_id")
        column = emb.get("column_name")
        if not asset_id or not column:
            continue
        parts = str(asset_id).split(".")
        if len(parts) < 3:
            continue
        emb_dataset, emb_table = parts[1], parts[2]
        source_key = (emb_dataset, emb_table, column)
        if source_key not in nodes:
            nodes[source_key] = ColumnNode(dataset_id=emb_dataset, table_name=emb_table, column_name=column)
        emb_node = ColumnNode(
            dataset_id=emb_dataset,
            table_name=emb_table,
            column_name=f"__embedding__:{column}",
            node_type_override="embedding",
        )
        nodes[emb_node.key] = emb_node
        embedding_edges.append(
            Edge(
                source=nodes[source_key],
                target=emb_node,
                relationship_type="EMBEDDING",
                impact_weight=1,
                discovery_method="column_embedding",
                transformation_logic=None,
                metadata={"embedding_model": emb.get("embedding_model")},
                bidirectional=False,
            )
        )

    for emb in embedding_impacts.get("table_embeddings", []):
        asset_id = emb.get("asset_id")
        if not asset_id:
            continue
        parts = str(asset_id).split(".")
        if len(parts) < 3:
            continue
        emb_dataset, emb_table = parts[1], parts[2]
        table_key = (emb_dataset, emb_table, "*")
        if table_key not in nodes:
            nodes[table_key] = ColumnNode(dataset_id=emb_dataset, table_name=emb_table, column_name="*")
        emb_node = ColumnNode(
            dataset_id=emb_dataset,
            table_name=emb_table,
            column_name="__table_embedding__",
            node_type_override="embedding",
        )
        nodes[emb_node.key] = emb_node
        embedding_edges.append(
            Edge(
                source=nodes[table_key],
                target=emb_node,
                relationship_type="EMBEDDING",
                impact_weight=1,
                discovery_method="table_embedding",
                transformation_logic=None,
                metadata={"embedding_model": emb.get("embedding_model")},
                bidirectional=False,
            )
        )

    dependency_nodes: list[dict[str, Any]] = []
    for node_key in impacted_nodes:
        node = nodes.get(node_key)
        if not node:
            continue
        direction = "target"
        hop = 0
        severity = "Ripple"
        if node_key in downstream:
            direction = "downstream"
            hop = downstream[node_key]["hop"]
            severity = _classify_severity(
                hop,
                downstream[node_key]["via"].relationship_type if downstream[node_key]["via"] else "UNKNOWN",
                node.dataset_id == "gold_layer",
            )
        elif node_key in upstream:
            direction = "upstream"
            hop = upstream[node_key]["hop"]
            severity = _classify_severity(
                hop,
                upstream[node_key]["via"].relationship_type if upstream[node_key]["via"] else "UNKNOWN",
                node.dataset_id == "gold_layer",
            )
        dependency_nodes.append(
            {
                "id": node.node_id,
                "type": node.node_type,
                "dataset_id": node.dataset_id,
                "table_name": node.table_name,
                "column_name": node.column_name,
                "direction": direction,
                "hop": hop,
                "severity": severity,
            }
        )

    seen_embeddings: set[str] = set()
    for edge in embedding_edges:
        node = edge.target
        if node.node_id in seen_embeddings:
            continue
        seen_embeddings.add(node.node_id)
        dependency_nodes.append(
            {
                "id": node.node_id,
                "type": node.node_type,
                "dataset_id": node.dataset_id,
                "table_name": node.table_name,
                "column_name": node.column_name,
                "direction": "downstream",
                "hop": 1,
                "severity": "Ripple",
            }
        )

    dependency_edges = []
    for edge in visited_edges + embedding_edges:
        dependency_edges.append(
            {
                "source": edge.source.node_id,
                "target": edge.target.node_id,
                "relationship_type": edge.relationship_type,
                "impact_weight": edge.impact_weight,
                "discovery_method": edge.discovery_method,
                "bidirectional": edge.bidirectional,
            }
        )

    target_asset_id = f"{config.project_id}.{dataset_id}.{table_name}"
    target_meta = column_metadata.get((target_asset_id, column_name), {})
    downstream_samples = [
        item
        for item in impacted_columns
        if item.get("direction") == "downstream"
    ][:5]
    upstream_samples = upstream_sources[:5]

    llm_payload = {
        "target_column": target.node_id,
        "target_column_metadata": {
            "data_type": target_meta.get("data_type"),
            "is_nullable": target_meta.get("is_nullable"),
            "description": target_meta.get("description") or "unknown",
        },
        "total_impacted": total_impacted,
        "severity_distribution": severity_counts,
        "risk_score": risk_score,
        "upstream_count": len(upstream),
        "downstream_count": len(downstream),
        "cycles_detected": bool(cycles),
        "upstream_columns_sample": upstream_samples,
        "downstream_columns_sample": downstream_samples,
        "impacted_tables": impacted_tables[:5],
        "impacted_embeddings": {
            "column_embeddings": len(embedding_impacts.get("column_embeddings", [])),
            "table_embeddings": len(embedding_impacts.get("table_embeddings", [])),
        },
    }
    llm_response = _call_llm_summary(config, llm_payload)
    exec_summary = llm_response.get("executive_summary", {}) if isinstance(llm_response, dict) else {}
    recommendations = llm_response.get("recommendations", {}) if isinstance(llm_response, dict) else {}

    if not exec_summary:
        exec_summary = {"summary": "unknown", "recommended_action": recommended_action}
    if "recommended_action" not in exec_summary:
        exec_summary["recommended_action"] = recommended_action

    report = {
        "status": "ok",
        "target": {
            "dataset_id": dataset_id,
            "table_name": table_name,
            "column_name": column_name,
        },
        "executive_summary": {
            "summary": exec_summary.get("summary", "unknown"),
            "total_impacted_columns": total_impacted,
            "severity_distribution": severity_counts,
            "risk_score": risk_score,
            "recommended_action": exec_summary.get("recommended_action", recommended_action),
        },
        "upstream_analysis": {
            "sources": upstream_sources,
            "data_quality_implications": data_quality_implications,
            "potential_data_loss_or_truncation": potential_risks,
        },
        "downstream_analysis": {
            "impacted_tables": impacted_tables,
            "impacted_embeddings": embedding_impacts,
            "query_patterns": query_patterns,
            "rag_retrieval_patterns": {
                "column_embeddings": len(embedding_impacts.get("column_embeddings", [])),
                "table_embeddings": len(embedding_impacts.get("table_embeddings", [])),
            },
            "generation_templates": prompt_templates,
        },
        "visualization": {
            "dependency_graph": {"nodes": dependency_nodes, "edges": dependency_edges},
            "flow_diagram": flow_paths,
            "heatmap": heatmap,
        },
        "recommendations": recommendations or {
            "breaking_change_warnings": ["unknown"],
            "migration_steps": ["unknown"],
            "testing_checklist": ["unknown"],
            "rollback_procedures": ["unknown"],
        },
        "cycles": cycles,
    }

    log_event(
        "ripple_report_generated",
        target=target.node_id,
        upstream=len(upstream),
        downstream=len(downstream),
        risk_score=risk_score,
    )
    return report
