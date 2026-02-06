from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.config import PipelineConfig
from src import ripple_report


@dataclass
class FakeQueryJob:
    rows: list[dict]

    def result(self) -> list[dict]:
        return self.rows


class FakeClient:
    def __init__(self, rows_by_table: dict[str, list[dict]]):
        self.rows_by_table = rows_by_table

    def query(self, sql: str, job_config=None) -> FakeQueryJob:
        for key, rows in self.rows_by_table.items():
            if key in sql:
                return FakeQueryJob(rows)
        return FakeQueryJob([])


def _config() -> PipelineConfig:
    return PipelineConfig(
        project_id="project-6ab0b570-446d-448e-882",
        bq_location="us",
        vertex_location="us-central1",
        datasets=["bronze_layer_mssql", "silver_layer", "gold_layer"],
        metadata_dataset="metadata_rag",
        ignore_datasets=set(),
        enrichment_version="v1",
        gemini_model="gemini-2.5-flash",
        gemini_pro_model="gemini-2.5-pro",
        embedding_model="gemini-embedding-001",
        embedding_task_type="RETRIEVAL_DOCUMENT",
        embedding_dim=None,
        max_bq_workers=1,
        max_enrich_workers=1,
        max_embed_workers=1,
        embed_batch_size=1,
        enrich_batch_size=1,
        doc_column_limit=10,
        search_candidate_limit=500,
    )


def test_ripple_report_basic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ripple_report,
        "_call_llm_summary",
        lambda config, payload: {"executive_summary": {"summary": "ok"}},
    )

    rows_by_table = {
        "metadata_rag.lineage_edges": [
            {
                "source_dataset": "bronze_layer_mssql",
                "source_table": "raw_orders",
                "source_column": "order_id",
                "target_dataset": "silver_layer",
                "target_table": "orders",
                "target_column": "order_id",
                "relationship_type": "TRANSFORMATION",
                "transformation_logic": "cast",
                "confidence_score": 1.0,
                "discovery_method": "view_sql",
                "impact_weight": 1,
                "metadata": {},
            },
            {
                "source_dataset": "silver_layer",
                "source_table": "orders",
                "source_column": "order_id",
                "target_dataset": "gold_layer",
                "target_table": "orders_fact",
                "target_column": "order_id",
                "relationship_type": "TRANSFORMATION",
                "transformation_logic": "select",
                "confidence_score": 1.0,
                "discovery_method": "view_sql",
                "impact_weight": 1,
                "metadata": {},
            },
            {
                "source_dataset": "gold_layer",
                "source_table": "orders_fact",
                "source_column": "order_id",
                "target_dataset": "gold_layer",
                "target_table": "revenue_summary",
                "target_column": "order_id",
                "relationship_type": "TRANSFORMATION",
                "transformation_logic": "agg",
                "confidence_score": 1.0,
                "discovery_method": "view_sql",
                "impact_weight": 1,
                "metadata": {},
            },
        ],
        "metadata_rag.columns": [
            {
                "asset_id": "project-6ab0b570-446d-448e-882.bronze_layer_mssql.raw_orders",
                "column_name": "order_id",
                "data_type": "STRING",
                "is_nullable": True,
                "column_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.orders",
                "column_name": "order_id",
                "data_type": "STRING",
                "is_nullable": False,
                "column_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.gold_layer.orders_fact",
                "column_name": "order_id",
                "data_type": "STRING",
                "is_nullable": False,
                "column_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.gold_layer.revenue_summary",
                "column_name": "order_id",
                "data_type": "STRING",
                "is_nullable": False,
                "column_description": "",
            },
        ],
        "metadata_rag.assets": [
            {
                "asset_id": "project-6ab0b570-446d-448e-882.bronze_layer_mssql.raw_orders",
                "dataset_id": "bronze_layer_mssql",
                "asset_name": "raw_orders",
                "asset_type": "TABLE",
                "table_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.orders",
                "dataset_id": "silver_layer",
                "asset_name": "orders",
                "asset_type": "TABLE",
                "table_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.gold_layer.orders_fact",
                "dataset_id": "gold_layer",
                "asset_name": "orders_fact",
                "asset_type": "TABLE",
                "table_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.gold_layer.revenue_summary",
                "dataset_id": "gold_layer",
                "asset_name": "revenue_summary",
                "asset_type": "TABLE",
                "table_description": "",
            },
        ],
        "metadata_rag.enriched_assets": [],
        "metadata_rag.column_embeddings": [],
        "metadata_rag.embeddings": [],
    }

    report = ripple_report.build_ripple_report(
        FakeClient(rows_by_table),
        _config(),
        dataset_id="silver_layer",
        table_name="orders",
        column_name="order_id",
        max_hops=3,
    )

    assert report["status"] == "ok"
    assert report["executive_summary"]["total_impacted_columns"] == 3
    assert report["executive_summary"]["risk_score"] >= 70
    assert report["executive_summary"]["recommended_action"] == "block"
    assert len(report["upstream_analysis"]["sources"]) == 1
    assert report["downstream_analysis"]["query_patterns"] == []
    assert report["downstream_analysis"]["generation_templates"] == []


def test_ripple_report_detects_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ripple_report,
        "_call_llm_summary",
        lambda config, payload: {"executive_summary": {"summary": "ok"}},
    )

    rows_by_table = {
        "metadata_rag.lineage_edges": [
            {
                "source_dataset": "silver_layer",
                "source_table": "table_a",
                "source_column": "id",
                "target_dataset": "silver_layer",
                "target_table": "table_b",
                "target_column": "id",
                "relationship_type": "TRANSFORMATION",
                "transformation_logic": "select",
                "confidence_score": 1.0,
                "discovery_method": "view_sql",
                "impact_weight": 1,
                "metadata": {},
            },
            {
                "source_dataset": "silver_layer",
                "source_table": "table_b",
                "source_column": "id",
                "target_dataset": "silver_layer",
                "target_table": "table_a",
                "target_column": "id",
                "relationship_type": "TRANSFORMATION",
                "transformation_logic": "select",
                "confidence_score": 1.0,
                "discovery_method": "view_sql",
                "impact_weight": 1,
                "metadata": {},
            },
        ],
        "metadata_rag.columns": [
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.table_a",
                "column_name": "id",
                "data_type": "STRING",
                "is_nullable": False,
                "column_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.table_b",
                "column_name": "id",
                "data_type": "STRING",
                "is_nullable": False,
                "column_description": "",
            },
        ],
        "metadata_rag.assets": [
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.table_a",
                "dataset_id": "silver_layer",
                "asset_name": "table_a",
                "asset_type": "TABLE",
                "table_description": "",
            },
            {
                "asset_id": "project-6ab0b570-446d-448e-882.silver_layer.table_b",
                "dataset_id": "silver_layer",
                "asset_name": "table_b",
                "asset_type": "TABLE",
                "table_description": "",
            },
        ],
        "metadata_rag.enriched_assets": [],
        "metadata_rag.column_embeddings": [],
        "metadata_rag.embeddings": [],
    }

    report = ripple_report.build_ripple_report(
        FakeClient(rows_by_table),
        _config(),
        dataset_id="silver_layer",
        table_name="table_a",
        column_name="id",
        max_hops=2,
    )

    assert report["cycles"]
