from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.api as api


class FakeRow:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def items(self):
        return self._data.items()


class FakeRowIterator:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [FakeRow(row) for row in rows]
        self.total_rows = len(rows)

    def __iter__(self):
        return iter(self._rows)


class FakeJob:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.job_id = "job_123"
        self._rows = rows

    def result(self, max_results: int | None = None):
        rows = self._rows[: max_results or len(self._rows)]
        return FakeRowIterator(rows)


class FakeClient:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or [{"ok": True}]

    def query(self, _sql: str, job_config=None):
        return FakeJob(self._rows)


class FakeValidatorResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self):
        return self._payload


class FakeOrchestratorAgent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def validate_table(self, **_kwargs):
        return FakeValidatorResult(
            {
                "table": "silver_layer.sample",
                "layer": "silver",
                "status": "ok",
                "rules_executed": 1,
                "rules_failed": 0,
                "rules_passed": 1,
                "violations": [],
                "metadata": {"grain_analysis": {"grain": "order_id"}},
            }
        )


@pytest.fixture()
def client():
    return TestClient(api.app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_matching_tables(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_query_matching_tables_cached",
        lambda query, top_k, bucket: (
            api.MatchTable(
                asset_id="project.dataset.table",
                dataset_id="silver_layer",
                table_name="table",
                semantic_score=0.9,
                keyword_score=0.8,
                hybrid_score=0.85,
            ),
        ),
    )
    resp = client.post("/get_matching_tables", json={"query": "sales", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()[0]["table_name"] == "table"


def test_get_matching_columns(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_query_matching_columns_cached",
        lambda query, top_k, bucket: (
            api.MatchColumn(
                asset_id="project.dataset.table",
                dataset_id="silver_layer",
                table_name="table",
                column_name="col",
                semantic_score=0.9,
                keyword_score=0.8,
                hybrid_score=0.85,
            ),
        ),
    )
    resp = client.post("/get_matching_columns", json={"query": "order", "top_k": 5})
    assert resp.status_code == 200
    assert resp.json()[0]["column_name"] == "col"


def test_get_lineage(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_resolve_table_candidates",
        lambda table_name, query: [
            {"asset_id": "p.silver_layer.orders", "dataset_id": "silver_layer", "table_name": "orders"}
        ],
    )
    monkeypatch.setattr(
        api,
        "_fetch_lineage_edges",
        lambda table_name, dataset_id: [
            api.LineageEdgeResponse(
                source_dataset="silver_layer",
                source_table="orders",
                source_column="id",
                target_dataset="gold_layer",
                target_table="orders_fact",
                target_column="id",
                relationship_type="TRANSFORMATION",
                transformation_logic="select",
                confidence_score=1.0,
                discovery_method="view_sql",
                metadata={},
            )
        ],
    )
    resp = client.post("/get_lineage", json={"table_name": "orders"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert len(data["edges"]) == 1


def test_generate_sql(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_build_sql_prompt",
        lambda user_query, tables: ("prompt", {f"{api.CONFIG.project_id}.silver_layer.orders": [{"column_name": "id"}]}),
    )
    monkeypatch.setattr(
        api,
        "_generate_sql_payload",
        lambda prompt: ({"sql": "SELECT 1", "notes": "", "tables_used": ["silver_layer.orders"]}, False, "{}", 1),
    )
    monkeypatch.setattr(
        api,
        "_validate_query_internal",
        lambda sql, budget_usd=5.0, warn_threshold_pct=80.0, project_id=None: {
            "status": "ok",
            "recommendation": "APPROVE",
            "approved": True,
            "is_suboptimal": False,
            "cost": {},
            "efficiency": {},
            "issues": [],
            "suggestions": [],
        },
    )
    resp = client.post(
        "/generate_sql",
        json={"user_query": "test", "tables": [{"dataset_id": "silver_layer", "table_name": "orders"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["sql"].lower().startswith("select")


def test_execute_sql(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_assess_sql", lambda sql: None)
    monkeypatch.setattr(api, "_get_bq_client", lambda: FakeClient([{"col": 1}]))
    resp = client.post("/execute_sql", json={"sql": "SELECT 1", "max_rows": 10, "dry_run": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"][0]["col"] == 1


def test_validate_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_validate_query_internal",
        lambda sql, budget_usd=5.0, warn_threshold_pct=80.0, project_id=None: {
            "status": "ok",
            "recommendation": "proceed",
            "approved": True,
            "is_suboptimal": False,
            "cost": {"usd": 0.1},
            "efficiency": {"bytes_processed": 123},
            "issues": [],
            "suggestions": [],
        },
    )
    resp = client.post("/validate_query", json={"sql": "SELECT 1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_classify_intent(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_classify_intent",
        lambda prompt: {"intent": "generate_sql", "confidence": 0.9, "entities": {}, "rationale": "ok"},
    )
    resp = client.post("/classify_intent", json={"query": "generate sql"})
    assert resp.status_code == 200
    assert resp.json()["intent"] == "generate_sql"


def test_get_table_schema(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_fetch_table_schema_single",
        lambda dataset_id, table_name: [{"name": "id", "data_type": "STRING", "is_nullable": False, "description": ""}],
    )
    resp = client.post("/get_table_schema", json={"dataset_id": "silver_layer", "table_name": "orders"})
    assert resp.status_code == 200
    assert resp.json()["columns"][0]["name"] == "id"


def test_validate_table(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_get_data_valid_tools", lambda: (None, None, FakeOrchestratorAgent))
    monkeypatch.setattr(
        api,
        "_generate_table_validation_summary",
        lambda prompt: {
            "summary": "ok",
            "risks": [],
            "recommendations": [],
            "next_steps": [],
        },
    )
    resp = client.post(
        "/validate_table",
        json={"dataset_id": "silver_layer", "table_name": "orders", "layer": "silver"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_validate_table_v2(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_get_data_valid_tools", lambda: (None, None, FakeOrchestratorAgent))
    resp = client.post(
        "/validate_table_v2",
        json={"table": "silver_layer.orders", "layer": "silver"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(state, _config):
        return {"response": {"reply": "ok", "intent": "chat"}}

    monkeypatch.setattr(api.CHAT_GRAPH, "invoke", fake_invoke)
    resp = client.post("/chat", json={"query": "hi", "session_id": "s1"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "ok"


def test_chat_validate_query(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        api,
        "_validate_query_internal",
        lambda sql, budget_usd=5.0, warn_threshold_pct=80.0, project_id=None: {
            "status": "ok",
            "recommendation": "proceed",
            "approved": True,
            "is_suboptimal": False,
            "cost": {"usd": 0.1},
            "efficiency": {"bytes_processed": 123},
            "issues": [],
            "suggestions": [],
        },
    )
    resp = client.post(
        "/chat",
        json={
            "query": "Validate sql query \"SELECT 1\"",
            "session_id": "s2",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "validate_query"
    assert data["query_validation"]["status"] == "ok"


def test_chat_validate_table(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_get_data_valid_tools", lambda: (None, None, FakeOrchestratorAgent))
    monkeypatch.setattr(
        api,
        "_generate_table_validation_summary",
        lambda prompt: {
            "summary": "ok",
            "risks": [],
            "recommendations": [],
            "next_steps": [],
        },
    )
    resp = client.post(
        "/chat",
        json={"query": "validate table silver_layer.orders", "session_id": "s3"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "validate_table"
    assert data["table_validation"]["validation_result"]["table"]


def test_ripple_report(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_column_exists", lambda dataset_id, table_name, column_name: True)
    monkeypatch.setattr(
        api,
        "build_ripple_report",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "target": {"dataset_id": "silver_layer", "table_name": "orders", "column_name": "id"},
            "executive_summary": {},
            "upstream_analysis": {},
            "downstream_analysis": {},
            "visualization": {},
            "recommendations": {},
            "cycles": [],
        },
    )
    resp = client.post(
        "/ripple_report",
        json={"dataset_id": "silver_layer", "table_name": "orders", "column_name": "id"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
