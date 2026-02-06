from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


DEFAULT_DATASETS = ["bronze_layer_mssql", "silver_layer", "gold_layer"]
DEFAULT_IGNORE_DATASETS = {"raw_billing_export"}


@dataclass(frozen=True)
class PipelineConfig:
    project_id: str
    bq_location: str
    vertex_location: str
    datasets: list[str]
    metadata_dataset: str
    ignore_datasets: set[str]
    enrichment_version: str
    gemini_model: str
    gemini_pro_model: str
    embedding_model: str
    embedding_task_type: str
    embedding_dim: int | None
    max_bq_workers: int
    max_enrich_workers: int
    max_embed_workers: int
    embed_batch_size: int
    enrich_batch_size: int
    doc_column_limit: int
    search_candidate_limit: int


def _env_list(key: str, default: Sequence[str]) -> list[str]:
    raw = os.getenv(key)
    if not raw:
        return list(default)
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def load_config() -> PipelineConfig:
    project_id = os.getenv("PROJECT_ID", "project-6ab0b570-446d-448e-882")
    bq_location = os.getenv("BQ_LOCATION", "us")
    vertex_location = os.getenv("VERTEX_LOCATION", "us-central1")
    metadata_dataset = os.getenv("METADATA_DATASET", "metadata_rag")
    datasets = _env_list("DATASETS", DEFAULT_DATASETS)
    ignore_datasets = set(_env_list("IGNORE_DATASETS", DEFAULT_IGNORE_DATASETS))

    enrichment_version = os.getenv("ENRICHMENT_VERSION", "v1")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_pro_model = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
    embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_task_type = os.getenv("EMBEDDING_TASK_TYPE", "RETRIEVAL_DOCUMENT")
    embedding_dim_raw = os.getenv("EMBEDDING_DIM")
    embedding_dim = int(embedding_dim_raw) if embedding_dim_raw else None

    max_bq_workers = int(os.getenv("MAX_BQ_WORKERS", "8"))
    max_enrich_workers = int(os.getenv("MAX_ENRICH_WORKERS", "6"))
    max_embed_workers = int(os.getenv("MAX_EMBED_WORKERS", "6"))
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    enrich_batch_size = int(os.getenv("ENRICH_BATCH_SIZE", "1"))
    doc_column_limit = int(os.getenv("DOC_COLUMN_LIMIT", "60"))
    search_candidate_limit = int(os.getenv("SEARCH_CANDIDATE_LIMIT", "500"))

    return PipelineConfig(
        project_id=project_id,
        bq_location=bq_location,
        vertex_location=vertex_location,
        datasets=datasets,
        metadata_dataset=metadata_dataset,
        ignore_datasets=ignore_datasets,
        enrichment_version=enrichment_version,
        gemini_model=gemini_model,
        gemini_pro_model=gemini_pro_model,
        embedding_model=embedding_model,
        embedding_task_type=embedding_task_type,
        embedding_dim=embedding_dim,
        max_bq_workers=max_bq_workers,
        max_enrich_workers=max_enrich_workers,
        max_embed_workers=max_embed_workers,
        embed_batch_size=embed_batch_size,
        enrich_batch_size=enrich_batch_size,
        doc_column_limit=doc_column_limit,
        search_candidate_limit=search_candidate_limit,
    )
