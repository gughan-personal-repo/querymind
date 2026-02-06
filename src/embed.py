from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from google import genai
from google.genai import types
from google.cloud import bigquery
from tenacity import retry, stop_after_attempt, wait_exponential

from .utils import log_event, utc_now


def _fetch_docs_to_embed(
    client: bigquery.Client,
    docs_table: str,
    embeddings_table: str,
    id_field: str,
    model: str,
    include_column_name: bool,
) -> list[dict[str, Any]]:
    column_select = "d.column_name" if include_column_name else "NULL AS column_name"
    query = f"""
    SELECT d.{id_field} AS doc_id,
           d.asset_id,
           d.doc_text,
           d.doc_hash,
           {column_select}
    FROM `{docs_table}` d
    LEFT JOIN `{embeddings_table}` e
      ON d.{id_field} = e.{id_field}
    WHERE e.{id_field} IS NULL
       OR e.doc_hash != d.doc_hash
       OR e.embedding_model != @model
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("model", "STRING", model)]
    )
    rows = client.query(query, job_config=job_config).result()
    return [dict(row.items()) for row in rows]


def fetch_table_docs_to_embed(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    model: str,
) -> list[dict[str, Any]]:
    docs_table = f"{project_id}.{metadata_dataset}.documents"
    embeddings_table = f"{project_id}.{metadata_dataset}.embeddings"
    return _fetch_docs_to_embed(
        client, docs_table, embeddings_table, "doc_id", model, include_column_name=False
    )


def fetch_column_docs_to_embed(
    client: bigquery.Client,
    project_id: str,
    metadata_dataset: str,
    model: str,
) -> list[dict[str, Any]]:
    docs_table = f"{project_id}.{metadata_dataset}.column_documents"
    embeddings_table = f"{project_id}.{metadata_dataset}.column_embeddings"
    return _fetch_docs_to_embed(
        client,
        docs_table,
        embeddings_table,
        "column_doc_id",
        model,
        include_column_name=True,
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _embed_batch(
    project_id: str,
    location: str,
    model: str,
    task_type: str,
    output_dim: int | None,
    texts: list[str],
) -> list[list[float]]:
    client = genai.Client(vertexai=True, project=project_id, location=location)
    config = types.EmbedContentConfig(task_type=task_type)
    if output_dim:
        config.output_dimensionality = output_dim
    response = client.models.embed_content(model=model, contents=texts, config=config)
    embeddings = []
    for item in response.embeddings:
        embeddings.append(list(item.values))
    return embeddings


def embed_documents(
    docs: list[dict[str, Any]],
    project_id: str,
    location: str,
    model: str,
    task_type: str,
    output_dim: int | None,
    max_workers: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not docs:
        return []

    batches: list[list[dict[str, Any]]] = [
        docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
    ]
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _embed_batch,
                project_id,
                location,
                model,
                task_type,
                output_dim,
                [item["doc_text"] for item in batch],
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            batch = future_map[future]
            try:
                embeddings = future.result()
                if len(embeddings) != len(batch):
                    log_event(
                        "embedding_batch_mismatch",
                        expected=len(batch),
                        received=len(embeddings),
                    )
                    continue
                for item, embedding in zip(batch, embeddings):
                    results.append(
                        {
                            "doc_id": item.get("doc_id"),
                            "asset_id": item.get("asset_id"),
                            "column_name": item.get("column_name"),
                            "embedding": embedding,
                            "embedding_model": model,
                            "embedding_dim": len(embedding),
                            "doc_hash": item.get("doc_hash"),
                            "updated_at": utc_now(),
                        }
                    )
            except Exception as exc:
                log_event("embedding_batch_failed", error=str(exc))

    log_event("embedding_batches_completed", batches=len(batches), embeddings=len(results))
    return results
