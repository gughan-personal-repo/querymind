from __future__ import annotations

import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .bq_extract import AssetRecord, ColumnRecord
from .utils import hash_text, json_dumps, log_event, utc_now


@dataclass(frozen=True)
class EnrichmentResult:
    asset_id: str
    enrichment_version: str
    concepts: list[str]
    grain: str
    synonyms: list[dict]
    join_hints: list[dict]
    pii_flags: list[str]
    notes: str
    enriched_hash: str


def _normalize_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    return []


def _normalize_synonyms(value: object) -> list[dict]:
    items: list[dict] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            maps_to = str(item.get("maps_to", "")).strip()
            if not term or not maps_to:
                continue
            items.append({"term": term, "maps_to": maps_to})
    return sorted(items, key=lambda item: (item["term"], item["maps_to"]))


def _normalize_join_hints(value: object) -> list[dict]:
    items: list[dict] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            other_asset_id = str(item.get("other_asset_id", "")).strip()
            keys = _normalize_list(item.get("keys"))
            confidence = item.get("confidence", 0.0)
            try:
                confidence_val = float(confidence)
            except Exception:
                confidence_val = 0.0
            confidence_val = min(confidence_val, 0.6)
            evidence = str(item.get("evidence", "")).strip()
            if not other_asset_id or not keys:
                continue
            items.append(
                {
                    "other_asset_id": other_asset_id,
                    "keys": keys,
                    "confidence": confidence_val,
                    "evidence": evidence or "unknown",
                }
            )
    return sorted(items, key=lambda item: (item["other_asset_id"], ",".join(item["keys"])))


def _normalize_text(value: object, default: str = "unknown") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_enrichment(data: dict, enrichment_version: str) -> EnrichmentResult:
    concepts = sorted(_normalize_list(data.get("concepts")))
    grain = _normalize_text(data.get("grain"))
    synonyms = _normalize_synonyms(data.get("synonyms"))
    join_hints = _normalize_join_hints(data.get("join_hints"))
    pii_flags = sorted(_normalize_list(data.get("pii_flags")))
    notes = _normalize_text(data.get("notes"))

    if not concepts:
        concepts = ["unknown"]
    if not pii_flags:
        pii_flags = ["unknown"]

    payload = {
        "enrichment_version": enrichment_version,
        "concepts": concepts,
        "grain": grain,
        "synonyms": synonyms,
        "join_hints": join_hints,
        "pii_flags": pii_flags,
        "notes": notes,
    }
    enriched_hash = hash_text(json_dumps(payload))
    return EnrichmentResult(
        asset_id="",
        enrichment_version=enrichment_version,
        concepts=concepts,
        grain=grain,
        synonyms=synonyms,
        join_hints=join_hints,
        pii_flags=pii_flags,
        notes=notes,
        enriched_hash=enriched_hash,
    )


def _extract_response_text(response: object) -> str:
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        if parts:
            return "".join(part.text for part in parts if hasattr(part, "text"))
    return ""


def _safe_json_parse(text: str) -> dict:
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


def _build_prompt(asset: AssetRecord, columns: Iterable[ColumnRecord]) -> str:
    column_lines = []
    for col in columns:
        column_lines.append(
            f"- {col.column_name} | {col.data_type} | {col.column_description or 'unknown'}"
        )
    column_text = "\n".join(column_lines)

    schema = {
        "concepts": ["string"],
        "grain": "string",
        "synonyms": [{"term": "string", "maps_to": "string"}],
        "join_hints": [
            {
                "other_asset_id": "string",
                "keys": ["string"],
                "confidence": 0.0,
                "evidence": "string",
            }
        ],
        "pii_flags": ["string"],
        "notes": "string",
    }

    return (
        "You are a data catalog assistant. Output STRICT JSON matching this schema:"
        f"\n{json_dumps(schema)}\n"
        "Rules: If you are unsure, output 'unknown' rather than guessing. "
        "Join confidence must be <= 0.6 unless explicit evidence exists. "
        "Do not hallucinate business meaning or sensitive data.\n\n"
        f"Table: {asset.asset_id}\n"
        f"Description: {asset.table_description or 'unknown'}\n"
        f"Columns:\n{column_text}\n"
    )


@retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(3))
def _call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            max_output_tokens=1024,
        ),
    )
    text = _extract_response_text(response)
    return _safe_json_parse(text)


def _enrich_one(
    project_id: str,
    location: str,
    model: str,
    enrichment_version: str,
    asset: AssetRecord,
    columns: list[ColumnRecord],
) -> dict | None:
    client = genai.Client(vertexai=True, project=project_id, location=location)
    prompt = _build_prompt(asset, columns)
    raw = _call_gemini(client, model, prompt)
    normalized = _normalize_enrichment(raw, enrichment_version)
    return {
        "asset_id": asset.asset_id,
        "enrichment_version": enrichment_version,
        "concepts": normalized.concepts,
        "grain": normalized.grain,
        "synonyms": normalized.synonyms,
        "join_hints": normalized.join_hints,
        "pii_flags": normalized.pii_flags,
        "notes": normalized.notes,
        "enriched_hash": normalized.enriched_hash,
        "updated_at": utc_now(),
    }


def enrich_assets(
    project_id: str,
    location: str,
    model: str,
    enrichment_version: str,
    assets: list[AssetRecord],
    columns_by_asset: dict[str, list[ColumnRecord]],
    max_workers: int = 4,
) -> dict[str, dict]:
    if not assets:
        return {}

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _enrich_one,
                project_id,
                location,
                model,
                enrichment_version,
                asset,
                columns_by_asset.get(asset.asset_id, []),
            ): asset.asset_id
            for asset in assets
        }
        for future in as_completed(future_map):
            asset_id = future_map[future]
            try:
                payload = future.result()
                if payload:
                    results[asset_id] = payload
            except Exception as exc:
                log_event("enrichment_failed", asset_id=asset_id, error=str(exc))

    log_event("enrichment_completed", assets=len(results))
    return results
