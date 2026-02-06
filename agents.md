# agents.md — Codex CLI Project Agents (Best Practices)

This repository is driven by Codex CLI. The orchestrator must use the sub-agents below to parallelize work and avoid blind spots. Agents should **implement**, not suggest.

---

## Global Rules (apply to every agent)

1) **Do the work**: never punt implementation back to the user. If something is missing, make a safe assumption and proceed.
2) **Incremental + idempotent**: every run should be safe to re-run; use hashes + MERGE upserts; avoid full rebuilds unless explicitly requested.
3) **Deterministic outputs**: doc templates, ordering (ordinal_position), truncation limits, and hashing must be consistent across runs.
4) **Guardrails over cleverness**: enrichment must return `unknown` when uncertain; do not hallucinate joins, PII, or business meaning.
5) **Cost-aware**: batch and cache LLM/enrichment; embed only changed docs; cap concurrency and implement retries.
6) **Logging**: structured logs with counts (assets scanned, changed, enriched, embedded) and timings per stage.
7) **Fail safe**: partial failures should not corrupt tables. Use staging tables or transactional-style merges where possible.
8) **Permissions minimal**: document required IAM roles and keep them narrow.

---

## Prompting Standards (Gemini Enrichment)

- Always request **strict JSON** with a schema.
- Always include: “If you are unsure, output `unknown` rather than guessing.”
- Cap join confidence unless explicit evidence exists in descriptions.
- Never include sensitive data samples.

---

## Code Standards

- Python 3.11+
- `google-cloud-bigquery`
- Vertex AI SDK for Gemini + embeddings
- Typed functions, dataclasses where useful
- Black + Ruff configuration
- Modular code, unit-testable helpers (hashing/doc formatting/JSON parsing)
- Always use latest libraries and versions. Do web search if you are not sure about latest developments
- Always create a virtual environment before exeucting a project 
- maintain a requirements document 

---

## What “Done” Means

- Repo builds and runs locally.
- BigQuery tables created and populated.
- Delta runs are incremental and fast.
- Silver/gold enrichment works and is stored.
- Embeddings exist for documents and are only regenerated when needed.

