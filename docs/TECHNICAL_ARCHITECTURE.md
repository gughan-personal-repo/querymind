# QueryMind Technical Architecture

## 1. Purpose
QueryMind is a BigQuery metadata indexing and AI-assistant system with three major runtime surfaces:
- Metadata pipeline (`src/main.py`) for extraction, enrichment, embeddings, and lineage persistence.
- FastAPI application (`src/api.py`) for retrieval, SQL generation/execution, validation, and ripple analysis.
- React UI (`ui_next/`) that calls API endpoints through `/api/*`.

Default medallion datasets are:
- `bronze_layer_mssql`
- `silver_layer`
- `gold_layer`

Metadata is materialized in:
- `<project>.metadata_rag.*`

## 2. Repository Structure
Key backend modules:
- `src/main.py`: pipeline orchestration and run modes.
- `src/bq_extract.py`: table/column extraction and FK lineage extraction.
- `src/smart_load.py`: idempotent MERGE upserts via staging tables.
- `src/enrich_gemini.py`: table-level metadata enrichment.
- `src/column_desc_enrich.py`: missing column description enrichment.
- `src/doc_builder.py`: table document construction.
- `src/column_doc_builder.py`: column document construction.
- `src/embed.py`: Vertex embedding batch jobs.
- `src/api.py`: all API endpoints, chat graph, SQL/validation flows.
- `src/ripple_report.py`: graph-based change impact analysis.
- `src/entrypoint.py`: process entrypoint for API or pipeline mode.
- `src/config.py`: environment-driven configuration.

Deployment assets:
- `deploy/Dockerfile`: backend container build.
- `ui_next/Dockerfile`: frontend build + Nginx runtime.
- `ui_next/nginx.conf.template`: SPA + `/api` reverse proxy.
- `deploy/setup_service_account.sh`: service account + IAM roles.
- `deploy/deploy_cloud_run.sh`: build and deploy API/UI services.

Vendor dependency:
- `vendor/data-valid-ai/src/data_valid_ai`: vendored data validation module used by API table-validation endpoints.

## 3. End-to-End Data Flow

### 3.1 Metadata Pipeline Flow
1. Ensure metadata dataset/tables exist.
2. Extract table and column metadata from target datasets.
3. Extract FK-based lineage edges from INFORMATION_SCHEMA constraints.
4. Load existing hashes/state from metadata tables.
5. Detect changed/new assets by schema/meta hash.
6. Enrich eligible silver/gold assets with Gemini.
7. Build table and column documents (hash-based).
8. MERGE all changed records into metadata tables.
9. Generate embeddings only for new/changed docs or model change.

### 3.2 API Flow
Requests hit FastAPI endpoints directly or via UI reverse proxy. For `/chat`, a LangGraph state machine classifies intent and routes to node-specific handlers for search, SQL, lineage, table validation, ripple report, etc.

### 3.3 UI Flow
`MetadataBotUI` and `ChatInterface` call `/api/*` endpoints using axios. In local dev Vite proxies to `http://localhost:8000`. In Cloud Run, Nginx in UI container proxies `/api/*` to API Cloud Run URL.

## 4. Metadata Storage Model
DDL source: `sql/create_metadata_tables.sql`.

Tables created:
- `assets`: table/view-level metadata and hashes.
- `columns`: flattened schema fields with column hash.
- `documents`: table-level retrieval docs.
- `column_documents`: column-level retrieval docs.
- `enriched_assets`: AI-enriched concepts/grain/join hints/PII notes.
- `embeddings`: table doc vectors.
- `column_embeddings`: column doc vectors.
- `lineage_edges`: normalized lineage graph edges.

Design properties:
- Clustered tables for common access paths.
- Hash-based incremental updates.
- JSON fields used for semi-structured metadata.
- MERGE-based idempotent upsert semantics.

## 5. Pipeline Runtime Modes
Entry: `python -m src.main --mode <mode>`

Supported modes in `src/main.py`:
- `backfill`: full extraction + enrichment + docs + merge + embeddings.
- `delta`: same code path as backfill; hash checks prevent unnecessary rewrites.
- `enrich`: extraction + enrichment/doc refresh + column description enrichment.
- `embed`: embedding-only pass for docs requiring vectors.

## 6. Extraction and Hashing
`src/bq_extract.py` responsibilities:
- Enumerates tables in configured datasets.
- Pulls full table metadata via `client.get_table`.
- Flattens nested schemas.
- Computes:
  - `table_meta_hash` from description/labels/partitioning/clustering.
  - `schema_hash` from ordered column signatures.
  - `column_hash` per column from type/nullability/description/policy_tags.
- Extracts FK relationships from INFORMATION_SCHEMA and emits lineage edges with deterministic `edge_id`.

## 7. Enrichment and Document Generation

### 7.1 Table Enrichment
`src/enrich_gemini.py`:
- Uses Gemini with strict JSON prompt.
- Normalizes outputs for `concepts`, `grain`, `synonyms`, `join_hints`, `pii_flags`, `notes`.
- Applies guardrails:
  - unknown fallback when uncertain.
  - join hint confidence capped at `0.6` unless explicit evidence.
- Computes `enriched_hash` for deterministic change detection.

### 7.2 Column Description Enrichment
`src/column_desc_enrich.py`:
- Targets silver/gold assets with blank column descriptions.
- Builds prompts with table context + existing enrichment.
- Updates only missing descriptions.
- Recomputes `column_hash` after description insertion.

### 7.3 Document Builders
- `src/doc_builder.py`: table docs with schema summary, partitioning/clustering, optional view SQL and enrichment payload.
- `src/column_doc_builder.py`: column docs with table context and optional parent enrichment context.
- Both return deterministic text + SHA256 hash for incremental behavior.

## 8. Embedding Pipeline
`src/embed.py`:
- Fetches docs needing vectors by comparing doc hash and embedding model.
- Batches requests to Vertex embedding model (`gemini-embedding-001` by default).
- Parallelizes by batch with retries.
- Writes vectors to `embeddings` and `column_embeddings` via MERGE.

## 9. Persistence and Idempotency
`src/smart_load.py` pattern for each table:
1. Create temporary staging table from target schema.
2. Load JSON rows into staging.
3. MERGE into target with change predicates on hashes and selected fields.
4. Drop staging table.

Implications:
- Re-runnable and safe for incremental operation.
- Avoids full table rebuild.
- Keeps writes bounded to changed rows.

## 10. API Surface
Primary API endpoints in `src/api.py`:
- `GET /health`
- `POST /get_matching_columns`
- `POST /get_matching_tables`
- `POST /get_lineage`
- `POST /generate_sql`
- `POST /execute_sql`
- `POST /validate_query`
- `POST /classify_intent`
- `POST /get_table_schema`
- `POST /validate_table`
- `POST /validate_table_v2`
- `POST /chat`
- `POST /ripple_report`

## 11. Search and Retrieval
- Query embeddings generated with Vertex (`_embed_query_cached`).
- Hybrid ranking combines semantic signal and token hits.
- Dataset weighting prioritizes queryable layers.
- Cached search functions with TTL bucketing reduce repeat compute.

## 12. Chat Orchestration (LangGraph)
`src/api.py` defines `ChatState` and graph nodes:
- `classify`
- `chat`
- `get_matching_tables`
- `get_matching_columns`
- `get_lineage`
- `generate_sql`
- `execute_sql`
- `validate_query`
- `validate_table`
- `ripple_report`

Flow:
- `START -> classify -> intent-specific node -> END`.
- If graph invocation fails, `/chat` has fallback behavior and returns safe helper response.

## 13. SQL Generation and Validation

### 13.1 SQL Generation
`/generate_sql` and chat SQL node:
- Builds schema context from selected tables.
- Augments candidate tables using FK graph neighbors (`_augment_sql_tables`).
- Injects FK relationship hints into prompt (`_collect_fk_relationships`).
- Calls Gemini for structured payload (`sql`, `notes`, `tables_used`).
- Handles parse retries and trace metadata (`SQLGenerationTrace`).

### 13.2 SQL Safety and Cost
- `_assess_sql` runs baseline checks.
- `_validate_query_internal` uses data-valid-ai cost/efficiency tools.
- Optional revision loop (`_maybe_revise_sql`) re-prompts if validation flags issues.

### 13.3 SQL Execution
`/execute_sql`:
- Supports dry-run mode.
- Returns `job_id`, `total_rows`, and result rows with max-row cap.

## 14. Table Validation Integration
`/validate_table` and `/validate_table_v2`:
- Loads `data_valid_ai` via `_load_data_valid_ai`.
- Path precedence:
  1. `DATA_VALID_AI_PATH` env var.
  2. vendored path `vendor/data-valid-ai/src`.
  3. legacy local fallback path.
- Executes `OrchestratorAgent.validate_table`.
- Returns:
  - `validation_result` (structured result payload).
  - `llm_summary` if present.
  - `cli_output` generated from data-valid-ai CLI formatter for parity with terminal output.

## 15. Ripple Report (Change Impact)
`src/ripple_report.py` implements graph-centric impact analysis:
- Pulls lineage edges and optional enrichment/query-pattern/template context.
- Builds directed/bidirectional adjacency graphs.
- Runs BFS upstream/downstream with hop limits.
- Detects cycles.
- Scores severity/risk and builds executive summary.
- Optional LLM summarization for natural-language explanation.
- Exposed via `/ripple_report` and chat `ripple_report` intent.

## 16. Frontend Architecture
`ui_next/`:
- React + Vite + Tailwind + axios.
- Main UX in `ui_next/src/components/MetadataBotUI.jsx`.
- Uses API endpoints for validation/chat/lineage/schema/query execution.
- Validation rendering prioritizes API `cli_output` for terminal-parity formatting.

## 17. Deployment Architecture

### 17.1 API Container
`deploy/Dockerfile`:
- Installs Python dependencies.
- Copies `src`, `sql`, and vendored `data_valid_ai`.
- Sets `DATA_VALID_AI_PATH=/app/vendor/data-valid-ai/src`.
- Starts via `python -m src.entrypoint`.

### 17.2 UI Container
`ui_next/Dockerfile` + `ui_next/nginx.conf.template`:
- Builds static assets with Node.
- Serves SPA via Nginx.
- Reverse proxies `/api/*` to configured `API_UPSTREAM`.

### 17.3 Cloud Run Scripts
- `deploy/setup_service_account.sh`: creates runtime SA and grants required roles.
- `deploy/deploy_cloud_run.sh`: builds images and deploys `querymind-api` and `querymind-ui`.

## 18. Authentication and Authorization Model
Current production model:
- Cloud Run service account identity for server-side access to BigQuery and Vertex.
- Recommended IAM roles configured by script:
  - `roles/bigquery.jobUser`
  - `roles/bigquery.dataViewer`
  - `roles/bigquery.metadataViewer`
  - `roles/aiplatform.user`
  - `roles/logging.logWriter`
- UI calls API through server-side proxy, minimizing client auth complexity.

## 19. Configuration
Core env vars (`src/config.py`, `src/entrypoint.py`, `src/api.py`):
- Project/region: `PROJECT_ID`, `BQ_LOCATION`, `VERTEX_LOCATION`, `METADATA_DATASET`
- Dataset scope: `DATASETS`, `IGNORE_DATASETS`
- Models: `GEMINI_MODEL`, `GEMINI_PRO_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_TASK_TYPE`, `EMBEDDING_DIM`
- Throughput: `MAX_BQ_WORKERS`, `MAX_ENRICH_WORKERS`, `MAX_EMBED_WORKERS`, `EMBED_BATCH_SIZE`, `ENRICH_BATCH_SIZE`
- Retrieval/build tuning: `DOC_COLUMN_LIMIT`, `SEARCH_CANDIDATE_LIMIT`
- Entrypoint: `APP_MODE`, `PIPELINE_MODE`, `API_HOST`, `API_PORT`, `PORT`
- Validation module: `DATA_VALID_AI_PATH`
- Optional CORS: `CORS_ALLOW_ORIGINS`

## 20. Observability and Reliability
- Structured JSON logging via `log_event`.
- Stage timing logs in pipeline (`stage_completed` events).
- Retry wrappers for LLM/embedding calls.
- Graceful fallback behavior for chat/orchestration failures.
- MERGE-based writes avoid corruption from partial retries.

## 21. Tests and Quality Gates
Current test suite:
- `tests/test_api_endpoints.py`: endpoint-level behavior and payload shape.
- `tests/test_ripple_report.py`: ripple graph behavior, risk scoring, and cycle handling.

Run:
```bash
PYTHONPATH=. pytest -q tests/test_api_endpoints.py
```

## 22. Known Constraints and Risks
- Some modules execute broad metadata scans; cost/perf depends on dataset size.
- SQL generation quality depends on metadata completeness and FK lineage quality.
- Optional metadata tables (`query_patterns`, `prompt_templates`) may not exist; ripple logic handles missing tables when configured as optional.
- API currently exposes broad functionality; hardening with explicit authn/authz may be needed for multi-tenant or internet-exposed use.

## 23. Extension Points
- Add richer lineage extractors beyond FK constraints (query logs, parser-based lineage).
- Expand intent taxonomy and chat nodes.
- Add CI checks for lint/type/coverage.
- Introduce explicit auth middleware and role-based API authorization.
- Add UI-level feature flags and per-endpoint timeout handling.
