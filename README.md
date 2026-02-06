# Metadata RAG Pipeline (BigQuery + Vertex AI)

This repo builds a Metadata RAG ingestion pipeline for BigQuery tables and columns, including enrichment (silver/gold only) and embeddings (tables + columns) with incremental hashing.

**Default project**: `project-6ab0b570-446d-448e-882`  
**Default datasets**: `bronze_layer_mssql`, `silver_layer`, `gold_layer`  
**Metadata dataset**: `metadata_rag`

## Prereqs
1. Enable APIs: BigQuery, Vertex AI.
2. Authenticate with ADC:
   - `gcloud auth application-default login`
   - `gcloud config set project project-6ab0b570-446d-448e-882`

## Local Run (exact commands)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

export PROJECT_ID=project-6ab0b570-446d-448e-882
export BQ_LOCATION=us
export VERTEX_LOCATION=us-central1
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=project-6ab0b570-446d-448e-882
export GOOGLE_CLOUD_LOCATION=us-central1

bq query --use_legacy_sql=false < sql/create_metadata_tables.sql

python3 -m src.main --mode backfill
python3 -m src.main --mode delta
python3 -m src.main --mode enrich
python3 -m src.main --mode embed
```

## API (FastAPI)
Run the API locally:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /classify_intent`
- `POST /get_matching_columns`
- `POST /get_matching_tables`
- `POST /get_lineage`
- `POST /get_table_schema`
- `POST /chat`
- `POST /ripple_report`
- `POST /generate_sql`
- `POST /execute_sql`

## UI (Vite)
The UI lives in `ui_next/` and is already wired to the API endpoints via the Vite proxy.

```bash
cd ui_next
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://localhost:8000`.

## Cloud Run
Build and deploy:
```bash
gcloud builds submit --tag gcr.io/project-6ab0b570-446d-448e-882/metadata-rag

gcloud run deploy metadata-rag \
  --image gcr.io/project-6ab0b570-446d-448e-882/metadata-rag \
  --region us-central1 \
  --set-env-vars PROJECT_ID=project-6ab0b570-446d-448e-882,BQ_LOCATION=us,VERTEX_LOCATION=us-central1,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=project-6ab0b570-446d-448e-882,GOOGLE_CLOUD_LOCATION=us-central1,APP_MODE=api,API_HOST=0.0.0.0,API_PORT=8000
```

Run the job by updating the container command to set the mode, for example:
```bash
gcloud run jobs create metadata-rag-delta \
  --image gcr.io/project-6ab0b570-446d-448e-882/metadata-rag \
  --region us-central1 \
  --command python \
  --args "-m","src.entrypoint" \
  --set-env-vars PROJECT_ID=project-6ab0b570-446d-448e-882,BQ_LOCATION=us,VERTEX_LOCATION=us-central1,GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=project-6ab0b570-446d-448e-882,GOOGLE_CLOUD_LOCATION=us-central1,APP_MODE=pipeline,PIPELINE_MODE=delta
```

## Notes
- The pipeline creates `metadata_rag` tables automatically from `sql/create_metadata_tables.sql`.
- Enrichment is only applied to `silver_layer` and `gold_layer`.
- Embeddings run only when doc hash or embedding model changes.
- Delta runs are idempotent and hash-based.
- Lineage edges are extracted from NOT ENFORCED foreign key constraints into `metadata_rag.lineage_edges`.
- If `metadata_rag.lineage_edges` already exists with a different schema, drop it and rerun `sql/create_metadata_tables.sql`.
