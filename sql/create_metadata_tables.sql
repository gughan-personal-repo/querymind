CREATE SCHEMA IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag`
OPTIONS (location="us");

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.assets` (
  asset_id STRING,
  project_id STRING,
  dataset_id STRING,
  asset_name STRING,
  asset_type STRING,
  location STRING,
  table_description STRING,
  labels JSON,
  partitioning STRING,
  clustering STRING,
  created_time TIMESTAMP,
  last_modified_time TIMESTAMP,
  table_meta_hash STRING,
  schema_hash STRING,
  doc_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY dataset_id, asset_type;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.columns` (
  asset_id STRING,
  column_name STRING,
  data_type STRING,
  is_nullable BOOL,
  ordinal_position INT64,
  column_description STRING,
  policy_tags STRING,
  column_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.documents` (
  doc_id STRING,
  asset_id STRING,
  doc_text STRING,
  doc_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.column_documents` (
  column_doc_id STRING,
  asset_id STRING,
  column_name STRING,
  doc_text STRING,
  doc_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.enriched_assets` (
  asset_id STRING,
  enrichment_version STRING,
  concepts ARRAY<STRING>,
  grain STRING,
  synonyms ARRAY<STRUCT<term STRING, maps_to STRING>>,
  join_hints ARRAY<STRUCT<other_asset_id STRING, keys ARRAY<STRING>, confidence FLOAT64, evidence STRING>>,
  pii_flags ARRAY<STRING>,
  notes STRING,
  enriched_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.embeddings` (
  doc_id STRING,
  asset_id STRING,
  embedding ARRAY<FLOAT64>,
  embedding_model STRING,
  embedding_dim INT64,
  doc_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.column_embeddings` (
  column_doc_id STRING,
  asset_id STRING,
  column_name STRING,
  embedding ARRAY<FLOAT64>,
  embedding_model STRING,
  embedding_dim INT64,
  doc_hash STRING,
  updated_at TIMESTAMP
)
CLUSTER BY asset_id;

CREATE TABLE IF NOT EXISTS `project-6ab0b570-446d-448e-882.metadata_rag.lineage_edges` (
  edge_id STRING,
  source_dataset STRING,
  source_table STRING,
  source_column STRING,
  target_dataset STRING,
  target_table STRING,
  target_column STRING,
  relationship_type STRING,
  transformation_logic STRING,
  confidence_score FLOAT64,
  discovery_method STRING,
  impact_weight INT64,
  metadata JSON,
  created_at TIMESTAMP,
  last_verified TIMESTAMP
)
CLUSTER BY source_dataset, target_dataset;
