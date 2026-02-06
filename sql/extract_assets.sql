-- Extract table/view metadata for bronze/silver/gold datasets.

WITH asset_tables AS (
  SELECT
    table_catalog AS project_id,
    table_schema AS dataset_id,
    table_name AS asset_name,
    table_type AS raw_table_type,
    creation_time AS created_time,
    last_modified_time AS last_modified_time
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.TABLES`
  UNION ALL
  SELECT
    table_catalog,
    table_schema,
    table_name,
    table_type,
    creation_time,
    last_modified_time
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.TABLES`
  UNION ALL
  SELECT
    table_catalog,
    table_schema,
    table_name,
    table_type,
    creation_time,
    last_modified_time
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.TABLES`
),
asset_options AS (
  SELECT
    table_schema AS dataset_id,
    table_name AS asset_name,
    MAX(IF(option_name = 'description', option_value, NULL)) AS table_description,
    MAX(IF(option_name = 'labels', option_value, NULL)) AS labels,
    MAX(IF(option_name = 'partitioning_type', option_value, NULL)) AS partitioning_type,
    MAX(IF(option_name = 'partitioning_field', option_value, NULL)) AS partitioning_field,
    MAX(IF(option_name = 'clustering_fields', option_value, NULL)) AS clustering_fields
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.TABLE_OPTIONS`
  GROUP BY dataset_id, asset_name
  UNION ALL
  SELECT
    table_schema,
    table_name,
    MAX(IF(option_name = 'description', option_value, NULL)),
    MAX(IF(option_name = 'labels', option_value, NULL)),
    MAX(IF(option_name = 'partitioning_type', option_value, NULL)),
    MAX(IF(option_name = 'partitioning_field', option_value, NULL)),
    MAX(IF(option_name = 'clustering_fields', option_value, NULL))
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.TABLE_OPTIONS`
  GROUP BY table_schema, table_name
  UNION ALL
  SELECT
    table_schema,
    table_name,
    MAX(IF(option_name = 'description', option_value, NULL)),
    MAX(IF(option_name = 'labels', option_value, NULL)),
    MAX(IF(option_name = 'partitioning_type', option_value, NULL)),
    MAX(IF(option_name = 'partitioning_field', option_value, NULL)),
    MAX(IF(option_name = 'clustering_fields', option_value, NULL))
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.TABLE_OPTIONS`
  GROUP BY table_schema, table_name
),
asset_views AS (
  SELECT table_schema AS dataset_id, table_name AS asset_name, view_definition
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.VIEWS`
  UNION ALL
  SELECT table_schema, table_name, view_definition
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.VIEWS`
  UNION ALL
  SELECT table_schema, table_name, view_definition
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.VIEWS`
)
SELECT
  CONCAT(a.project_id, '.', a.dataset_id, '.', a.asset_name) AS asset_id,
  a.project_id,
  a.dataset_id,
  a.asset_name,
  IF(a.raw_table_type = 'VIEW', 'VIEW', 'TABLE') AS asset_type,
  a.created_time,
  a.last_modified_time,
  o.table_description,
  o.labels,
  o.partitioning_type,
  o.partitioning_field,
  o.clustering_fields,
  v.view_definition
FROM asset_tables a
LEFT JOIN asset_options o
  ON a.dataset_id = o.dataset_id AND a.asset_name = o.asset_name
LEFT JOIN asset_views v
  ON a.dataset_id = v.dataset_id AND a.asset_name = v.asset_name
ORDER BY dataset_id, asset_name;
