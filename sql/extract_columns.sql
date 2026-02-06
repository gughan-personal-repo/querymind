-- Extract column metadata for bronze/silver/gold datasets.

SELECT
  CONCAT(table_catalog, '.', table_schema, '.', table_name) AS asset_id,
  column_name,
  data_type,
  is_nullable = 'YES' AS is_nullable,
  ordinal_position,
  description AS column_description,
  policy_tags
FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
UNION ALL
SELECT
  CONCAT(table_catalog, '.', table_schema, '.', table_name) AS asset_id,
  column_name,
  data_type,
  is_nullable = 'YES' AS is_nullable,
  ordinal_position,
  description,
  policy_tags
FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
UNION ALL
SELECT
  CONCAT(table_catalog, '.', table_schema, '.', table_name) AS asset_id,
  column_name,
  data_type,
  is_nullable = 'YES' AS is_nullable,
  ordinal_position,
  description,
  policy_tags
FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
ORDER BY asset_id, ordinal_position;
