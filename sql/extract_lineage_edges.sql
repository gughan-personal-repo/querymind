-- Extract foreign key lineage edges from NOT ENFORCED constraints.

WITH fk_constraints AS (
  SELECT constraint_catalog, constraint_schema, constraint_name, constraint_type, enforced, table_catalog, table_schema, table_name
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
  WHERE constraint_type = 'FOREIGN KEY'
  UNION ALL
  SELECT constraint_catalog, constraint_schema, constraint_name, constraint_type, enforced, table_catalog, table_schema, table_name
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
  WHERE constraint_type = 'FOREIGN KEY'
  UNION ALL
  SELECT constraint_catalog, constraint_schema, constraint_name, constraint_type, enforced, table_catalog, table_schema, table_name
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.TABLE_CONSTRAINTS`
  WHERE constraint_type = 'FOREIGN KEY'
),
fk_source AS (
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY ordinal_position) AS rn
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.KEY_COLUMN_USAGE`
  UNION ALL
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY ordinal_position) AS rn
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.KEY_COLUMN_USAGE`
  UNION ALL
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY ordinal_position) AS rn
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.KEY_COLUMN_USAGE`
),
fk_target AS (
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY column_name) AS rn
  FROM `project-6ab0b570-446d-448e-882.bronze_layer_mssql.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE`
  UNION ALL
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY column_name) AS rn
  FROM `project-6ab0b570-446d-448e-882.silver_layer.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE`
  UNION ALL
  SELECT
    constraint_name,
    constraint_schema,
    table_catalog,
    table_schema,
    table_name,
    column_name,
    ROW_NUMBER() OVER (PARTITION BY constraint_name, constraint_schema, table_schema, table_name ORDER BY column_name) AS rn
  FROM `project-6ab0b570-446d-448e-882.gold_layer.INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE`
)
SELECT
  CONCAT(fk.table_catalog, '.', fk.table_schema, '.', fk.table_name) AS source_asset_id,
  fs.column_name AS source_column,
  CONCAT(ft.table_catalog, '.', ft.table_schema, '.', ft.table_name) AS target_asset_id,
  ft.column_name AS target_column,
  fk.constraint_name,
  fk.constraint_schema,
  fk.constraint_type,
  fk.enforced
FROM fk_constraints fk
LEFT JOIN fk_source fs
  ON fk.constraint_name = fs.constraint_name
 AND fk.constraint_schema = fs.constraint_schema
 AND fk.table_schema = fs.table_schema
 AND fk.table_name = fs.table_name
LEFT JOIN fk_target ft
  ON fk.constraint_name = ft.constraint_name
 AND fk.constraint_schema = ft.constraint_schema
 AND fs.rn = ft.rn
ORDER BY source_asset_id, source_column;
