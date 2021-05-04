SELECT SCHEMA_NAME(schema_id) as schema_name, *
FROM sys.tables
WHERE 1=1
ORDER BY name;
