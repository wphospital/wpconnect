SELECT SCHEMA_NAME(schema_id) as schema_name, *
FROM sys.objects
WHERE 1=1
ORDER BY name;
