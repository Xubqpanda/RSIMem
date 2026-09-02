# DB Migration Risk Correlation Rule

Flag a migration as high risk when all three factors coincide: the region is
EU, the schema transition is version 7 to version 8, and the shard identifier
has the odd-shard pattern. Apply the conjunction to new entities, distinguish
safe near-misses, and do not memorize the original ticket IDs or surface names.
