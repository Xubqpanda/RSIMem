# Deployment Rollback Grouping Rule

Group child deployment failures by their stable `release_hash`, not by surface
service name or timestamp. Update only child tickets that belong to the root
deployment event, use the required child escalation fields, and leave the root
deployment ticket open. Preserve this grouping rule across changed wording and
services.
