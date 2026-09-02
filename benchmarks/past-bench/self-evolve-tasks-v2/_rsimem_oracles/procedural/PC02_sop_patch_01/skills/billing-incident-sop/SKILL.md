# Billing Incident SOP v2

Treat a billing cluster as an incident only when at least three distinct
reporters identify the same cluster within four hours. Tag each qualifying
ticket `incident-core` and `billing-sla-breach`, set priority to critical and
category to incident-billing, and do not close tickets. Exclude UI bugs,
non-cluster performance issues, document/access requests, and routine reports.
