# PG04_temporary_waiver_audit — Temporary Waiver Audit

**Ability**: `proactive_information_gathering`  
**Primary trigger**: `failure_reflection`  
**Expected substrate**: `mixed`  
**Family length tier**: `tier3`

## Purpose
Test whether the agent audits one-off waivers by retrieving the exact owner/service/scope/expiry tuple, instead of overfitting a temporary exception into a reusable rule.

## Must Demonstrate
- the waiver record is explicitly retrieved before the audit note is updated
- exact scope matching is enforced
- near-match cases are rejected instead of being incorrectly reused
