# PG03_oncall_handoff_lookup — Oncall Handoff Lookup

**Ability**: `proactive_information_gathering`  
**Primary trigger**: `failure_reflection`  
**Expected substrate**: `session_search`  
**Family length tier**: `tier3`

## Purpose
Test whether the agent consults the last handoff record before touching incident tickets, and whether it retrieves exact IDs, owners, windows, and blocked actions.

## Must Demonstrate
- the handoff record is retrieved before the first ticket update
- exact owner, window, and ticket set are recalled
- stale or similar handoff records are not reused by mistake
