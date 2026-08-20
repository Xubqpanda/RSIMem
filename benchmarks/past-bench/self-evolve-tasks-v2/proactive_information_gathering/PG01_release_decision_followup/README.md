# PG01_release_decision_followup — Release Decision Followup

**Ability**: `proactive_information_gathering`  
**Primary trigger**: `failure_reflection`  
**Expected substrate**: `session_search`  
**Family length tier**: `tier3`  

## Purpose
Test whether the agent proactively searches prior decision records and retrieves the exact environment, expiry, and reference before sharing release guidance.

## Must Demonstrate
- explicit retrieval happens before the first external action
- exact decision fields are recovered from session history, not guessed from memory preload
- later probes improve after reflection and policy-note updates
