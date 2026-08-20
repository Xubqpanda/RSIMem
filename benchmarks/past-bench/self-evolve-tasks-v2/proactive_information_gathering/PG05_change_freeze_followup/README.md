# PG05_change_freeze_followup — Change Freeze Followup

**Ability**: `proactive_information_gathering`  
**Primary trigger**: `failure_reflection`  
**Expected substrate**: `session_search`  
**Family length tier**: `tier3`  

## Purpose
Test whether the agent proactively searches prior CAB freeze packets and retrieves the exact packet ID, UTC freeze window, and exception projects before updating the current freeze note.

## Must Demonstrate
- explicit retrieval happens before the first external action
- exact packet, window, and exception fields are recovered from session history, not guessed from visible hints
- later probes improve after reflection and policy-note updates
