# EP02 Exception-List Recall Sensitivity Pilot Attempt - 2026-09-04

This document records an infrastructure-interrupted replicate-2 attempt for
`EP02_exception_list_recall`. It is excluded from sensitivity, mechanism,
quality, and resource denominators. No candidate policy, optimizer input, or
N+1 update was produced.

The provider completion probe passed and all five runner commands exited with
code `0`. However, the content-free audit reported `usage_incomplete` for the
`shortcut_current_input` and `wrong_mechanism` conditions. Their traces include
retry activity without a complete usage record, so the five-condition pilot is
not accepted. The complete native, oracle, and no-persistence conditions remain
available as audit-plane evidence.

A fresh retry must use a new batch identity and isolated state. Raw resources
remain audit data only and are not policy rewards.
