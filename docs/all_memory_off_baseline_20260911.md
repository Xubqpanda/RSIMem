# AllMemoryOff Baseline Preparation

Status: `frozen; 78 accepted provider-backed runs`
Date: 2026-09-12

`AllMemoryOff` is the new formal control. It is not the historical
`NoMemory` result: the historical condition disabled Hermes semantic memory
only, while leaving family-selected session search and skills available.

The new condition preserves the `with_persistence` execution topology but
sets all four Hermes switches to `false`:

- `memory_enabled`
- `user_profile_enabled`
- `skills_enabled`
- `session_search_enabled`

Its run descriptor records semantic, episodic, and procedural Memory as
disabled. Per-episode mechanism routing is fail-closed and cannot re-enable a
toolset. The audit rejects any Memory operation/injection evidence, nonzero
`session_search`, `skills_list`, `skill_view`, or semantic writer evidence.

## Frozen Inputs

- Source formal manifest:
  `outputs/adamem_full_suite_20260908/formal_manifest.json`
- Source manifest SHA-256:
  `c2a349320414c73d856d3ec770ad36a9f723d7970f39f194678b007c3c012001`
- Source protocol digest:
  `153689a96e7f6245d1771ae5079da76af0fc3332d3e15e48a1e7d8b156835f76`
- New formal manifest:
  `outputs/all_memory_off_formal_20260911/formal_manifest.json`
- New formal manifest digest:
  `81b4cdd9480b8414bb6d3fad3cfc397c841f2a772b3c476f09c7694470d2344d`

The new manifest contains 26 frozen families and three required isolated
replicates per family, for 78 required accepted runs. It does not alter any
historical output root.

## Deterministic Evidence

The complete 26-family no-provider dry-run passed and wrote receipts under
`outputs/all_memory_off_formal_20260911/dry_run/`. It confirms the frozen
family config digests, condition identity, isolated roots, and the
AllMemoryOff backend descriptor for every family.

Four representative provider-backed smoke runs, covering Semantic, EP, PC,
and PG, passed the strict v2 audit. The serial formal suite then completed
26 family batches with three isolated accepted replicates per family. Every
formal batch was audited before the next family started; incomplete and
superseded attempt roots remain in the output directory as provenance.

## Frozen Results

The independent result ledger contains exactly `26` families and `78`
accepted replicates. It retains per-task Near/Far scores, complete input and
output usage, isolated run roots, and the zero-evidence AllMemoryOff audit
record for every replicate:

- Ledger: `outputs/all_memory_off_formal_20260911/all_memory_off_result_ledger.json`
- Main-table input: `outputs/all_memory_off_formal_20260911/all_memory_off_main_table_input.json`
- Ledger digest: `b768499ba7987249f76b71d8de376a8dc47272d9c66db22c3e14425d94164455`

The formal batch roots remain the raw recomputation source. Historical
`NoMemory` outputs are not included in this denominator and continue to mean
`SemanticDisabled`. Compatibility-shim removal can proceed only through the
canonical import and reachability audit in the active checklist.
