# RSIMem Documentation

Start here for current work:

1. [Current checklist](implementation_handoff_checklist.md): the only active
   execution protocol.
2. [AllMemoryOff baseline preparation](all_memory_off_baseline_20260911.md):
   current formal-control definition, frozen manifest, and execution gate.
3. [Current goal](current_goal.md): immediate result-analysis and refactor
   objective.
4. [Current progress](progress.md): implemented boundary, accepted evidence,
   and next blocked or executable step.
5. [AdaMem adaptation audit](adamem_adapter_audit_20260908.md): upstream
   mechanism, permitted adaptation boundary, and leakage boundary for the
   current Semantic RSI baseline.
6. [SM01 formal results](sm01_adamem_trajectory_results_20260908.md): final
   provenance-complete B0/B1/B2 comparison and its limits.
7. [Current experiment results](current_experiment_results.md): formal-result
   ledger, full-suite screening coverage, and explicit interpretation boundary.
8. [Planned main tables](main_table.md): paper-facing quality, resource, and
   per-family table templates.
9. [Codebase refactor checklist](codebase_refactor_checklist.md): staged
   package ownership, compatibility, and verification plan.
10. [Documentation status inventory](document_status_inventory_20260908.md):
   current, reference, historical, and generated-evidence classification.
11. [Refactor reachability audit](refactor_reachability_audit_20260911.md):
   completed canonical import and compatibility-shim removal evidence.
12. [Refactor release notes](refactor_release_notes_20260912.md): breaking
   module-path change and replay guidance.

The completed 234-run `NoMemory`/`HermesNative`/`Mem0Static` comparison is
historical. Its `NoMemory` control is now named `SemanticDisabled`: it did not
disable episodic or procedural surfaces. The active formal control is the new
`AllMemoryOff` baseline; its 26-family, 78-replicate provider-backed run is
frozen and independently aggregated.

## Status Labels

`current` documents define an active protocol or current result.
`reference` documents describe reusable runtime contracts or a fixed upstream
adapter boundary. `historical` documents retain prior experiments, attempts,
or design discussions. `generated evidence` is an immutable, dated experiment
record. A historical document is not an entrypoint for the active protocol.

## Historical Evidence

The following document families are retained under [`archive/`](archive/) so
old results remain reproducible, but they are not active experiment routes:

- `extraction_*`: earlier extraction-only feedback and optimizer work.
- `sensitivity_*`: earlier five-condition sensitivity pilots.
- `stage0*_native_*`, `stage1_*`, and
  `native_attribution_repair_protocol_v1.json`: native-attribution-repair
  infrastructure and dated evidence.
- `lifecycle_implementation_plan.md`, `current_checkpoint_20260901.md`, and
  `experiment_plan.md`: prior planning or reusable implementation reference.

The archive preserves original filenames and contents; only relative Markdown
links were adjusted for the new directory depth.

Do not infer a current conclusion from those documents without checking the
current checklist and progress summary first.
