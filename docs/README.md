# RSIMem Documentation

Start here for current work:

1. [Current checklist](implementation_handoff_checklist.md): the only active
   execution protocol.
2. [Current progress](progress.md): implemented boundary, accepted evidence,
   and next blocked or executable step.
3. [AdaMem adaptation audit](adamem_adapter_audit_20260908.md): upstream
   mechanism, permitted adaptation boundary, and leakage boundary for the
   current Semantic RSI baseline.
4. [Documentation status inventory](document_status_inventory_20260908.md):
   current, reference, historical, and generated-evidence classification.

## Status Labels

`current` documents define an active protocol or current result.
`reference` documents describe reusable runtime contracts or a fixed upstream
adapter boundary. `historical` documents retain prior experiments, attempts,
or design discussions. `generated evidence` is an immutable, dated experiment
record. A historical document is not an entrypoint for the active protocol.

## Historical Evidence

The following document families are retained at their existing paths so that
old results remain reproducible, but they are not active experiment routes:

- `extraction_*`: earlier extraction-only feedback and optimizer work.
- `sensitivity_*`: earlier five-condition sensitivity pilots.
- `stage0*_native_*`, `stage1_*`, and
  `native_attribution_repair_protocol_v1.json`: native-attribution-repair
  infrastructure and dated evidence.
- `lifecycle_implementation_plan.md`, `current_checkpoint_20260901.md`, and
  `experiment_plan.md`: prior planning or reusable implementation reference.

Do not infer a current conclusion from those documents without checking the
current checklist and progress summary first.
