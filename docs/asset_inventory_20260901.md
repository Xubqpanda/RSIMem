# Asset Inventory: Pre-cleanup Classification

Baseline: [`baseline_manifest_20260901.json`](baseline_manifest_20260901.json)
at source commit `250cd28e23c6b563fd6edaa924db210666f1a9c4`.

This inventory is an initial 0B classification.  It does not authorize
deletion.  A `DELETE` row must pass the import/call-site audit in 0C first;
`EVIDENCE_KEEP` rows are historical provenance and may only be consolidated
after their evidence index remains reachable.

| Path | Disposition | Reason | Known dependents / migration target |
| --- | --- | --- | --- |
| `src/rsimem/adaptive_activation.py` | GENERALIZE | Current experiment manifest and validation still import activation/state checks; extraction-only entry point cannot be deleted before migration | `src/rsimem/experiment_manifest.py`, adaptive validation; future method activation API |
| `src/rsimem/adaptive_analysis.py` | GENERALIZE | Raw resource and matched analysis are reusable reporting primitives | adaptive tests/scripts; future panel reporter |
| `src/rsimem/adaptive_preparation.py` | GENERALIZE | Split/preparation identity can back feedback conditions | adaptive validation launcher/tests; future condition preparation |
| `src/rsimem/adaptive_validation_evidence.py` | GENERALIZE | Evidence assembly pattern is host/method neutral | adaptive validation tests; future matched evidence |
| `src/rsimem/adaptive_validation_runtime.py` | GENERALIZE | Trial/runtime binding and state isolation are reusable | adaptive validation launcher/tests; future method trial runtime |
| `src/rsimem/extraction_experiment_analysis.py` | GENERALIZE | Audit classification and denominator rules can serve sensitivity panels | extraction launcher/tests; future benchmark reporter |
| `src/rsimem/extraction_experiment_manifest.py` | GENERALIZE | Immutable run/attempt identity is needed by protocol manifests | extraction launchers/tests; future BenchmarkAdapter manifest |
| `src/rsimem/extraction_experiment_preflight.py` | GENERALIZE | Clean-tree, revision and config preflight is reusable | extraction feedback launcher/tests; future stage preflight |
| `src/rsimem/extraction_matched_preflight.py` | GENERALIZE | Matched identity and state checks are reusable | matched launcher/tests; future sensitivity runner |
| `src/rsimem/extraction_preparation.py` | GENERALIZE | Durable observation reload and corpus preparation are generic | feedback launcher/tests; future FeedbackCondition builder |
| `src/rsimem/extraction_proposal.py` | DELETE | Removed in Stage 0C; stopped extraction-only N+1 proposal CLI had no runtime import after launcher cleanup | future MemoryMethodAdapter proposal boundary |
| `src/rsimem/extraction_split_plan.py` | GENERALIZE | Frozen split and template identity apply to all memory panels | split-plan tests/configs; future protocol manifest |
| `src/rsimem/extraction_validation_evidence.py` | GENERALIZE | Matched evidence and raw resource joins are reusable | validation tests/launcher; future sensitivity report |
| `src/rsimem/extraction_validation_runtime.py` | GENERALIZE | Attempt-local trial binding and artifact fingerprints are reusable | validation tests/PAST CLI; future method adapter runtime |
| `src/rsimem/matched_analysis.py` | GENERALIZE | Content-free matched comparison and usage accounting are generic | matched tests/console script; future panel comparison |
| `src/rsimem/memory/adaptive_matched_validation.py` | GENERALIZE | Current adaptive validation and experiment manifest still import matched identity checks | adaptive validation/activation modules; future method-specific validation |
| `src/rsimem/memory/adaptive_mem0_binding.py` | GENERALIZE | Active state binding and capability checks are reusable | adaptive binding tests/runtime; future method state binder |
| `src/rsimem/memory/adaptive_policy.py` | GENERALIZE | Versioned policy state is a candidate MethodStateArtifact | adaptive policy tests/runtime; future method state contract |
| `src/rsimem/memory/adaptive_policy_store.py` | GENERALIZE | Crash-safe state store and activation pointer are reusable | adaptive policy tests/runtime; future method state store |
| `src/rsimem/memory/adaptive_policy_validation.py` | GENERALIZE | Stale/revision/safety validation patterns are reusable | adaptive validation tests; future method validation |
| `src/rsimem/memory/extraction_feedback.py` | GENERALIZE | Attribution, opportunity, use and outcome contracts are framework evidence | bridge, process corpus and feedback tests |
| `src/rsimem/memory/extraction_matched_activation.py` | GENERALIZE | Current validation evidence/runtime still import artifact activation checks | validation evidence/runtime and Hermes integration tests; future generic activation contract |
| `src/rsimem/memory/extraction_offline_validation.py` | GENERALIZE | Safety and offline validation primitives apply to method artifacts | validation tests; future MemoryMethodAdapter validation |
| `src/rsimem/memory/extraction_optimizer_audit.py` | GENERALIZE | Evidence-plane and contamination audit is framework-level | optimizer tests; future FeedbackCondition allowlist |
| `src/rsimem/memory/extraction_optimizer_builder.py` | GENERALIZE | Content-free optimizer view and weighting can be generalized | extraction/pure optimizer tests; future updater view |
| `src/rsimem/memory/extraction_optimizer_capture.py` | GENERALIZE | Owner-controlled capture separation is reusable | bridge and capture tests; future method corpus capture |
| `src/rsimem/memory/extraction_optimizer_contracts.py` | GENERALIZE | Candidate safety/lineage fields inform MethodStateArtifact | optimizer/proposal tests; future method contract |
| `src/rsimem/memory/extraction_optimizer_corpus.py` | GENERALIZE | Pure process corpus and evidence-plane gates are framework assets | process/preparation tests; future feedback corpus |
| `src/rsimem/memory/extraction_optimizer_provider.py` | GENERALIZE | OpenAI-compatible transport and raw usage projection remain reusable for future method adapters | future MemoryMethodAdapter provider boundary |
| `src/rsimem/memory/extraction_optimizer_store.py` | GENERALIZE | Restart-safe corpus store and revocation checks are reusable | preparation/store tests; future updater store |
| `src/rsimem/memory/extraction_policy_artifact.py` | GENERALIZE | Versioned artifact, lineage and binding identity map to method state | policy artifact/store tests; future method artifact |
| `src/rsimem/memory/extraction_policy_store.py` | GENERALIZE | Immutable artifact store and ACTIVE pointer are reusable | policy store/activation tests |
| `src/rsimem/memory/extraction_projection.py` | GENERALIZE | Source projection and extraction identity are canonical evidence | Hermes bridge and projection tests |
| `src/rsimem/memory/extraction_prompt_optimizer.py` | GENERALIZE | Pure-process feasibility and optimizer fixtures still import the safety/optimizer primitives | policy feasibility and pure optimizer paths; future updater state contract |
| `src/rsimem/memory/extraction_prompt_validation.py` | GENERALIZE | Grounding/privacy/shortcut gates apply to any generated method artifact | validation tests; future method safety |
| `src/rsimem/memory/extraction_source.py` | GENERALIZE | Source boundary and provenance are memory-method neutral | semantic ingestion/feedback tests |
| `src/rsimem/memory/extraction_validation_adapter.py` | GENERALIZE | Validation adapter boundary is reusable | validation tests; future BenchmarkAdapter |
| `src/rsimem/memory/pure_extraction_optimizer.py` | GENERALIZE | Pure-process corpus materialization is reusable after renaming | pure optimizer tests; future updater view |
| `src/rsimem/memory/pure_extraction.py` | GENERALIZE | Delayed feedback and source/future joins are framework evidence | Hermes bridge and pure extraction tests |
| `configs/extraction_feedback_sm01.json` | DELETE | Stopped SM01 extraction feedback launcher config | old launcher only |
| `configs/extraction_feedback_sm02.json` | DELETE | Removed in Stage 0C; no active consumer after the finite SM02 attempt closed | historical baseline/evidence only |
| `configs/extraction_feedback_sm05.json` | DELETE | Removed in Stage 0C; no active consumer after the finite SM05 attempt closed | historical baseline/evidence only |
| `configs/extraction_split_plan_sm02_sm03_sm04.json` | DELETE | Extraction-only split config superseded by new protocol manifest | extraction preflight/launcher |
| `configs/extraction_split_plan_sm05_sm03_sm04.json` | DELETE | Removed in Stage 0C; superseded by the forthcoming Stage 1 protocol manifest and had no active consumer | historical split evidence only |
| `configs/extraction_validation_sm03.json` | DELETE | Stopped extraction validation config | validation launcher only |
| `scripts/run_luna_adaptive_sm01.sh` | DELETE | Stopped extraction/adaptive experiment launcher | docs and adaptive tests; no new entry point |
| `scripts/run_luna_adaptive_validation_sm01.sh` | DELETE | Stopped extraction validation launcher | docs and adaptive tests |
| `scripts/run_luna_extraction_feedback_sm01.sh` | DELETE | Stopped extraction process-signal launcher | historical reports only |
| `scripts/run_luna_extraction_matched.sh` | DELETE | Stopped extraction prompt matched launcher | historical reports only |
| `scripts/run_luna_static_sm01.sh` | DELETE | Legacy semantic static launcher | historical static report only |
| `tests/test_adaptive_activation.py` | GENERALIZE | Exercises activation/state checks still used by current experiment manifest | adaptive activation and manifest |
| `tests/test_adaptive_analysis.py` | GENERALIZE | Raw resource/reporting regression remains useful | adaptive analysis module |
| `tests/test_adaptive_matched_validation.py` | GENERALIZE | Matched validation identity fixtures are reusable | adaptive validation module |
| `tests/test_adaptive_mem0_binding.py` | GENERALIZE | State binding/revision fixtures are reusable | adaptive binding module |
| `tests/test_adaptive_policy.py` | GENERALIZE | Versioned policy state contract can become method state | adaptive policy module |
| `tests/test_adaptive_policy_store.py` | GENERALIZE | Crash-safe store/restart fixtures are reusable | adaptive policy store |
| `tests/test_adaptive_policy_validation.py` | GENERALIZE | Safety and stale-state fixtures are reusable | adaptive policy validation |
| `tests/test_adaptive_preparation.py` | GENERALIZE | Frozen preparation and split fixtures are reusable | adaptive preparation |
| `tests/test_adaptive_validation_evidence.py` | GENERALIZE | Evidence join fixtures are reusable | adaptive validation evidence |
| `tests/test_adaptive_validation_runtime.py` | GENERALIZE | Attempt-local runtime fixtures are reusable | adaptive validation runtime |
| `tests/test_extraction_experiment_analysis.py` | GENERALIZE | Audit/denominator fixtures apply to sensitivity | experiment analysis |
| `tests/test_extraction_experiment_manifest.py` | GENERALIZE | Immutable manifest/attempt fixtures are reusable | experiment manifest |
| `tests/test_extraction_experiment_preflight.py` | GENERALIZE | Revision and clean-tree preflight fixtures are reusable | experiment preflight |
| `tests/test_extraction_feedback.py` | GENERALIZE | Generic opportunity/use/outcome attribution fixtures | extraction feedback |
| `tests/test_extraction_launcher.py` | DELETE | Launcher-specific extraction-only assertions | stopped launchers |
| `tests/test_extraction_matched_activation.py` | GENERALIZE | Protects activation checks still imported by validation runtime | extraction validation runtime/evidence |
| `tests/test_extraction_matched_preflight.py` | GENERALIZE | Matched identity fixtures are reusable | matched preflight |
| `tests/test_extraction_offline_validation.py` | GENERALIZE | Candidate safety gates inform method validation | offline validation |
| `tests/test_extraction_optimizer_builder.py` | GENERALIZE | Allowlist and weighting fixtures inform F0-F5 | optimizer builder |
| `tests/test_extraction_optimizer_capture.py` | GENERALIZE | Owner-content separation fixtures are generic | optimizer capture |
| `tests/test_extraction_optimizer_contracts.py` | GENERALIZE | Lineage and contamination fixtures inform method state | optimizer contracts |
| `tests/test_extraction_optimizer_corpus.py` | GENERALIZE | Corpus/revocation fixtures are generic | optimizer corpus |
| `tests/test_extraction_policy_artifact.py` | GENERALIZE | Artifact identity fixtures are generic | policy artifact |
| `tests/test_extraction_policy_store.py` | GENERALIZE | Store/ACTIVE/restart fixtures are generic | policy store |
| `tests/test_extraction_preparation.py` | GENERALIZE | Durable observation preparation fixtures are generic | preparation |
| `tests/test_extraction_projection.py` | GENERALIZE | Source projection fixtures are generic | projection |
| `tests/test_extraction_prompt_validation.py` | GENERALIZE | Safety/shortcut fixtures apply to generated artifacts | prompt validation |
| `tests/test_extraction_proposal.py` | DELETE | Removed in Stage 0C with stopped proposal CLI; assertions were extraction-only | future generic proposal contract tests |
| `tests/test_extraction_source.py` | GENERALIZE | Source provenance fixtures are generic | extraction source |
| `tests/test_extraction_split_plan.py` | GENERALIZE | Split identity fixtures are generic | split plan |
| `tests/test_extraction_validation_evidence.py` | GENERALIZE | Matched evidence fixtures are generic | validation evidence |
| `tests/test_extraction_validation_runtime.py` | GENERALIZE | Runtime binding fixtures are generic | validation runtime |
| `tests/test_policy_feasibility.py` | GENERALIZE | Surface intervention/replay fixtures inform sensitivity | policy feasibility |
| `tests/test_policy_feasibility_nplus1.py` | DELETE | Removed in Stage 0C; prompt N+1-specific assertions are stopped while generic intervention/replay coverage remains in `test_policy_feasibility.py` | old N+1 path |
| `tests/test_pure_extraction_optimizer.py` | GENERALIZE | Pure process corpus fixtures are reusable | pure optimizer |
| `tests/test_pure_extraction.py` | GENERALIZE | Delayed feedback and attribution fixtures are reusable | pure extraction |
| `docs/extraction_stage1_acceptance_20260828.md` | EVIDENCE_KEEP | Historical infrastructure evidence | current checkpoint/case index |
| `docs/extraction_stage2_clean_parent_20260901.md` | EVIDENCE_KEEP | Clean-parent provenance and no-signal result | current checkpoint/case index |
| `docs/extraction_stage2e_feedback_v10_20260828.md` | EVIDENCE_KEEP | Historical provider/runtime diagnosis | current checkpoint |
| `docs/extraction_stage2e_provider_attempts_20260828.md` | EVIDENCE_KEEP | Historical provider failure evidence | current checkpoint |
| `docs/extraction_stage2f_acceptance_20260828.md` | EVIDENCE_KEEP | Historical acceptance evidence | current checkpoint |
| `docs/extraction_stage2_production_reruns_20260831.md` | EVIDENCE_KEEP | Formal raw census and STOP_NO_SIGNAL evidence | case analysis |
| `docs/extraction_stage2_sm02_process_signal_20260830.md` | EVIDENCE_KEEP | Historical process-signal diagnosis | case analysis |
| `docs/extraction_stage2_sm02_process_signal_final_20260830.md` | EVIDENCE_KEEP | Historical final process-signal diagnosis | case analysis |
| `docs/extraction_stage2_sm05_process_signal_20260830.md` | EVIDENCE_KEEP | Historical process-signal diagnosis | case analysis |
| `docs/extraction_stage3_process_signal_census_20260829.md` | EVIDENCE_KEEP | Historical census and revocation context | case analysis |
| `docs/extraction_stage3_s1_feedback_20260829.md` | EVIDENCE_KEEP | Historical feedback attempt | current checkpoint |
| `docs/extraction_stage3_sm01_feedback_attempts_20260829_v5_v8.md` | EVIDENCE_KEEP | Historical provider attempts | current checkpoint |
| `docs/extraction_stage3_sm01_feedback_v9a_20260829.md` | EVIDENCE_KEEP | Historical provider attempt | current checkpoint |
| `docs/extraction_stage3_sm02_feedback_rerun_20260829.md` | EVIDENCE_KEEP | Historical revoked evidence context | revocation/case index |
| `docs/extraction_stage3_sm02_feedback_v5_20260829.md` | EVIDENCE_KEEP | Historical revoked evidence context | revocation/case index |
| `docs/extraction_stage3_sm02_optimizer_retry_20260829.md` | EVIDENCE_KEEP | Historical optimizer retry diagnosis | current checkpoint |
| `docs/extraction_stage3_sm02_process_pilot_20260829.md` | EVIDENCE_KEEP | Historical process pilot | current checkpoint |
| `docs/extraction_stage3_sm02_provider_attempts_20260829_v3_v4.md` | EVIDENCE_KEEP | Historical provider failures | current checkpoint |
| `docs/extraction_stage3_sm03_heldout_preflight_20260829.md` | EVIDENCE_KEEP | Historical stopped held-out preflight | current checkpoint |
| `docs/extraction_stage3_sm03_offline_validation_20260829.md` | EVIDENCE_KEEP | Historical stopped validation | current checkpoint |
| `docs/extraction_stage3_sm05_optimizer_20260829.md` | EVIDENCE_KEEP | Historical optimizer diagnosis | current checkpoint |
| `docs/extraction_stage3_split_audit_20260829.md` | EVIDENCE_KEEP | Historical split audit | current checkpoint |
| `docs/matched_20260827.md` | EVIDENCE_KEEP | Historical matched runtime evidence | current checkpoint |
| `docs/matched_phase1c_20260827.md` | EVIDENCE_KEEP | Historical adapter equivalence evidence | current checkpoint |
| `docs/phase1_acceptance_20260827.md` | EVIDENCE_KEEP | Historical acceptance evidence | current checkpoint |
| `docs/phase2k_acceptance_review_20260828.md` | EVIDENCE_KEEP | Historical acceptance evidence | current checkpoint |
| `docs/static_sm01_20260827.md` | EVIDENCE_KEEP | Historical static semantic result | current checkpoint |
| `docs/static_utility_sm01_20260827.md` | EVIDENCE_KEEP | Historical static utility result | current checkpoint |
| `docs/provider_probe.md` | KEEP | Generic pre-run connectivity gate | launchers and baseline protocol |

## 0B decision

The initial table did not authorize deletion.  Stage 0C now executes a
mechanical audit of every `DELETE` row against tracked imports, console-script
exports, shell references, tests, README/docs links, and packaging metadata.
A row is deleted only when that audit has no live dependency and its
replacement boundary is identified.

## Dependency audit snapshot

The audit was run against `src`, `tests`, `scripts`, `configs`,
`pyproject.toml`, `README.md`, `docs`, and the vendored PAST source/tests.
The launcher group and extraction proposal entry point had no active runtime
consumer after the extraction-only path was stopped, so they were removed in
Stage 0C commits `b1c9970` and the proposal-cleanup commit.  The provider
transport was reclassified as `GENERALIZE` because its OpenAI-compatible
usage projection is reusable.  Remaining DELETE rows still have generalized
preflight/test/config consumers or require the Stage 1 protocol manifest;
they remain pending until migration is complete.  Historical evidence may
mention removed paths, but current CLI, packaging, and production imports may
not.
