# AdaMem Upstream And Hermes Adaptation Audit

Status: `reference`  
Date: 2026-09-08  
Upstream: `/mnt/20t/xubuqiang/Study/AdaMem`  
Remote: `https://github.com/galaxyChen/AdaMem.git`  
Commit: `9dba25d` (`Camera-Ready public release`)

## Upstream Mechanism

The allowed upstream mechanism is implemented in
`run_adamem.py`:

| Concern | Upstream behavior | RSIMem adaptation boundary |
| --- | --- | --- |
| Policy state | `general_policy` plus `by_character` rules | Preserve this JSON policy shape as the AdaMem policy artifact. |
| Reflection input | Current policy plus a rendered, deployment-visible QA dialogue history | Substitute only a PAST/Hermes pure-process feedback view; do not add benchmark-only fields. |
| Reflection output | JSON patch with optional `general_policy`, `set`, and `remove` fields | Preserve incremental patch semantics and empty-object no-op. |
| Failure behavior | Parse or shape failure retains the old policy | Preserve as `NO_UPDATE`; no speculative policy replacement. |
| Render point | Per-policy rules render into Mem0 `custom_instructions` for later `add()` extraction | Render only at the Mem0 extraction-instruction boundary. Retrieval, host route, task prompt, and base model remain fixed. |
| Timing | A policy update after a feedback block affects subsequent memory writes | The `P_n -> P_{n+1}` receipt must bind the update to later Mem0 extraction, never retroactively rewrite old state. |

The upstream renderer intentionally keeps `general_policy` in reflection state
while it renders the per-character preference list into Mem0 instructions. The
RSIMem adapter must preserve that distinction rather than treating the whole
policy object as a host prompt.

The RSIMem binding is implemented by `rsimem.adamem_adapter.bind_to_mem0_flat`.
It creates a versioned component for the existing
`mem0-flat.semantic.extraction` slot, appending the rendered AdaMem preference
block to the fixed Mem0-flat extraction policy. The component provenance is the
AdaMem policy digest and the binding fingerprint identifies the exact rendered
template. No retrieval configuration, host route, task prompt, or base-model
field is writable through this adapter.

## Required RSIMem Records

For every AdaMem update attempt, record content-safe identifiers for:

- parent and candidate policy versions;
- policy render digest and Mem0 extraction binding;
- allowed feedback-view schema and request digest;
- patch/proposal digest, `NO_UPDATE` or parse/shape reason;
- validation, activation/rejection, and rollback receipt.

`B1_mem0_adamem_terminal` and `B2_mem0_adamem_full_trajectory` share the same
policy schema, updater prompt, patch space, update budget, validation, and
rollback. They differ only in the allowed pure-process feedback view.

## Prohibited Upstream Inputs

AdaMem-Bench uses `gold_answer`, `golden_feedback`, its story data, and an
upstream judge to build its own QA records. None may enter Hermes/PAST updater
or reflection input. RSIMem also does not import AdaMem-Bench stories, the
upstream runner, Full Context/Ideal Memory baselines, API configuration, or
evaluation scripts.

PAST hidden answers, grader output, official score, future evaluation content,
and task-specific reference answers are likewise forbidden from the reflection
request. Any detection is fail-closed.

## Fidelity Boundary

The first fidelity condition is `AdaMem-native`: it uses the upstream policy
contract and its native-style feedback shape after forbidden benchmark fields
have been removed. If adapting that input changes AdaMem's patch space or
failure semantics, RSIMem must record the coupling and use a separately named
SemanticRSI fallback. Such a fallback is an engineering control, not an
AdaMem-native result.
