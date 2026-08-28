# Extraction-Prompt Stage 2F Deterministic Acceptance

Date: 2026-08-28

## Decision

Stage 2F deterministic runtime binding and activation-fingerprint contracts pass.
Execution returns to the unfinished Stage 2E live matched validation batch.

This acceptance does not activate a production candidate and does not establish
an adaptive quality improvement. No provider request was made while completing
Stage 2F.

Implementation commits:

- `984a12d feat(runtime): bind matched extraction candidates`
- `4169674 feat(past): transport extraction trial profiles`
- `2aeb9ae feat(runtime): trace extraction activation fingerprints`

## Runtime Binding

- Plain static execution explicitly binds the trusted root extraction artifact.
- Matched validation loads the unique validation-only ACTIVE candidate through
  `load_extraction_matched_trial_profile()` and cannot use that store under a
  production scope.
- PAST-Bench keeps the content-free trial profile in sequence/manifest state,
  excludes the machine source path, copies the three immutable bundle files
  into each attempt's capture directory, and revalidates the copied profile.
- `prompt_slot(...)` is a one-line, registry-owned adapter entry. It has no
  global state or monkey patch and executes the same slot, owner, artifact, and
  binding checks as explicit registry registration.
- A configured candidate/root mismatch, bundle/profile mismatch, path escape,
  missing ACTIVE artifact, or slot/owner/contract/wrapper/schema mismatch fails
  before the extraction model call. Formal matched validation never silently
  falls back to root while retaining an adaptive label.

## Activation Fingerprint

Every completed extraction source now records content-free joins for:

- rich extraction policy artifact ID, digest, and version;
- actual prompt-component ID and body digest;
- adapter, slot, slot contract, wrapper, input schema, output schema, model,
  binding, and rendered-template identities;
- render ID, render-input digest, and raw model-output digest;
- parsed extraction-set digest and extraction operation ID;
- full semantic policy manifest, including frozen update/retrieval components,
  route, boundary, backend, framework, and model profile;
- mutation IDs and persisted artifact IDs.

The source-record schema is version 3. Version 2 records are rejected rather
than silently loaded under the stronger semantics. Prompt bodies, source text,
model output, credentials, and machine paths are not written into the
fingerprint.

Formal evidence assembly verifies the rich parent/candidate artifact, actual
runtime scope, complete binding contract, and complete semantic policy manifest.
Analysis reports `eligible`, `renderedNPlus1`, `changedExtraction`,
`noIntervention`, `changedArtifact`, `futureExposure`, `attributableUse`, and
`attributableOutcome`. Equal parent/candidate extraction is counted as no
intervention and cannot satisfy the adaptation claim gate.

## Failure Coverage

Deterministic tests reject:

- config declares N+1 while runtime binds N;
- wrong rich artifact ID or digest;
- adapter, slot contract, wrapper, input schema, or output schema drift;
- route, boundary, backend, model, update component, or retrieval component
  drift;
- malformed or conflicting source/fingerprint records;
- attempt-local trial bundle conflicts and manifest/source profile drift.

Two independently constructed runtimes bind the same trial artifact to the
same fingerprint after restart. Root static remains an explicit non-formal
configuration.

## Verification

- Focused Stage 2F tests: `102 passed`.
- RSIMem: `546 passed`.
- Vendored PAST-Bench: `397 passed, 2 skipped`.
- Python compileall: passed.
- Tracked secret scan: no newly introduced credential or provider endpoint.
- `git diff --check`: passed before documentation synchronization.

## Next Boundary

Run the independent predeclared Stage 2E parent N / proposal N+1 matched
validation batch. The run must use clean fixed revisions, attempt-local homes,
the frozen task/model/budget/feedback contracts, and the validation-only trial
profile. Activation remains contingent on strict positive resolved useful-rate
delta plus every anti-collapse and safety gate.
