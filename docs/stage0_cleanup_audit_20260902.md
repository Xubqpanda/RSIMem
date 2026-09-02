# Stage 0C/0D Cleanup Audit

Date: 2026-09-02

This audit records the second call-site pass required before removing
extraction-only assets.  The cleanup-ready baseline is
`baseline_manifest_stage0_cleanup_20260901.json`; its verifier is run again
after the final cleanup commit is recorded.

## Completed removals

The following stopped assets had no active production import, console-script
entry, shell caller, or non-historical test dependency after the extraction
path was closed:

- five `run_luna_*` extraction/adaptive/static launchers;
- `src/rsimem/extraction_proposal.py` and its proposal-only test module;
- `tests/test_extraction_launcher.py`;
- `tests/test_policy_feasibility_nplus1.py`;
- `configs/extraction_feedback_sm02.json`;
- `configs/extraction_feedback_sm05.json`;
- `configs/extraction_split_plan_sm05_sm03_sm04.json`.

The removal commits are `b1c9970`, `480f77b`, `3b2cbb4`, and `e7e214e`.
`rsimem-propose-extraction` was removed from `pyproject.toml`.  The
OpenAI-compatible optimizer transport was retained as a generalized provider
boundary; it is no longer reachable through an extraction-specific CLI.

## Retained pending migration

Three checked-in extraction configs remain because current generalized
preflight and split-contract tests still use them as deterministic fixtures:

- `configs/extraction_feedback_sm01.json`;
- `configs/extraction_split_plan_sm02_sm03_sm04.json`;
- `configs/extraction_validation_sm03.json`.

They are not production launch entry points.  They remain `DELETE` candidates
until Stage 1 supplies a versioned protocol manifest and the tests migrate to
temporary or generic fixtures.  Deleting them now would remove active test
coverage rather than remove dead runtime behavior.

## Audit scope and results

The second pass searched tracked `src`, `tests`, `scripts`, `configs`,
`pyproject.toml`, `README.md`, current docs, and vendored PAST source/tests for
imports, module paths, console scripts, shell references, and packaging
metadata.  No current runtime or CLI path references a removed file.  Mentions
in the immutable pre-cleanup manifest and dated historical evidence are
retained intentionally for provenance and are not executable references.

The following checks were run after each code/config removal commit:

```text
git diff --check                         passed
pytest -q tests                          passed
python -m compileall -q src tests        passed
bash -n scripts/*.sh                     passed
```

The final clean-tree Stage 0D run must additionally include the vendored
PAST-Bench suite, `pip check`, tracked-secret scan, baseline preflight, and a
current-reference scan that excludes only immutable baseline and historical
evidence paths.

## Remaining Stage 0D work

Stage 0D is not closed until the retained fixture configs are either migrated
to the Stage 1 protocol boundary or explicitly recorded as generalized test
fixtures, the current-reference scan is clean, and the full acceptance matrix
is rerun from a clean tree.  No provider benchmark run is required for this
cleanup gate.
