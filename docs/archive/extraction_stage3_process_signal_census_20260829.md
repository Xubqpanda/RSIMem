# Stage 3B/3D process-signal census — 2026-08-29

This census summarizes the content-free process evidence available from the
clean real-provider plain-parent pilots. It is a feasibility diagnostic, not a
quality score and not a matched parent/candidate effect result. No grader,
answer key, official score or cost value is used.

## Family-level process coverage

Each row is the union of the completed replicate `process_corpus.json` files
for that batch. The `policy coverage` and `receipt coverage` values come from
`census_process_events`; they measure observability, not semantic correctness.

| batch / family | replicates | process events | policy coverage | receipt coverage | stage action variation | process outcome signals |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `s1-sm01-feedback-20260829-v9a` / SM01 | 3 | 231 | 0.597 | 0.714 | Trigger/Source/Extraction/Admission/Commit all RUN; Exposure RUN+SKIP | `task_completed`, `absence`, `retrieval_miss` |
| `s1-sm02-feedback-20260829-v5` / SM02 | 3 | 231 | 0.597 | 0.714 | Trigger/Source/Extraction/Admission/Commit all RUN; Exposure RUN+SKIP | `task_completed`, `absence`, `retrieval_miss` |
| `s1-sm05-feedback-20260829-v1` / SM05 | 3 | 267 | 0.573 | 0.719 | Trigger/Source/Extraction/Admission/Commit all RUN; Exposure RUN+SKIP | `task_completed`, `absence`, `non_use`, `retrieval_miss` |
| `sm02-feedback-20260829-rerun-main` / SM02 | 3 | 231 | 0.597 | 0.714 | Trigger/Source/Extraction/Admission/Commit all RUN; Exposure RUN+SKIP | `task_completed`, `absence`, `retrieval_miss` |

Across these parent-only runs, the fixed policy produces stable process
signals and the expected exposure action variation. The absence of RUN/SKIP
variation for Trigger, Source, Extraction, Admission and Commit is why those
layers remain `validation-only` in the current census; it is not evidence that
the layers lack a useful future intervention. The deterministic feasibility
fixture supplies the required replayable parent/candidate action variation
for every layer, while Extraction additionally has resolved useful/missed
outcome cases and is therefore `optimization-ready`.

## Interpretation and next order

- Extraction is the first layer to open: it has a real source/extraction
  process boundary, a content-bearing train corpus, resolved missed signals in
  SM02, and a replay-stable N+1 candidate.
- SM02 provides the densest semantic process signal among the current pilots:
  recipient-boundary tool events, memory exposure, admission and task outcomes.
  Its unresolved records remain unresolved and are not converted to negatives.
- SM01 and SM05 expose useful tool/process events but their preference-format
  reuse signal is sparse; they remain pilot/evaluation substrates rather than
  strict process-only training labels.
- Trigger, Source selection, Admission, Commit and Exposure should remain
  fixed or shadow-only until a predeclared case shows independent outcome
  variation. Joint six-layer modification and matched validation are deferred
  to the later effect-experiment stage.

The corresponding raw process corpora are retained under the ignored
`outputs/extraction_feedback/hermes_luna/` tree. Replaying them through the
host-neutral corpus parser reproduces event IDs and the counts above without
reading evaluation-only fields.
