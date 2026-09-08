# SM05 process pilot and optimizer result — 2026-08-29

The clean parent batch `s1-sm05-feedback-20260829-v1` completed three
replicates over the nine-episode `SM05_weak_trigger_preference_adoption`
family. All three attempts passed the runtime audit and produced 36 feedback
records and 21 source captures. Each run had 89 canonical process events;
the batch-level canonical process corpus contains 267 events. The strict
primary distribution was:

| label | count |
| --- | ---: |
| missed | 24 |
| unresolved | 12 |
| useful | 0 |
| harmful | 0 |
| censored | 0 |

The 24 extraction-owned `missed` examples make the corpus optimizer-eligible,
but there is no useful/harmful variation yet. The first request exceeded the
frozen 160,000-character budget because replicated source/evidence units were
expanded separately. Commit `f194bf8` adds a deterministic, budget-triggered
replica compaction that retains all primary IDs, replica counts, level counts,
and delayed identities. The resulting request is 158,538 characters and keeps
the full corpus digest unchanged.

Two schema-valid optimizer completions were attempted after compaction. Both
were rejected by the existing candidate content-safety gate because the model
copied a corpus-specific value into the proposed rule. Response bodies were
not stored or printed; the diagnostic completion had one `PROPOSE` edit and
complete model usage. No candidate artifact was written, and no activation or
matched validation was started.

This batch is useful process-signal evidence, but remains a no-candidate
provider/model-output result rather than a task negative or an effect claim.

The 267-event process corpus also gives the following layer census (across all
three replicates):

| layer | events | distinct output digests | status/action observation |
| --- | ---: | ---: | --- |
| trigger | 21 | 1 | all `pending`; no action variation |
| source selection | 21 | 21 | identity variation, no resolved outcome |
| extraction | 21 | 7 | output variation, outcomes unresolved/missed only |
| admission | 21 | 2 | ADD/NONE variation, no resolved outcome |
| commit | 21 | 21 | receipt identity variation, no resolved outcome |
| exposure | 60 | 9 | `success`/`skipped`/`pending`, no resolved useful/harmful variation |

These process observations support the current `validation-only` status for the
non-extraction layers; they do not turn status or digest variation into a
quality label.
