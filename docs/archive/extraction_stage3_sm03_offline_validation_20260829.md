# SM03 Extraction Offline Validation (2026-08-29)

This report records the first real-provider SM03 held-out observation run for
the frozen extraction candidate. It is an offline validation result only; no
matched-trial or production pointer was created.

## Runtime identities

- Family: `SM03_fact_correction`
- Split role: `validation`
- Task-template group: `sm03-correction-heldout-validation-v1`
- Task manifest digest: `cece4019357f08d7bde746e012683542a699e61756ede76c91d4f6641dced54c`
- Parent run: `outputs/extraction_offline/sm03-heldout-v1/parent-contract`
- Candidate run: `outputs/extraction_offline/sm03-heldout-v1/candidate-contract`
- Candidate artifact: `extraction-prompt.a45dca366abeb0cd24bf6dacbe8859014caaaf0b`
- Candidate artifact digest: `a45dca366abeb0cd24bf6dacbe8859014caaaf0bffb2a065cce7f8afa94e2403`
- Static safety report: `candidate-safety.05a51ca68d8f4f4ec571193864bf141c2945d1be`
- Deterministic suite report: `deterministic-suite.9f74b50fa95110df5c0d2253e5b193e5fa78d68c`

The provider completion probe for `https://coding.tu-zi.com/v1` returned HTTP
200 with non-empty completion and usage available. The API credential is not
stored in this report or in any run manifest.

## Evidence checks

Both runs used the registered `sm03_fact_correction_v1` feedback contract and
the same local Hermes/PAST-Bench configuration. Each run produced 7 task
traces, semantic operation evidence, policy/process evidence, and a ledger.
The resulting audits reported no schema, privacy, policy, or process errors.
Process corpora were written with split role `validation` and audited before
observation assembly.

The strict observation join retained one non-empty extraction source per
evaluation task and collapsed duplicate feedback records by logical task/source
identity. It produced four observations:

- `eval_near`: parent and proposal
- `eval_far`: parent and proposal

All four primary labels were `unresolved` with reason `use_not_bound_to_memory`.
No unresolved or provider/task failure was converted into a negative extraction
label.

## Offline decision

The frozen acceptance criteria require at least 3 matched pairs and 2 resolved
examples. The actual run supplied 2 matched pairs and 0 resolved examples, so
the deterministic validator returned:

- Decision: `offline-validation.c15901146a57f73bfd1237fa6b63298e5b538be9`
- Status: `rejected`
- Reasons: `insufficient_matched_pairs`, `insufficient_resolved_examples`,
  `useful_rate_not_improved`, `missed_rate_unknown`
- Parent/proposal resolved useful rate: `unknown` (zero resolved denominator)
- Parent/proposal non-empty coverage: `1.0`
- Parent/proposal observed harmful rate: `0.0`

The complete decision payload and raw observation payload are retained under
`outputs/extraction_offline/sm03-heldout-v1/`. This rejected decision is not an
activation artifact and cannot satisfy the matched-trial loader. The candidate
therefore remains offline-validation-only; the next valid path is to improve
the registered family process/use contract or collect a family with sufficient
resolved signal, then rerun independent validation.
