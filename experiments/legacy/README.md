# Archived Experiment Routes

These modules retain superseded research protocols for evidence replay and
regression coverage. They are not current experiment entrypoints.

| Group | Historical route |
| --- | --- |
| `adaptive/` | adaptive utility-policy preparation, activation, validation, and analysis |
| `extraction/` | extraction-only optimizer, manifest, preflight, and validation evidence |
| `native/` | native attribution, observation, repair, and scheduler protocols |
| `sensitivity/` | sensitivity matrix, PAST catalog, seed registry, and static utility routes |

The legacy `rsimem.<module>` imports and `python -m rsimem.<module>` commands
remain forwarding shims for documented replay and test coverage. New active
work must use `experiments/base_memory/` or `experiments/adamem/`; reusable
runtime belongs under `src/rsimem/`.
