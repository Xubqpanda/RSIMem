# update_ability

Families where the main requirement is replacing, preserving, expiring,
patching, or locally modifying existing state.

Runnable update families initialize old state through native Hermes home
fixtures under `_shared/home_fixtures/update_ability/`. Wrong-mechanism
controls keep using `_shared/preseed/` overlays as separate ablations.

Included families:

- `SM03_fact_correction`
- `SM04_rule_migration`
- `SM06_temporary_exception_pollution`
- `SM07_scoped_rule_migration`
- `PC02_sop_patch_01`
- `PC02_sop_patch_02`
- `EP03_recall_then_modify`
