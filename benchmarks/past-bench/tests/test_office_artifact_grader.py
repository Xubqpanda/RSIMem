import base64
from pathlib import Path

import pytest

from past_bench.graders.office_artifact_checks import (
    compare_docx_expected_paragraphs,
    compare_docx_tables,
    compare_pdf_layout,
    compare_pptx_changed_content,
    compare_pptx_files,
    compare_xlsx_changed_cells,
    compare_xlsx_changed_chart_props,
)
from past_bench.graders.registry import get_grader
from past_bench.models.task import TaskDefinition


ROOT = Path(__file__).resolve().parents[1]


def _load_task(task_dir_name: str) -> TaskDefinition:
    return TaskDefinition.from_yaml(ROOT / "past_bench_tasks" / task_dir_name / "task.yaml")


def _binary_snapshot(path: Path) -> dict[str, str | int]:
    return {
        "content": base64.b64encode(path.read_bytes()).decode("ascii"),
        "encoding": "base64",
        "size_bytes": path.stat().st_size,
    }


def _env_snapshot_for_mode(task_dir: Path, task: TaskDefinition, *, use_gold: bool) -> dict[str, dict[str, str | int]]:
    env_snapshot: dict[str, dict[str, str | int]] = {}
    for target in task.env_snapshot_files:
        basename = Path(target).name
        direct = (task_dir / ("gold" if use_gold else "fixtures")) / basename
        alt = task_dir / "gold" / basename.replace("_source.", "_gold.")
        source = direct if direct.exists() else alt if use_gold and alt.exists() else None
        if source is not None and source.exists():
            env_snapshot[f"file:{target}"] = _binary_snapshot(source)
    return env_snapshot


def test_task_definition_parses_office_artifact_checks():
    task = _load_task("T23_writer_heading_system_reconstruction")

    assert task.task_id == "T23_writer_heading_system_reconstruction"
    assert len(task.artifact_checks) == 1
    assert task.artifact_checks[0].func == "compare_docx_expected_paragraphs"


def test_writer_office_grader_scores_gold_snapshot():
    task_dir = ROOT / "past_bench_tasks" / "T23_writer_heading_system_reconstruction"
    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    grader = get_grader(task.task_id, tasks_dir=task_dir.parent, task_dir=task_dir)

    env_snapshot = {
        "file:/workspace/fixtures/heading_system_reconstruction_source.docx": _binary_snapshot(
            task_dir / "gold" / "heading_system_reconstruction_gold.docx"
        )
    }

    scores = grader.grade([], [], task, env_snapshot=env_snapshot)

    assert scores.completion == 1.0
    assert scores.robustness == 1.0


def test_docx_changed_paragraphs_zero_source_baseline():
    task_dir = ROOT / "past_bench_tasks" / "T23_writer_heading_system_reconstruction"
    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    check = task.artifact_checks[0]
    options = dict(check.options)
    options["source_path"] = str(task_dir / options["source_path"])

    assert compare_docx_expected_paragraphs(
        str(task_dir / check.expected),
        str(task_dir / "fixtures" / "heading_system_reconstruction_source.docx"),
        **options,
    ) == 0.0
    assert compare_docx_expected_paragraphs(
        str(task_dir / check.expected),
        str(task_dir / "gold" / "heading_system_reconstruction_gold.docx"),
        **options,
    ) == 1.0


def test_xlsx_changed_cells_detect_calc_source_mismatch():
    task = _load_task("T25_calc_revenue_model_reconciliation")
    check = task.artifact_checks[0]
    task_dir = ROOT / "past_bench_tasks" / "T25_calc_revenue_model_reconciliation"
    options = dict(check.options)
    options["source_path"] = str(task_dir / options["source_path"])

    score = compare_xlsx_changed_cells(
        str(task_dir / check.expected),
        str(task_dir / "fixtures" / "revenue_model_reconciliation_source.xlsx"),
        **options,
    )

    assert score == 0.0


def test_print_ready_grader_scores_partial_when_pdf_missing():
    task_dir = ROOT / "past_bench_tasks" / "T26_calc_print_ready_regional_brief"
    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    grader = get_grader(task.task_id, tasks_dir=task_dir.parent, task_dir=task_dir)

    env_snapshot = {
        "file:/workspace/fixtures/print_ready_regional_brief_source.xlsx": _binary_snapshot(
            task_dir / "gold" / "print_ready_regional_brief_gold.xlsx"
        )
    }

    scores = grader.grade([], [], task, env_snapshot=env_snapshot)

    assert scores.completion == 0.55


def test_docx_table_comparator_detects_source_mismatch():
    task_dir = ROOT / "past_bench_tasks" / "T24_writer_meeting_notes_to_formal_minutes"

    score = compare_docx_tables(
        str(task_dir / "gold" / "formal_minutes_gold.docx"),
        str(task_dir / "fixtures" / "formal_minutes_source.docx"),
    )

    assert score == 0.0


def test_meeting_minutes_changed_text_zeroes_source_baseline():
    task_dir = ROOT / "past_bench_tasks" / "T24_writer_meeting_notes_to_formal_minutes"
    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    check = task.artifact_checks[0]
    options = dict(check.options)
    options["source_path"] = str(task_dir / options["source_path"])

    assert compare_docx_expected_paragraphs(
        str(task_dir / check.expected),
        str(task_dir / "fixtures" / "formal_minutes_source.docx"),
        **options,
    ) == 0.0
    assert compare_docx_expected_paragraphs(
        str(task_dir / check.expected),
        str(task_dir / "gold" / "formal_minutes_gold.docx"),
        **options,
    ) == 1.0


def test_impress_and_pdf_comparators_distinguish_gold_from_source():
    pptx_task_dir = ROOT / "past_bench_tasks" / "T27_impress_agenda_summary_pair"
    assert compare_pptx_files(
        str(pptx_task_dir / "gold" / "agenda_summary_pair_gold.pptx"),
        str(pptx_task_dir / "gold" / "agenda_summary_pair_gold.pptx"),
        approximate=True,
    ) == 1.0
    assert compare_pptx_files(
        str(pptx_task_dir / "gold" / "agenda_summary_pair_gold.pptx"),
        str(pptx_task_dir / "fixtures" / "agenda_summary_pair_source.pptx"),
        approximate=True,
    ) < 1.0

    pdf_task_dir = ROOT / "past_bench_tasks" / "T26_calc_print_ready_regional_brief"
    assert compare_pdf_layout(
        str(pdf_task_dir / "gold" / "print_ready_regional_brief_gold.pdf"),
        str(pdf_task_dir / "gold" / "print_ready_regional_brief_gold.pdf"),
    ) == 1.0


def test_pptx_changed_content_comparator_zeroes_source_baseline():
    for task_name, source_name, gold_name, changed_slides, unchanged_slides in [
        (
            "T27_impress_agenda_summary_pair",
            "agenda_summary_pair_source.pptx",
            "agenda_summary_pair_gold.pptx",
            [2, 7],
            [1, 3, 4, 5, 6],
        ),
        (
            "T28_impress_stale_chart_replacement",
            "chart_refresh_source.pptx",
            "chart_refresh_gold.pptx",
            [2, 3, 5],
            [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        ),
    ]:
        task_dir = ROOT / "past_bench_tasks" / task_name
        source_path = task_dir / "fixtures" / source_name
        gold_path = task_dir / "gold" / gold_name
        assert compare_pptx_changed_content(
            str(gold_path),
            str(source_path),
            source_path=str(source_path),
            changed_slides=changed_slides,
            unchanged_slides=unchanged_slides,
        ) == 0.0
        assert compare_pptx_changed_content(
            str(gold_path),
            str(gold_path),
            source_path=str(source_path),
            changed_slides=changed_slides,
            unchanged_slides=unchanged_slides,
        ) == 1.0


def test_xlsx_changed_chart_comparators_zero_source_baseline():
    for task_name, source_name, gold_name, sheet_name in [
        (
            "T25_calc_revenue_model_reconciliation",
            "revenue_model_reconciliation_source.xlsx",
            "revenue_model_reconciliation_gold.xlsx",
            "Summary",
        ),
        (
            "T26_calc_print_ready_regional_brief",
            "print_ready_regional_brief_source.xlsx",
            "print_ready_regional_brief_gold.xlsx",
            "RegionalBrief",
        ),
    ]:
        task_dir = ROOT / "past_bench_tasks" / task_name
        source_path = task_dir / "fixtures" / source_name
        gold_path = task_dir / "gold" / gold_name
        assert compare_xlsx_changed_chart_props(
            str(gold_path),
            str(source_path),
            source_path=str(source_path),
            sheet_name=sheet_name,
            chart_props=["type", "title"],
        ) == 0.0
        assert compare_xlsx_changed_chart_props(
            str(gold_path),
            str(gold_path),
            source_path=str(source_path),
            sheet_name=sheet_name,
            chart_props=["type", "title"],
        ) == 1.0


@pytest.mark.parametrize(
    ("task_name", "expected_source", "expected_gold"),
    [
        ("T29_writer_citation_repair_under_distractors", 0.0, 1.0),
        ("T30_writer_appendix_cross_reference_repair", 0.0, 1.0),
        ("T31_writer_contract_redline_resolution", 0.0, 1.0),
        ("T32_writer_data_to_narrative_results_summary", 0.0, 1.0),
        ("T33_writer_calc_to_writer_executive_memo", 0.0, 1.0),
        ("T34_writer_to_impress_briefing_source", 0.0, 1.0),
        ("T35_writer_bilingual_glossary_control", 0.0, 1.0),
        ("T36_writer_submission_ready_report_bundle", 0.0, 1.0),
        ("T37_writer_template_parameter_fill_with_decoys", 0.0, 1.0),
        ("T38_writer_figure_and_table_renumber_after_insertions", 0.0, 1.0),
        ("T39_writer_cross_source_synthesis_brief", 0.0, 1.0),
        ("T40_calc_ambiguous_entity_resolution", 0.0, 1.0),
        ("T41_calc_decoy_summary_repair", 0.0, 1.0),
        ("T42_calc_approval_ops_workbook", 0.0, 1.0),
        ("T43_calc_audit_rollforward_hidden_locked", 0.0, 1.0),
        ("T44_calc_parameterized_month_close_pack", 0.0, 1.0),
        ("T45_calc_to_writer_variance_memo", 0.0, 1.0),
        ("T46_calc_to_impress_kpi_slide", 0.0, 1.0),
        ("T47_calc_multi_source_consolidation", 0.0, 1.0),
        ("T48_calc_what_if_scenario_manager", 0.0, 1.0),
        ("T49_calc_exception_tagging_evidence_trace", 0.0, 1.0),
        ("T50_calc_export_controlled_deliverable_pack", 0.0, 1.0),
        ("T51_calc_template_drift_recovery", 0.0, 1.0),
        ("T52_calc_cross_suite_board_pack", 0.0, 1.0),
        ("T53_impress_conditional_slide_selection_under_distractors", 0.0, 1.0),
        ("T54_impress_calc_to_impress_dashboard_slide", 0.0, 1.0),
        ("T55_impress_writer_to_impress_executive_summary_deck", 0.0, 1.0),
        ("T56_impress_variant_sales_deck_from_asset_bundle", 0.0, 1.0),
        ("T57_impress_large_deck_cleanup_deduplication", 0.0, 1.0),
        ("T58_impress_cross_suite_launch_readout", 0.0, 1.0),
        ("T59_impress_evidence_backed_decision_deck", 0.0, 1.0),
    ],
)
def test_new_office_tasks_source_and_gold_baselines(task_name: str, expected_source: float, expected_gold: float):
    task_dir = ROOT / "past_bench_tasks" / task_name
    task = TaskDefinition.from_yaml(task_dir / "task.yaml")
    grader = get_grader(task.task_id, tasks_dir=task_dir.parent, task_dir=task_dir)

    source_scores = grader.grade([], [], task, env_snapshot=_env_snapshot_for_mode(task_dir, task, use_gold=False))
    gold_scores = grader.grade([], [], task, env_snapshot=_env_snapshot_for_mode(task_dir, task, use_gold=True))

    assert source_scores.completion == expected_source
    assert gold_scores.completion == expected_gold
