from __future__ import annotations

import json
from pathlib import Path

from rsimem.adamem_formal_launcher import FORMAL_SCHEMA, build_formal_manifest


def test_formal_manifest_requires_all_current_screening_families(tmp_path: Path) -> None:
    screening = {
        "schema": "rsimem-adamem-full-suite-screening-manifest-v1",
        "manifest_digest": "screening-digest",
        "execution_order": ["SM01_preference_adoption", "SM02_constraint_retention"],
        "records": [
            {"family_id": "SM01_preference_adoption", "config": "/tmp/sm01.yaml", "config_digest": "a", "cutover_label": "learn", "category": "semantic-targeted"},
            {"family_id": "SM02_constraint_retention", "config": "/tmp/sm02.yaml", "config_digest": "b", "cutover_label": "learn", "category": "semantic-targeted"},
        ],
    }
    result = {"manifest_digest": "screening-digest", "results": [
        {"family_id": "SM01_preference_adoption", "status": "precovered"},
        {"family_id": "SM02_constraint_retention", "status": "accepted", "screening_topology": "single_full_sequence"},
    ]}
    mp = tmp_path / "manifest.json"; rp = tmp_path / "result.json"
    mp.write_text(json.dumps(screening), encoding="utf-8"); rp.write_text(json.dumps(result), encoding="utf-8")
    try:
        build_formal_manifest(screening_manifest=mp, screening_result=rp, output_root=tmp_path, repo_root=tmp_path)
    except ValueError as exc:
        assert "25 accepted" in str(exc)
    else:
        raise AssertionError("incomplete screening must not freeze formal manifest")
