#!/usr/bin/env python3
"""Build the three public-input episodic oracle homes with Hermes SessionDB.

The resulting state is a real native search database. Do not add benchmark
expectations, graders, answers, or non-episodic memory surfaces here.
"""

from __future__ import annotations

import json
from pathlib import Path

from hermes_state import SessionDB


ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = ROOT / "benchmarks/past-bench/self-evolve-tasks-v2/_rsimem_oracles/episodic"


SEEDS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "EP01_prior_case_recall": (
        "oracle-ep01-vendor-decision-v1",
        (
            ("user", "Vendor X Contract Decision: after review, Vendor X is not approved for production until SOC2 is complete. A staging-only pilot is allowed through 2026-05-31."),
            ("assistant", "Recorded the Vendor X decision for future reference: production is not approved; only the staging pilot through 2026-05-31 is allowed."),
        ),
    ),
    "EP02_exception_list_recall": (
        "oracle-ep02-wave-cedar-v1",
        (
            ("user", "Wave Cedar session-bounded restart exception: activate and approve INTG-EU-002, INTG-US-004, and INTG-AP-006 only. Keep INTG-EU-013, INTG-US-014, and INTG-AP-015 suspended."),
            ("assistant", "Recorded the Wave Cedar carry-forward subset as a session-bounded restart exception only; the remaining listed connectors stay suspended."),
        ),
    ),
    "EP03_recall_then_modify": (
        "oracle-ep03-onboarding-playbook-v1",
        (
            ("user", "Settled onboarding playbook: HR emails the welcome packet, the manager books the kickoff call, IT provisions the laptop, and the mentor handoff SLA is 24 hours. Preserve every other approved playbook field."),
            ("assistant", "Recorded the approved onboarding playbook with the mentor handoff SLA updated from 48 hours to 24 hours; all other fields remain unchanged."),
        ),
    ),
}


def build_seed(family_id: str, session_id: str, messages: tuple[tuple[str, str], ...]) -> None:
    home = SEED_ROOT / family_id
    sessions = home / "sessions"
    home.mkdir(parents=True, exist_ok=True)
    sessions.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id=session_id, source="rsimem-oracle", model="oracle")
        for role, content in messages:
            db.append_message(session_id=session_id, role=role, content=content)
    finally:
        db.close()
    (sessions / f"{session_id}.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "model": "oracle",
                "platform": "rsimem-oracle",
                "messages": [{"role": role, "content": content} for role, content in messages],
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    for family_id, (session_id, messages) in SEEDS.items():
        build_seed(family_id, session_id, messages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
