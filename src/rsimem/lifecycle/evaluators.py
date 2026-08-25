"""Evaluator implementations and the LLM integration boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..memory.contracts import MemoryKind
from .contracts import (
    ContextAction,
    CompletionStatus,
    ContextEvaluation,
    ContextEvaluationRequest,
    ContextEvaluator,
    EvaluationSignal,
    MemoryScope,
    TemporalValidity,
    WritebackAction,
)


class JsonLlmContextEvaluator:
    """Adapt a JSON-returning model call to the context evaluator contract.

    The model client is injected by the host, so this package stays independent
    of OpenAI, Anthropic, Hermes, or a local inference runtime. The evaluator
    requires one signal per segment and rejects unknown or duplicate IDs before
    a future writeback coordinator can act on the result.
    """

    def __init__(self, complete: Callable[[str], str], *, name: str = "llm-context-evaluator") -> None:
        self._complete = complete
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, request: ContextEvaluationRequest) -> ContextEvaluation:
        raw = self._complete(self.build_prompt(request))
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("context evaluator returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("signals"), list):
            raise ValueError("context evaluator response must contain a signals list")

        expected = {segment.segment_id for segment in request.segments}
        seen: set[str] = set()
        signals: list[EvaluationSignal] = []
        for item in payload["signals"]:
            if not isinstance(item, dict):
                raise ValueError("each context evaluation signal must be an object")
            segment_id = item.get("segment_id")
            if not isinstance(segment_id, str) or segment_id not in expected:
                raise ValueError("context evaluator returned an unknown segment_id")
            if segment_id in seen:
                raise ValueError("context evaluator returned duplicate segment_id")
            seen.add(segment_id)
            signals.append(EvaluationSignal(
                segment_id=segment_id,
                context_action=item.get("context_action", ContextAction.RETAIN),
                writeback_action=item.get("writeback_action", WritebackAction.DEFER),
                utility_estimate=float(item.get("utility_estimate", 0.0)),
                confidence=float(item.get("confidence", 0.0)),
                memory_kind=(
                    MemoryKind(item["memory_kind"])
                    if item.get("memory_kind") is not None
                    else None
                ),
                reason_codes=tuple(item.get("reason_codes", ())),
                completion_status=item.get("completion_status", CompletionStatus.UNKNOWN),
                completion_evidence=tuple(item.get("completion_evidence", ())),
                scope=(MemoryScope(item["scope"]) if item.get("scope") else None),
                temporal_validity=(
                    TemporalValidity(item["temporal_validity"])
                    if item.get("temporal_validity")
                    else None
                ),
                reusable_facts=tuple(item.get("reusable_facts", ())),
                reusable_procedures=tuple(item.get("reusable_procedures", ())),
                update_hints=tuple(item.get("update_hints", ())),
            ))
        if seen != expected:
            missing = ", ".join(sorted(expected - seen))
            raise ValueError(f"context evaluator omitted segment IDs: {missing}")
        return ContextEvaluation(
            evaluation_id=request.evaluation_id,
            evaluator=self.name,
            trigger=request.trigger,
            signals=tuple(signals),
            policy_version=str(payload.get("policy_version", "llm")),
            input_chars=sum(len(segment.content) for segment in request.segments),
        )

    @staticmethod
    def build_prompt(request: ContextEvaluationRequest) -> str:
        segments = [
            {
                "segment_id": segment.segment_id,
                "role": segment.role,
                "content": segment.content,
                "token_count": segment.token_count,
                "completed": segment.completed,
            }
            for segment in request.segments
        ]
        schema = {
            "signals": [
                {
                    "segment_id": "string",
                    "context_action": "retain|evict",
                    "writeback_action": "defer|discard|add|update",
                    "memory_kind": "semantic|episodic|procedural|null",
                    "utility_estimate": "number in [0,1]",
                    "confidence": "number in [0,1]",
                    "reason_codes": ["short_machine_readable_code"],
                    "completion_status": "unknown|in_progress|completed|blocked",
                    "completion_evidence": ["short evidence"],
                    "scope": "turn|task|session|user|global|null",
                    "temporal_validity": "transient|current|durable|expired|null",
                    "reusable_facts": ["candidate fact"],
                    "reusable_procedures": ["candidate procedure"],
                    "update_hints": ["candidate update hint"],
                }
            ],
            "policy_version": "string",
        }
        return json.dumps(
            {
                "instruction": (
                    "Evaluate every context segment independently but make the context and memory "
                    "decision jointly. Never evict an active segment. Return JSON only."
                ),
                "trigger": request.trigger.value,
                "active_segment_ids": request.active_segment_ids,
                "response_schema": schema,
                "segments": segments,
            },
            ensure_ascii=True,
        )


class ConservativeContextEvaluator:
    """Deterministic baseline that never evicts or writes back content."""

    name = "conservative-baseline"

    def evaluate(self, request: ContextEvaluationRequest) -> ContextEvaluation:
        return ContextEvaluation(
            evaluation_id=request.evaluation_id,
            evaluator=self.name,
            trigger=request.trigger,
            signals=tuple(
                EvaluationSignal(
                    segment_id=segment.segment_id,
                    context_action=ContextAction.RETAIN,
                    writeback_action=WritebackAction.DEFER,
                    utility_estimate=0.0,
                    confidence=1.0,
                    completion_status=(
                        CompletionStatus.COMPLETED
                        if segment.completed
                        else CompletionStatus.IN_PROGRESS
                    ),
                    completion_evidence=(
                        ("host_segment_completed",)
                        if segment.completed
                        else ("host_segment_unresolved",)
                    ),
                    safe_to_evict=False,
                    unresolved_state=None if segment.completed else "host_unresolved",
                    provenance=(
                        str(request.metadata.get("snapshot_id") or "snapshot_unknown"),
                        segment.segment_id,
                    ),
                    reason_codes=("baseline_no_action",),
                )
                for segment in request.segments
            ),
            policy_version="baseline",
            input_chars=sum(len(segment.content) for segment in request.segments),
        )
