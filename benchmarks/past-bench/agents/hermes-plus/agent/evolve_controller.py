from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Iterable, Sequence


_ALL_MECHANISMS = {"planning", "memory", "skill", "gate", "closeout"}
_HEADER_RE = re.compile(r"^\[(tags|meta):\s*(.*?)\]\s*$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_/-]{2,}")

_BINDING_KINDS = {"constraint", "format", "policy", "preference", "rule"}
_PROCEDURE_KINDS = {"procedure", "workflow"}
_SESSION_SEARCH_HINTS = (
    "approved",
    "chosen",
    "decided",
    "earlier",
    "history",
    "last session",
    "last time",
    "latest",
    "previous",
    "prior",
    "recall",
    "record",
    "session",
    "settled",
    "version",
)
_PROCEDURE_HINTS = (
    "checklist",
    "workflow",
    "procedure",
    "runbook",
    "playbook",
    "sop",
    "steps",
    "triage",
    "handoff",
    "migration",
    "incident",
    "followup",
)
_UPDATE_HINTS = (
    "changed",
    "corrected",
    "deprecated",
    "expire",
    "instead",
    "migrate",
    "no longer",
    "outdated",
    "patch",
    "preserve",
    "replace",
    "revise",
    "scope",
    "stale",
    "supersede",
    "temporary",
    "unchanged",
    "update",
    "updating",
)
_STOPWORDS = {
    "about",
    "after",
    "agent",
    "all",
    "and",
    "before",
    "call",
    "conversation",
    "current",
    "details",
    "exact",
    "follow",
    "from",
    "have",
    "into",
    "latest",
    "message",
    "note",
    "only",
    "past",
    "previous",
    "rule",
    "same",
    "session",
    "task",
    "that",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "update",
    "user",
    "using",
    "with",
    "work",
}


def active_mechanisms() -> set[str]:
    raw = os.getenv("HERMES_PLUS_MECHANISMS", "all").strip().lower()
    if raw == "all":
        return set(_ALL_MECHANISMS)
    if raw == "none":
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def mechanism_enabled(name: str) -> bool:
    return name in active_mechanisms()


@dataclass(frozen=True)
class ParsedMemoryEntry:
    raw: str
    body: str
    tags: tuple[str, ...]
    meta: dict[str, str]

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "").strip().lower()

    @property
    def binding(self) -> bool:
        raw = self.meta.get("binding", "").strip().lower()
        if raw in {"1", "true", "yes"}:
            return True
        if raw in {"0", "false", "no"}:
            return False
        return self.kind in _BINDING_KINDS or "must " in self.body.lower()

    @property
    def scope(self) -> str:
        return self.meta.get("scope", "").strip()

    @property
    def entity(self) -> str:
        return self.meta.get("entity", "").strip()

    @property
    def supersedes(self) -> str:
        return self.meta.get("supersedes", "").strip()

    @property
    def expires_at(self) -> str:
        return self.meta.get("expires_at", "").strip()

    @property
    def summary(self) -> str:
        first_line = self.body.splitlines()[0].strip() if self.body.strip() else ""
        if len(first_line) <= 140:
            return first_line
        return first_line[:137].rstrip() + "..."


@dataclass(frozen=True)
class EvolvePlan:
    relevant_entries: tuple[ParsedMemoryEntry, ...]
    relevant_skills: tuple[str, ...]
    exact_recall: bool
    update_task: bool
    procedure_task: bool
    needs_session_search: bool
    needs_skill_lookup: bool


def encode_memory_entry(
    *,
    content: str,
    tags: str | None = None,
    kind: str | None = None,
    scope: str | None = None,
    entity: str | None = None,
    source: str | None = None,
    confidence: str | None = None,
    binding: bool | None = None,
    expires_at: str | None = None,
    supersedes: str | None = None,
    example: str | None = None,
) -> str:
    body = (content or "").strip()
    if example:
        body = f"{body}\n\nExample:\n{example.strip()}"

    headers: list[str] = []
    normalized_tags = _normalize_csv(tags)
    if normalized_tags:
        headers.append(f"[tags: {', '.join(normalized_tags)}]")

    meta_items: list[str] = []
    for key, value in (
        ("kind", _clean_scalar(kind)),
        ("scope", _clean_scalar(scope)),
        ("entity", _clean_scalar(entity)),
        ("source", _clean_scalar(source)),
        ("confidence", _clean_scalar(confidence)),
        ("expires_at", _clean_scalar(expires_at)),
        ("supersedes", _clean_scalar(supersedes)),
    ):
        if value:
            meta_items.append(f"{key}={value}")
    if binding is not None:
        meta_items.append(f"binding={'true' if binding else 'false'}")
    if meta_items:
        headers.append(f"[meta: {', '.join(meta_items)}]")

    if not headers:
        return body
    return "\n".join(headers + [body])


def parse_memory_entry(raw: str) -> ParsedMemoryEntry:
    tags: list[str] = []
    meta: dict[str, str] = {}
    body_lines: list[str] = []

    in_header = True
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if in_header:
            match = _HEADER_RE.match(stripped)
            if match:
                kind, payload = match.groups()
                if kind.lower() == "tags":
                    tags.extend(_normalize_csv(payload))
                else:
                    meta.update(_parse_meta_items(payload))
                continue
        in_header = False
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    return ParsedMemoryEntry(
        raw=raw or "",
        body=body,
        tags=tuple(dict.fromkeys(tags)),
        meta=meta,
    )


def render_memory_block(target: str, entries: Sequence[str], current: int, limit: int) -> str:
    if not entries:
        return ""

    pct = int((current / limit) * 100) if limit > 0 else 0
    separator = "=" * 46
    if not mechanism_enabled("memory"):
        separator = "═" * 46
        content = "\n§\n".join(entries)
        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"
        return f"{separator}\n{header}\n{separator}\n{content}"

    if target == "user":
        header = f"USER PROFILE (who the user is) [{pct}% - {current:,}/{limit:,} chars]"
        rendered = [parse_memory_entry(entry).body or entry.strip() for entry in entries]
        content = "\n\n".join(item for item in rendered if item)
        return f"{separator}\n{header}\n{separator}\n{content}"

    parsed = [parse_memory_entry(entry) for entry in entries]
    active = [entry for entry in parsed if not _is_expired(entry)]
    binding = [entry for entry in active if entry.binding]
    archival = [entry for entry in active if not entry.binding]
    expired_count = len(parsed) - len(active)

    lines = [
        separator,
        f"MEMORY (binding rules + archival context) [{pct}% - {current:,}/{limit:,} chars]",
        separator,
    ]
    if binding:
        lines.append(
            "BINDING RULES: follow these defaults unless the current user message explicitly overrides them."
        )
        for index, entry in enumerate(binding, start=1):
            lines.append(_render_entry(entry, index, label="RULE"))
    if archival:
        if binding:
            lines.append("")
        lines.append("ARCHIVAL CONTEXT: useful prior state and cases; verify exact historical details when needed.")
        for index, entry in enumerate(archival, start=1):
            lines.append(_render_entry(entry, index, label="CONTEXT"))
    if expired_count:
        lines.append("")
        lines.append(
            f"Note: {expired_count} expired memory entr{'y' if expired_count == 1 else 'ies'} were kept on disk but not injected."
        )
    return "\n".join(lines).strip()


def build_evolve_plan(
    *,
    user_message: str,
    memory_entries: Sequence[str],
    available_tools: Iterable[str],
    skill_names: Sequence[str] | None = None,
) -> EvolvePlan:
    tools = {tool.strip() for tool in available_tools if tool}
    parsed_entries = [parse_memory_entry(entry) for entry in memory_entries]
    active_entries = [entry for entry in parsed_entries if not _is_expired(entry)]
    relevant_entries = _rank_entries_for_message(active_entries, user_message)
    exact_recall = _looks_like_exact_recall_task(user_message)
    update_task = _looks_like_update_task(user_message)
    procedure_task = _looks_like_procedure_task(user_message, relevant_entries)
    has_skill_tools = any(tool in tools for tool in ("skills_list", "skill_view", "skill_manage"))
    relevant_skills = (
        _rank_skill_names(skill_names or (), user_message, relevant_entries)
        if has_skill_tools and (procedure_task or exact_recall or update_task)
        else []
    )

    return EvolvePlan(
        relevant_entries=tuple(relevant_entries),
        relevant_skills=tuple(relevant_skills),
        exact_recall=exact_recall,
        update_task=update_task,
        procedure_task=procedure_task,
        needs_session_search=exact_recall and "session_search" in tools,
        needs_skill_lookup=has_skill_tools and (procedure_task or bool(relevant_skills)),
    )


def build_turn_guidance(
    *,
    user_message: str,
    memory_entries: Sequence[str],
    available_tools: Iterable[str],
    skill_names: Sequence[str] | None = None,
) -> str:
    if not user_message:
        return ""

    tools = {tool.strip() for tool in available_tools if tool}
    plan = build_evolve_plan(
        user_message=user_message,
        memory_entries=memory_entries,
        available_tools=tools,
        skill_names=skill_names,
    )
    relevant_entries = list(plan.relevant_entries)
    relevant_skills = list(plan.relevant_skills)

    lines = ["TURN-SPECIFIC EVOLVE PLAN:"]

    if relevant_entries and "memory" in tools:
        lines.append(
            "Relevant saved defaults for this task:"
        )
        for entry in relevant_entries[:3]:
            scope_bits = [bit for bit in (entry.kind, entry.scope, entry.entity) if bit]
            prefix = f"- {' | '.join(scope_bits)}: " if scope_bits else "- "
            lines.append(prefix + entry.summary)
        lines.append(
            "Use these as defaults. If they conflict with the current task or look stale, verify before reusing them."
        )

    if relevant_skills:
        lines.append("Existing skills worth checking first:")
        for name in relevant_skills[:3]:
            lines.append(f"- {name}")
        lines.append(
            "Inspect a close skill before re-deriving the workflow. Reuse it as the default path, and patch it if only part of the procedure changed."
        )
    elif plan.procedure_task and any(tool in tools for tool in ("skills_list", "skill_view", "skill_manage")):
        lines.append(
            "This looks like a reusable workflow task. Check for an existing skill before inventing a new procedure from scratch."
        )

    if plan.needs_session_search:
        query = build_session_search_query(user_message, relevant_entries)
        lines.append(
            "This task likely depends on the latest exact prior state. Before any external action or final answer, call "
            f"`session_search` with a focused query such as: {query!r}."
        )
        lines.append(
            "Use session_search for one-off historical records, approvals, settled versions, exact IDs, dates, and exception lists."
        )
        if plan.procedure_task:
            lines.append(
                "Recover the exact record first, then apply the reusable workflow. Do not let a generic skill override the latest factual state."
            )

    if plan.update_task:
        lines.append(
            "This looks like an update/correction task. Replace or supersede stale state, preserve unaffected scope, "
            "and do not promote temporary exceptions into durable defaults."
        )

    if (plan.procedure_task or any(entry.kind in _PROCEDURE_KINDS for entry in relevant_entries)) and "skill_manage" in tools:
        lines.append(
            "If an existing skill is close but outdated, patch it instead of creating a second overlapping skill."
        )

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_closeout_guidance(
    *,
    user_message: str,
    memory_entries: Sequence[str],
    available_tools: Iterable[str],
    skill_names: Sequence[str] | None = None,
) -> str:
    if not user_message:
        return ""

    tools = {tool.strip() for tool in available_tools if tool}
    if "skill_manage" not in tools:
        return ""

    plan = build_evolve_plan(
        user_message=user_message,
        memory_entries=memory_entries,
        available_tools=tools,
        skill_names=skill_names,
    )
    relevant_entries = list(plan.relevant_entries)
    relevant_skills = list(plan.relevant_skills)

    if not plan.procedure_task and not relevant_skills and not any(
        entry.kind in _PROCEDURE_KINDS for entry in relevant_entries
    ):
        return ""

    lines = ["TASK CLOSEOUT EVOLVE CHECK:"]
    if relevant_skills:
        lines.append("Existing skills close to this workflow:")
        for name in relevant_skills[:3]:
            lines.append(f"- {name}")
        lines.append(
            "Before deciding to create anything new, review one of these skills and patch it if the workflow mostly matches."
        )
    else:
        lines.append(
            "If this task produced a reusable workflow, save it as a skill before ending the episode."
        )

    if plan.update_task:
        lines.append(
            "This task looks like a procedure update. Preserve unaffected steps and patch the existing skill instead of creating a parallel variant."
        )

    lines.append(
        "Capture the stable workflow, scope, and boundaries - not the one-off entity names, dates, IDs, or approvals."
    )
    if plan.exact_recall:
        lines.append(
            "Keep exact historical facts in session_search/memory. Only save the reusable retrieval procedure as a skill."
        )
    return "\n".join(lines)


def build_action_gate_guidance(
    *,
    user_message: str,
    memory_entries: Sequence[str],
    available_tools: Iterable[str],
    skill_names: Sequence[str] | None = None,
    completed_tool_names: Iterable[str] = (),
) -> str:
    if not user_message:
        return ""

    tools = {tool.strip() for tool in available_tools if tool}
    completed = {tool.strip() for tool in completed_tool_names if tool}
    plan = build_evolve_plan(
        user_message=user_message,
        memory_entries=memory_entries,
        available_tools=tools,
        skill_names=skill_names,
    )

    missing: list[str] = []
    if plan.needs_session_search and "session_search" not in completed:
        query = build_session_search_query(user_message, plan.relevant_entries)
        missing.append(
            f"call `session_search` first with a focused query such as {query!r} to recover exact prior state"
        )

    skill_lookup_done = bool(completed & {"skills_list", "skill_view", "skill_manage"})
    if plan.needs_skill_lookup and not skill_lookup_done:
        if plan.relevant_skills:
            names = ", ".join(plan.relevant_skills[:3])
            missing.append(
                f"inspect the closest existing skill before deriving the workflow: {names}"
            )
        else:
            missing.append(
                "check the skills index before deriving a reusable workflow from scratch"
            )

    if not missing:
        return ""

    lines = ["EVOLVE ACTION GATE:"]
    lines.append(
        "Do not give the final answer or take irreversible external actions until these evolve preconditions are satisfied:"
    )
    for item in missing:
        lines.append(f"- {item}")
    lines.append(
        "This gate is based on ability signals in the user request."
    )
    return "\n".join(lines)


def build_session_search_query(user_message: str, relevant_entries: Sequence[ParsedMemoryEntry]) -> str:
    seed = user_message or ""

    query_terms: list[str] = []

    for entry in relevant_entries[:3]:
        for item in (entry.entity, entry.scope):
            cleaned = item.strip()
            if cleaned and cleaned not in query_terms:
                query_terms.append(cleaned)

    for token in _salient_tokens(seed):
        if token not in query_terms:
            query_terms.append(token)
        if len(query_terms) >= 6:
            break

    if not query_terms:
        query_terms = ["previous", "approved", "latest", "history"]
    return " ".join(query_terms[:6])


def _parse_meta_items(payload: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for item in payload.split(","):
        piece = item.strip()
        if not piece or "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            meta[key] = value
    return meta


def _normalize_csv(value: str | None) -> list[str]:
    if not value:
        return []
    normalized: list[str] = []
    for item in value.split(","):
        cleaned = item.strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _clean_scalar(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().replace("\n", " ")


def _is_expired(entry: ParsedMemoryEntry, now: datetime | None = None) -> bool:
    expires_at = entry.expires_at
    if not expires_at:
        return False
    expiry = _parse_datetime(expires_at)
    if expiry is None:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    return expiry <= now


def _parse_datetime(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _render_entry(entry: ParsedMemoryEntry, index: int, *, label: str) -> str:
    descriptor: list[str] = []
    for field in (entry.kind, entry.scope, entry.entity):
        cleaned = field.strip()
        if cleaned:
            descriptor.append(cleaned)
    if entry.supersedes:
        descriptor.append(f"supersedes={entry.supersedes}")
    header = f"[{label} {index}"
    if descriptor:
        header += " | " + " | ".join(descriptor[:3])
    header += "]"
    body = entry.body or entry.raw.strip()
    return f"{header}\n{body}".strip()


def _rank_entries_for_message(entries: Sequence[ParsedMemoryEntry], user_message: str) -> list[ParsedMemoryEntry]:
    terms = set(_salient_tokens(user_message))
    if not terms:
        return []

    scored: list[tuple[int, ParsedMemoryEntry]] = []
    for entry in entries:
        haystack_parts = [entry.body]
        haystack_parts.extend(entry.tags)
        haystack_parts.extend(
            entry.meta.get(key, "")
            for key in ("kind", "scope", "entity", "source", "supersedes")
        )
        entry_terms = {token.lower() for token in _salient_tokens(" ".join(part for part in haystack_parts if part))}
        score = 0
        for term in terms:
            lowered = term.lower()
            if lowered in entry_terms:
                entity_terms = {token.lower() for token in _salient_tokens(entry.meta.get("entity", ""))}
                score += 3 if lowered in entity_terms else 1
        if entry.binding:
            score += 1
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored] if scored else []


def _salient_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text or ""):
        lowered = match.lower()
        if lowered in _STOPWORDS or lowered.isdigit():
            continue
        if len(lowered) < 4:
            continue
        if match not in tokens:
            tokens.append(match)
    return tokens[:12]


def _looks_like_exact_recall_task(user_message: str) -> bool:
    lowered = (user_message or "").lower()
    return any(hint in lowered for hint in _SESSION_SEARCH_HINTS)


def _looks_like_procedure_task(
    user_message: str,
    relevant_entries: Sequence[ParsedMemoryEntry] | None = None,
) -> bool:
    lowered = (user_message or "").lower()
    if any(hint in lowered for hint in _PROCEDURE_HINTS):
        return True
    if relevant_entries:
        for entry in relevant_entries:
            if entry.kind in _PROCEDURE_KINDS:
                return True
            body = entry.body.lower()
            if any(hint in body for hint in _PROCEDURE_HINTS):
                return True
    return False


def _looks_like_update_task(user_message: str) -> bool:
    lowered = (user_message or "").lower()
    return any(hint in lowered for hint in _UPDATE_HINTS)


def _rank_skill_names(
    skill_names: Sequence[str],
    user_message: str,
    relevant_entries: Sequence[ParsedMemoryEntry],
) -> list[str]:
    terms = {token.lower() for token in _salient_tokens(user_message)}
    for entry in relevant_entries[:3]:
        terms.update(
            token.lower()
            for token in _salient_tokens(
                " ".join(
                    piece
                    for piece in (entry.kind, entry.scope, entry.entity, entry.supersedes)
                    if piece
                )
            )
        )
    if not terms:
        return []

    scored: list[tuple[int, str]] = []
    for raw_name in skill_names:
        name = (raw_name or "").strip()
        if not name:
            continue
        normalized = name.replace("_", " ").replace("-", " ")
        name_terms = {token.lower() for token in _salient_tokens(normalized)}
        score = len(terms & name_terms)
        if score == 0:
            continue
        if any(term in name.lower() for term in ("sop", "workflow", "playbook", "runbook")):
            score += 1
        scored.append((score, name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered: list[str] = []
    for _, name in scored:
        if name not in ordered:
            ordered.append(name)
    return ordered
