#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall

Searches past session transcripts in SQLite via FTS5, then summarizes the top
matching sessions using a cheap/fast model (same pattern as web_extract).
Returns focused summaries of past conversations rather than raw transcripts,
keeping the main model's context window clean.

Flow:
  1. FTS5 search finds matching messages ranked by relevance
  2. Groups by session, takes the top N unique sessions (default 3)
  3. Loads each session's conversation, truncates to ~100k chars centered on matches
  4. Sends to Gemini Flash with a focused summarization prompt
  5. Returns per-session summaries with metadata
"""

import asyncio
import concurrent.futures
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union

from agent.auxiliary_client import async_call_llm
MAX_SESSION_CHARS = 100_000
MAX_SUMMARY_TOKENS = 10000
_INTEGRATION_ID_RE = re.compile(r"\bINTG-[A-Z]{2,}-\d+\b")


def _clean_excerpt(text: str, limit: int = 280) -> str:
    """Normalize whitespace and truncate a transcript excerpt for fallback recall."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _safe_json_loads(text: str) -> Optional[Any]:
    """Parse a tool payload when it looks like JSON; otherwise return None."""
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate or candidate[0] not in "{[":
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _looks_like_nested_session_search(payload: Any) -> bool:
    """Return True when a tool payload is itself a prior session_search result."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("results"), list)
        and "query" in payload
        and "success" in payload
    )


def _extract_integration_ids(text: str) -> List[str]:
    """Extract ordered unique integration IDs from free text."""
    seen = set()
    ordered: list[str] = []
    for integration_id in _INTEGRATION_ID_RE.findall(text or ""):
        if integration_id in seen:
            continue
        ordered.append(integration_id)
        seen.add(integration_id)
    return ordered


def _integration_matches_query(integration: Dict[str, Any], query_terms: List[str]) -> bool:
    """Heuristic relevance check for integration snapshots in fallback recall."""
    haystack = " ".join(
        str(integration.get(key, ""))
        for key in ("integration_id", "name", "region", "status", "notes")
    ).lower()
    return any(term in haystack for term in query_terms if len(term) > 2)


def _collect_structured_recall_lines(
    *,
    query_terms: List[str],
    messages: List[Dict[str, Any]],
) -> List[str]:
    """Extract deterministic, structured facts from the transcript.

    This is designed to preserve key tool-recorded state changes such as
    integration approvals, which are easy to lose in excerpt-only fallback.
    """
    approved_updates: list[str] = []
    other_updates: list[str] = []
    snapshot_lines: list[str] = []
    assistant_subset_lines: list[str] = []
    seen_update_ids: set[str] = set()
    seen_snapshot_ids: set[str] = set()
    seen_subset_lines: set[str] = set()

    for msg in messages:
        role = str(msg.get("role", "")).upper()
        content = str(msg.get("content") or "")
        if not content.strip():
            continue

        payload = _safe_json_loads(content) if role == "TOOL" else None
        if _looks_like_nested_session_search(payload):
            continue

        if isinstance(payload, dict):
            integration = payload.get("integration")
            if isinstance(integration, dict):
                integration_id = str(integration.get("integration_id") or "").strip()
                if integration_id and integration_id not in seen_update_ids:
                    seen_update_ids.add(integration_id)
                    parts: list[str] = []
                    if integration.get("status") is not None:
                        parts.append(f"status={integration.get('status')}")
                    if "exception_approved" in integration:
                        approved = str(bool(integration.get("exception_approved"))).lower()
                        parts.append(f"exception_approved={approved}")
                    notes = _clean_excerpt(str(integration.get("notes") or ""), limit=140)
                    if notes:
                        parts.append(f"notes={notes}")
                    line = f"- {integration_id}: " + "; ".join(parts)
                    if integration.get("exception_approved") is True:
                        approved_updates.append(line)
                    else:
                        other_updates.append(line)

            integrations = payload.get("integrations")
            if isinstance(integrations, list):
                for integration in integrations:
                    if not isinstance(integration, dict):
                        continue
                    integration_id = str(integration.get("integration_id") or "").strip()
                    if (
                        not integration_id
                        or integration_id in seen_snapshot_ids
                        or not _integration_matches_query(integration, query_terms)
                    ):
                        continue
                    seen_snapshot_ids.add(integration_id)
                    parts = []
                    if integration.get("region"):
                        parts.append(str(integration.get("region")))
                    if integration.get("status"):
                        parts.append(str(integration.get("status")))
                    notes = _clean_excerpt(str(integration.get("notes") or ""), limit=110)
                    if notes:
                        parts.append(notes)
                    snapshot_lines.append(f"- {integration_id}: " + " | ".join(parts))

        if role == "ASSISTANT":
            ids = _extract_integration_ids(content)
            content_lower = content.lower()
            if ids and any(term in content_lower for term in ("approved", "reactivated", "subset", "allow", "exception")):
                line = f"- Assistant referenced subset: {', '.join(ids)}"
                if line not in seen_subset_lines:
                    assistant_subset_lines.append(line)
                    seen_subset_lines.add(line)

    lines: list[str] = []
    approved_ids = [
        line.split(": ", 1)[0].removeprefix("- ")
        for line in approved_updates
    ]
    if approved_ids:
        lines.append(
            "Recovered approved/allowed IDs from tool-recorded state changes: "
            + ", ".join(approved_ids)
        )
    if approved_updates:
        lines.append("Observed tool-recorded approved updates:")
        lines.extend(approved_updates[:6])
    if other_updates:
        lines.append("Observed other tool-recorded integration updates:")
        lines.extend(other_updates[:4])
    if snapshot_lines and not approved_updates:
        lines.append("Relevant integration snapshots matching the query:")
        lines.extend(snapshot_lines[:5])
    if assistant_subset_lines and not approved_updates:
        lines.append("Assistant-stated subset references:")
        lines.extend(assistant_subset_lines[:3])
    return lines


def _fallback_session_summary(
    query: str,
    session_meta: Dict[str, Any],
    match_info: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> str:
    """Build a deterministic recall summary when the LLM summarizer is unavailable.

    The fallback intentionally preserves concrete snippets from the underlying
    transcript so downstream agents still get grounded recall instead of an
    empty result set.
    """
    query_terms = [term for term in re.findall(r"[A-Za-z0-9_-]+", query.lower()) if len(term) > 1]
    title = session_meta.get("title") or "Untitled session"
    source = session_meta.get("source", "unknown")
    when = _format_timestamp(session_meta.get("started_at"))
    snippet = _clean_excerpt(match_info.get("snippet") or "")
    structured_lines = _collect_structured_recall_lines(query_terms=query_terms, messages=messages)

    scored_excerpts: list[tuple[int, str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).upper()
        content = str(msg.get("content") or "")
        if not content.strip():
            continue
        payload = _safe_json_loads(content) if role == "TOOL" else None
        if _looks_like_nested_session_search(payload):
            continue
        content_lower = content.lower()
        score = 0
        if any(term in content_lower for term in query_terms):
            score += 3
        if role in {"TOOL", "ASSISTANT"}:
            score += 2
        if any(term in content_lower for term in ("decision", "approved", "exception", "staging", "soc2", "future reference")):
            score += 2
        if score <= 0:
            continue
        scored_excerpts.append((score, role, _clean_excerpt(content)))

    scored_excerpts.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str]] = []
    seen = set()
    for _, role, excerpt in scored_excerpts:
        if excerpt in seen:
            continue
        selected.append((role, excerpt))
        seen.add(excerpt)
        if len(selected) >= 3:
            break

    if not selected:
        for msg in messages:
            role = str(msg.get("role", "unknown")).upper()
            excerpt = _clean_excerpt(str(msg.get("content") or ""))
            if excerpt:
                selected.append((role, excerpt))
            if len(selected) >= 2:
                break

    lines = [
        f"Fallback recall for query '{query}' from session '{title}' ({when}, source={source}).",
    ]
    if snippet:
        lines.append(f"Matched snippet: {snippet}")
    if structured_lines:
        lines.append("Recovered structured facts:")
        lines.extend(structured_lines)
    if selected:
        lines.append("Relevant transcript excerpts:")
        for role, excerpt in selected:
            lines.append(f"- {role}: {excerpt}")
    return "\n".join(lines)


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date.

    Returns "unknown" for None, str(ts) if conversion fails.
    """
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError) as e:
        # Log specific errors for debugging while gracefully handling edge cases
        logging.debug("Failed to format timestamp %s: %s", ts, e, exc_info=True)
    except Exception as e:
        logging.debug("Unexpected error formatting timestamp %s: %s", ts, e, exc_info=True)
    return str(ts)


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format session messages into a readable transcript for summarization."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        tool_name = msg.get("tool_name")

        if role == "TOOL" and tool_name:
            # Truncate long tool outputs
            if len(content) > 500:
                content = content[:250] + "\n...[truncated]...\n" + content[-250:]
            parts.append(f"[TOOL:{tool_name}]: {content}")
        elif role == "ASSISTANT":
            # Include tool call names if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                tc_names = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "?")
                        tc_names.append(name)
                if tc_names:
                    parts.append(f"[ASSISTANT]: [Called: {', '.join(tc_names)}]")
                if content:
                    parts.append(f"[ASSISTANT]: {content}")
            else:
                parts.append(f"[ASSISTANT]: {content}")
        else:
            parts.append(f"[{role}]: {content}")

    return "\n\n".join(parts)


def _build_broadened_queries(query: str) -> List[str]:
    """Return broader fallback queries when a strict FTS search misses.

    Hermes guidance tells the model to use OR joins, but models still often
    send whitespace-only queries that FTS5 interprets as AND. When that misses,
    try a small set of broader alternatives before giving up.
    """
    normalized = (query or "").strip()
    if not normalized:
        return []

    # If the user already used explicit operators or quotes, avoid rewriting
    # the query semantics behind their back.
    lowered = normalized.lower()
    if any(op in lowered for op in ('"', " or ", " and ", " not ", "*", "(", ")")):
        return []

    terms = [term for term in re.findall(r"[A-Za-z0-9_-]+", normalized) if len(term) > 1]
    if len(terms) <= 1:
        return []

    fallbacks: List[str] = [" OR ".join(terms)]
    # Individual-term retries help when one anchor term is sufficient.
    fallbacks.extend(terms[:5])

    deduped: List[str] = []
    seen = {normalized}
    for candidate in fallbacks:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _truncate_around_matches(
    full_text: str, query: str, max_chars: int = MAX_SESSION_CHARS
) -> str:
    """
    Truncate a conversation transcript to max_chars, centered around
    where the query terms appear. Keeps content near matches, trims the edges.
    """
    if len(full_text) <= max_chars:
        return full_text

    # Find the first occurrence of any query term
    query_terms = query.lower().split()
    text_lower = full_text.lower()
    first_match = len(full_text)
    for term in query_terms:
        pos = text_lower.find(term)
        if pos != -1 and pos < first_match:
            first_match = pos

    if first_match == len(full_text):
        # No match found, take from the start
        first_match = 0

    # Center the window around the first match
    half = max_chars // 2
    start = max(0, first_match - half)
    end = min(len(full_text), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    truncated = full_text[start:end]
    prefix = "...[earlier conversation truncated]...\n\n" if start > 0 else ""
    suffix = "\n\n...[later conversation truncated]..." if end < len(full_text) else ""
    return prefix + truncated + suffix


async def _summarize_session(
    conversation_text: str, query: str, session_meta: Dict[str, Any]
) -> Optional[str]:
    """Summarize a single session conversation focused on the search query."""
    system_prompt = (
        "You are reviewing a past conversation transcript to help recall what happened. "
        "Summarize the conversation with a focus on the search topic. Include:\n"
        "1. What the user asked about or wanted to accomplish\n"
        "2. What actions were taken and what the outcomes were\n"
        "3. Key decisions, solutions found, or conclusions reached\n"
        "4. Any specific commands, files, URLs, or technical details that were important\n"
        "5. Anything left unresolved or notable\n\n"
        "Be thorough but concise. Preserve specific details (commands, paths, error messages) "
        "that would be useful to recall. Write in past tense as a factual recap."
    )

    source = session_meta.get("source", "unknown")
    started = _format_timestamp(session_meta.get("started_at"))

    user_prompt = (
        f"Search topic: {query}\n"
        f"Session source: {source}\n"
        f"Session date: {started}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_text}\n\n"
        f"Summarize this conversation with focus on: {query}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await async_call_llm(
                task="session_search",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_SUMMARY_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except RuntimeError:
            logging.warning("No auxiliary model available for session summarization")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.warning(
                    "Session summarization failed after %d attempts: %s",
                    max_retries,
                    e,
                    exc_info=True,
                )
                return None


# Sources that are excluded from session browsing/searching by default.
# Third-party integrations (Paperclip agents, etc.) tag their sessions with
# HERMES_SESSION_SOURCE=tool so they don't clutter the user's session history.
_HIDDEN_SESSION_SOURCES = ("tool",)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls)."""
    try:
        sessions = db.list_sessions_rich(limit=limit + 5, exclude_sources=list(_HIDDEN_SESSION_SOURCES))  # fetch extra to skip current

        # Resolve current session lineage to exclude it
        current_root = None
        if current_session_id:
            try:
                sid = current_session_id
                visited = set()
                while sid and sid not in visited:
                    visited.add(sid)
                    s = db.get_session(sid)
                    parent = s.get("parent_session_id") if s else None
                    sid = parent if parent else None
                current_root = max(visited, key=len) if visited else current_session_id
            except Exception:
                current_root = current_session_id

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            # Skip child/delegation sessions (they have parent_session_id)
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "recent",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Use a keyword query to search specific topics.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": f"Failed to list recent sessions: {e}"}, ensure_ascii=False)


def session_search(
    query: str,
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
) -> str:
    """
    Search past sessions and return focused summaries of matching conversations.

    Uses FTS5 to find matches, then summarizes the top sessions with Gemini Flash.
    The current session is excluded from results since the agent already has that context.
    """
    if db is None:
        return json.dumps({"success": False, "error": "Session database not available."}, ensure_ascii=False)

    limit = min(limit, 5)  # Cap at 5 sessions to avoid excessive LLM calls

    # Recent sessions mode: when query is empty, return metadata for recent sessions.
    # No LLM calls — just DB queries for titles, previews, timestamps.
    if not query or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    query = query.strip()

    try:
        # Parse role filter
        role_list = None
        if role_filter and role_filter.strip():
            role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

        # FTS5 search -- get matches ranked by relevance
        search_kwargs = {
            "role_filter": role_list,
            "exclude_sources": list(_HIDDEN_SESSION_SOURCES),
            "limit": 50,
            "offset": 0,
        }
        raw_results = db.search_messages(query=query, **search_kwargs)

        if not raw_results:
            for fallback_query in _build_broadened_queries(query):
                raw_results = db.search_messages(query=fallback_query, **search_kwargs)
                if raw_results:
                    logging.debug(
                        "Session search broadened query from %r to %r after 0 results",
                        query,
                        fallback_query,
                    )
                    break

        if not raw_results:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "message": "No matching sessions found.",
            }, ensure_ascii=False)

        # Resolve child sessions to their parent — delegation stores detailed
        # content in child sessions, but the user's conversation is the parent.
        def _resolve_to_parent(session_id: str) -> str:
            """Walk delegation chain to find the root parent session ID."""
            visited = set()
            sid = session_id
            while sid and sid not in visited:
                visited.add(sid)
                try:
                    session = db.get_session(sid)
                    if not session:
                        break
                    parent = session.get("parent_session_id")
                    if parent:
                        sid = parent
                    else:
                        break
                except Exception as e:
                    logging.debug(
                        "Error resolving parent for session %s: %s",
                        sid,
                        e,
                        exc_info=True,
                    )
                    break
            return sid

        current_lineage_root = (
            _resolve_to_parent(current_session_id) if current_session_id else None
        )

        # Group by resolved (parent) session_id, dedup, skip the current
        # session lineage. Compression and delegation create child sessions
        # that still belong to the same active conversation.
        seen_sessions = {}
        for result in raw_results:
            raw_sid = result["session_id"]
            resolved_sid = _resolve_to_parent(raw_sid)
            # Skip the current session lineage — the agent already has that
            # context, even if older turns live in parent fragments.
            if current_lineage_root and resolved_sid == current_lineage_root:
                continue
            if current_session_id and raw_sid == current_session_id:
                continue
            if resolved_sid not in seen_sessions:
                result = dict(result)
                result["session_id"] = resolved_sid
                seen_sessions[resolved_sid] = result
            if len(seen_sessions) >= limit:
                break

        # Prepare all sessions for parallel summarization
        tasks = []
        for session_id, match_info in seen_sessions.items():
            try:
                messages = db.get_messages_as_conversation(session_id)
                if not messages:
                    continue
                session_meta = db.get_session(session_id) or {}
                conversation_text = _format_conversation(messages)
                conversation_text = _truncate_around_matches(conversation_text, query)
                tasks.append((session_id, match_info, conversation_text, session_meta, messages))
            except Exception as e:
                logging.warning(
                    "Failed to prepare session %s: %s",
                    session_id,
                    e,
                    exc_info=True,
                )

        # Summarize all sessions in parallel
        async def _summarize_all() -> List[Union[str, Exception]]:
            """Summarize all sessions in parallel."""
            coros = [
                _summarize_session(text, query, meta)
                for _, _, text, meta, _ in tasks
            ]
            return await asyncio.gather(*coros, return_exceptions=True)

        try:
            # Use _run_async() which properly manages event loops across
            # CLI, gateway, and worker-thread contexts.  The previous
            # pattern (asyncio.run() in a ThreadPoolExecutor) created a
            # disposable event loop that conflicted with cached
            # AsyncOpenAI/httpx clients bound to a different loop,
            # causing deadlocks in gateway mode (#2681).
            from model_tools import _run_async
            results = _run_async(_summarize_all())
        except concurrent.futures.TimeoutError:
            logging.warning(
                "Session summarization timed out after 60 seconds",
                exc_info=True,
            )
            return json.dumps({
                "success": False,
                "error": "Session summarization timed out. Try a more specific query or reduce the limit.",
            }, ensure_ascii=False)

        summaries = []
        for (session_id, match_info, _, session_meta, messages), result in zip(tasks, results):
            summary_text = None
            summary_mode = "llm"
            if isinstance(result, Exception):
                logging.warning(
                    "Failed to summarize session %s: %s",
                    session_id,
                    result,
                    exc_info=True,
                )
            elif result:
                summary_text = result

            if not summary_text:
                summary_text = _fallback_session_summary(query, session_meta, match_info, messages)
                summary_mode = "fallback_excerpt"

            summaries.append({
                "session_id": session_id,
                "when": _format_timestamp(match_info.get("session_started")),
                "source": match_info.get("source", "unknown"),
                "model": match_info.get("model"),
                "summary": summary_text,
                "summary_mode": summary_mode,
            })

        return json.dumps({
            "success": True,
            "query": query,
            "results": summaries,
            "count": len(summaries),
            "sessions_searched": len(seen_sessions),
        }, ensure_ascii=False)

    except Exception as e:
        logging.error("Session search failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": f"Search failed: {str(e)}"}, ensure_ascii=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database and an auxiliary text model."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- "
        "every past session is searchable, and this tool summarizes what happened.\n\n"
        "TWO MODES:\n"
        "1. Recent sessions (no query): Call with no arguments to see what was worked on recently. "
        "Returns titles, previews, and timestamps. Zero LLM cost, instant. "
        "Start here when the user asks what were we working on or what did we do recently.\n"
        "2. Keyword search (with query): Search for specific topics across all past sessions. "
        "Returns LLM-generated summaries of matching sessions.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search when it is actually cross-session -- it's fast and cheap. "
        "Better to search and confirm than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), "
        "phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel. Returns summaries of the top matching sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead (returns titles, previews, timestamps with no LLM cost).",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to summarize (default: 3, max: 5).",
                "default": 3,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id")),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
