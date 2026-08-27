"""Adapters for Hermes' native semantic, episodic, and procedural stores.

These adapters intentionally do not import Hermes at module import time. They
operate on the stable on-disk formats used by Hermes, which keeps RSIMem
usable when the benchmark agent is not installed and makes backend replacement
explicit in experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import fcntl

from ..contracts import (
    MemoryAccessMode,
    MemoryArtifact,
    MemoryBackendDescriptor,
    MemoryHit,
    MemoryKind,
    MemoryKindCapability,
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationResult,
    MemoryQuery,
    MemoryResource,
)
from ..runtime import MemoryBackendRegistry


_ENTRY_DELIMITER = "\n§\n"
_SKILL_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_SEMANTIC_CHAR_LIMITS = {"memory": 2200, "user": 1375}
_ALLOWED_RESOURCE_ROOTS = {"references", "templates", "scripts", "assets"}


def _opaque_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def semantic_artifact_id(namespace: str, content: str) -> str:
    """Return the stable identity used by the Hermes semantic projection."""

    return _opaque_id("hermes-semantic", f"{namespace}\0{content}")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _read_entries(path: Path) -> list[str]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    entries = [entry.strip() for entry in content.split(_ENTRY_DELIMITER) if entry.strip()]
    return list(dict.fromkeys(entries))


def _write_entries(path: Path, entries: list[str]) -> None:
    _atomic_write(path, _ENTRY_DELIMITER.join(entries) if entries else "")


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class HermesSemanticBackend:
    """Native Hermes semantic memory backed by MEMORY.md and USER.md."""

    def __init__(self, memories_dir: Path) -> None:
        self.memories_dir = memories_dir.expanduser().resolve()
        self._paths = {
            "memory": self.memories_dir / "MEMORY.md",
            "user": self.memories_dir / "USER.md",
        }

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            name="hermes-native-semantic",
            capabilities=(MemoryKindCapability(
                kind=MemoryKind.SEMANTIC,
                access_mode=MemoryAccessMode.EAGER,
            ),),
        )

    def _artifact(self, namespace: str, index: int, content: str) -> MemoryArtifact:
        artifact_id = semantic_artifact_id(namespace, content)
        return MemoryArtifact(
            artifact_id=artifact_id,
            kind=MemoryKind.SEMANTIC,
            namespace=namespace,
            content=content,
            revision=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
            metadata={"target": namespace, "entry_index": index},
        )

    @staticmethod
    def _namespace(namespace: str) -> str:
        normalized = "memory" if namespace == "default" else namespace
        if normalized not in _SEMANTIC_CHAR_LIMITS:
            raise ValueError(f"unknown Hermes semantic namespace: {namespace}")
        return normalized

    def _entries(self, namespace: str) -> list[str]:
        namespace = self._namespace(namespace)
        path = self._paths[namespace]
        return _read_entries(path)

    def get(self, artifact_id: str) -> MemoryArtifact | None:
        for namespace in self._paths:
            artifact = self._find_in_namespace(namespace, artifact_id)
            if artifact is not None:
                return artifact
        return None

    def _find_in_namespace(
        self,
        namespace: str,
        artifact_id: str,
    ) -> MemoryArtifact | None:
        for index, content in enumerate(self._entries(namespace)):
            artifact = self._artifact(namespace, index, content)
            if artifact.artifact_id == artifact_id:
                return artifact
        return None

    def query(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        namespace = self._namespace(query.namespace)
        terms = [term.lower() for term in query.text.split() if term.strip()]
        candidates: list[tuple[int, MemoryArtifact]] = []
        for index, content in enumerate(self._entries(namespace)):
            haystack = content.lower()
            score = sum(haystack.count(term) for term in terms) if terms else 1
            if terms and score == 0:
                continue
            candidates.append((score, self._artifact(namespace, index, content)))
        if terms:
            candidates.sort(key=lambda item: (-item[0], item[1].artifact_id))
        return tuple(
            MemoryHit(artifact, rank=index, score=float(score), backend=self.descriptor.name)
            for index, (score, artifact) in enumerate(candidates[:query.limit], start=1)
        )

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult:
        artifact_id = mutation.resolved_artifact_id
        if mutation.action == MemoryMutationAction.ADD:
            assert mutation.artifact is not None
            namespace = self._namespace(mutation.artifact.namespace)
            if _ENTRY_DELIMITER in mutation.artifact.content:
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="invalid_entry_delimiter")
            with _exclusive_file_lock(self._paths[namespace]):
                entries = self._entries(namespace)
                if mutation.artifact.content in entries:
                    return MemoryMutationResult(
                        accepted=True,
                        backend=self.descriptor.name,
                        action=mutation.action,
                        artifact_id=self._artifact(namespace, entries.index(mutation.artifact.content), mutation.artifact.content).artifact_id,
                        reason_code="already_present",
                    )
                entries.append(mutation.artifact.content)
                if len(_ENTRY_DELIMITER.join(entries)) > _SEMANTIC_CHAR_LIMITS[namespace]:
                    return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="character_limit_exceeded")
                _write_entries(self._paths[namespace], entries)
            artifact = self._artifact(namespace, len(entries) - 1, mutation.artifact.content)
            return MemoryMutationResult(True, self.descriptor.name, mutation.action, artifact.artifact_id)

        if mutation.action == MemoryMutationAction.DELETE:
            target = self.get(artifact_id or "")
            if target is None:
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="not_found")
            with _exclusive_file_lock(self._paths[target.namespace]):
                target = self._find_in_namespace(target.namespace, artifact_id or "")
                if target is None:
                    return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="not_found")
                if mutation.expected_revision and mutation.expected_revision != target.revision:
                    return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="revision_conflict")
                entries = self._entries(target.namespace)
                entries.remove(target.content)
                _write_entries(self._paths[target.namespace], entries)
            return MemoryMutationResult(True, self.descriptor.name, mutation.action, artifact_id)

        assert mutation.artifact is not None
        target = self.get(artifact_id or mutation.artifact.artifact_id)
        if target is None:
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="not_found")
        namespace = self._namespace(mutation.artifact.namespace)
        if namespace != target.namespace:
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="namespace_change_not_supported")
        if _ENTRY_DELIMITER in mutation.artifact.content:
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="invalid_entry_delimiter")
        with _exclusive_file_lock(self._paths[target.namespace]):
            target = self._find_in_namespace(target.namespace, artifact_id or "")
            if target is None:
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="not_found")
            if mutation.expected_revision and mutation.expected_revision != target.revision:
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="revision_conflict")
            entries = self._entries(target.namespace)
            entries[entries.index(target.content)] = mutation.artifact.content
            if len(_ENTRY_DELIMITER.join(entries)) > _SEMANTIC_CHAR_LIMITS[target.namespace]:
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, artifact_id, reason_code="character_limit_exceeded")
            _write_entries(self._paths[target.namespace], entries)
        updated = self._artifact(target.namespace, entries.index(mutation.artifact.content), mutation.artifact.content)
        return MemoryMutationResult(True, self.descriptor.name, mutation.action, updated.artifact_id)

    def close(self) -> None:
        return None


class HermesEpisodicBackend:
    """Read-only adapter over Hermes' SQLite session transcript store."""

    def __init__(self, state_db: Path) -> None:
        self.state_db = state_db.expanduser().resolve()

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            name="hermes-native-episodic",
            capabilities=(MemoryKindCapability(
                kind=MemoryKind.EPISODIC,
                access_mode=MemoryAccessMode.SEARCH,
                writable=False,
                updatable=False,
                deletable=False,
            ),),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.state_db}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _session_metadata_projection(connection: sqlite3.Connection) -> str:
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }
        return ", ".join((
            (
                "s.started_at AS session_started"
                if "started_at" in columns
                else "NULL AS session_started"
            ),
            (
                "s.title AS session_title"
                if "title" in columns
                else "NULL AS session_title"
            ),
            (
                "s.parent_session_id AS parent_session_id"
                if "parent_session_id" in columns
                else "NULL AS parent_session_id"
            ),
        ))

    @staticmethod
    def _artifact(row: sqlite3.Row) -> MemoryArtifact:
        message_id = str(row["id"])
        session_id = str(row["session_id"])
        content = str(row["content"] or "").strip()
        row_keys = set(row.keys())
        metadata = {
            "message_id": int(row["id"]),
            "session_id": session_id,
            "role": row["role"],
            "timestamp": row["timestamp"],
            "tool_name": row["tool_name"],
            "source": row["source"],
            "model": row["model"],
        }
        for key in ("session_started", "session_title", "parent_session_id"):
            if key in row_keys:
                metadata[key] = row[key]
        return MemoryArtifact(
            artifact_id=f"hermes-episodic:message:{message_id}",
            kind=MemoryKind.EPISODIC,
            namespace=session_id,
            title=str(row["tool_name"] or row["role"] or "session message"),
            content=content or "(empty message)",
            metadata=metadata,
        )

    def get(self, artifact_id: str) -> MemoryArtifact | None:
        prefix = "hermes-episodic:message:"
        if not artifact_id.startswith(prefix):
            return None
        try:
            message_id = int(artifact_id[len(prefix):])
        except ValueError:
            return None
        if not self.state_db.exists():
            return None
        with self._connect() as connection:
            session_metadata = self._session_metadata_projection(connection)
            row = connection.execute(
                f"""SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                          m.tool_name, s.source, s.model,
                          {session_metadata}
                   FROM messages m JOIN sessions s ON s.id = m.session_id
                   WHERE m.id = ?""",  # nosec B608 - projection is hardcoded above
                (message_id,),
            ).fetchone()
        return self._artifact(row) if row is not None else None

    @staticmethod
    def _conversation(connection: sqlite3.Connection, session_id: str) -> tuple[dict[str, Any], ...]:
        rows = connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            keys = set(row.keys())
            message: dict[str, Any] = {
                "role": row["role"],
                "content": row["content"],
            }
            for field in ("tool_call_id", "tool_name"):
                if field in keys and row[field]:
                    message[field] = row[field]
            if "tool_calls" in keys and row["tool_calls"]:
                try:
                    message["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if row["role"] == "assistant":
                if "reasoning" in keys and row["reasoning"]:
                    message["reasoning"] = row["reasoning"]
                for field in ("reasoning_details", "codex_reasoning_items"):
                    if field not in keys or not row[field]:
                        continue
                    try:
                        message[field] = json.loads(row[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            messages.append(message)
        return tuple(messages)

    @classmethod
    def _session_lineage(
        cls,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> tuple[tuple[str, dict[str, Any], tuple[dict[str, Any], ...]], ...]:
        lineage = []
        visited: set[str] = set()
        current = session_id
        while current and current not in visited:
            visited.add(current)
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            session = dict(row)
            lineage.append((current, session, cls._conversation(connection, current)))
            current = str(session.get("parent_session_id") or "")
        return tuple(lineage)

    @staticmethod
    def _filter_values(filters: Any, name: str) -> tuple[str, ...] | None:
        value = filters.get(name)
        if value is None:
            return None
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(f"episodic query filter {name} must be a list")
        return tuple(str(item) for item in value)

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Mirror Hermes SessionDB FTS5 normalization without importing Hermes."""

        quoted_parts: list[str] = []

        def preserve_quoted(match: re.Match[str]) -> str:
            quoted_parts.append(match.group(0))
            return f"\x00Q{len(quoted_parts) - 1}\x00"

        sanitized = re.sub(r'"[^"]*"', preserve_quoted, query)
        sanitized = re.sub(r'[+{}()"^]', " ", sanitized)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)
        sanitized = re.sub(
            r"(?i)^(AND|OR|NOT)\b\s*",
            "",
            sanitized.strip(),
        )
        sanitized = re.sub(
            r"(?i)\s+(AND|OR|NOT)\s*$",
            "",
            sanitized.strip(),
        )
        sanitized = re.sub(r"\b(\w+(?:-\w+)+)\b", r'"\1"', sanitized)
        for index, quoted in enumerate(quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{index}\x00", quoted)
        return sanitized.strip()

    def query(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        if not self.state_db.exists():
            return ()
        terms = self._sanitize_fts5_query(query.text.strip())
        if not terms:
            return ()
        source_filter = self._filter_values(query.filters, "source_filter")
        exclude_sources = self._filter_values(query.filters, "exclude_sources")
        role_filter = self._filter_values(query.filters, "role_filter")
        if role_filter == ():
            role_filter = None
        offset = query.filters.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("episodic query offset must be a non-negative integer")
        try:
            with self._connect() as connection:
                session_metadata = self._session_metadata_projection(connection)
                where_clauses = ["messages_fts MATCH ?"]
                parameters: list[Any] = [terms]
                for column, values in (
                    ("s.source", source_filter),
                    ("m.role", role_filter),
                ):
                    if values is not None:
                        if not values:
                            return ()
                        placeholders = ",".join("?" for _ in values)
                        where_clauses.append(f"{column} IN ({placeholders})")
                        parameters.extend(values)
                if exclude_sources:
                    placeholders = ",".join("?" for _ in exclude_sources)
                    where_clauses.append(f"s.source NOT IN ({placeholders})")
                    parameters.extend(exclude_sources)
                if query.namespace != "default":
                    where_clauses.append("m.session_id = ?")
                    parameters.append(query.namespace)
                parameters.extend((query.limit, offset))
                sql = f"""SELECT m.id, m.session_id, m.role, m.content, m.timestamp,
                                  m.tool_name, s.source, s.model,
                                  {session_metadata},
                                  snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                                  bm25(messages_fts) AS rank_score
                           FROM messages_fts
                           JOIN messages m ON m.id = messages_fts.rowid
                           JOIN sessions s ON s.id = m.session_id
                           WHERE {' AND '.join(where_clauses)}
                           ORDER BY rank_score
                           LIMIT ? OFFSET ?"""  # nosec B608 - columns are hardcoded above
                rows = connection.execute(sql, parameters).fetchall()
                enriched_rows = []
                for row in rows:
                    context_rows = connection.execute(
                        """SELECT id, role, content FROM messages
                           WHERE session_id = ? AND id >= ? - 1 AND id <= ? + 1
                           ORDER BY id""",
                        (row["session_id"], row["id"], row["id"]),
                    ).fetchall()
                    enriched_rows.append((
                        row,
                        context_rows,
                        self._session_lineage(connection, str(row["session_id"])),
                    ))
        except sqlite3.OperationalError:
            return ()
        hits: list[MemoryHit] = []
        for rank, (row, context_rows, lineage) in enumerate(enriched_rows, start=1):
            artifact = self._artifact(row)
            metadata = dict(artifact.metadata)
            metadata["snippet"] = row["snippet"]
            metadata["context"] = tuple(
                (str(item["role"]), str(item["content"] or "")[:200])
                for item in context_rows
            )
            metadata["context_messages"] = tuple(
                (int(item["id"]), str(item["role"]), str(item["content"] or "")[:200])
                for item in context_rows
            )
            metadata["session_lineage"] = lineage
            artifact = MemoryArtifact(
                artifact_id=artifact.artifact_id,
                kind=artifact.kind,
                namespace=artifact.namespace,
                title=artifact.title,
                content=artifact.content,
                revision=artifact.revision,
                metadata=metadata,
            )
            hits.append(MemoryHit(artifact, rank=rank, score=float(row["rank_score"] or 0.0), backend=self.descriptor.name))
        return tuple(hits)

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult:
        return MemoryMutationResult(
            accepted=False,
            backend=self.descriptor.name,
            action=mutation.action,
            artifact_id=mutation.resolved_artifact_id,
            reason_code="read_only_backend",
        )

    def close(self) -> None:
        return None


class HermesProceduralBackend:
    """Native Hermes procedural memory backed by skill directories."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir.expanduser().resolve()

    @property
    def descriptor(self) -> MemoryBackendDescriptor:
        return MemoryBackendDescriptor(
            name="hermes-native-procedural",
            capabilities=(MemoryKindCapability(
                kind=MemoryKind.PROCEDURAL,
                access_mode=MemoryAccessMode.PROGRESSIVE,
            ),),
        )

    def _skill_paths(self) -> list[Path]:
        return sorted(path for path in self.skills_dir.rglob("SKILL.md") if path.is_file())

    @staticmethod
    def _parse_name(content: str, fallback: str) -> tuple[str, str]:
        match = _SKILL_FRONTMATTER.match(content)
        if not match:
            return fallback, ""
        name = fallback
        description = ""
        for line in match.group(1).splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip() == "name":
                name = value.strip().strip("\"'") or fallback
            elif key.strip() == "description":
                description = value.strip().strip("\"'")
        return name, description

    def _artifact(self, skill_path: Path) -> MemoryArtifact:
        content = skill_path.read_text(encoding="utf-8")
        relative = skill_path.parent.relative_to(self.skills_dir).as_posix()
        name, description = self._parse_name(content, skill_path.parent.name)
        resources = tuple(
            MemoryResource(
                path=file.relative_to(skill_path.parent).as_posix(),
                content=file.read_bytes(),
            )
            for file in sorted(skill_path.parent.rglob("*"))
            if file.is_file() and not file.is_symlink() and file.name != "SKILL.md"
        )
        return MemoryArtifact(
            artifact_id=_opaque_id("hermes-procedural", relative),
            kind=MemoryKind.PROCEDURAL,
            namespace=relative.split("/", 1)[0] if "/" in relative else "default",
            title=name,
            content=content,
            revision=hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
            metadata={"skill_name": name, "description": description, "relative_path": relative},
            resources=resources,
        )

    def get(self, artifact_id: str) -> MemoryArtifact | None:
        for path in self._skill_paths():
            artifact = self._artifact(path)
            if artifact.artifact_id == artifact_id:
                return artifact
        return None

    @staticmethod
    def _valid_resources(resources: tuple[MemoryResource, ...]) -> bool:
        return all(
            PurePosixPath(resource.path).parts[0] in _ALLOWED_RESOURCE_ROOTS
            for resource in resources
        )

    @staticmethod
    def _valid_category(category: str) -> bool:
        if not category:
            return True
        path = PurePosixPath(category)
        return (
            not path.is_absolute()
            and ".." not in path.parts
            and all(_SAFE_NAME.fullmatch(part) for part in path.parts)
        )

    def query(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        terms = [term.lower() for term in query.text.split() if term.strip()]
        candidates: list[tuple[int, MemoryArtifact]] = []
        for path in self._skill_paths():
            artifact = self._artifact(path)
            haystack = f"{artifact.title or ''}\n{artifact.content}".lower()
            score = sum(haystack.count(term) for term in terms) if terms else 1
            if terms and score == 0:
                continue
            candidates.append((score, artifact))
        if terms:
            candidates.sort(key=lambda item: (-item[0], item[1].artifact_id))
        else:
            candidates.sort(key=lambda item: (
                "" if item[1].namespace == "default" else item[1].namespace,
                item[1].title or "",
            ))
        return tuple(
            MemoryHit(artifact, rank=index, score=float(score), backend=self.descriptor.name)
            for index, (score, artifact) in enumerate(candidates[:query.limit], start=1)
        )

    def mutate(self, mutation: MemoryMutation) -> MemoryMutationResult:
        if mutation.action == MemoryMutationAction.ADD:
            assert mutation.artifact is not None
            name = str(mutation.artifact.metadata.get("skill_name") or mutation.artifact.title or "")
            if not _SAFE_NAME.fullmatch(name):
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="invalid_skill_name")
            if not self._valid_resources(mutation.artifact.resources):
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="invalid_resource_path")
            category = mutation.artifact.namespace if mutation.artifact.namespace != "default" else ""
            if not self._valid_category(category):
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="invalid_category")
            skill_dir = self.skills_dir / category / name if category else self.skills_dir / name
            if skill_dir.exists():
                return MemoryMutationResult(False, self.descriptor.name, mutation.action, reason_code="already_exists")
            skill_dir.mkdir(parents=True, exist_ok=False)
            _atomic_write(skill_dir / "SKILL.md", mutation.artifact.content)
            for resource in mutation.artifact.resources:
                target = skill_dir / PurePosixPath(resource.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resource.content)
            return MemoryMutationResult(True, self.descriptor.name, mutation.action, _opaque_id("hermes-procedural", skill_dir.relative_to(self.skills_dir).as_posix()))

        target = self.get(mutation.resolved_artifact_id or "")
        if target is None:
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, mutation.resolved_artifact_id, reason_code="not_found")
        relative = str(target.metadata["relative_path"])
        skill_dir = self.skills_dir / relative
        if mutation.expected_revision and mutation.expected_revision != target.revision:
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, target.artifact_id, reason_code="revision_conflict")
        if mutation.action == MemoryMutationAction.DELETE:
            shutil.rmtree(skill_dir)
            return MemoryMutationResult(True, self.descriptor.name, mutation.action, target.artifact_id)

        assert mutation.artifact is not None
        if not self._valid_resources(mutation.artifact.resources):
            return MemoryMutationResult(False, self.descriptor.name, mutation.action, target.artifact_id, reason_code="invalid_resource_path")
        _atomic_write(skill_dir / "SKILL.md", mutation.artifact.content)
        for child in skill_dir.iterdir():
            if child.name == "SKILL.md":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for resource in mutation.artifact.resources:
            target_path = skill_dir / PurePosixPath(resource.path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(resource.content)
        updated = self._artifact(skill_dir / "SKILL.md")
        return MemoryMutationResult(True, self.descriptor.name, mutation.action, updated.artifact_id, revision=updated.revision)

    def close(self) -> None:
        return None


def build_hermes_native_registry(hermes_home: Path) -> MemoryBackendRegistry:
    """Build the default three-route registry for one Hermes home."""

    home = hermes_home.expanduser().resolve()
    registry = MemoryBackendRegistry()
    registry.register(HermesSemanticBackend(home / "memories"))
    registry.register(HermesEpisodicBackend(home / "state.db"))
    registry.register(HermesProceduralBackend(home / "skills"))
    return registry
