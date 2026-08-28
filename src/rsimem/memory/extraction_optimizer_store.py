"""Attempt-local private persistence for content-bearing optimizer corpora."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .extraction_optimizer_corpus import (
    EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
    ExtractionOptimizerCorpus,
    OptimizerCorpusRetention,
    OptimizerCorpusSplit,
)
from .prompt_components import canonical_json


EXTRACTION_OPTIMIZER_STORE_SCHEMA = "extraction-optimizer-corpus-store-v3"


class JsonExtractionOptimizerCorpusStore:
    """One immutable corpus per split under an owner-controlled attempt root."""

    def __init__(
        self,
        attempt_root: Path,
        *,
        owner_controlled_root: Path,
        attempt_id: str,
        split: OptimizerCorpusSplit,
    ) -> None:
        root = attempt_root.expanduser().resolve()
        owner_root = owner_controlled_root.expanduser().resolve()
        if owner_controlled_root.exists() and owner_controlled_root.is_symlink():
            raise ValueError("optimizer owner-controlled root cannot be a symlink")
        if attempt_root.exists() and attempt_root.is_symlink():
            raise ValueError("optimizer attempt root cannot be a symlink")
        try:
            relative = root.relative_to(owner_root)
        except ValueError as exc:
            raise ValueError(
                "optimizer attempt must be under its owner-controlled root"
            ) from exc
        if not relative.parts:
            raise ValueError("optimizer attempt must be below its owner-controlled root")
        self.attempt_root = root
        self.owner_controlled_root = owner_root
        self.attempt_id = attempt_id
        self.split = OptimizerCorpusSplit(split)
        self.private_root = root / "private" / "optimizer-corpus"
        self.path = self.private_root / f"{self.split.value}.json"
        self.lock_path = self.private_root / f".{self.split.value}.lock"

    @contextmanager
    def _lock(self, operation: int) -> Iterator[None]:
        if self.private_root.exists() and self.private_root.is_symlink():
            raise ValueError("optimizer private corpus directory cannot be a symlink")
        self.private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.private_root, 0o700)
        descriptor = os.open(
            self.lock_path,
            os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            os.chmod(self.lock_path, 0o600)
            with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), operation)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise

    def write(self, corpus: ExtractionOptimizerCorpus) -> bool:
        self._validate_identity(corpus)
        serialized = canonical_json(self._wrapper(corpus))
        with self._lock(fcntl.LOCK_EX):
            existing = self._read_unlocked()
            if existing is not None:
                if canonical_json(existing.payload()) != canonical_json(corpus.payload()):
                    raise ValueError("optimizer corpus store already contains a conflict")
                return False
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{self.split.value}.",
                suffix=".tmp",
                dir=self.private_root,
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                os.chmod(self.path, 0o600)
                directory = os.open(self.private_root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return True

    def read_for_optimizer(self) -> ExtractionOptimizerCorpus:
        if self.split != OptimizerCorpusSplit.TRAIN:
            raise PermissionError("optimizer can read only the training corpus")
        return self._read_required()

    def read_for_validation(self) -> ExtractionOptimizerCorpus:
        if self.split != OptimizerCorpusSplit.VALIDATION:
            raise PermissionError("validator can read only the validation corpus")
        return self._read_required()

    def read_for_future_evaluation(
        self,
        *,
        active_artifact_id: str | None,
    ) -> ExtractionOptimizerCorpus:
        if self.split != OptimizerCorpusSplit.FUTURE_TEST:
            raise PermissionError("future evaluator can read only future-test corpus")
        if active_artifact_id is None:
            raise PermissionError("future-test corpus is unavailable before activation")
        with self._lock(fcntl.LOCK_SH):
            corpus = self._read_unlocked()
        if corpus is None:
            raise FileNotFoundError("optimizer corpus has not been persisted")
        if corpus.activation_artifact_id != active_artifact_id:
            raise PermissionError("future-test activation artifact mismatch")
        return corpus

    def purge(self, *, retention: OptimizerCorpusRetention) -> bool:
        with self._lock(fcntl.LOCK_EX):
            corpus = self._read_unlocked()
            if corpus is None:
                return False
            if corpus.retention != OptimizerCorpusRetention(retention):
                raise ValueError("optimizer corpus retention policy mismatch")
            self.path.unlink()
            directory = os.open(self.private_root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return True

    def _read_required(self) -> ExtractionOptimizerCorpus:
        with self._lock(fcntl.LOCK_SH):
            corpus = self._read_unlocked()
        if corpus is None:
            raise FileNotFoundError("optimizer corpus has not been persisted")
        return corpus

    def _read_unlocked(self) -> ExtractionOptimizerCorpus | None:
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise ValueError("optimizer corpus file cannot be a symlink")
        if stat.S_IMODE(self.path.stat().st_mode) & 0o077:
            raise PermissionError("optimizer corpus file permissions are too broad")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("malformed optimizer corpus store") from exc
        fields = {
            "schema_version",
            "store_schema",
            "attempt_id",
            "split",
            "corpus",
        }
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or value["schema_version"] != EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION
            or value["store_schema"] != EXTRACTION_OPTIMIZER_STORE_SCHEMA
            or value["attempt_id"] != self.attempt_id
            or value["split"] != self.split.value
        ):
            raise ValueError("malformed optimizer corpus store")
        try:
            corpus = ExtractionOptimizerCorpus.from_payload(value["corpus"])
        except (TypeError, ValueError) as exc:
            raise ValueError("malformed optimizer corpus store") from exc
        self._validate_identity(corpus)
        return corpus

    def _validate_identity(self, corpus: ExtractionOptimizerCorpus) -> None:
        if corpus.attempt_id != self.attempt_id or corpus.split != self.split:
            raise ValueError("optimizer corpus belongs to another attempt or split")

    def _wrapper(self, corpus: ExtractionOptimizerCorpus) -> dict[str, object]:
        return {
            "schema_version": EXTRACTION_OPTIMIZER_CORPUS_SCHEMA_VERSION,
            "store_schema": EXTRACTION_OPTIMIZER_STORE_SCHEMA,
            "attempt_id": self.attempt_id,
            "split": self.split.value,
            "corpus": corpus.payload(),
        }
