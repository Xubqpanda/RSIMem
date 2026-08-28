"""Least-privilege content projection for the extraction prompt optimizer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from .prompt_components import text_digest


OPTIMIZER_CONTENT_SCHEMA_VERSION = 1
OPTIMIZER_UNTRUSTED_TEXT_SCHEMA = "optimizer-untrusted-text-v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9_])(?:sk-[A-Za-z0-9_-]{20,}|"
    r"(?:api|access|secret)[_-]?key\s*[:=]\s*[^\s,;]+)",
    re.IGNORECASE,
)
_AUTHORIZATION = re.compile(
    r"authorization\s*:\s*(?:bearer|basic)\s+[^\s,;]+",
    re.IGNORECASE,
)
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|mnt|tmp|var|etc|opt|Users)/[^\s\"'<>]+"
)
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\[^\s\"'<>]+")
_FORBIDDEN_EVIDENCE = re.compile(
    r"(?:official[ _-]?grader|answer[ _-]?key|hidden[ _-]?expectation|"
    r"judge[ _-]?feedback|official[ _-]?score)",
    re.IGNORECASE,
)


class OptimizerTextRedaction(StrEnum):
    AUTHORIZATION = "authorization"
    CREDENTIAL = "credential"
    MACHINE_PATH = "machine_path"


@dataclass(frozen=True, slots=True)
class OptimizerUntrustedText:
    text: str
    source_digest: str
    projected_digest: str
    redactions: tuple[OptimizerTextRedaction, ...]
    trust: str = "untrusted_data"
    text_schema: str = OPTIMIZER_UNTRUSTED_TEXT_SCHEMA
    schema_version: int = OPTIMIZER_CONTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != OPTIMIZER_CONTENT_SCHEMA_VERSION
            or self.text_schema != OPTIMIZER_UNTRUSTED_TEXT_SCHEMA
            or self.trust != "untrusted_data"
        ):
            raise ValueError("unsupported optimizer text contract")
        if not isinstance(self.text, str):
            raise TypeError("optimizer text must be a string")
        for value in (self.source_digest, self.projected_digest):
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise ValueError("optimizer text digest must be sha256")
        if self.projected_digest != text_digest(self.text):
            raise ValueError("optimizer projected text digest mismatch")
        object.__setattr__(
            self,
            "redactions",
            tuple(OptimizerTextRedaction(value) for value in self.redactions),
        )
        values = tuple(value.value for value in self.redactions)
        if values != tuple(sorted(set(values))):
            raise ValueError("optimizer text redactions must be sorted and unique")
        if _FORBIDDEN_EVIDENCE.search(self.text):
            raise ValueError("forbidden evaluation evidence reached optimizer text")
        if (
            _CREDENTIAL.search(self.text)
            or _AUTHORIZATION.search(self.text)
            or _POSIX_PATH.search(self.text)
            or _WINDOWS_PATH.search(self.text)
        ):
            raise ValueError("optimizer text still contains protected content")

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "text_schema": self.text_schema,
            "trust": self.trust,
            "text": self.text,
            "source_digest": self.source_digest,
            "projected_digest": self.projected_digest,
            "redactions": [value.value for value in self.redactions],
        }


class OptimizerSecretBoundary:
    """Project a protected optimizer-only copy without changing agent context."""

    def project(
        self,
        value: str,
        *,
        forbidden_values: tuple[str, ...] = (),
    ) -> OptimizerUntrustedText:
        if not isinstance(value, str):
            raise TypeError("optimizer boundary input must be text")
        if _FORBIDDEN_EVIDENCE.search(value) or any(
            forbidden and forbidden in value for forbidden in forbidden_values
        ):
            raise ValueError("forbidden evaluation evidence cannot enter optimizer corpus")
        projected = value
        redactions: set[OptimizerTextRedaction] = set()
        projected, count = _AUTHORIZATION.subn(
            "Authorization: [REDACTED_CREDENTIAL]",
            projected,
        )
        if count:
            redactions.add(OptimizerTextRedaction.AUTHORIZATION)
        projected, count = _CREDENTIAL.subn("[REDACTED_CREDENTIAL]", projected)
        if count:
            redactions.add(OptimizerTextRedaction.CREDENTIAL)
        projected, posix_count = _POSIX_PATH.subn("[REDACTED_MACHINE_PATH]", projected)
        projected, windows_count = _WINDOWS_PATH.subn(
            "[REDACTED_MACHINE_PATH]",
            projected,
        )
        if posix_count or windows_count:
            redactions.add(OptimizerTextRedaction.MACHINE_PATH)
        return OptimizerUntrustedText(
            projected,
            text_digest(value),
            text_digest(projected),
            tuple(sorted(redactions, key=lambda item: item.value)),
        )
