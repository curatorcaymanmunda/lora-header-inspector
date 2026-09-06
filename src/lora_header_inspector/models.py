"""Typed records shared by classification, inspection, and file management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


class EvidenceSource(str, Enum):
    STRUCTURE = "structure"
    METADATA = "metadata"
    FILENAME = "filename"
    IMPLIED = "implied"
    SYSTEM = "system"


class EvidenceCategory(str, Enum):
    ARCHITECTURE = "architecture"
    STRUCTURAL_FAMILY = "structural_family"
    MODEL_VERSION = "model_version"
    TASK_VARIANT = "task_variant"
    NOISE_STAGE = "noise_stage"
    MODEL_VARIANT = "model_variant"
    ECOSYSTEM = "ecosystem"
    TECHNIQUE = "technique"
    WARNING = "warning"


class ConflictSeverity(str, Enum):
    HARD = "hard"
    WARNING = "warning"


@dataclass(frozen=True)
class Evidence:
    """One machine-readable observation contributing to a classification."""

    rule_id: str
    source: str
    category: str
    value: str
    confidence: str = "low"
    priority: int = 0
    detail: str = ""
    field: str = ""
    precise: bool = False

    @property
    def message(self) -> str:
        return self.detail

    @property
    def architecture(self) -> str:
        return self.value if self.category == EvidenceCategory.ARCHITECTURE.value else ""

    @property
    def severity(self) -> str:
        return "info"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["message"] = self.detail
        payload["architecture"] = self.architecture
        return payload


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    category: str
    values: Tuple[str, ...]
    severity: str
    detail: str
    evidence_rule_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "category": self.category,
            "values": list(self.values),
            "severity": self.severity,
            "detail": self.detail,
            "evidence_rule_ids": list(self.evidence_rule_ids),
        }


@dataclass(frozen=True)
class SignatureMatch:
    """One structural signature match returned by the registry."""

    rule_id: str
    family: str
    architecture: str = ""
    confidence: str = "low"
    priority: int = 0
    ambiguous: bool = False
    message: str = ""
    details: Tuple[str, ...] = ()

    @property
    def arch(self) -> str:
        return self.architecture

    @property
    def decisive(self) -> bool:
        return bool(self.architecture and self.confidence == "high" and not self.ambiguous)

    def evidence(self) -> List[Evidence]:
        category = (
            EvidenceCategory.ARCHITECTURE.value
            if self.architecture
            else EvidenceCategory.STRUCTURAL_FAMILY.value
        )
        value = self.architecture or self.family
        messages = [self.message] if self.message else []
        messages.extend(self.details)
        if not messages:
            messages = [f"matched structural rule {self.rule_id}"]
        return [
            Evidence(
                rule_id=self.rule_id,
                source=EvidenceSource.STRUCTURE.value,
                category=category,
                value=value,
                confidence=self.confidence,
                priority=self.priority,
                detail=message,
                precise=bool(self.architecture and not self.ambiguous),
            )
            for message in messages
        ]

    def evidence_items(self) -> List[Evidence]:
        return self.evidence()


@dataclass
class StructuralAssessment:
    family: str = "Unknown"
    architecture: str = ""
    confidence: str = "low"
    ambiguous: bool = True
    matches: List[SignatureMatch] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class Classification:
    arch: str = "Unknown"
    structural_family: str = "Unknown"
    model_version: str = ""
    task_variant: str = ""
    noise_stage: str = ""
    confidence: str = "low"
    status: str = "ambiguous"
    ecosystems: List[str] = field(default_factory=list)
    model_variants: List[str] = field(default_factory=list)
    unresolved_hints: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    sort_subfolder: str = ""
    format: str = "unknown"
    evidence: List[Evidence] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def evidence_json(self) -> List[Dict[str, Any]]:
        return [value.to_dict() for value in self.evidence]

    def conflicts_json(self) -> List[Dict[str, Any]]:
        return [value.to_dict() for value in self.conflicts]

    def evidence_summary(self) -> str:
        return "; ".join(unique_strings(value.detail for value in self.evidence if value.detail))


# Compatibility alias retained for early modular prototypes.
ClassificationResult = Classification


@dataclass(frozen=True)
class HeaderData:
    tensors: Dict[str, Dict[str, Any]]
    metadata: Dict[str, str]
    header_sha256: str
    header_bytes: int
    file_bytes: int
    warnings: Tuple[str, ...] = ()

    @property
    def payload_bytes(self) -> int:
        return max(0, self.file_bytes - 8 - self.header_bytes)


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileIdentity":
        stat = path.stat()
        return cls(
            device=int(stat.st_dev), inode=int(stat.st_ino), size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns), ctime_ns=int(stat.st_ctime_ns),
        )

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class FileFingerprint:
    identity: FileIdentity
    header_sha256: str = ""
    content_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "header_sha256": self.header_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class SidecarSnapshot:
    path: Path
    name: str
    role: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "role": self.role,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SidecarInfo:
    path: Path
    kind: str
    identity: FileIdentity
    content_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path), "kind": self.kind,
            "identity": self.identity.to_dict(),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class HintObservation:
    dimension: str
    value: str
    confidence: str
    source: str
    priority: int
    rule_id: str
    detail: str = ""
    field: str = ""
    precise: bool = False
