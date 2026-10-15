"""SwiftMev Dataset Studio - Data Schema & Models.

Defines typed data structures for scientific micrograph inventory,
rigorous 4-state annotations, experimental metadata taxonomy,
quality control alerts, and leakage-safe dataset manifests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnnotationState(str, Enum):
    """Rigorous 4-state lifecycle for human-curated ground truth."""
    DRAFT = "draft"
    ANNOTATED = "annotated"
    REVIEWED = "reviewed"
    APPROVED = "approved"


class AnnotationSourceType(str, Enum):
    """Source classification for segmentation masks."""
    MANUAL = "manual"
    PRELIMINARY_AUTO_OTSU = "preliminary_automatic_otsu"
    PRELIMINARY_AUTO_SAUVOLA = "preliminary_automatic_sauvola"
    PRELIMINARY_AUTO_AI = "preliminary_automatic_ai"
    REVISED_AUTOMATIC = "revised_automatic"


class SplitAssignment(str, Enum):
    """Split partition assignments."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EXCLUDED = "excluded"


@dataclass
class QCAlert:
    """Quality control alert identifying potential annotation flaws."""
    code: str
    severity: str  # "info", "warning", "caution"
    message: str
    count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QCAlert:
        return cls(
            code=data["code"],
            severity=data["severity"],
            message=data["message"],
            count=data.get("count", 0),
            details=data.get("details", {}),
        )


@dataclass
class AnnotationHistoryEntry:
    """Audit log entry tracking author, reviewer, date and state change."""
    version: int
    timestamp: str
    author: str
    action: str
    state: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationHistoryEntry:
        return cls(
            version=data["version"],
            timestamp=data["timestamp"],
            author=data["author"],
            action=data["action"],
            state=data["state"],
            notes=data.get("notes", ""),
        )


@dataclass
class AnnotationRecord:
    """Versioned ground truth annotation record with full provenance."""
    annotation_id: str
    micrograph_id: str
    mask_filename: str
    state: AnnotationState = AnnotationState.DRAFT
    source_type: AnnotationSourceType = AnnotationSourceType.MANUAL
    version: int = 1
    author: str = "unknown"
    reviewer: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    history: List[AnnotationHistoryEntry] = field(default_factory=list)
    qc_alerts: List[QCAlert] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value if isinstance(self.state, AnnotationState) else self.state
        d["source_type"] = self.source_type.value if isinstance(self.source_type, AnnotationSourceType) else self.source_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotationRecord:
        history = [AnnotationHistoryEntry.from_dict(h) for h in data.get("history", [])]
        qc_alerts = [QCAlert.from_dict(q) for q in data.get("qc_alerts", [])]
        return cls(
            annotation_id=data["annotation_id"],
            micrograph_id=data["micrograph_id"],
            mask_filename=data["mask_filename"],
            state=AnnotationState(data.get("state", AnnotationState.DRAFT)),
            source_type=AnnotationSourceType(data.get("source_type", AnnotationSourceType.MANUAL)),
            version=data.get("version", 1),
            author=data.get("author", "unknown"),
            reviewer=data.get("reviewer"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            history=history,
            qc_alerts=qc_alerts,
            notes=data.get("notes", ""),
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> AnnotationRecord:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class MicrographRecord:
    """Scientific micrograph record distinguishing physical vs derived identity."""
    micrograph_id: str
    file_path: str
    file_name: str
    file_size_bytes: int
    sha256: str
    dhash: str  # 64-bit hexadecimal perceptual hash
    width: int
    height: int
    channels: int
    bit_depth: int = 8
    format: str = "TIFF"

    # Explicit experimental taxonomy - DO NOT infer sample identity from folder names!
    formulation: str = "unknown"
    batch_id: str = "unknown"
    physical_sample_id: str = "unknown"  # "unknown" or "requires_review" until confirmed
    acquisition_session_id: str = "unknown"
    conservative_group_id: str = "GRP-UNKNOWN"  # Provisional cluster to prevent leakage, NOT confirmed sample
    experimental_independence_reviewed: bool = False
    metadata_review_status: str = "pending_review"  # "pending_review" | "reviewed" | "confirmed"

    # Calibration isolation
    um_per_pixel: Optional[float] = None
    calibration_source: str = "uncalibrated"
    is_calibrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MicrographRecord:
        return cls(**data)


@dataclass
class DatasetSplitRecord:
    """Leakage-checked split partitions with pairwise disjointness verification."""
    train_ids: List[str] = field(default_factory=list)
    val_ids: List[str] = field(default_factory=list)
    test_ids: List[str] = field(default_factory=list)
    excluded_ids: List[str] = field(default_factory=list)

    # Verification status flags
    automated_leakage_checks_passed: bool = False
    pairwise_disjoint_verified: bool = False
    hash_duplicates_verified: bool = False
    group_separation_verified: bool = False
    experimental_independence_reviewed: bool = False
    validation_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetSplitRecord:
        return cls(**data)


@dataclass
class DatasetManifest:
    """Manifest for exported dataset releases with full integrity hashes."""
    dataset_name: str
    dataset_version: str
    created_at: str
    generator: str = "SwiftMev Dataset Studio"
    total_micrographs: int = 0
    total_annotations: int = 0
    splits_count: Dict[str, int] = field(default_factory=dict)

    # Rigorous scientific claims
    automated_leakage_checks_passed: bool = False
    experimental_independence_reviewed: bool = False
    integrity_hashes: Dict[str, str] = field(default_factory=dict)  # relative_file_path -> sha256
    scientific_notes: str = ""
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetManifest:
        return cls(**data)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> DatasetManifest:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)
