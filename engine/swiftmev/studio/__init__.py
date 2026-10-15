"""SwiftMev Dataset Studio Package.

Provides scientific micrograph cataloging, 4-state ground truth annotation,
pairwise leakage-free split curation, and versioned dataset exports.
"""

from swiftmev.studio.schema import (
    AnnotationHistoryEntry,
    AnnotationRecord,
    AnnotationSourceType,
    AnnotationState,
    DatasetManifest,
    DatasetSplitRecord,
    MicrographRecord,
    QCAlert,
    SplitAssignment,
)
from swiftmev.studio.inventory import (
    compute_dhash,
    compute_sha256,
    export_inventory_csv,
    hamming_distance,
    reconcile_inventory,
    scan_directory,
    scan_micrograph,
)
from swiftmev.studio.curation import (
    LeakageError,
    partition_groups_disjoint,
    validate_splits,
)
from swiftmev.studio.annotations import (
    AnnotationWorkflowError,
    create_or_update_annotation,
    inspect_annotation_quality,
    save_annotation_bundle,
)
from swiftmev.studio.exporter import (
    ExportSecurityError,
    ExportValidationError,
    ReleaseExistsError,
    export_dataset_release,
    safe_resolve_path,
)

__all__ = [
    "AnnotationHistoryEntry",
    "AnnotationRecord",
    "AnnotationSourceType",
    "AnnotationState",
    "DatasetManifest",
    "DatasetSplitRecord",
    "MicrographRecord",
    "QCAlert",
    "SplitAssignment",
    "compute_dhash",
    "compute_sha256",
    "export_inventory_csv",
    "hamming_distance",
    "reconcile_inventory",
    "scan_directory",
    "scan_micrograph",
    "LeakageError",
    "partition_groups_disjoint",
    "validate_splits",
    "AnnotationWorkflowError",
    "create_or_update_annotation",
    "inspect_annotation_quality",
    "save_annotation_bundle",
    "ExportSecurityError",
    "ExportValidationError",
    "ReleaseExistsError",
    "export_dataset_release",
    "safe_resolve_path",
]
