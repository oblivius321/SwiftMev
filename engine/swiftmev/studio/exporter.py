"""SwiftMev Dataset Studio - Dataset Release Exporter & Integrity Packager.

Safely packages scientific datasets for neural network training and validation:
- Preserves raw source micrographs without modification.
- Guards against silent release overwrites.
- Blocks directory traversal attacks.
- Computes SHA-256 manifest integrity hashes for all exported assets.
- Separates automated leakage status from human experimental independence review.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Set

from swiftmev.studio.inventory import compute_sha256, export_inventory_csv
from swiftmev.studio.schema import (
    AnnotationRecord,
    AnnotationState,
    DatasetManifest,
    DatasetSplitRecord,
    MicrographRecord,
)


class ExportSecurityError(Exception):
    """Raised when an unsafe path traversal or unauthorized export is detected."""
    pass


class ReleaseExistsError(FileExistsError):
    """Raised when attempting to overwrite an existing release directory without explicit authorization."""
    pass


class ExportValidationError(ValueError):
    """Raised when dataset constraints (leakage, unreviewed annotations, shape mismatch) fail."""
    pass


def safe_resolve_path(base_dir: Path, relative_path: str) -> Path:
    """Safely resolves relative_path inside base_dir, preventing directory traversal."""
    resolved_base = base_dir.resolve()
    target = (resolved_base / relative_path).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError:
        raise ExportSecurityError(
            f"Tentativa de path traversal detectada! O caminho '{relative_path}' escapa de '{base_dir}'."
        )
    return target


def export_dataset_release(
    destination_dir: Path | str,
    dataset_name: str,
    dataset_version: str,
    micrographs: List[MicrographRecord],
    split_record: DatasetSplitRecord,
    annotations_map: Optional[Dict[str, AnnotationRecord]] = None,
    masks_dir: Optional[Path | str] = None,
    source_root: Optional[Path | str] = None,
    require_reviewed_annotations: bool = False,
    allow_unverified_splits: bool = False,
    overwrite: bool = False,
    scientific_notes: str = "",
    limitations: str = "",
) -> DatasetManifest:
    """Exports a reproducible, versioned dataset bundle."""
    dest = Path(destination_dir).resolve()
    release_dir = dest / f"{dataset_name}_v{dataset_version}"

    if release_dir.exists():
        if not overwrite:
            raise ReleaseExistsError(
                f"O diretório da release '{release_dir}' já existe. Sobrescrita silenciosa foi bloqueada. "
                "Para substituir, defina overwrite=True explicitamente."
            )
        shutil.rmtree(release_dir)

    # 1. Leakage verification gate
    if not split_record.automated_leakage_checks_passed and not allow_unverified_splits:
        raise ExportValidationError(
            f"Exportação bloqueada: Os splits falharam na verificação de vazamento. "
            f"Mensagens: {split_record.validation_messages}"
        )

    # 2. Annotation review gate
    ann_map = annotations_map or {}
    all_assigned_ids = set(split_record.train_ids + split_record.val_ids + split_record.test_ids)
    if require_reviewed_annotations:
        unapproved = []
        for m_id in all_assigned_ids:
            if m_id in ann_map:
                ann = ann_map[m_id]
                if ann.state not in (AnnotationState.REVIEWED, AnnotationState.APPROVED):
                    unapproved.append((m_id, ann.state.value))
            else:
                unapproved.append((m_id, "missing_annotation"))

        if unapproved:
            raise ExportValidationError(
                f"Exportação bloqueada: Existem {len(unapproved)} anotações não revisadas/aprovadas "
                f"no conjunto selecionado ({unapproved[:3]}...)."
            )

    # 3. Create release directory layout
    release_dir.mkdir(parents=True, exist_ok=True)
    images_dir = release_dir / "images"
    masks_out_dir = release_dir / "masks"
    annotations_out_dir = release_dir / "annotations"
    splits_dir = release_dir / "splits"

    images_dir.mkdir()
    masks_out_dir.mkdir()
    annotations_out_dir.mkdir()
    splits_dir.mkdir()

    micro_dict = {m.micrograph_id: m for m in micrographs}
    integrity_hashes: Dict[str, str] = {}

    # 4. Copy micrographs (read-only from source)
    for m_id in all_assigned_ids:
        if m_id not in micro_dict:
            continue
        m = micro_dict[m_id]
        if source_root:
            src_file = safe_resolve_path(Path(source_root), m.file_path)
        else:
            src_file = Path(m.file_path).resolve()

        if not src_file.exists():
            raise FileNotFoundError(f"Micrografia de origem não encontrada: {src_file}")

        target_img = images_dir / f"{m.micrograph_id}{src_file.suffix}"
        shutil.copy2(src_file, target_img)
        rel_img = str(target_img.relative_to(release_dir)).replace("\\", "/")
        integrity_hashes[rel_img] = compute_sha256(target_img)

        # Copy mask if available
        if masks_dir:
            mask_src = Path(masks_dir) / f"{m.micrograph_id}_mask.png"
            if mask_src.exists():
                target_mask = masks_out_dir / f"{m.micrograph_id}_mask.png"
                shutil.copy2(mask_src, target_mask)
                rel_mask = str(target_mask.relative_to(release_dir)).replace("\\", "/")
                integrity_hashes[rel_mask] = compute_sha256(target_mask)

        # Save annotation sidecar JSON if available
        if m_id in ann_map:
            ann = ann_map[m_id]
            ann_file = annotations_out_dir / f"{m.micrograph_id}_ann.json"
            ann.save(ann_file)
            rel_ann = str(ann_file.relative_to(release_dir)).replace("\\", "/")
            integrity_hashes[rel_ann] = compute_sha256(ann_file)

    # 5. Write split lists
    for split_name, ids in [
        ("train", split_record.train_ids),
        ("val", split_record.val_ids),
        ("test", split_record.test_ids),
    ]:
        split_file = splits_dir / f"{split_name}.txt"
        split_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        rel_split = str(split_file.relative_to(release_dir)).replace("\\", "/")
        integrity_hashes[rel_split] = compute_sha256(split_file)

    # 6. Write inventory CSV
    assigned_records = [micro_dict[i] for i in all_assigned_ids if i in micro_dict]
    inv_path = release_dir / "inventory.csv"
    export_inventory_csv(assigned_records, inv_path)
    rel_inv = str(inv_path.relative_to(release_dir)).replace("\\", "/")
    integrity_hashes[rel_inv] = compute_sha256(inv_path)

    # 7. Generate and save DatasetManifest
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        created_at=now_iso,
        generator="SwiftMev Dataset Studio",
        total_micrographs=len(assigned_records),
        total_annotations=len(ann_map),
        splits_count={
            "train": len(split_record.train_ids),
            "val": len(split_record.val_ids),
            "test": len(split_record.test_ids),
        },
        automated_leakage_checks_passed=split_record.automated_leakage_checks_passed,
        experimental_independence_reviewed=split_record.experimental_independence_reviewed,
        integrity_hashes=integrity_hashes,
        scientific_notes=scientific_notes or "Release exportada com checagens de vazamento e agrupamento conservador.",
        limitations=limitations or "Amostras físicas dependentes de validação experimental presencial.",
    )

    manifest_path = release_dir / "manifest.json"
    manifest.save(manifest_path)

    return manifest
