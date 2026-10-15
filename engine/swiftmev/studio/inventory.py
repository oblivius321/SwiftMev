"""SwiftMev Dataset Studio - Micrograph Inventory & Integrity Engine.

Provides read-only scanning of scientific micrographs, computing cryptographic
SHA-256 and perceptual dHash signatures, extracting image metadata, reconciling
discrepancies, and cataloging experimental records without assuming sample independence.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from swiftmev.studio.schema import MicrographRecord


def compute_sha256(file_path: Path | str) -> str:
    """Computes SHA-256 checksum reading in 64 KB chunks."""
    p = Path(file_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_gray: np.ndarray, hash_size: int = 8) -> str:
    """Computes a 64-bit difference hash (dHash) for perceptual similarity comparison."""
    # Resize to (width=hash_size + 1, height=hash_size)
    resized = cv2.resize(image_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert bool array (8x8) to 64-bit integer
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    int_val = int(bit_string, 2)
    return f"{int_val:016x}"


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Calculates the Hamming distance (number of bit differences) between two 64-bit hex hashes."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    return bin(val1 ^ val2).count("1")


def scan_micrograph(
    file_path: Path,
    base_dir: Optional[Path] = None,
    default_um_per_pixel: Optional[float] = None,
    calibration_source: str = "uncalibrated",
) -> MicrographRecord:
    """Scans a single micrograph file (read-only) and extracts metadata."""
    p = Path(file_path).resolve()
    base = Path(base_dir).resolve() if base_dir else p.parent

    try:
        rel_path = str(p.relative_to(base)).replace("\\", "/")
    except ValueError:
        rel_path = p.name

    file_size = p.stat().st_size
    sha256 = compute_sha256(p)

    # Read image in color or grayscale
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível decodificar a micrografia: {p}")

    if len(img.shape) == 2:
        h, w = img.shape
        c = 1
        gray = img
    elif len(img.shape) == 3:
        h, w, c = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if c in (3, 4) else img[:, :, 0]
    else:
        raise ValueError(f"Dimensões de imagem não suportadas: {img.shape}")

    dhash = compute_dhash(gray)

    # Clean ID based on relative path
    clean_id = rel_path.replace("/", "__").replace("\\", "__").replace(".", "_")
    micrograph_id = f"MG_{clean_id}"

    # Extract provisional grouping from directory structure (conservative, non-verified)
    parts = Path(rel_path).parts
    formulation_hint = "unknown"
    group_parts = []
    for part in parts[:-1]:
        clean_part = part.strip()
        if clean_part:
            group_parts.append(clean_part)
        if any(f in clean_part.lower() for f in ["5_", "10_", "15_", "20_", "control"]):
            formulation_hint = clean_part

    conservative_group = "GRP-" + "-".join(group_parts) if group_parts else "GRP-ROOT"

    is_calibrated = default_um_per_pixel is not None and default_um_per_pixel > 0

    return MicrographRecord(
        micrograph_id=micrograph_id,
        file_path=rel_path,
        file_name=p.name,
        file_size_bytes=file_size,
        sha256=sha256,
        dhash=dhash,
        width=w,
        height=h,
        channels=c,
        bit_depth=8 if img.dtype == np.uint8 else 16,
        format=p.suffix.lstrip(".").upper(),
        formulation=formulation_hint,
        batch_id="unknown",
        physical_sample_id="unknown",  # Must remain unknown until human physical review
        acquisition_session_id="unknown",
        conservative_group_id=conservative_group,
        experimental_independence_reviewed=False,
        metadata_review_status="pending_review",
        um_per_pixel=default_um_per_pixel if is_calibrated else None,
        calibration_source=calibration_source if is_calibrated else "uncalibrated",
        is_calibrated=is_calibrated,
    )


def scan_directory(
    root_dir: Path | str,
    valid_extensions: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    exclude_filenames: Tuple[str, ...] = ("getIT_1_relatorio.png",),
) -> List[MicrographRecord]:
    """Scans directory recursively for micrographs, ignoring non-micrograph files."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {root_dir}")

    records: List[MicrographRecord] = []
    # Sort files for deterministic ordering
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in exclude_filenames:
            continue
        if p.suffix.lower() not in valid_extensions:
            continue

        record = scan_micrograph(p, base_dir=root)
        records.append(record)

    return records


def reconcile_inventory(root_dir: Path | str) -> Dict[str, Any]:
    """Reconciles the full inventory of a micrograph repository.

    Detects valid micrographs vs auxiliary report files, exact SHA-256 duplicates,
    and near-duplicates by perceptual dHash.
    """
    root = Path(root_dir)
    all_files = sorted([p for p in root.rglob("*") if p.is_file()])

    micrographs: List[MicrographRecord] = []
    auxiliary_files: List[Dict[str, Any]] = []

    for p in all_files:
        rel = str(p.relative_to(root)).replace("\\", "/")
        # Reconcile auxiliary files (e.g., reports, CSVs, graphs)
        if p.suffix.lower() in (".csv", ".txt", ".json", ".log") or p.name == "getIT_1_relatorio.png":
            auxiliary_files.append({
                "file_path": rel,
                "file_name": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": compute_sha256(p),
                "type": "auxiliary_report" if "relatorio" in p.name.lower() else "metadata",
            })
        elif p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            try:
                rec = scan_micrograph(p, base_dir=root)
                micrographs.append(rec)
            except Exception as e:
                auxiliary_files.append({
                    "file_path": rel,
                    "file_name": p.name,
                    "size_bytes": p.stat().st_size,
                    "error": str(e),
                    "type": "corrupted_or_unreadable",
                })
        else:
            auxiliary_files.append({
                "file_path": rel,
                "file_name": p.name,
                "size_bytes": p.stat().st_size,
                "type": "other",
            })

    # Detect exact duplicates by SHA-256
    sha_map: Dict[str, List[MicrographRecord]] = {}
    for m in micrographs:
        sha_map.setdefault(m.sha256, []).append(m)

    exact_duplicate_groups = [
        [{"micrograph_id": item.micrograph_id, "file_path": item.file_path} for item in group]
        for group in sha_map.values()
        if len(group) > 1
    ]

    # Detect perceptual near-duplicates (Hamming distance <= 2 bits on 64-bit dHash)
    perceptual_duplicates = []
    for i in range(len(micrographs)):
        for j in range(i + 1, len(micrographs)):
            m1 = micrographs[i]
            m2 = micrographs[j]
            if m1.sha256 == m2.sha256:
                continue  # Already recorded in exact duplicates
            dist = hamming_distance(m1.dhash, m2.dhash)
            if dist <= 2:
                perceptual_duplicates.append({
                    "id1": m1.micrograph_id,
                    "path1": m1.file_path,
                    "id2": m2.micrograph_id,
                    "path2": m2.file_path,
                    "hamming_distance": dist,
                })

    return {
        "root_directory": str(root.resolve()),
        "total_files": len(all_files),
        "total_micrographs": len(micrographs),
        "total_auxiliary_files": len(auxiliary_files),
        "exact_duplicate_groups_count": len(exact_duplicate_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
        "perceptual_near_duplicates_count": len(perceptual_duplicates),
        "perceptual_near_duplicates": perceptual_duplicates,
        "auxiliary_files": auxiliary_files,
        "micrographs": micrographs,
    }


def export_inventory_csv(micrographs: List[MicrographRecord], out_path: Path | str) -> Path:
    """Exports inventory records to a standard scientific CSV."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "micrograph_id",
        "file_path",
        "file_name",
        "file_size_bytes",
        "sha256",
        "dhash",
        "width",
        "height",
        "channels",
        "bit_depth",
        "format",
        "formulation",
        "batch_id",
        "physical_sample_id",
        "acquisition_session_id",
        "conservative_group_id",
        "experimental_independence_reviewed",
        "metadata_review_status",
        "um_per_pixel",
        "calibration_source",
        "is_calibrated",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in micrographs:
            writer.writerow(m.to_dict())

    return p
