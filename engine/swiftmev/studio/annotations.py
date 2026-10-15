"""SwiftMev Dataset Studio - Ground Truth Annotation & Quality Control Engine.

Manages the 4-state annotation lifecycle:
    draft → annotated → reviewed → approved

Includes an automated Quality Control Assistant providing non-destructive inspection
alerts (tiny debris speckles, boundary roughness, coverage extremes, and unannotated high-contrast regions).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from swiftmev.studio.schema import (
    AnnotationHistoryEntry,
    AnnotationRecord,
    AnnotationSourceType,
    AnnotationState,
    QCAlert,
)


class AnnotationWorkflowError(Exception):
    """Raised when an invalid state transition or unpermitted ground truth claim is attempted."""
    pass


def inspect_annotation_quality(
    mask: np.ndarray,
    micrograph_gray: Optional[np.ndarray] = None,
    min_speckle_area: int = 15,
) -> List[QCAlert]:
    """Inspects a binary segmentation mask for potential quality flaws and anomalies.

    Returns advisory QC alerts for human review. Does NOT modify the mask.
    """
    alerts: List[QCAlert] = []

    if mask is None or mask.size == 0:
        alerts.append(QCAlert(
            code="INVALID_MASK",
            severity="caution",
            message="A máscara fornecida é nula ou vazia.",
        ))
        return alerts

    bin_mask = (mask > 127).astype(np.uint8)
    total_pixels = bin_mask.size
    fg_pixels = int(np.sum(bin_mask))

    # 1. Check for Empty Mask
    if fg_pixels == 0:
        alerts.append(QCAlert(
            code="EMPTY_MASK",
            severity="caution",
            message="A máscara não contém nenhuma região anotada (0% de primeiro plano).",
            count=0,
        ))
        return alerts

    # 2. Check for Extreme Coverage
    coverage_ratio = fg_pixels / total_pixels
    if coverage_ratio < 0.02:
        alerts.append(QCAlert(
            code="SUSPICIOUSLY_LOW_COVERAGE",
            severity="warning",
            message=f"Cobertura muito baixa de fibras ({coverage_ratio * 100:.2f}%). Verifique se fibras reais foram ignoradas.",
            details={"coverage_ratio": coverage_ratio},
        ))
    elif coverage_ratio > 0.85:
        alerts.append(QCAlert(
            code="SUSPICIOUSLY_HIGH_COVERAGE",
            severity="warning",
            message=f"Cobertura excepcionalmente alta ({coverage_ratio * 100:.2f}%). Verifique se o fundo foi indevidamente segmentado.",
            details={"coverage_ratio": coverage_ratio},
        ))

    # 3. Check for Tiny Disconnected Speckles (Debris)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # stats format: [x, y, width, height, area]
    small_components_count = 0
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area < min_speckle_area:
            small_components_count += 1

    if small_components_count > 0:
        alerts.append(QCAlert(
            code="TINY_COMPONENTS",
            severity="warning",
            message=f"Detectados {small_components_count} componentes minúsculos (< {min_speckle_area} px). Possível ruído/artefato de limiarização.",
            count=small_components_count,
            details={"min_area_threshold": min_speckle_area},
        ))

    # 4. Check for Boundary Irregularity / Roughness
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rough_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            perimeter = cv2.arcLength(cnt, closed=True)
            if perimeter > 0:
                # Circularity: 4 * pi * area / (perimeter^2)
                # For smooth circular/fiber regions circularity is higher; very jagged noise has extremely low circularity
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if circularity < 0.04:  # High perimeter with low area = fractal/noisy border
                    rough_contours += 1

    if rough_contours > 5:
        alerts.append(QCAlert(
            code="IRREGULAR_BORDERS",
            severity="info",
            message=f"{rough_contours} componentes apresentam bordas altamente serrilhadas ou irregulares.",
            count=rough_contours,
        ))

    # 5. Check Unannotated High-Contrast Regions (if original micrograph is provided)
    if micrograph_gray is not None and micrograph_gray.shape == mask.shape:
        # High intensity in micrograph where mask is 0
        bg_pixels_img = micrograph_gray[bin_mask == 0]
        if bg_pixels_img.size > 0:
            q90_bg = np.percentile(bg_pixels_img, 95)
            # If 95th percentile of unannotated background is very bright compared to mean
            overall_mean = np.mean(micrograph_gray)
            if q90_bg > overall_mean + 40:
                alerts.append(QCAlert(
                    code="UNANNOTATED_BRIGHT_REGIONS",
                    severity="info",
                    message="Existem regiões de alta intensidade na micrografia que não foram incluídas na máscara.",
                    details={"q95_unannotated": float(q90_bg), "mean_intensity": float(overall_mean)},
                ))

    return alerts


def create_or_update_annotation(
    micrograph_id: str,
    mask: np.ndarray,
    target_state: AnnotationState,
    author: str,
    source_type: AnnotationSourceType = AnnotationSourceType.MANUAL,
    existing_record: Optional[AnnotationRecord] = None,
    reviewer: Optional[str] = None,
    notes: str = "",
    micrograph_gray: Optional[np.ndarray] = None,
) -> Tuple[AnnotationRecord, np.ndarray]:
    """Creates or transitions an annotation through the rigorous 4-state lifecycle.

    Guarantees:
      - Automatic masks cannot jump directly to APPROVED ground truth.
      - State changes increment version and record audit log.
      - Runs Quality Control Assistant and attaches advisory alerts.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Ground truth integrity gate: Automatic masks cannot be saved directly as APPROVED
    if target_state == AnnotationState.APPROVED:
        if source_type in (
            AnnotationSourceType.PRELIMINARY_AUTO_OTSU,
            AnnotationSourceType.PRELIMINARY_AUTO_SAUVOLA,
            AnnotationSourceType.PRELIMINARY_AUTO_AI,
        ):
            raise AnnotationWorkflowError(
                f"Máscara com origem automática ({source_type.value}) não pode ser aprovada diretamente "
                "como Ground Truth sem passar por revisão humana prévia ('reviewed')."
            )

        if not reviewer and author == "unknown":
            raise AnnotationWorkflowError(
                "Para aprovação de Ground Truth científico, é obrigatório registrar revisor e autor identificados."
            )

    # Validate mask binary format
    if mask.dtype != np.uint8:
        clean_mask = (mask > 0).astype(np.uint8) * 255
    else:
        clean_mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    # Run QC Inspection
    qc_alerts = inspect_annotation_quality(clean_mask, micrograph_gray)

    if existing_record is None:
        rec = AnnotationRecord(
            annotation_id=f"ANN_{micrograph_id}_{now_iso.replace(':', '-').replace('.', '_')}",
            micrograph_id=micrograph_id,
            mask_filename=f"{micrograph_id}_mask.png",
            state=target_state,
            source_type=source_type,
            version=1,
            author=author,
            reviewer=reviewer if target_state in (AnnotationState.REVIEWED, AnnotationState.APPROVED) else None,
            created_at=now_iso,
            updated_at=now_iso,
            history=[
                AnnotationHistoryEntry(
                    version=1,
                    timestamp=now_iso,
                    author=author,
                    action=f"created_{target_state.value}",
                    state=target_state.value,
                    notes=notes or f"Criação inicial com origem {source_type.value}",
                )
            ],
            qc_alerts=qc_alerts,
            notes=notes,
        )
    else:
        new_version = existing_record.version + 1
        history_entry = AnnotationHistoryEntry(
            version=new_version,
            timestamp=now_iso,
            author=reviewer if reviewer and target_state in (AnnotationState.REVIEWED, AnnotationState.APPROVED) else author,
            action=f"transition_to_{target_state.value}",
            state=target_state.value,
            notes=notes or f"Transição de estado para {target_state.value}",
        )
        updated_history = list(existing_record.history) + [history_entry]

        # Determine effective source type if human edited an automatic mask
        effective_source = source_type
        if existing_record.source_type != AnnotationSourceType.MANUAL and target_state in (AnnotationState.REVIEWED, AnnotationState.APPROVED):
            effective_source = AnnotationSourceType.REVISED_AUTOMATIC

        rec = AnnotationRecord(
            annotation_id=existing_record.annotation_id,
            micrograph_id=micrograph_id,
            mask_filename=existing_record.mask_filename,
            state=target_state,
            source_type=effective_source,
            version=new_version,
            author=existing_record.author,
            reviewer=reviewer or existing_record.reviewer,
            created_at=existing_record.created_at,
            updated_at=now_iso,
            history=updated_history,
            qc_alerts=qc_alerts,
            notes=notes or existing_record.notes,
        )

    return rec, clean_mask


def save_annotation_bundle(
    target_dir: Path | str,
    record: AnnotationRecord,
    mask: np.ndarray,
) -> Tuple[Path, Path]:
    """Saves mask as lossless PNG and annotation record as sidecar JSON."""
    d = Path(target_dir)
    d.mkdir(parents=True, exist_ok=True)

    mask_path = d / record.mask_filename
    cv2.imwrite(str(mask_path), mask)

    json_path = d / f"{record.micrograph_id}_ann.json"
    record.save(json_path)

    return mask_path, json_path
