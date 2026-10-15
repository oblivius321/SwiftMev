"""
SwiftMev — Pipeline Unificado (Alpha 0.3)

Orquestra o fluxo completo:
  Imagem → Pré-processamento → Máscara → Métricas → Overlay

Cada etapa é chamada exatamente uma vez, resolvendo a duplicação de
processamento da Alpha 0.2 (onde fiber() e segment() recalculavam a máscara).
"""
from __future__ import annotations
import numpy as np

from swiftmev.imaging import ensure_gray_uint8
from swiftmev.segmentation import segment
from swiftmev.measurements import measure_migration, measure_fibers, overlay_contours
from swiftmev.validation import validate_calibration


def run_migration(
    img: np.ndarray,
    um_per_pixel: float,
    method: str = "otsu",
    use_clahe: bool = False,
    invert: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2,
    **kwargs,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Pipeline completo de migração celular (scratch assay).
    """
    if method.lower() == "ai":
        from swiftmev.ai.exceptions import TaskIncompatibilityError
        raise TaskIncompatibilityError(
            model_task="fiber_segmentation",
            requested_task="migration_segmentation"
        )

    validate_calibration(um_per_pixel)

    # 1. Segmentação (executada uma única vez)
    mask = segment(
        img,
        method=method,
        invert=invert,
        use_clahe=use_clahe,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k,
    )

    # 2. Métricas (recebe a máscara já computada)
    metrics = measure_migration(mask, um_per_pixel, image_shape=img.shape[:2])
    metrics["method"] = method
    metrics["clahe_applied"] = use_clahe
    metrics["inverted"] = invert

    # 3. Overlay (reutiliza a mesma máscara)
    gray = ensure_gray_uint8(img)
    annotated = overlay_contours(gray, mask)

    return metrics, mask, annotated


def run_fiber(
    img: np.ndarray,
    um_per_pixel: float,
    method: str = "otsu",
    use_clahe: bool = False,
    invert: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2,
    model_id: str | None = None,
    models_dir: str | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Pipeline completo de análise de fibras.

    Returns
    -------
    tuple[dict, np.ndarray, np.ndarray]
        (metrics, mask, annotated_image)
    """
    validate_calibration(um_per_pixel)

    # 1. Segmentação (executada uma única vez)
    mask = segment(
        img,
        method=method,
        invert=invert,
        use_clahe=use_clahe,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k,
        model_id=model_id,
        models_dir=models_dir,
        expected_task="fiber_segmentation",
    )

    # 2. Métricas (recebe a máscara já computada)
    metrics = measure_fibers(mask, um_per_pixel)
    metrics["method"] = method
    metrics["clahe_applied"] = use_clahe
    metrics["inverted"] = invert

    if method.lower() == "ai":
        from swiftmev.ai import get_model
        ai_m = get_model(model_id or "test_fiber_onnx_mock", expected_task="fiber_segmentation", models_dir=models_dir)
        metrics["ai_model"] = {
            "model_id": ai_m.manifest.model_id,
            "name": ai_m.manifest.name,
            "architecture": ai_m.manifest.architecture,
            "model_version": ai_m.manifest.model_version,
            "checksum_sha256": ai_m.manifest.checksum_sha256,
            "license": ai_m.manifest.license,
            "provenance": ai_m.manifest.provenance,
            "scientific_limitations": ai_m.manifest.scientific_limitations,
            "status": "experimental_unvalidated",
        }

    # 3. Overlay (reutiliza a mesma máscara)
    gray = ensure_gray_uint8(img)
    annotated = overlay_contours(gray, mask)

    return metrics, mask, annotated


def run_fiber_from_mask(
    mask: np.ndarray,
    um_per_pixel: float,
    img_for_overlay: np.ndarray | None = None,
) -> tuple[dict, np.ndarray | None]:
    """
    Análise de fibras a partir de máscara pré-computada.

    Útil para benchmarks onde a segmentação já foi realizada e queremos
    medir apenas os diâmetros sem reprocessar.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binária uint8.
    um_per_pixel : float
        Fator de calibração.
    img_for_overlay : np.ndarray | None
        Imagem base para overlay. Se None, overlay não é gerado.

    Returns
    -------
    tuple[dict, np.ndarray | None]
        (metrics, annotated_image_or_None)
    """
    validate_calibration(um_per_pixel)
    metrics = measure_fibers(mask, um_per_pixel)

    annotated = None
    if img_for_overlay is not None:
        gray = ensure_gray_uint8(img_for_overlay)
        annotated = overlay_contours(gray, mask)

    return metrics, annotated


def run_migration_from_mask(
    mask: np.ndarray,
    um_per_pixel: float,
    image_shape: tuple[int, int] | None = None,
    img_for_overlay: np.ndarray | None = None,
) -> tuple[dict, np.ndarray | None]:
    """
    Análise de migração celular a partir de máscara pré-computada ou editada.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binária uint8.
    um_per_pixel : float
        Fator de calibração.
    image_shape : tuple[int, int] | None
        Dimensões (height, width) originais.
    img_for_overlay : np.ndarray | None
        Imagem base para overlay.

    Returns
    -------
    tuple[dict, np.ndarray | None]
        (metrics, annotated_image_or_None)
    """
    validate_calibration(um_per_pixel)
    metrics = measure_migration(mask, um_per_pixel, image_shape=image_shape)

    annotated = None
    if img_for_overlay is not None:
        gray = ensure_gray_uint8(img_for_overlay)
        annotated = overlay_contours(gray, mask)

    return metrics, annotated

