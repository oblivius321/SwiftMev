"""
SwiftMev — Módulo de Medições (Alpha 0.3)

Todas as funções de medição recebem uma máscara binária já computada.
A segmentação NÃO é realizada aqui — isso elimina a duplicação de
processamento identificada na Alpha 0.2.

Referências de medição para fibras (três conceitos separados):
  1. Diâmetro nominal (D_geom): espessura definida na geometria contínua.
  2. Largura rasterizada central (W_rast): medida perpendicular na máscara
     binária digital, na região central da fibra.
  3. Diâmetro estimado (D_est): valor calculado pelo algoritmo
     (distanceTransform + skeletonize).
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.morphology import skeletonize

from swiftmev.validation import (
    validate_calibration,
    validate_mask,
    check_resolution_adequacy,
)


def measure_migration(
    mask: np.ndarray,
    um_per_pixel: float,
    image_shape: tuple[int, int] | None = None,
) -> dict:
    """
    Calcula métricas de migração celular (scratch assay) a partir de
    uma máscara binária pré-computada.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binária uint8 (0/255). Foreground = região segmentada.
    um_per_pixel : float
        Fator de calibração (µm/pixel). Validado obrigatoriamente.
    image_shape : tuple[int, int] | None
        (height, width) da imagem original. Se None, usa mask.shape.

    Returns
    -------
    dict
        Métricas da análise de migração.
    """
    validate_calibration(um_per_pixel)
    validate_mask(mask)

    h, w = image_shape if image_shape else mask.shape[:2]
    total = int(h * w)
    area_px = int(np.count_nonzero(mask))

    return {
        "empty_area_px2": area_px,
        "empty_area_um2": round(area_px * (um_per_pixel ** 2), 3),
        "empty_area_fraction": round(area_px / total, 6),
        "total_pixels": total,
        "warning": (
            "Máscara heurística. Confirmar região livre e orientação "
            "do contraste manualmente."
        ),
    }


def measure_fibers(
    mask: np.ndarray,
    um_per_pixel: float,
) -> dict:
    """
    Mede o diâmetro local de fibras via transformada de distância L2
    combinada com esqueletização.

    O resultado (D_est) é o diâmetro estimado pelo algoritmo.
    Para validação científica, compare com D_geom e W_rast separadamente.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binária uint8 (0/255). Foreground = fibra.
    um_per_pixel : float
        Fator de calibração (µm/pixel). Validado obrigatoriamente.

    Returns
    -------
    dict
        Métricas de diâmetro de fibras com avisos de resolução.
    """
    validate_calibration(um_per_pixel)
    validate_mask(mask)

    # Transformada de distância L2 + esqueletização
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask > 0)
    skeleton_count = int(np.count_nonzero(skeleton))

    # Diâmetros locais: 2 * distância no esqueleto
    raw_d_px = 2.0 * distance[skeleton]
    valid = raw_d_px[np.isfinite(raw_d_px) & (raw_d_px > 0)]

    if len(valid) == 0:
        return {
            "skeleton_points": skeleton_count,
            "mean_diameter_px": None,
            "median_diameter_px": None,
            "std_diameter_px": None,
            "mean_diameter_um": None,
            "median_diameter_um": None,
            "std_diameter_um": None,
            "resolution_warning": None,
            "warning": (
                "Nenhum ponto válido no esqueleto. "
                "A fibra pode ser muito fina ou a segmentação inadequada."
            ),
        }

    mean_px = float(np.mean(valid))
    median_px = float(np.median(valid))
    std_px = float(np.std(valid))

    # Aviso de resolução (sem usar o termo Nyquist)
    res_warning = check_resolution_adequacy(median_px)

    return {
        "skeleton_points": skeleton_count,
        "mean_diameter_px": round(mean_px, 3),
        "median_diameter_px": round(median_px, 3),
        "std_diameter_px": round(std_px, 3),
        "mean_diameter_um": round(mean_px * um_per_pixel, 3),
        "median_diameter_um": round(median_px * um_per_pixel, 3),
        "std_diameter_um": round(std_px * um_per_pixel, 3),
        "resolution_warning": res_warning,
        "warning": (
            "Medição exploratória de espessura local via distanceTransform + "
            "skeletonize. Cruzamentos, fibras sobrepostas e efeitos de borda "
            "podem enviesar os resultados."
        ),
    }


def overlay_contours(
    img_gray: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Desenha os contornos da máscara sobre a imagem em escala de cinza.
    Retorna imagem BGR para visualização.
    """
    if img_gray.ndim == 2:
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    else:
        rgb = img_gray.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(rgb, contours, -1, color, thickness)
    return rgb
