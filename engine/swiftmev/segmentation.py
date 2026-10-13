"""
SwiftMev — Módulo de Segmentação (Alpha 0.3)

Responsável exclusivamente por converter uma imagem pré-processada em uma
máscara binária. Não realiza medições — apenas segmentação.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from swiftmev.imaging import preprocess


def segment(
    img: np.ndarray,
    method: str = "otsu",
    invert: bool = False,
    use_clahe: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2,
    apply_morphology: bool = True
) -> np.ndarray:
    """
    Segmenta a imagem utilizando método global (Otsu) ou adaptativo local (Sauvola).

    Parameters
    ----------
    img : np.ndarray
        Imagem de entrada (qualquer profundidade suportada por preprocess).
    method : str
        'otsu' para limiarização global, 'sauvola' para adaptativa local.
    invert : bool
        Inverter polaridade da máscara (foreground ↔ background).
    use_clahe : bool
        Aplicar CLAHE antes da segmentação.
    sauvola_window : int
        Tamanho da janela local para Sauvola (deve ser ímpar).
    sauvola_k : float
        Parâmetro k para Sauvola.
    apply_morphology : bool
        Aplicar abertura + fechamento morfológico para remoção de ruído.

    Returns
    -------
    np.ndarray
        Máscara binária uint8 (0 ou 255).
    """
    gray = preprocess(img, use_clahe=use_clahe)
    method_lower = method.lower()

    if method_lower == "otsu":
        mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    elif method_lower == "sauvola":
        window_size = sauvola_window if sauvola_window % 2 != 0 else sauvola_window + 1
        thresh_map = threshold_sauvola(gray, window_size=window_size, k=sauvola_k)
        if invert:
            mask = np.where(gray < thresh_map, 255, 0).astype(np.uint8)
        else:
            mask = np.where(gray > thresh_map, 255, 0).astype(np.uint8)
    else:
        raise ValueError(
            f"Método de segmentação desconhecido: '{method}'. "
            f"Opções válidas: 'otsu', 'sauvola'."
        )

    if apply_morphology:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
