"""
SwiftMev — Módulo de Imaging (Alpha 0.3)

Carregamento, conversão de profundidade de bits e pré-processamento de imagens.
Todas as funções são puras: recebem arrays e devolvem arrays, sem estado global.
"""
from __future__ import annotations
import cv2
import numpy as np


def ensure_gray_uint8_with_meta(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Garante que a imagem esteja em escala de cinza e com profundidade uint8,
    registrando metadados da transformação aplicada para fins de reprodutibilidade científica.

    Suporta:
      - uint8 BGR / BGRA / grayscale
      - uint16 (microscopia TIFF)
      - float32 / float64 (normalizado)
    """
    orig_dtype = str(img.dtype)
    orig_shape = list(img.shape)
    channels = img.shape[2] if img.ndim == 3 else 1

    meta = {
        "original_dtype": orig_dtype,
        "original_shape": orig_shape,
        "channels": channels,
        "min_raw_val": float(img.min()),
        "max_raw_val": float(img.max()),
        "scaling_applied": "none"
    }

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if orig_dtype == "uint16":
        min_val, max_val = meta["min_raw_val"], meta["max_raw_val"]
        if max_val > min_val:
            img = ((img.astype(np.float32) - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            meta["scaling_applied"] = "min_max_linear_16bit_to_8bit"
        else:
            img = np.zeros(img.shape, dtype=np.uint8)
            meta["scaling_applied"] = "zero_constant"
    elif orig_dtype in ("float32", "float64"):
        min_val, max_val = meta["min_raw_val"], meta["max_raw_val"]
        if max_val > min_val:
            img = ((img - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            meta["scaling_applied"] = "min_max_linear_float_to_8bit"
        else:
            img = np.zeros(img.shape, dtype=np.uint8)
            meta["scaling_applied"] = "zero_constant"
    elif img.dtype != np.uint8:
        raise ValueError(f"Tipo de dados de imagem não suportado: {img.dtype}")

    return img, meta


def ensure_gray_uint8(img: np.ndarray) -> np.ndarray:
    """Garante que a imagem esteja em escala de cinza e com profundidade uint8."""
    gray, _ = ensure_gray_uint8_with_meta(img)
    return gray


def load_gray_with_meta(path: str) -> tuple[np.ndarray, dict]:
    """Carrega imagem do disco e retorna matriz uint8 acompanhada dos metadados de transformação."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")
    return ensure_gray_uint8_with_meta(img)


def load_gray(path: str) -> np.ndarray:
    """Carrega imagem do disco e converte para escala de cinza uint8."""
    img, _ = load_gray_with_meta(path)
    return img


def preprocess(
    image: np.ndarray,
    use_clahe: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: tuple[int, int] = (8, 8),
    blur_ksize: tuple[int, int] = (3, 3)
) -> np.ndarray:
    """
    Aplica pré-processamento configurável.

    Pipeline:
      1. Conversão para grayscale uint8 (se necessário)
      2. CLAHE (opcional) — equalização adaptativa de histograma
      3. Suavização Gaussiana (se ksize > 0)
    """
    gray = ensure_gray_uint8(image)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)
    if blur_ksize[0] > 0 and blur_ksize[1] > 0:
        gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    return gray
