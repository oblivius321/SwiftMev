"""
SwiftMev AI — Pré-processamento e Tiling com Sobreposição (Alpha 0.6A)

Implementa inferência por blocos sobrepostos (overlapping tiling) em resolução nativa,
com janelamento trapezoidal para eliminação de artefatos de borda e preservação geométrica estrita.
"""
from __future__ import annotations
import cv2
import numpy as np


def normalize_tensor_tile(
    tile: np.ndarray,
    target_channels: int = 1,
    normalization_config: dict | None = None
) -> np.ndarray:
    """
    Normaliza um recorte (tile) para tensor NumPy float32 no formato (1, C, H, W).
    """
    if normalization_config is None:
        normalization_config = {"type": "min_max", "scale": 1.0 / 255.0}

    # Ajuste de canais
    if tile.ndim == 2:
        if target_channels == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)
        else:
            tile = tile[:, :, np.newaxis]
    elif tile.ndim == 3:
        if tile.shape[2] == 4:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGRA2GRAY) if target_channels == 1 else cv2.cvtColor(tile, cv2.COLOR_BGRA2RGB)
        elif tile.shape[2] == 3 and target_channels == 1:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            tile = tile[:, :, np.newaxis]
        elif tile.shape[2] == 1 and target_channels == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)

    norm_type = normalization_config.get("type", "min_max")
    scale = float(normalization_config.get("scale", 1.0 / 255.0))

    tile_f = tile.astype(np.float32)

    if norm_type == "min_max":
        tile_norm = tile_f * scale
    elif norm_type == "standard":
        mean = float(normalization_config.get("mean", 0.5))
        std = float(normalization_config.get("std", 0.5))
        tile_norm = (tile_f * scale - mean) / (std + 1e-7)
    else:
        tile_norm = tile_f * scale

    # Transpor para formato NCHW: (1, C, H, W)
    if tile_norm.ndim == 3:
        tile_nchw = np.transpose(tile_norm, (2, 0, 1))
    else:
        tile_nchw = tile_norm[np.newaxis, :, :]

    return np.ascontiguousarray(tile_nchw[np.newaxis, ...], dtype=np.float32)


def compute_taper_window(
    tile_h: int,
    tile_w: int,
    overlap: int,
    is_top: bool,
    is_bottom: bool,
    is_left: bool,
    is_right: bool,
) -> np.ndarray:
    """
    Gera máscara de pesos trapezoidal 2D para ponderação de sobreposição.
    Bordas que tocam a extremidade real da imagem mantêm peso 1.0 para não degradar a margem.
    Bordas internas sofrem decaimento suave para fusão perfeita entre blocos adjacentes.
    """
    def ramp_1d(length: int, ramp_len: int, touch_start: bool, touch_end: bool) -> np.ndarray:
        w = np.ones(length, dtype=np.float32)
        r = min(ramp_len, length // 2)
        if r > 0:
            linear = np.linspace(0.05, 1.0, r, dtype=np.float32)
            if not touch_start:
                w[:r] = linear
            if not touch_end:
                w[-r:] = linear[::-1]
        return w

    wy = ramp_1d(tile_h, overlap, is_top, is_bottom)
    wx = ramp_1d(tile_w, overlap, is_left, is_right)
    return np.outer(wy, wx).astype(np.float32)


def generate_tile_coordinates(
    img_h: int,
    img_w: int,
    tile_h: int,
    tile_w: int,
    overlap: int,
) -> list[tuple[int, int, int, int, bool, bool, bool, bool]]:
    """
    Calcula as coordenadas de corte (y0, y1, x0, x1) e flags de borda externa
    para cobrir toda a imagem em resolução nativa.
    """
    overlap_y = min(overlap, tile_h // 2)
    overlap_x = min(overlap, tile_w // 2)
    step_y = max(1, tile_h - overlap_y)
    step_x = max(1, tile_w - overlap_x)

    y_points = list(range(0, max(1, img_h - tile_h + 1), step_y))
    if not y_points or y_points[-1] + tile_h < img_h:
        y_points.append(max(0, img_h - tile_h))
    y_points = sorted(list(set(y_points)))

    x_points = list(range(0, max(1, img_w - tile_w + 1), step_x))
    if not x_points or x_points[-1] + tile_w < img_w:
        x_points.append(max(0, img_w - tile_w))
    x_points = sorted(list(set(x_points)))

    coords = []
    for yi, y in enumerate(y_points):
        y0 = y
        y1 = min(img_h, y0 + tile_h)
        is_top = (yi == 0)
        is_bottom = (y1 >= img_h)

        for xi, x in enumerate(x_points):
            x0 = x
            x1 = min(img_w, x0 + tile_w)
            is_left = (xi == 0)
            is_right = (x1 >= img_w)

            coords.append((y0, y1, x0, x1, is_top, is_bottom, is_left, is_right))

    return coords
