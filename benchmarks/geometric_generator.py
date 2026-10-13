"""
SwiftMev — Gerador Geométrico Controlado de Fibras (Alpha 0.3)

Mudanças em relação à Alpha 0.2:
  - Jitter estocástico REAL: posição subpixel, ângulo, comprimento e centroide
  - Medição de W_rast na região central (excluindo extremidades)
  - Metadados completos para separar D_geom, W_rast e D_est
  - Geração de imagens de migração celular (scratch assay)
"""
from __future__ import annotations
import cv2
import numpy as np


def measure_perpendicular_width(
    mask: np.ndarray,
    center: tuple[float, float],
    theta_deg: float,
    max_search: float = 80.0,
    step_resolution: float = 0.04,
) -> float:
    """
    Mede a largura transversal contínua da máscara binária rasterizada no ponto 'center'
    ao longo do vetor perpendicular à orientação da fibra.

    A resolução do passo (step_resolution) controla a precisão subpixel.
    """
    theta = np.deg2rad(theta_deg)
    perp = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    c = np.array(center, dtype=np.float64)
    n_steps = int(max_search / step_resolution)

    # Varredura na direção +perp
    step_plus = 0.0
    for s in np.linspace(0.0, max_search, n_steps):
        pt = c + s * perp
        y, x = int(round(pt[1])), int(round(pt[0]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] == 0:
                step_plus = float(s)
                break
        else:
            step_plus = float(s)
            break

    # Varredura na direção -perp
    step_minus = 0.0
    for s in np.linspace(0.0, max_search, n_steps):
        pt = c - s * perp
        y, x = int(round(pt[1])), int(round(pt[0]))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] == 0:
                step_minus = float(s)
                break
        else:
            step_minus = float(s)
            break

    return round(float(step_plus + step_minus), 4)


def generate_geometric_fiber(
    size: int = 256,
    nominal_diameter: float = 16.0,
    angle_deg: float = 0.0,
    length: float = 180.0,
    condition: str = "ideal",
    seed: int = 42,
    jitter_center: float = 0.0,
    jitter_angle_deg: float = 0.0,
    jitter_length: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Gera uma imagem sintética e sua máscara binária ground truth para uma fibra
    com orientação theta_deg e diâmetro nominal D_geom.

    Jitter estocástico real (Alpha 0.3):
      - jitter_center: desvio padrão em pixels para deslocamento do centroide (x, y)
      - jitter_angle_deg: desvio padrão em graus para perturbação angular
      - jitter_length: desvio padrão em pixels para perturbação do comprimento

    Parameters
    ----------
    size : int
        Dimensão da imagem quadrada (size x size).
    nominal_diameter : float
        Diâmetro geométrico contínuo da fibra (D_geom), em pixels.
    angle_deg : float
        Orientação base da fibra em graus.
    length : float
        Comprimento base da fibra em pixels.
    condition : str
        Condição de imagem: 'ideal', 'noise', 'gradient', 'low_contrast'.
    seed : int
        Semente para reprodutibilidade.
    jitter_center : float
        σ do jitter de posição do centroide (pixels).
    jitter_angle_deg : float
        σ do jitter angular (graus).
    jitter_length : float
        σ do jitter de comprimento (pixels).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        (image, gt_mask, metadata)
    """
    rng = np.random.default_rng(seed)

    # Aplicar jitter estocástico
    actual_angle = angle_deg + rng.normal(0, jitter_angle_deg) if jitter_angle_deg > 0 else angle_deg
    actual_length = max(10.0, length + rng.normal(0, jitter_length)) if jitter_length > 0 else length

    center_x = size / 2.0 + (rng.normal(0, jitter_center) if jitter_center > 0 else 0.0)
    center_y = size / 2.0 + (rng.normal(0, jitter_center) if jitter_center > 0 else 0.0)

    theta = np.deg2rad(actual_angle)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)   # longitudinal
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)  # transversal

    center = np.array([center_x, center_y], dtype=np.float64)

    # 4 vértices do retângulo contínuo (geometria exata)
    half_l = actual_length / 2.0
    half_d = nominal_diameter / 2.0

    p1 = center + half_l * u + half_d * v
    p2 = center - half_l * u + half_d * v
    p3 = center - half_l * u - half_d * v
    p4 = center + half_l * u - half_d * v

    # Rasterização com fillConvexPoly para polígonos convexos (mais precisa que fillPoly)
    pts_float = np.array([p1, p2, p3, p4], dtype=np.float64)
    pts_int = np.round(pts_float).astype(np.int32)

    gt_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(gt_mask, pts_int, 255)

    # Medição de W_rast na região central (excluindo extremidades)
    # Usa 5 pontos no terço central da fibra
    w_rast_samples = []
    for offset_factor in [-0.15, -0.075, 0.0, 0.075, 0.15]:
        sample_pt = center + (offset_factor * actual_length) * u
        w = measure_perpendicular_width(
            gt_mask,
            (float(sample_pt[0]), float(sample_pt[1])),
            actual_angle,
        )
        if w > 0:
            w_rast_samples.append(w)

    w_rast = float(np.median(w_rast_samples)) if w_rast_samples else float(nominal_diameter)

    # Geração da imagem com condições de iluminação
    img = _apply_imaging_condition(gt_mask, size, condition, rng)

    meta = {
        "nominal_diameter_px": float(nominal_diameter),
        "raster_width_central_px": round(w_rast, 4),
        "rasterization_error_px": round(w_rast - nominal_diameter, 4),
        "angle_deg_nominal": float(angle_deg),
        "angle_deg_actual": round(float(actual_angle), 4),
        "length_nominal": float(length),
        "length_actual": round(float(actual_length), 4),
        "center_x": round(float(center_x), 4),
        "center_y": round(float(center_y), 4),
        "condition": condition,
        "seed": seed,
        "jitter_center_sigma": float(jitter_center),
        "jitter_angle_sigma_deg": float(jitter_angle_deg),
        "jitter_length_sigma": float(jitter_length),
        "mask_area_px": int(np.count_nonzero(gt_mask)),
    }

    return img, gt_mask, meta


def _apply_imaging_condition(
    gt_mask: np.ndarray,
    size: int,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gera a imagem sintética com a condição de iluminação especificada."""
    n_fg = int(np.count_nonzero(gt_mask))

    if condition == "ideal":
        bg = rng.integers(15, 30, (size, size), dtype=np.uint8)
        img = bg.astype(np.float32)
        img[gt_mask > 0] = rng.integers(220, 245, n_fg, dtype=np.uint8)

    elif condition == "noise":
        bg = rng.integers(15, 30, (size, size), dtype=np.uint8)
        img = bg.astype(np.float32)
        img[gt_mask > 0] = rng.integers(215, 245, n_fg, dtype=np.uint8)
        noise_gauss = rng.normal(0, 20, (size, size)).astype(np.float32)
        img += noise_gauss
        sp = rng.random((size, size))
        img[sp < 0.015] = 0.0
        img[sp > 0.985] = 255.0

    elif condition == "gradient":
        grad_y = np.linspace(0, 50, size, dtype=np.float32).reshape(size, 1)
        grad_x = np.linspace(0, 50, size, dtype=np.float32).reshape(1, size)
        bg = rng.integers(15, 30, (size, size), dtype=np.uint8).astype(np.float32) + grad_y + grad_x
        img = bg
        img[gt_mask > 0] = np.clip(img[gt_mask > 0] + 170.0, 0, 255.0)

    elif condition == "low_contrast":
        bg = rng.integers(110, 130, (size, size), dtype=np.uint8).astype(np.float32)
        img = bg
        img[gt_mask > 0] = rng.integers(160, 185, n_fg, dtype=np.uint8)

    else:
        raise ValueError(
            f"Condição desconhecida: '{condition}'. "
            f"Opções: 'ideal', 'noise', 'gradient', 'low_contrast'."
        )

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_scratch_assay(
    size: tuple[int, int] = (256, 256),
    scratch_fraction: float = 0.30,
    condition: str = "ideal",
    seed: int = 42,
    jitter_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Gera uma imagem sintética de scratch assay (migração celular).

    Parameters
    ----------
    size : tuple[int, int]
        (height, width) da imagem.
    scratch_fraction : float
        Fração da largura da imagem ocupada pelo scratch (região livre).
    condition : str
        Condição de imagem: 'ideal', 'noise', 'gradient', 'low_contrast'.
    seed : int
        Semente para reprodutibilidade.
    jitter_width : float
        σ do jitter na largura do scratch (pixels).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        (image, gt_mask, metadata)
        - gt_mask: foreground (255) = região livre (scratch)
    """
    rng = np.random.default_rng(seed)
    h, w = size

    scratch_w = int(w * scratch_fraction)
    if jitter_width > 0:
        scratch_w = max(4, scratch_w + int(rng.normal(0, jitter_width)))
    scratch_w = min(scratch_w, w - 4)

    x_start = (w - scratch_w) // 2

    # Ground truth: scratch = foreground (255)
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    gt_mask[:, x_start:x_start + scratch_w] = 255

    actual_fraction = float(np.count_nonzero(gt_mask)) / (h * w)

    # Gerar imagem
    if condition == "ideal":
        img = rng.integers(180, 220, (h, w), dtype=np.uint8).astype(np.float32)
        img[:, x_start:x_start + scratch_w] = rng.integers(
            20, 40, (h, scratch_w), dtype=np.uint8
        )
    elif condition == "noise":
        img = rng.integers(180, 220, (h, w), dtype=np.uint8).astype(np.float32)
        img[:, x_start:x_start + scratch_w] = rng.integers(
            20, 40, (h, scratch_w), dtype=np.uint8
        )
        img += rng.normal(0, 15, (h, w)).astype(np.float32)
    elif condition == "gradient":
        grad = np.linspace(0, 60, w, dtype=np.float32).reshape(1, w)
        img = rng.integers(160, 200, (h, w), dtype=np.uint8).astype(np.float32) + grad
        img[:, x_start:x_start + scratch_w] = (
            rng.integers(20, 40, (h, scratch_w), dtype=np.uint8).astype(np.float32) + grad[:, x_start:x_start + scratch_w]
        )
    elif condition == "low_contrast":
        img = rng.integers(130, 160, (h, w), dtype=np.uint8).astype(np.float32)
        img[:, x_start:x_start + scratch_w] = rng.integers(
            80, 110, (h, scratch_w), dtype=np.uint8
        )
    else:
        raise ValueError(f"Condição desconhecida: '{condition}'.")

    img = np.clip(img, 0, 255).astype(np.uint8)

    meta = {
        "scratch_fraction_nominal": float(scratch_fraction),
        "scratch_fraction_actual": round(actual_fraction, 6),
        "scratch_width_px": scratch_w,
        "condition": condition,
        "seed": seed,
        "jitter_width_sigma": float(jitter_width),
        "image_size": list(size),
    }

    return img, gt_mask, meta
