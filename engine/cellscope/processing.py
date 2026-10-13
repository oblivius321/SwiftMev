from __future__ import annotations
import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize


def ensure_gray_uint8_with_meta(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Garante que a imagem esteja em escala de cinza e com profundidade uint8,
    registrando metadados da transformação aplicada para fins de reprodutibilidade científica.
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
    """Aplica pré-processamento configurável (conversão de tipo, CLAHE opcional e suavização)."""
    gray = ensure_gray_uint8(image)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        gray = clahe.apply(gray)
    if blur_ksize[0] > 0 and blur_ksize[1] > 0:
        gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    return gray


def segment(
    img: np.ndarray,
    method: str = "otsu",
    invert: bool = False,
    use_clahe: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2,
    apply_morphology: bool = True
) -> np.ndarray:
    """Segmenta a imagem utilizando método global (Otsu) ou adaptativo local (Sauvola)."""
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
        raise ValueError(f"Método de segmentação desconhecido: '{method}'. Opções válidas: 'otsu', 'sauvola'.")

    if apply_morphology:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Desenha os contornos da máscara em verde sobre a imagem original."""
    gray = ensure_gray_uint8(img)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    return rgb


def migration(
    img: np.ndarray,
    um_per_pixel: float,
    method: str = "otsu",
    use_clahe: bool = False,
    invert: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2
) -> tuple[dict, np.ndarray]:
    """Análise de migração celular (scratch assay)."""
    mask = segment(
        img,
        method=method,
        invert=invert,
        use_clahe=use_clahe,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k
    )
    area_px = int(np.count_nonzero(mask))
    total = int(img.shape[0] * img.shape[1])
    metrics = {
        "method": method,
        "clahe_applied": use_clahe,
        "inverted": invert,
        "empty_area_px2": area_px,
        "empty_area_um2": round(area_px * (um_per_pixel ** 2), 3),
        "empty_area_fraction": round(area_px / total, 6),
        "warning": "Máscara heurística. Confirmar região livre e orientação do contraste manualmente."
    }
    return metrics, overlay(img, mask)


def fiber(
    img: np.ndarray,
    um_per_pixel: float,
    method: str = "otsu",
    use_clahe: bool = False,
    invert: bool = False,
    sauvola_window: int = 35,
    sauvola_k: float = 0.2
) -> tuple[dict, np.ndarray]:
    """Análise do diâmetro local de fibras longitudinais via transformada de distância e esqueletização."""
    mask = segment(
        img,
        method=method,
        invert=invert,
        use_clahe=use_clahe,
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k
    )
    # Se a máscara for vazia ou preencher 100% da imagem, não há contornos válidos
    if not np.any(mask > 0) or not np.any(mask == 0):
        metrics = {
            "method": method,
            "clahe_applied": use_clahe,
            "inverted": invert,
            "skeleton_points": 0,
            "mean_local_diameter_um": None,
            "median_local_diameter_um": None,
            "std_local_diameter_um": None,
            "warning": "Segmentação degenerada (sem primeiro plano ou sem fundo detectado). Medição não aplicável."
        }
        return metrics, overlay(img, mask)

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask > 0)
    raw_d = 2.0 * distance[skeleton] * um_per_pixel
    local_diameters = raw_d[np.isfinite(raw_d) & (raw_d > 0)]

    metrics = {
        "method": method,
        "clahe_applied": use_clahe,
        "inverted": invert,
        "skeleton_points": int(len(local_diameters)),
        "mean_local_diameter_um": round(float(np.mean(local_diameters)), 3) if len(local_diameters) else None,
        "median_local_diameter_um": round(float(np.median(local_diameters)), 3) if len(local_diameters) else None,
        "std_local_diameter_um": round(float(np.std(local_diameters)), 3) if len(local_diameters) else None,
        "warning": "Medição exploratória de espessura local. Cruzamentos e fibras sobrepostas podem enviesar os resultados."
    }
    return metrics, overlay(img, mask)
