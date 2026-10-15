"""
SwiftMev AI — Pós-processamento e Binarização (Alpha 0.6A)

Converte mapas de probabilidade contínuos para máscaras binárias uint8 (0 ou 255)
em total conformidade com o motor de medições e o editor interativo de máscaras.
"""
from __future__ import annotations
import numpy as np


def extract_probability_tile(output_tensor: np.ndarray) -> np.ndarray:
    """
    Extrai a matriz 2D de probabilidades [0.0, 1.0] de saída do modelo ONNX.
    Suporta saídas com formato:
      - (1, 1, H, W): sigmoide ou logit monocanal
      - (1, 2, H, W): softmax de 2 classes (canal 1 = foreground)
      - (1, H, W) ou (H, W)
    """
    arr = output_tensor
    if arr.ndim == 4:
        if arr.shape[1] == 2:
            # Softmax 2 classes: softmax(arr, axis=1)[:, 1, :, :]
            exp_arr = np.exp(arr - np.max(arr, axis=1, keepdims=True))
            probs = exp_arr[:, 1, :, :] / np.sum(exp_arr, axis=1)
            prob_tile = probs[0]
        else:
            prob_tile = arr[0, 0]
    elif arr.ndim == 3:
        prob_tile = arr[0]
    elif arr.ndim == 2:
        prob_tile = arr
    else:
        raise ValueError(f"Dimensão de saída ONNX não suportada: {arr.shape}")

    # Se os valores não estiverem no intervalo [0, 1], aplicar sigmoide numérica estável
    min_v, max_v = float(np.min(prob_tile)), float(np.max(prob_tile))
    if min_v < -0.05 or max_v > 1.05:
        prob_tile = 1.0 / (1.0 + np.exp(-np.clip(prob_tile, -30.0, 30.0)))

    return np.clip(prob_tile, 0.0, 1.0).astype(np.float32)


def binarize_probability_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binariza o mapa de probabilidades na resolução nativa.

    Parameters
    ----------
    prob_map : np.ndarray
        Matriz 2D de probabilidades contínuas [0.0, 1.0].
    threshold : float
        Limiar de corte para classificação de nanofibra (padrão: 0.5).

    Returns
    -------
    np.ndarray
        Máscara binária uint8 estritamente com valores 0 e 255.
    """
    if prob_map.ndim != 2:
        raise ValueError(f"Mapa de probabilidades deve ser 2D (H, W). Recebido: {prob_map.shape}")

    mask = np.where(prob_map >= threshold, 255, 0).astype(np.uint8)
    return mask
