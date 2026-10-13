"""
SwiftMev — Módulo de Validação Científica (Alpha 0.3)

Validações obrigatórias que bloqueiam o processamento quando os dados de
entrada não atendem aos requisitos mínimos para uma medição confiável.
"""
from __future__ import annotations
import math
import numpy as np


class CalibrationError(ValueError):
    """Erro de calibração: um_per_pixel inválido ou ausente."""
    pass


class DegenerateMaskError(ValueError):
    """Máscara degenerada: 100% foreground, 100% background ou vazia."""
    pass


def validate_calibration(um_per_pixel: float) -> None:
    """
    Bloqueia o processamento se o fator de calibração for inválido.

    Critérios:
      - Deve ser um número real (não NaN, não infinito)
      - Deve ser estritamente positivo

    Raises
    ------
    CalibrationError
        Se um_per_pixel for None, NaN, infinito, zero ou negativo.
    """
    if um_per_pixel is None:
        raise CalibrationError(
            "Fator de calibração (um_per_pixel) não informado. "
            "Informe um valor positivo e finito antes de executar a análise."
        )
    if not isinstance(um_per_pixel, (int, float)):
        raise CalibrationError(
            f"Fator de calibração deve ser numérico, recebido: {type(um_per_pixel).__name__}"
        )
    if math.isnan(um_per_pixel) or math.isinf(um_per_pixel):
        raise CalibrationError(
            f"Fator de calibração deve ser finito, recebido: {um_per_pixel}"
        )
    if um_per_pixel <= 0:
        raise CalibrationError(
            f"Fator de calibração deve ser positivo (> 0), recebido: {um_per_pixel}"
        )


def validate_mask(mask: np.ndarray) -> None:
    """
    Verifica se a máscara binária é válida para análise.

    Rejeita máscaras inteiramente brancas ou inteiramente pretas,
    pois indicam falha de segmentação.

    Raises
    ------
    DegenerateMaskError
        Se a máscara não contiver foreground e background simultaneamente.
    """
    if mask is None or mask.size == 0:
        raise DegenerateMaskError(
            "Máscara vazia ou nula recebida."
        )
    has_fg = np.any(mask > 0)
    has_bg = np.any(mask == 0)
    if not has_fg:
        raise DegenerateMaskError(
            "Segmentação degenerada: máscara 100% background (sem estrutura detectada). "
            "Verifique os parâmetros de segmentação e a polaridade (--invert)."
        )
    if not has_bg:
        raise DegenerateMaskError(
            "Segmentação degenerada: máscara 100% foreground (toda a imagem classificada como estrutura). "
            "Verifique os parâmetros de segmentação e a polaridade (--invert)."
        )


def check_resolution_adequacy(
    diameter_px: float,
    threshold_px: float = 6.0
) -> dict | None:
    """
    Emite aviso quando a espessura detectada em pixels é baixa o suficiente
    para comprometer a confiabilidade da medição.

    NÃO usa o termo 'Nyquist' porque este critério não é uma aplicação
    formal do teorema de Nyquist-Shannon.

    Parameters
    ----------
    diameter_px : float
        Diâmetro estimado em pixels.
    threshold_px : float
        Limite abaixo do qual o aviso é emitido (default: 6 px).

    Returns
    -------
    dict | None
        Dicionário com aviso, ou None se a resolução for adequada.
    """
    if diameter_px < threshold_px:
        return {
            "level": "warning",
            "code": "LOW_PIXEL_COUNT",
            "message": (
                f"Precisão de medição potencialmente reduzida. "
                f"A espessura detectada ({diameter_px:.1f} px) corresponde a poucos pixels. "
                f"Efeitos de discretização e resolução óptica podem comprometer "
                f"a confiabilidade da medição."
            ),
            "detected_diameter_px": round(diameter_px, 3),
            "threshold_px": threshold_px
        }
    return None
