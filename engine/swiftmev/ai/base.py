"""
SwiftMev AI — Interface Abstrata de Segmentação (Alpha 0.6A)

Contrato padrão para qualquer arquitetura de modelo de inteligência artificial.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from swiftmev.ai.model_manifest import ModelManifest


class BaseSegmentationModel(ABC):
    """Classe base abstrata para todos os segmentadores científicos do SwiftMev."""

    def __init__(self, manifest: ModelManifest):
        self.manifest = manifest

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Indica se a sessão/pesos estão em memória."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Carrega os pesos e prepara o motor de inferência."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Libera tensores e memória RAM alocada."""
        pass

    @abstractmethod
    def predict_probability_map(self, image: np.ndarray) -> np.ndarray:
        """
        Executa a inferência e retorna mapa de probabilidades contínuo [0.0, 1.0]
        na resolução original da micrografia (H, W).
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Executa a inferência e retorna máscara binária uint8 (0 ou 255)
        com dimensões idênticas à imagem de entrada (H, W).
        """
        pass

    def get_manifest(self) -> ModelManifest:
        return self.manifest
