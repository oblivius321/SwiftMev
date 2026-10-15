"""
SwiftMev AI — Plataforma de Inteligência Artificial Científica (Alpha 0.6A)

Módulo desacoplado para inferência offline de redes neurais via ONNX Runtime CPU.
"""
from __future__ import annotations

from swiftmev.ai.exceptions import (
    AIError,
    AINotAvailableError,
    AIModelNotFoundError,
    ModelManifestError,
    ChecksumMismatchError,
    TaskIncompatibilityError,
    TensorMismatchError,
)
from swiftmev.ai.model_manifest import ModelManifest
from swiftmev.ai.base import BaseSegmentationModel
from swiftmev.ai.onnx_engine import ONNXSegmentationEngine
from swiftmev.ai.registry import ModelRegistry, get_default_models_dir


def is_ai_available() -> bool:
    """Verifica se o ambiente possui suporte à inferência por IA (onnxruntime instalado)."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def list_available_models(models_dir: str | None = None) -> list[dict]:
    """Retorna lista de modelos de IA disponíveis no disco local."""
    reg = ModelRegistry(models_dir=models_dir)
    return reg.list_models()


def get_model(
    model_id: str,
    expected_task: str | None = None,
    models_dir: str | None = None
) -> BaseSegmentationModel:
    """Instancia um modelo pelo seu ID, validando compatibilidade de tarefa."""
    reg = ModelRegistry(models_dir=models_dir)
    return reg.get_model(model_id=model_id, expected_task=expected_task)


__all__ = [
    "is_ai_available",
    "list_available_models",
    "get_model",
    "ModelRegistry",
    "ModelManifest",
    "BaseSegmentationModel",
    "ONNXSegmentationEngine",
    "AIError",
    "AINotAvailableError",
    "AIModelNotFoundError",
    "ModelManifestError",
    "ChecksumMismatchError",
    "TaskIncompatibilityError",
    "TensorMismatchError",
    "get_default_models_dir",
]
