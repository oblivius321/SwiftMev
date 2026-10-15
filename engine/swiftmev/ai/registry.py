"""
SwiftMev AI — Catálogo e Descoberta Local de Modelos (Alpha 0.6A)

Gerenciamento de ciclo de vida, descoberta em disco, verificação de compatibilidade de tarefas
e carregamento preguiçoso (Lazy Loading).
"""
from __future__ import annotations
import os
from pathlib import Path

from swiftmev.ai.base import BaseSegmentationModel
from swiftmev.ai.exceptions import (
    AIModelNotFoundError,
    TaskIncompatibilityError,
    ModelManifestError,
)
from swiftmev.ai.model_manifest import ModelManifest
from swiftmev.ai.onnx_engine import ONNXSegmentationEngine


def get_default_models_dir() -> Path:
    """Identifica o diretório padrão de modelos locais."""
    env_dir = os.environ.get("SWIFTMEV_MODELS_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir).resolve()

    # Procura a pasta models/ a partir do repositório
    curr = Path(__file__).resolve().parent
    for _ in range(5):
        cand = curr / "models"
        if cand.exists() and cand.is_dir():
            return cand.resolve()
        curr = curr.parent

    # Fallback relativo à raiz
    return (Path(__file__).resolve().parent.parent.parent.parent / "models").resolve()


class ModelRegistry:
    """Registro e gerenciador de modelos locais de inteligência artificial."""

    def __init__(self, models_dir: Path | str | None = None):
        if models_dir is not None:
            self.models_dir = Path(models_dir).resolve()
        else:
            self.models_dir = get_default_models_dir()

    def list_models(self) -> list[dict]:
        """Escaneia a pasta models/ e lista todos os modelos válidos com seus manifestos."""
        if not self.models_dir.exists():
            return []

        models = []
        # Suporta modelos organizados em pastas (ex: models/unet_fiber/manifest.json) ou arquivos diretos
        manifest_paths = list(self.models_dir.glob("**/manifest.json")) + list(self.models_dir.glob("*_manifest.json"))
        # Evita duplicatas se algum arquivo foi coletado duas vezes
        manifest_paths = sorted(list(set(manifest_paths)))

        for mp in manifest_paths:
            try:
                manifest = ModelManifest.load(mp)
                onnx_file = mp.parent / manifest.onnx_filename
                item = manifest.to_dict()
                item["installed"] = onnx_file.exists()
                models.append(item)
            except ModelManifestError:
                continue

        return models

    def get_model(
        self,
        model_id: str,
        expected_task: str | None = None
    ) -> BaseSegmentationModel:
        """
        Localiza, valida a tarefa e instancia o motor de inferência para o model_id.

        Parameters
        ----------
        model_id : str
            Identificador do modelo.
        expected_task : str | None
            Tarefa requerida pela pipeline (ex: 'fiber_segmentation').
            Se o modelo for incompatible, lança TaskIncompatibilityError.
        """
        if not self.models_dir.exists():
            raise AIModelNotFoundError(model_id, str(self.models_dir))

        # Busca pelo manifest.json onde model_id confere
        manifest_paths = list(self.models_dir.glob("**/manifest.json")) + list(self.models_dir.glob("*_manifest.json"))
        manifest_paths = sorted(list(set(manifest_paths)))

        target_manifest: ModelManifest | None = None

        for mp in manifest_paths:
            try:
                m = ModelManifest.load(mp)
                if m.model_id == model_id:
                    target_manifest = m
                    break
            except ModelManifestError:
                continue

        if target_manifest is None:
            raise AIModelNotFoundError(model_id, str(self.models_dir))

        # Validação estrita de compatibilidade de tarefa
        if expected_task is not None and target_manifest.task != expected_task:
            raise TaskIncompatibilityError(
                model_task=target_manifest.task,
                requested_task=expected_task,
            )

        return ONNXSegmentationEngine(target_manifest)
