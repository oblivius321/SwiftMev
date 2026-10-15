"""
SwiftMev AI — Motor de Inferência ONNX Runtime CPU (Alpha 0.6A)

Execução estritamente local (offline), com validação criptográfica SHA-256,
verificação dos tensores do modelo e reconstrução por blocos sobrepostos (tiling).
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import numpy as np

from swiftmev.ai.base import BaseSegmentationModel
from swiftmev.ai.exceptions import (
    AINotAvailableError,
    AIModelNotFoundError,
    ChecksumMismatchError,
    TensorMismatchError,
)
from swiftmev.ai.model_manifest import ModelManifest
from swiftmev.ai.preprocessing import (
    normalize_tensor_tile,
    generate_tile_coordinates,
    compute_taper_window,
)
from swiftmev.ai.postprocessing import extract_probability_tile, binarize_probability_map

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def compute_file_sha256(path: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo em disco."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


class ONNXSegmentationEngine(BaseSegmentationModel):
    """Segmentador baseado em ONNX Runtime otimizado para inferência de CPU."""

    def __init__(self, manifest: ModelManifest, model_dir: Path | str | None = None):
        super().__init__(manifest)
        if model_dir is not None:
            self.model_dir = Path(model_dir).resolve()
        elif manifest.manifest_path is not None:
            self.model_dir = manifest.manifest_path.parent.resolve()
        else:
            self.model_dir = Path(".").resolve()

        self.onnx_path = self.model_dir / manifest.onnx_filename
        self._session = None
        self._input_name = None
        self._output_name = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        """Carrega e valida o modelo ONNX sob demanda (Lazy Loading)."""
        if self._session is not None:
            return

        if ort is None:
            raise AINotAvailableError()

        if not self.onnx_path.exists():
            raise AIModelNotFoundError(self.manifest.model_id, str(self.onnx_path))

        # 1. Validação de Integridade Criptográfica (SHA-256)
        actual_sha256 = compute_file_sha256(self.onnx_path)
        if actual_sha256 != self.manifest.checksum_sha256.lower():
            raise ChecksumMismatchError(
                expected=self.manifest.checksum_sha256,
                actual=actual_sha256,
                file_path=str(self.onnx_path),
            )

        # 2. Configurações de Sessão Otimizadas para CPU
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 2
        sess_opts.inter_op_num_threads = 1
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise TensorMismatchError(f"Falha ao instanciar sessão ONNX para '{self.onnx_path}': {exc}")

        # 3. Validação dos Tensores Efetivos do Modelo vs. Manifesto
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise TensorMismatchError("O modelo ONNX não possui entradas ou saídas válidas.")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        # Validar canais e dimensões esperadas
        actual_shape = inputs[0].shape
        exp_shape = self.manifest.input_shape

        # Verifica canais (dimensão 1)
        if len(actual_shape) == 4 and isinstance(actual_shape[1], int) and actual_shape[1] != exp_shape[1]:
            raise TensorMismatchError(
                f"Divergência de canais no ONNX: modelo espera {actual_shape[1]} canais, "
                f"mas manifesto declara {exp_shape[1]}."
            )

    def unload(self) -> None:
        """Libera a sessão ONNX e descarrega os pesos da memória RAM."""
        self._session = None
        self._input_name = None
        self._output_name = None

    def predict_probability_map(self, image: np.ndarray) -> np.ndarray:
        """
        Gera mapa de probabilidade [0.0, 1.0] na resolução nativa da imagem (H, W)
        utilizando inferência por blocos sobrepostos (tiling).
        """
        if not self.is_loaded:
            self.load()

        img_h, img_w = image.shape[:2]
        tile_h = self.manifest.input_shape[2]
        tile_w = self.manifest.input_shape[3]
        target_channels = self.manifest.input_shape[1]
        overlap = self.manifest.tile_overlap

        # Caso especial: se a imagem for menor que as dimensões do bloco de entrada
        if img_h < tile_h or img_w < tile_w:
            pad_bottom = max(0, tile_h - img_h)
            pad_right = max(0, tile_w - img_w)
            padded = cv2.copyMakeBorder(
                image, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
            )
            tensor = normalize_tensor_tile(
                padded[:tile_h, :tile_w],
                target_channels=target_channels,
                normalization_config=self.manifest.normalization,
            )
            raw_out = self._session.run([self._output_name], {self._input_name: tensor})[0]
            prob_tile = extract_probability_tile(raw_out)
            return prob_tile[:img_h, :img_w].astype(np.float32)

        coords = generate_tile_coordinates(img_h, img_w, tile_h, tile_w, overlap)

        accumulator = np.zeros((img_h, img_w), dtype=np.float32)
        weight_map = np.zeros((img_h, img_w), dtype=np.float32)

        for (y0, y1, x0, x1, is_top, is_bottom, is_left, is_right) in coords:
            crop = image[y0:y1, x0:x1]

            # Caso haja ajuste de margem
            ch, cw = crop.shape[:2]
            if ch != tile_h or cw != tile_w:
                crop = cv2.copyMakeBorder(
                    crop, 0, tile_h - ch, 0, tile_w - cw, cv2.BORDER_REFLECT_101
                )

            tensor = normalize_tensor_tile(
                crop,
                target_channels=target_channels,
                normalization_config=self.manifest.normalization,
            )
            raw_out = self._session.run([self._output_name], {self._input_name: tensor})[0]
            prob_tile = extract_probability_tile(raw_out)[:ch, :cw]

            window = compute_taper_window(
                ch, cw, overlap, is_top, is_bottom, is_left, is_right
            )

            accumulator[y0:y1, x0:x1] += prob_tile * window
            weight_map[y0:y1, x0:x1] += window

        prob_map = accumulator / np.maximum(weight_map, 1e-6)
        return np.clip(prob_map, 0.0, 1.0).astype(np.float32)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Retorna máscara binária uint8 (0 ou 255) em resolução nativa."""
        prob_map = self.predict_probability_map(image)
        return binarize_probability_map(prob_map, threshold=self.manifest.threshold)
