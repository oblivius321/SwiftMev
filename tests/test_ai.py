"""
SwiftMev AI — Suíte de Testes Automatizados da Infraestrutura de IA (Alpha 0.6A)

Validação rigorosa de:
  - Descoberta e carregamento sob demanda
  - Modelos ausentes, manifestos inválidos e corrupção de hash SHA-256
  - Incompatibilidade de tarefas científicas (bloqueio de IA para migração celular)
  - Inferência por blocos sobrepostos (tiling) em resoluções arbitrárias
  - Preservação geométrica sem distorções anamórficas
  - Compatibilidade estrita da máscara produzida com measurements.py
  - Independência e não-regressão dos métodos clássicos (Otsu e Sauvola)
  - Limites de memória e latência em CPU
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from swiftmev.ai import (
    is_ai_available,
    list_available_models,
    get_model,
    ModelManifest,
    AIModelNotFoundError,
    ModelManifestError,
    ChecksumMismatchError,
    TaskIncompatibilityError,
    TensorMismatchError,
)
from swiftmev.pipeline import run_fiber, run_migration
from swiftmev.segmentation import segment
from swiftmev.measurements import measure_fibers

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable
_ENV = {**os.environ, "PYTHONPATH": str(_ROOT / "engine")}


@pytest.fixture
def synthetic_micrograph():
    """Gera micrografia sintética com fibras horizontais simuladas (400x600)."""
    img = np.full((400, 600), 40, dtype=np.uint8)
    # Fibras com contraste claro sobre fundo escuro
    cv2.line(img, (50, 100), (550, 100), 210, thickness=8)
    cv2.line(img, (50, 250), (550, 250), 220, thickness=12)
    return img


class TestAIInfrastructure:

    def test_ai_availability(self):
        """Verifica se o subsistema detecta corretamente a presença de onnxruntime."""
        assert is_ai_available() is True

    def test_list_models_contains_test_model(self):
        """Verifica se o catálogo descobre o modelo de teste na pasta models/."""
        models = list_available_models()
        assert len(models) >= 1
        ids = [m["model_id"] for m in models]
        assert "test_fiber_onnx_mock" in ids

    def test_missing_model_raises(self):
        """Garante que solicitar modelo inexistente gera AIModelNotFoundError."""
        with pytest.raises(AIModelNotFoundError):
            get_model("modelo_inexistente_xyz_123")

    def test_invalid_manifest_missing_required_keys(self, tmp_path):
        """Garante que manifesto sem campos obrigatórios seja rejeitado."""
        bad_manifest = tmp_path / "manifest.json"
        bad_manifest.write_text(json.dumps({"model_id": "bad"}), encoding="utf-8")
        with pytest.raises(ModelManifestError):
            ModelManifest.load(bad_manifest)

    def test_invalid_manifest_corrupted_json(self, tmp_path):
        """Garante que JSON corrompido seja rejeitado com ModelManifestError."""
        bad_manifest = tmp_path / "manifest.json"
        bad_manifest.write_text("{ chave_sem_fechar:", encoding="utf-8")
        with pytest.raises(ModelManifestError):
            ModelManifest.load(bad_manifest)

    def test_checksum_mismatch_raises(self, tmp_path):
        """Garante que alteração em arquivo ONNX seja detectada por divergência de SHA-256."""
        # Cria arquivo ONNX falso
        fake_onnx = tmp_path / "model.onnx"
        fake_onnx.write_bytes(b"conteudo_falso_tampered")

        manifest_data = {
            "model_id": "tampered_model",
            "name": "Tampered Model Test",
            "task": "fiber_segmentation",
            "architecture": "Mock",
            "model_version": "0.1.0",
            "input_shape": [1, 1, 256, 256],
            "input_dtype": "float32",
            "normalization": {"type": "min_max", "scale": 1.0 / 255.0},
            "output_type": "probability_map",
            "threshold": 0.5,
            "license": "MIT",
            "provenance": "Mock test",
            "scientific_limitations": "None",
            "checksum_sha256": "0" * 64,  # Hash deliberadamente divergente
            "onnx_filename": "model.onnx",
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        m = ModelManifest.load(manifest_path)
        from swiftmev.ai.onnx_engine import ONNXSegmentationEngine
        engine = ONNXSegmentationEngine(m, model_dir=tmp_path)
        with pytest.raises(ChecksumMismatchError):
            engine.load()

    def test_task_incompatibility_raises(self):
        """Garante que modelo de fibras seja rejeitado para tarefas de migração/células."""
        with pytest.raises(TaskIncompatibilityError):
            get_model("test_fiber_onnx_mock", expected_task="migration_segmentation")

    def test_pipeline_migration_blocks_ai(self, synthetic_micrograph):
        """Garante que run_migration bloqueie o uso do método 'ai' na Alpha 0.6A."""
        with pytest.raises(TaskIncompatibilityError):
            run_migration(synthetic_micrograph, um_per_pixel=0.5, method="ai")


class TestTilingAndReconstruction:

    @pytest.mark.parametrize("h,w", [
        (128, 128),   # Menor que o bloco padrão (256)
        (256, 256),   # Exatamente 1 bloco
        (350, 570),   # Resolução arbitrária não-potência de 2
        (1024, 768),  # Micrografia panorâmica retangular
    ])
    def test_arbitrary_resolutions_exact_shape(self, h, w):
        """Garante que imagens de qualquer dimensão preservem o tamanho original exato."""
        ai_model = get_model("test_fiber_onnx_mock", expected_task="fiber_segmentation")
        test_img = np.full((h, w), 50, dtype=np.uint8)
        mask = ai_model.predict(test_img)

        assert mask.shape == (h, w), f"Divergência de formato: esperado ({h}, {w}), obtido {mask.shape}"
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_tile_reconstruction_no_geometric_distortion(self):
        """Garante que a posição das fibras não sofra desvio durante a fusão por blocos."""
        ai_model = get_model("test_fiber_onnx_mock", expected_task="fiber_segmentation")
        # Imagem com linha horizontal contínua cruzando múltiplos blocos
        h, w = 512, 512
        img = np.full((h, w), 30, dtype=np.uint8)
        line_y = 250
        cv2.line(img, (20, line_y), (490, line_y), 230, thickness=1)

        prob_map = ai_model.predict_probability_map(img)
        assert prob_map.shape == (h, w)
        # O pico de probabilidade deve estar exatamente centrado na linha y=250
        col_slice = prob_map[:, 256]
        max_y = int(np.argmax(col_slice))
        assert abs(max_y - line_y) <= 1, f"Desvio geométrico no eixo Y: max_y={max_y}, esperado={line_y}"


class TestIntegrationWithMeasurements:

    def test_mask_compatibility_with_measure_fibers(self, synthetic_micrograph):
        """Verifica se a máscara da IA se integra perfeitamente ao cálculo de diâmetros."""
        metrics, mask, annotated = run_fiber(
            synthetic_micrograph, um_per_pixel=0.25, method="ai", model_id="test_fiber_onnx_mock"
        )
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
        assert "mean_diameter_um" in metrics
        assert "ai_model" in metrics
        assert metrics["ai_model"]["model_id"] == "test_fiber_onnx_mock"
        assert metrics["ai_model"]["status"] == "experimental_unvalidated"
        assert annotated.shape == (synthetic_micrograph.shape[0], synthetic_micrograph.shape[1], 3)

    def test_classical_methods_remain_fully_functional(self, synthetic_micrograph):
        """Garante regressão zero: Otsu e Sauvola continuam 100% operacionais."""
        m_otsu = segment(synthetic_micrograph, method="otsu")
        m_sauvola = segment(synthetic_micrograph, method="sauvola")

        assert m_otsu.shape == synthetic_micrograph.shape
        assert m_sauvola.shape == synthetic_micrograph.shape
        assert set(np.unique(m_otsu)).issubset({0, 255})
        assert set(np.unique(m_sauvola)).issubset({0, 255})


class TestCLIAndExitCodes:

    def test_cli_list_models_returns_zero(self):
        """Valida que 'python -m swiftmev.cli --list-models' retorna 0 e JSON válido."""
        cmd = [_PYTHON, "-m", "swiftmev.cli", "--list-models"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), env=_ENV)
        assert res.returncode == 0
        data = json.loads(res.stdout)
        assert data["ai_available"] is True
        assert any(m["model_id"] == "test_fiber_onnx_mock" for m in data["models"])

    def test_cli_ai_fiber_success(self, tmp_path, synthetic_micrograph):
        """Valida execução completa da CLI com --method ai."""
        img_path = tmp_path / "fiber.png"
        cv2.imwrite(str(img_path), synthetic_micrograph)
        out_dir = tmp_path / "out"

        cmd = [
            _PYTHON, "-m", "swiftmev.cli",
            "--module", "fiber",
            "--input", str(img_path),
            "--output", str(out_dir),
            "--um-per-pixel", "0.5",
            "--method", "ai",
            "--model-id", "test_fiber_onnx_mock",
            "--save-mask"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), env=_ENV)
        assert res.returncode == 0, f"Stderr: {res.stderr}"

        data = json.loads(res.stdout)
        assert data["method"] == "ai"
        assert "ai_model_info" in data
        assert Path(data["overlay"]).exists()
        assert Path(data["mask"]).exists()

    def test_cli_ai_migration_incompatible_exit_3(self, tmp_path, synthetic_micrograph):
        """Garante que solicitar IA para migração retorne Exit Code 3 e TASK_INCOMPATIBLE."""
        img_path = tmp_path / "sample.png"
        cv2.imwrite(str(img_path), synthetic_micrograph)
        out_dir = tmp_path / "out"

        cmd = [
            _PYTHON, "-m", "swiftmev.cli",
            "--module", "migration",
            "--input", str(img_path),
            "--output", str(out_dir),
            "--um-per-pixel", "0.5",
            "--method", "ai",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), env=_ENV)
        assert res.returncode == 3
        err = json.loads(res.stderr)
        assert err["status"] == "ai_error"
        assert err["error_code"] == "TASK_INCOMPATIBLE"

    def test_cli_missing_model_exit_3(self, tmp_path, synthetic_micrograph):
        """Garante que solicitar modelo ausente retorne Exit Code 3 e MODEL_NOT_FOUND."""
        img_path = tmp_path / "sample.png"
        cv2.imwrite(str(img_path), synthetic_micrograph)
        out_dir = tmp_path / "out"

        cmd = [
            _PYTHON, "-m", "swiftmev.cli",
            "--module", "fiber",
            "--input", str(img_path),
            "--output", str(out_dir),
            "--um-per-pixel", "0.5",
            "--method", "ai",
            "--model-id", "modelo_fantasma_nao_existe"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT), env=_ENV)
        assert res.returncode == 3
        err = json.loads(res.stderr)
        assert err["status"] == "ai_error"
        assert err["error_code"] == "MODEL_NOT_FOUND"


class TestPerformanceLimits:

    def test_memory_and_latency_on_large_micrograph(self):
        """Verifica que imagem 1024x1024 processe em CPU com baixa latência e consumo seguro de RAM."""
        import psutil
        proc = psutil.Process()
        rss_before_mb = proc.memory_info().rss / (1024 * 1024)

        large_img = np.full((1024, 1024), 60, dtype=np.uint8)
        ai_model = get_model("test_fiber_onnx_mock", expected_task="fiber_segmentation")

        t0 = time.perf_counter()
        mask = ai_model.predict(large_img)
        elapsed_s = time.perf_counter() - t0

        rss_after_mb = proc.memory_info().rss / (1024 * 1024)
        rss_delta_mb = rss_after_mb - rss_before_mb

        assert mask.shape == (1024, 1024)
        # Latência deve ser inferior a 3.0s em CPU para 1 Megapixel
        assert elapsed_s < 3.0, f"Latência excessiva: {elapsed_s:.2f}s"
        # Incremento de memória de processo deve ser bem abaixo do teto de 1 GB (idealmente < 250 MB)
        assert rss_delta_mb < 250.0, f"Consumo de memória elevado: {rss_delta_mb:.1f} MB"
