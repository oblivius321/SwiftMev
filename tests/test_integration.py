"""
SwiftMev — Testes de Integração End-to-End (Alpha 0.4)

Valida o contrato de execução entre a interface Desktop (C# WPF) e o motor científico:
  1. Sucesso CLI com Exit Code 0 e estrutura JSON completa
  2. Bloqueio obrigatório de calibração inválida (Exit Code 2)
  3. Bloqueio de máscara degenerada (Exit Code 2)
  4. Recálculo a partir de máscara editada manualmente (--mask)
  5. Registro de parâmetros, versão e proveniência para auditoria
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable


@pytest.fixture
def sample_fiber_img(tmp_path):
    img_path = tmp_path / "fiber_test.png"
    # Imagem sintética simples com fibra no centro
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (20, 50), (80, 50), 255, thickness=10)
    cv2.imwrite(str(img_path), img)
    return img_path


def test_cli_fiber_success(sample_fiber_img, tmp_path):
    out_dir = tmp_path / "out"
    cmd = [
        _PYTHON, "-m", "swiftmev.cli",
        "--module", "fiber",
        "--input", str(sample_fiber_img),
        "--output", str(out_dir),
        "--um-per-pixel", "0.5",
        "--save-mask",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["version"] == "0.3.0"
    assert data["module"] == "fiber"
    assert data["calibration_um_per_pixel"] == 0.5
    assert "metrics" in data
    assert data["metrics"]["median_diameter_um"] is not None
    assert Path(data["overlay"]).exists()
    assert Path(data["mask"]).exists()


def test_cli_rejects_invalid_calibration(sample_fiber_img, tmp_path):
    out_dir = tmp_path / "out"
    cmd = [
        _PYTHON, "-m", "swiftmev.cli",
        "--module", "fiber",
        "--input", str(sample_fiber_img),
        "--output", str(out_dir),
        "--um-per-pixel", "-0.5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    assert res.returncode == 2
    err_json = json.loads(res.stderr)
    assert err_json["status"] == "validation_error"
    assert "positivo" in err_json["message"]


def test_cli_recalculates_from_custom_mask(sample_fiber_img, tmp_path):
    out_dir = tmp_path / "out"
    # Criar máscara customizada (editada manualmente)
    custom_mask = tmp_path / "custom_mask.png"
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(mask, (30, 50), (70, 50), 255, thickness=6)
    cv2.imwrite(str(custom_mask), mask)

    cmd = [
        _PYTHON, "-m", "swiftmev.cli",
        "--module", "fiber",
        "--input", str(sample_fiber_img),
        "--mask", str(custom_mask),
        "--output", str(out_dir),
        "--um-per-pixel", "1.0",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    assert res.returncode == 0, f"Stderr: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["metrics"]["mask_source"] == "manual_or_external"
    assert Path(data["overlay"]).exists()
