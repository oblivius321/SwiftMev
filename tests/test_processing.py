"""
SwiftMev — Testes Unitários (Alpha 0.3)

Cobertura:
  1. Validação obrigatória de um_per_pixel (bloqueio antes de qualquer medição)
  2. Validação de máscara degenerada
  3. Imaging: conversão de tipos (uint8, uint16, float32)
  4. Imaging: pré-processamento com CLAHE
  5. Segmentação: Otsu e Sauvola
  6. Medições: migração e fibras a partir de máscara pré-computada
  7. Pipeline: fluxo completo sem duplicação
  8. Aviso de resolução (sem termo Nyquist)
"""
import math
import numpy as np
import pytest

from swiftmev.validation import (
    validate_calibration,
    validate_mask,
    check_resolution_adequacy,
    CalibrationError,
    DegenerateMaskError,
)
from swiftmev.imaging import (
    ensure_gray_uint8,
    ensure_gray_uint8_with_meta,
    preprocess,
)
from swiftmev.segmentation import segment
from swiftmev.measurements import measure_migration, measure_fibers, overlay_contours
from swiftmev.pipeline import run_migration, run_fiber, run_fiber_from_mask


# ────────────────────────────────────────────────────────
# 1. Validação de calibração (bloqueio obrigatório)
# ────────────────────────────────────────────────────────

class TestCalibrationValidation:
    def test_valid_calibration(self):
        """Valores positivos e finitos devem passar sem erro."""
        validate_calibration(1.0)
        validate_calibration(0.001)
        validate_calibration(100.0)

    def test_zero_raises(self):
        with pytest.raises(CalibrationError, match="positivo"):
            validate_calibration(0.0)

    def test_negative_raises(self):
        with pytest.raises(CalibrationError, match="positivo"):
            validate_calibration(-1.0)

    def test_nan_raises(self):
        with pytest.raises(CalibrationError, match="finito"):
            validate_calibration(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(CalibrationError, match="finito"):
            validate_calibration(float("inf"))

    def test_negative_inf_raises(self):
        with pytest.raises(CalibrationError, match="finito"):
            validate_calibration(float("-inf"))

    def test_none_raises(self):
        with pytest.raises(CalibrationError, match="não informado"):
            validate_calibration(None)

    def test_string_raises(self):
        with pytest.raises(CalibrationError, match="numérico"):
            validate_calibration("1.0")


# ────────────────────────────────────────────────────────
# 2. Validação de máscara degenerada
# ────────────────────────────────────────────────────────

class TestMaskValidation:
    def test_valid_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        validate_mask(mask)  # Não deve levantar exceção

    def test_all_black_raises(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(DegenerateMaskError, match="background"):
            validate_mask(mask)

    def test_all_white_raises(self):
        mask = np.full((50, 50), 255, dtype=np.uint8)
        with pytest.raises(DegenerateMaskError, match="foreground"):
            validate_mask(mask)

    def test_empty_raises(self):
        mask = np.array([], dtype=np.uint8)
        with pytest.raises(DegenerateMaskError, match="vazia"):
            validate_mask(mask)

    def test_none_raises(self):
        with pytest.raises(DegenerateMaskError, match="vazia"):
            validate_mask(None)


# ────────────────────────────────────────────────────────
# 3. Imaging: conversão de tipos
# ────────────────────────────────────────────────────────

class TestImaging:
    def test_bgr_to_gray(self):
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        bgr[:, :, 0] = 100
        gray = ensure_gray_uint8(bgr)
        assert gray.shape == (50, 50)
        assert gray.dtype == np.uint8

    def test_uint16_to_uint8(self):
        u16 = np.array([[0, 32768], [65535, 1000]], dtype=np.uint16)
        gray, meta = ensure_gray_uint8_with_meta(u16)
        assert gray.shape == (2, 2)
        assert gray.dtype == np.uint8
        assert gray.max() == 255
        assert gray.min() == 0
        assert meta["scaling_applied"] == "min_max_linear_16bit_to_8bit"

    def test_float32_to_uint8(self):
        f32 = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
        gray, meta = ensure_gray_uint8_with_meta(f32)
        assert gray.dtype == np.uint8
        assert gray.max() == 255
        assert meta["scaling_applied"] == "min_max_linear_float_to_8bit"

    def test_constant_image_zero(self):
        """Imagem constante deve produzir zeros (sem divisão por zero)."""
        const = np.full((10, 10), 42, dtype=np.uint16)
        gray, meta = ensure_gray_uint8_with_meta(const)
        assert np.all(gray == 0)
        assert meta["scaling_applied"] == "zero_constant"

    def test_preprocess_clahe_changes_histogram(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[20:40, 20:40] = 50
        out_no_clahe = preprocess(img, use_clahe=False)
        out_clahe = preprocess(img, use_clahe=True)
        assert out_no_clahe.shape == (100, 100)
        assert out_clahe.shape == (100, 100)
        assert not np.array_equal(out_no_clahe, out_clahe)


# ────────────────────────────────────────────────────────
# 4. Segmentação
# ────────────────────────────────────────────────────────

class TestSegmentation:
    def test_otsu_segments_square(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 200
        mask = segment(img, method="otsu")
        assert mask.shape == (100, 100)
        assert np.count_nonzero(mask) > 0

    def test_sauvola_segments_square(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[30:70, 30:70] = 200
        mask = segment(img, method="sauvola", sauvola_window=25, sauvola_k=0.2)
        assert mask.shape == (100, 100)
        assert np.count_nonzero(mask) > 0

    def test_invalid_method_raises(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="desconhecido"):
            segment(img, method="invalid_method")


# ────────────────────────────────────────────────────────
# 5. Medições com máscara pré-computada
# ────────────────────────────────────────────────────────

class TestMeasurements:
    def test_migration_from_mask(self):
        """Medição de migração a partir de máscara sem re-segmentar."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, 35:65] = 255
        metrics = measure_migration(mask, um_per_pixel=0.5)
        assert metrics["empty_area_px2"] == 100 * 30
        assert metrics["empty_area_fraction"] == pytest.approx(0.30, abs=0.01)
        assert metrics["empty_area_um2"] > 0

    def test_migration_rejects_invalid_calibration(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        with pytest.raises(CalibrationError):
            measure_migration(mask, um_per_pixel=0.0)
        with pytest.raises(CalibrationError):
            measure_migration(mask, um_per_pixel=float("nan"))

    def test_fiber_from_mask(self):
        """Medição de fibras a partir de máscara sem re-segmentar."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 10:90] = 255
        metrics = measure_fibers(mask, um_per_pixel=1.0)
        assert metrics["mean_diameter_px"] is not None
        assert metrics["std_diameter_px"] is not None
        assert metrics["mean_diameter_um"] is not None
        # Uma fibra de 10px de largura
        assert 8.0 < metrics["mean_diameter_px"] < 12.0

    def test_fiber_rejects_degenerate_mask(self):
        with pytest.raises(DegenerateMaskError):
            measure_fibers(np.zeros((50, 50), dtype=np.uint8), um_per_pixel=1.0)
        with pytest.raises(DegenerateMaskError):
            measure_fibers(np.full((50, 50), 255, dtype=np.uint8), um_per_pixel=1.0)

    def test_overlay_produces_bgr(self):
        gray = np.full((50, 50), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        rgb = overlay_contours(gray, mask)
        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8


# ────────────────────────────────────────────────────────
# 6. Pipeline unificado
# ────────────────────────────────────────────────────────

class TestPipeline:
    def test_migration_pipeline(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 35:65] = 250
        metrics, mask, annotated = run_migration(img, um_per_pixel=0.5, method="otsu")
        assert metrics["empty_area_px2"] > 0
        assert metrics["method"] == "otsu"
        assert mask.shape == (100, 100)
        assert annotated.shape == (100, 100, 3)

    def test_fiber_pipeline(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[45:55, 10:90] = 240
        metrics, mask, annotated = run_fiber(img, um_per_pixel=1.0, method="otsu")
        assert metrics["mean_diameter_px"] is not None
        assert metrics["method"] == "otsu"
        assert mask.shape == (100, 100)
        assert annotated.shape == (100, 100, 3)

    def test_fiber_from_mask_pipeline(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 10:90] = 255
        img = np.zeros((100, 100), dtype=np.uint8)
        img[45:55, 10:90] = 200
        metrics, annotated = run_fiber_from_mask(mask, um_per_pixel=1.0, img_for_overlay=img)
        assert metrics["mean_diameter_px"] is not None
        assert annotated is not None
        assert annotated.shape == (100, 100, 3)

    def test_pipeline_rejects_bad_calibration(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 200
        with pytest.raises(CalibrationError):
            run_migration(img, um_per_pixel=-1.0)
        with pytest.raises(CalibrationError):
            run_fiber(img, um_per_pixel=float("nan"))


# ────────────────────────────────────────────────────────
# 7. Aviso de resolução (sem Nyquist)
# ────────────────────────────────────────────────────────

class TestResolutionWarning:
    def test_thin_fiber_triggers_warning(self):
        result = check_resolution_adequacy(3.0, threshold_px=6.0)
        assert result is not None
        assert result["level"] == "warning"
        assert result["code"] == "LOW_PIXEL_COUNT"
        assert "Nyquist" not in result["message"]
        assert "discretização" in result["message"]

    def test_thick_fiber_no_warning(self):
        result = check_resolution_adequacy(10.0, threshold_px=6.0)
        assert result is None

    def test_exactly_at_threshold(self):
        result = check_resolution_adequacy(6.0, threshold_px=6.0)
        assert result is None

    def test_just_below_threshold(self):
        result = check_resolution_adequacy(5.99, threshold_px=6.0)
        assert result is not None
