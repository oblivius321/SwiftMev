"""Tests for SwiftMev Dataset Studio (Alpha 0.6B).

Validates:
1. Pairwise disjoint split leakage detection (Train ∩ Val, Train ∩ Test, Val ∩ Test).
2. Cryptographic (SHA-256) and perceptual (dHash) duplicate detection across splits.
3. Missing file reference detection in split assignments.
4. Annotation 4-state lifecycle (draft → annotated → reviewed → approved) and ground truth integrity gates.
5. Quality Control Assistant alerts (tiny speckles, irregular roughness, extreme coverage).
6. Missing calibration isolation (allows segmentation/annotation, blocks physical measurements).
7. Release export safety: overwrite guard, directory traversal guard, manifest integrity hashes.
8. Reconciled inventory scan of real micrographs (85 TIFFs + 2 report files).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from swiftmev.studio.schema import (
    AnnotationRecord,
    AnnotationSourceType,
    AnnotationState,
    DatasetManifest,
    DatasetSplitRecord,
    MicrographRecord,
    QCAlert,
)
from swiftmev.studio.inventory import (
    compute_dhash,
    compute_sha256,
    export_inventory_csv,
    hamming_distance,
    reconcile_inventory,
    scan_directory,
    scan_micrograph,
)
from swiftmev.studio.curation import (
    partition_groups_disjoint,
    validate_splits,
)
from swiftmev.studio.annotations import (
    AnnotationWorkflowError,
    create_or_update_annotation,
    inspect_annotation_quality,
    save_annotation_bundle,
)
from swiftmev.studio.exporter import (
    ExportSecurityError,
    ExportValidationError,
    ReleaseExistsError,
    export_dataset_release,
    safe_resolve_path,
)
from swiftmev.validation import CalibrationError


@pytest.fixture
def sample_micrographs(tmp_path) -> list[MicrographRecord]:
    """Generates a list of synthetic MicrographRecord objects with known hashes and groups."""
    recs = []
    for i in range(10):
        img = np.zeros((100, 100), dtype=np.uint8)
        # Draw distinct geometric features so each has a unique perceptual dHash
        cv2.circle(img, (20 + (i * 7) % 60, 20 + (i * 9) % 60), 10 + (i % 5), 200, -1)
        cv2.line(img, (0, i * 10), (100, (i * 11) % 100), 150, 2)
        p = tmp_path / f"img_{i:02d}.tif"
        cv2.imwrite(str(p), img)
        rec = scan_micrograph(p, base_dir=tmp_path)
        # Assign conservative groups
        rec.conservative_group_id = f"GRP_{i % 3}"
        recs.append(rec)
    return recs


class TestLeakageAndPairwiseValidation:

    def test_pairwise_disjointness_train_val(self, sample_micrographs):
        """Train ∩ Val ≠ ∅ deve falhar a validação de vazamento."""
        train_ids = [m.micrograph_id for m in sample_micrographs[:5]]
        val_ids = [sample_micrographs[4].micrograph_id, sample_micrographs[5].micrograph_id]  # sample 4 shared
        test_ids = [sample_micrographs[6].micrograph_id]

        res = validate_splits(sample_micrographs, train_ids, val_ids, test_ids)
        assert res.automated_leakage_checks_passed is False
        assert res.pairwise_disjoint_verified is False
        assert any("Train ∩ Val ≠ ∅" in msg for msg in res.validation_messages)

    def test_pairwise_disjointness_train_test(self, sample_micrographs):
        """Train ∩ Test ≠ ∅ deve falhar a validação de vazamento."""
        train_ids = [sample_micrographs[0].micrograph_id, sample_micrographs[1].micrograph_id]
        val_ids = [sample_micrographs[2].micrograph_id]
        test_ids = [sample_micrographs[1].micrograph_id, sample_micrographs[3].micrograph_id]  # sample 1 shared

        res = validate_splits(sample_micrographs, train_ids, val_ids, test_ids)
        assert res.automated_leakage_checks_passed is False
        assert res.pairwise_disjoint_verified is False
        assert any("Train ∩ Test ≠ ∅" in msg for msg in res.validation_messages)

    def test_pairwise_disjointness_val_test(self, sample_micrographs):
        """Val ∩ Test ≠ ∅ deve falhar mesmo se Train não compartilhar nada (Train ∩ Val ∩ Test = ∅)."""
        train_ids = [sample_micrographs[0].micrograph_id]
        val_ids = [sample_micrographs[1].micrograph_id, sample_micrographs[2].micrograph_id]
        test_ids = [sample_micrographs[2].micrograph_id, sample_micrographs[3].micrograph_id]  # sample 2 shared

        res = validate_splits(sample_micrographs, train_ids, val_ids, test_ids)
        assert res.automated_leakage_checks_passed is False
        assert res.pairwise_disjoint_verified is False
        assert any("Val ∩ Test ≠ ∅" in msg for msg in res.validation_messages)

    def test_sha256_duplicate_across_splits_blocked(self, tmp_path):
        """Imagens com mesmo hash SHA-256 em splits diferentes devem ser bloqueadas."""
        img_data = np.full((64, 64), 100, dtype=np.uint8)
        p1 = tmp_path / "copy_a.tif"
        p2 = tmp_path / "copy_b.tif"
        cv2.imwrite(str(p1), img_data)
        cv2.imwrite(str(p2), img_data)

        rec1 = scan_micrograph(p1, base_dir=tmp_path)
        rec2 = scan_micrograph(p2, base_dir=tmp_path)
        assert rec1.sha256 == rec2.sha256

        res = validate_splits([rec1, rec2], train_ids=[rec1.micrograph_id], val_ids=[rec2.micrograph_id], test_ids=[])
        assert res.automated_leakage_checks_passed is False
        assert res.hash_duplicates_verified is False
        assert any("VAZAMENTO POR HASH SHA-256" in msg for msg in res.validation_messages)

    def test_dhash_near_duplicate_across_splits_blocked(self, tmp_path):
        """Imagens perceptualmente quase idênticas (dHash <= 2) em splits diferentes devem ser bloqueadas."""
        # Cria imagem base e versão com 1 pixel de diferença (diferença imperceptível)
        img1 = np.full((64, 64), 120, dtype=np.uint8)
        img2 = img1.copy()
        img2[0, 0] = 121

        p1 = tmp_path / "perceptual_a.tif"
        p2 = tmp_path / "perceptual_b.tif"
        cv2.imwrite(str(p1), img1)
        cv2.imwrite(str(p2), img2)

        rec1 = scan_micrograph(p1, base_dir=tmp_path)
        rec2 = scan_micrograph(p2, base_dir=tmp_path)
        dist = hamming_distance(rec1.dhash, rec2.dhash)
        assert dist <= 2

        res = validate_splits([rec1, rec2], train_ids=[rec1.micrograph_id], val_ids=[rec2.micrograph_id], test_ids=[])
        assert res.automated_leakage_checks_passed is False
        assert res.hash_duplicates_verified is False
        assert any("VAZAMENTO PERCEPTUAL dHash" in msg for msg in res.validation_messages)

    def test_group_contamination_across_splits_blocked(self, sample_micrographs):
        """Amostras do mesmo grupo experimental não podem atravessar splits."""
        # Força mesmo grupo em duas imagens
        sample_micrographs[0].conservative_group_id = "GRP_CRITICAL"
        sample_micrographs[1].conservative_group_id = "GRP_CRITICAL"

        res = validate_splits(
            sample_micrographs,
            train_ids=[sample_micrographs[0].micrograph_id],
            val_ids=[sample_micrographs[1].micrograph_id],
            test_ids=[],
            strict_group_isolation=True,
        )
        assert res.automated_leakage_checks_passed is False
        assert res.group_separation_verified is False
        assert any("CONTAMINAÇÃO DE GRUPO EXPERIMENTAL" in msg for msg in res.validation_messages)

    def test_missing_ids_in_split_detected(self, sample_micrographs):
        """Identificadores no split que não existem no inventário geram erro de integridade."""
        res = validate_splits(sample_micrographs, train_ids=["ID_INEXISTENTE_999"], val_ids=[], test_ids=[])
        assert res.automated_leakage_checks_passed is False
        assert any("ERRO DE INTEGRIDADE" in msg for msg in res.validation_messages)

    def test_partition_groups_disjoint_guarantees_pairwise_isolation(self, sample_micrographs):
        """O particionador automático garante isolamento de grupos e splits disjuntos aos pares."""
        split = partition_groups_disjoint(sample_micrographs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert split.automated_leakage_checks_passed is True
        assert split.pairwise_disjoint_verified is True
        assert split.hash_duplicates_verified is True
        assert split.group_separation_verified is True


class TestAnnotationWorkflowAndQC:

    def test_automatic_mask_cannot_be_approved_directly(self):
        """Máscaras automáticas preliminares (Otsu/Sauvola/AI) não podem ser salvas diretamente como Ground Truth Aprovado."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(AnnotationWorkflowError, match="não pode ser aprovada diretamente"):
            create_or_update_annotation(
                micrograph_id="MG_001",
                mask=mask,
                target_state=AnnotationState.APPROVED,
                author="AutoEngine",
                source_type=AnnotationSourceType.PRELIMINARY_AUTO_OTSU,
            )

    def test_annotation_lifecycle_transitions(self):
        """A anotação transita rigorosamente: draft → annotated → reviewed → approved com histórico."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 20:40] = 255  # Alguma área anotada

        # 1. Draft
        ann, _ = create_or_update_annotation(
            micrograph_id="MG_001",
            mask=mask,
            target_state=AnnotationState.DRAFT,
            author="Pesquisador A",
            source_type=AnnotationSourceType.MANUAL,
        )
        assert ann.state == AnnotationState.DRAFT
        assert ann.version == 1
        assert len(ann.history) == 1

        # 2. Annotated
        ann, _ = create_or_update_annotation(
            micrograph_id="MG_001",
            mask=mask,
            target_state=AnnotationState.ANNOTATED,
            author="Pesquisador A",
            existing_record=ann,
        )
        assert ann.state == AnnotationState.ANNOTATED
        assert ann.version == 2

        # 3. Reviewed
        ann, _ = create_or_update_annotation(
            micrograph_id="MG_001",
            mask=mask,
            target_state=AnnotationState.REVIEWED,
            author="Pesquisador A",
            reviewer="Dr. Revisor Científico",
            existing_record=ann,
        )
        assert ann.state == AnnotationState.REVIEWED
        assert ann.version == 3
        assert ann.reviewer == "Dr. Revisor Científico"

        # 4. Approved
        ann, _ = create_or_update_annotation(
            micrograph_id="MG_001",
            mask=mask,
            target_state=AnnotationState.APPROVED,
            author="Pesquisador A",
            reviewer="Dr. Revisor Científico",
            existing_record=ann,
        )
        assert ann.state == AnnotationState.APPROVED
        assert ann.version == 4
        assert len(ann.history) == 4

    def test_qc_assistant_detects_tiny_components(self):
        """O assistente de controle de qualidade detecta speckles/ruídos muito pequenos."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Componente principal
        mask[50:150, 50:150] = 255
        # 3 speckles minúsculos de 3 pixels cada
        mask[10, 10:13] = 255
        mask[20, 20:23] = 255
        mask[30, 30:33] = 255

        alerts = inspect_annotation_quality(mask, min_speckle_area=15)
        codes = [a.code for a in alerts]
        assert "TINY_COMPONENTS" in codes
        tiny_alert = next(a for a in alerts if a.code == "TINY_COMPONENTS")
        assert tiny_alert.count == 3

    def test_qc_assistant_detects_empty_and_extreme_coverage(self):
        """QC detecta máscaras vazias ou com densidade anormal de cobertura."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        alerts_empty = inspect_annotation_quality(empty_mask)
        assert any(a.code == "EMPTY_MASK" for a in alerts_empty)

        full_mask = np.full((100, 100), 255, dtype=np.uint8)
        alerts_full = inspect_annotation_quality(full_mask)
        assert any(a.code == "SUSPICIOUSLY_HIGH_COVERAGE" for a in alerts_full)


class TestCalibrationIsolation:

    def test_uncalibrated_micrograph_blocks_physical_metrics(self):
        """Micrografias sem calibração documentada devem barrar medidas em micrômetros com CalibrationError."""
        from swiftmev.measurements import measure_fibers

        synthetic_mask = np.zeros((100, 100), dtype=np.uint8)
        synthetic_mask[40:60, :] = 255

        # Sem calibração (None ou <= 0) deve levantar CalibrationError
        with pytest.raises(CalibrationError):
            measure_fibers(synthetic_mask, um_per_pixel=None)

        with pytest.raises(CalibrationError):
            measure_fibers(synthetic_mask, um_per_pixel=0.0)

    def test_uncalibrated_micrograph_allows_annotation_and_segmentation(self, tmp_path):
        """Micrografias sem calibração podem ser anotadas, segmentadas e salvas em pixels."""
        img = np.full((100, 100), 50, dtype=np.uint8)
        p = tmp_path / "uncalibrated.tif"
        cv2.imwrite(str(p), img)

        rec = scan_micrograph(p)
        assert rec.is_calibrated is False
        assert rec.um_per_pixel is None

        # Criação de anotação funcional
        mask = (img > 30).astype(np.uint8) * 255
        ann, clean_m = create_or_update_annotation(
            micrograph_id=rec.micrograph_id,
            mask=mask,
            target_state=AnnotationState.DRAFT,
            author="Pesquisador",
        )
        assert ann.state == AnnotationState.DRAFT
        assert clean_m.shape == (100, 100)


class TestExporterAndSecurity:

    def test_path_traversal_blocked(self, tmp_path):
        """Tentativa de path traversal deve levantar ExportSecurityError."""
        with pytest.raises(ExportSecurityError):
            safe_resolve_path(tmp_path, "../../etc/passwd")

    def test_silent_release_overwrite_blocked(self, tmp_path, sample_micrographs):
        """Tentativa de sobrescrever release existente sem overwrite=True deve levantar ReleaseExistsError."""
        split = partition_groups_disjoint(sample_micrographs)
        export_dataset_release(
            destination_dir=tmp_path,
            dataset_name="test_ds",
            dataset_version="1.0.0",
            micrographs=sample_micrographs,
            split_record=split,
            source_root=tmp_path,
        )

        # Segunda tentativa sem overwrite
        with pytest.raises(ReleaseExistsError):
            export_dataset_release(
                destination_dir=tmp_path,
                dataset_name="test_ds",
                dataset_version="1.0.0",
                micrographs=sample_micrographs,
                split_record=split,
                source_root=tmp_path,
                overwrite=False,
            )

    def test_unreviewed_annotations_blocked_when_required(self, tmp_path, sample_micrographs):
        """Exportação com require_reviewed_annotations=True deve barrar anotações em draft."""
        split = partition_groups_disjoint(sample_micrographs)
        ann_map = {
            sample_micrographs[0].micrograph_id: AnnotationRecord(
                annotation_id="ANN_0",
                micrograph_id=sample_micrographs[0].micrograph_id,
                mask_filename="mask.png",
                state=AnnotationState.DRAFT,  # Apenas rascunho
            )
        }

        with pytest.raises(ExportValidationError, match="anotações não revisadas"):
            export_dataset_release(
                destination_dir=tmp_path,
                dataset_name="test_ds",
                dataset_version="1.0.0",
                micrographs=sample_micrographs,
                split_record=split,
                annotations_map=ann_map,
                source_root=tmp_path,
                require_reviewed_annotations=True,
            )

    def test_successful_export_generates_manifest_and_hashes(self, tmp_path, sample_micrographs):
        """Exportação autorizada gera manifest.json com hashes SHA-256 e splits intactos."""
        split = partition_groups_disjoint(sample_micrographs)
        manifest = export_dataset_release(
            destination_dir=tmp_path,
            dataset_name="alpha_export",
            dataset_version="0.6.0",
            micrographs=sample_micrographs,
            split_record=split,
            source_root=tmp_path,
        )

        assert manifest.dataset_name == "alpha_export"
        assert manifest.automated_leakage_checks_passed is True
        assert len(manifest.integrity_hashes) > 0

        release_folder = tmp_path / "alpha_export_v0.6.0"
        assert (release_folder / "manifest.json").exists()
        assert (release_folder / "inventory.csv").exists()
        assert (release_folder / "splits" / "train.txt").exists()


class TestInventoryReconciliation:

    def test_reconcile_real_fotos_inventory(self):
        """Reconcilia o inventário real de fotos/: exatamente 85 TIFFs + 2 arquivos auxiliares de relatório = 87 arquivos."""
        fotos_dir = Path("fotos")
        if not fotos_dir.exists():
            pytest.skip("Diretório fotos/ não encontrado no ambiente de execução.")

        summary = reconcile_inventory(fotos_dir)
        assert summary["total_files"] == 87, f"Total inesperado de arquivos: {summary['total_files']}"
        assert summary["total_micrographs"] == 85, f"Total inesperado de micrografias: {summary['total_micrographs']}"
        assert summary["total_auxiliary_files"] == 2, f"Total inesperado de arquivos de relatório: {summary['total_auxiliary_files']}"

        # Verifica detecção das 4 duplicatas exatas entre 5_\40-60 e 10_\40-60
        assert summary["exact_duplicate_groups_count"] == 4
