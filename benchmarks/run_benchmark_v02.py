"""
SwiftMev — Benchmark Científico V0.2 (Rigor Metodológico para SoftwareX)
Validação de 7 eixos:
1. RSS e pico de memória do processo com psutil.
2. 5 aquecimentos e 30 repetições por cenário (média, std, mediana, min, max).
3. Coeficientes Dice, IoU e erro absoluto/relativo de área na migração.
4. Erro absoluto e relativo nas medidas de diâmetro de fibras.
5. Imagens sintéticas desafiadoras (ruído Gaussiano, salt & pepper, gradiente e contraste).
6. Exportação em CSV e JSON com metadados do ambiente de hardware/software.
7. Registro de imagens originais, máscaras Ground Truth e predições.
"""
from __future__ import annotations
import csv
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import psutil
from skimage.filters import threshold_otsu, threshold_sauvola
from cellscope.processing import migration, fiber, segment, ensure_gray_uint8


# ---------------------------------------------------------------------------
# 1. FUNÇÕES DE MÉTRICAS DE SEGMENTAÇÃO (DICE, IoU, ERROS)
# ---------------------------------------------------------------------------
def calculate_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    p = mask_pred > 0
    g = mask_gt > 0
    intersection = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def calculate_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    p = mask_pred > 0
    g = mask_gt > 0
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# 2. GERADORES DE IMAGENS SINTÉTICAS COM GROUND TRUTH EXATO
# ---------------------------------------------------------------------------
def create_synthetic_migration_datasets(dataset_dir: Path, size: int = 512) -> list[dict]:
    """
    Gera 4 desafios sintéticos de migração com ground truth exato:
    1. Ideal: fenda nítida
    2. Ruído: Gaussiano + Salt & Pepper
    3. Gradiente: Iluminação heterogênea (vignetting/gradiente linear)
    4. Baixo Contraste: Pequena diferença de intensidade entre células e fenda
    """
    np.random.seed(101)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datasets = []

    # Fenda central: colunas 35% até 65%
    gap_start = int(size * 0.35)
    gap_end = int(size * 0.65)
    gt_mask = np.zeros((size, size), dtype=np.uint8)
    gt_mask[:, gap_start:gap_end] = 255
    gt_mask_path = dataset_dir / "migration_gt_mask.png"
    cv2.imwrite(str(gt_mask_path), gt_mask)
    gt_area_px = int(np.count_nonzero(gt_mask))

    # Cenário 1: Ideal
    img_ideal = np.random.randint(140, 190, (size, size), dtype=np.uint8)
    img_ideal[:, gap_start:gap_end] = np.random.randint(230, 255, (size, gap_end - gap_start), dtype=np.uint8)
    p1 = dataset_dir / "migration_scenario_1_ideal.png"
    cv2.imwrite(str(p1), img_ideal)
    datasets.append({
        "scenario": "1_Ideal",
        "path": p1,
        "gt_mask": gt_mask,
        "gt_area_px": gt_area_px
    })

    # Cenário 2: Ruído (Gaussiano + Salt & Pepper)
    noise_gauss = np.random.normal(0, 25, (size, size)).astype(np.float32)
    img_noise = np.clip(img_ideal.astype(np.float32) + noise_gauss, 0, 255).astype(np.uint8)
    sp_mask = np.random.rand(size, size)
    img_noise[sp_mask < 0.02] = 0
    img_noise[sp_mask > 0.98] = 255
    p2 = dataset_dir / "migration_scenario_2_noise.png"
    cv2.imwrite(str(p2), img_noise)
    datasets.append({
        "scenario": "2_Noise_Gaussian_SP",
        "path": p2,
        "gt_mask": gt_mask,
        "gt_area_px": gt_area_px
    })

    # Cenário 3: Gradiente de Iluminação
    grad_y = np.linspace(0, 70, size, dtype=np.float32).reshape(size, 1)
    grad_x = np.linspace(0, 70, size, dtype=np.float32).reshape(1, size)
    illumination = grad_y + grad_x
    img_grad = np.clip(img_ideal.astype(np.float32) * 0.7 + illumination, 0, 255).astype(np.uint8)
    p3 = dataset_dir / "migration_scenario_3_gradient.png"
    cv2.imwrite(str(p3), img_grad)
    datasets.append({
        "scenario": "3_Illumination_Gradient",
        "path": p3,
        "gt_mask": gt_mask,
        "gt_area_px": gt_area_px
    })

    # Cenário 4: Baixo Contraste
    img_low = np.full((size, size), 160, dtype=np.uint8)
    img_low += np.random.randint(-10, 10, (size, size), dtype=np.int8).astype(np.uint8)
    img_low[:, gap_start:gap_end] = 190 + np.random.randint(-10, 10, (size, gap_end - gap_start), dtype=np.int8).astype(np.uint8)
    p4 = dataset_dir / "migration_scenario_4_low_contrast.png"
    cv2.imwrite(str(p4), img_low)
    datasets.append({
        "scenario": "4_Low_Contrast",
        "path": p4,
        "gt_mask": gt_mask,
        "gt_area_px": gt_area_px
    })

    return datasets


def create_synthetic_fiber_datasets(dataset_dir: Path, size: int = 512) -> list[dict]:
    """
    Gera 4 desafios sintéticos de fibras longitudinais com diâmetro ground truth calibrado:
    1. Fibras Finas: diâmetro = 6 px
    2. Fibras Médias: diâmetro = 16 px
    3. Fibras Grossas: diâmetro = 30 px
    4. Fibras Ruidosas com gradiente de fundo: diâmetro = 16 px
    """
    np.random.seed(202)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    datasets = []

    def make_fibers(diam: int, add_noise: bool = False, add_grad: bool = False):
        bg = np.random.randint(15, 30, (size, size), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        ys = [int(size * 0.25), int(size * 0.50), int(size * 0.75)]
        for y in ys:
            cv2.line(mask, (int(size * 0.05), y), (int(size * 0.95), y), 255, thickness=diam)
            cv2.line(bg, (int(size * 0.05), y), (int(size * 0.95), y), 235, thickness=diam)

        img = bg.astype(np.float32)
        if add_grad:
            grad = np.linspace(0, 60, size, dtype=np.float32).reshape(size, 1)
            img += grad
        if add_noise:
            noise = np.random.normal(0, 20, (size, size)).astype(np.float32)
            img += noise

        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, mask

    configs = [
        ("1_Thin_Fibers_6px", 6, False, False),
        ("2_Medium_Fibers_16px", 16, False, False),
        ("3_Thick_Fibers_30px", 30, False, False),
        ("4_Noisy_Fibers_16px", 16, True, True),
    ]

    for name, diam, noise, grad in configs:
        img, mask = make_fibers(diam, noise, grad)
        img_p = dataset_dir / f"fiber_{name}.png"
        mask_p = dataset_dir / f"fiber_{name}_gt_mask.png"
        cv2.imwrite(str(img_p), img)
        cv2.imwrite(str(mask_p), mask)
        datasets.append({
            "scenario": name,
            "path": img_p,
            "gt_mask": mask,
            "gt_diameter_px": float(diam)
        })

    return datasets


# ---------------------------------------------------------------------------
# 3. EXECUTOR DE BENCHMARK RIGOROSO (5 WARM-UPS + 30 REPETIÇÕES)
# ---------------------------------------------------------------------------
def run_benchmark_rigorous(
    func,
    img: np.ndarray,
    um_per_pixel: float,
    warmups: int = 5,
    repeats: int = 30,
    **kwargs
) -> dict:
    proc = psutil.Process()

    # Aquecimento (warm-up descartado)
    for _ in range(warmups):
        _ = func(img, um_per_pixel=um_per_pixel, **kwargs)

    wall_times = []
    cpu_times = []
    rss_readings = []

    metrics_result = None
    annotated_result = None

    gc.collect()
    rss_initial_mb = proc.memory_info().rss / (1024 * 1024)

    for _ in range(repeats):
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        metrics_result, annotated_result = func(img, um_per_pixel=um_per_pixel, **kwargs)

        t_wall_elapsed = (time.perf_counter() - t_wall_start) * 1000.0
        t_cpu_elapsed = (time.process_time() - t_cpu_start) * 1000.0

        wall_times.append(t_wall_elapsed)
        cpu_times.append(t_cpu_elapsed)
        rss_readings.append(proc.memory_info().rss / (1024 * 1024))

    rss_final_mb = proc.memory_info().rss / (1024 * 1024)
    rss_peak_mb = max(rss_readings) if rss_readings else rss_final_mb

    return {
        "wall_time_ms": {
            "mean": round(float(np.mean(wall_times)), 3),
            "std": round(float(np.std(wall_times)), 3),
            "median": round(float(np.median(wall_times)), 3),
            "min": round(float(np.min(wall_times)), 3),
            "max": round(float(np.max(wall_times)), 3),
        },
        "cpu_time_ms": {
            "mean": round(float(np.mean(cpu_times)), 3),
            "std": round(float(np.std(cpu_times)), 3),
            "median": round(float(np.median(cpu_times)), 3),
            "min": round(float(np.min(cpu_times)), 3),
            "max": round(float(np.max(cpu_times)), 3),
        },
        "memory_rss_mb": {
            "initial": round(rss_initial_mb, 2),
            "final": round(rss_final_mb, 2),
            "peak": round(rss_peak_mb, 2),
            "delta": round(rss_final_mb - rss_initial_mb, 2)
        },
        "metrics": metrics_result,
        "annotated": annotated_result
    }


# ---------------------------------------------------------------------------
# 4. SUÍTE PRINCIPAL DE BENCHMARK V0.2
# ---------------------------------------------------------------------------
def execute_benchmark_v02():
    dataset_dir = Path("datasets/benchmark_v02").resolve()
    out_dir = Path("outputs/benchmark_v02").resolve()
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("Gerando datasets sintéticos controlados...")
    migration_cases = create_synthetic_migration_datasets(dataset_dir)
    fiber_cases = create_synthetic_fiber_datasets(dataset_dir)

    pipelines = [
        ("Otsu", {"method": "otsu", "use_clahe": False}),
        ("Otsu_CLAHE", {"method": "otsu", "use_clahe": True}),
        ("Sauvola", {"method": "sauvola", "use_clahe": False}),
        ("Sauvola_CLAHE", {"method": "sauvola", "use_clahe": True}),
    ]

    env_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "version_tag": "0.2.0-alpha"
    }

    all_records = []
    csv_rows = []

    print("\n" + "=" * 90)
    print("SWIFTMEV — BENCHMARK V0.2: RIGOR METODOLÓGICO (5 WARM-UPS + 30 REPETIÇÕES)")
    print("=" * 90)

    # Execução para Migração Celular
    print("\n>>> Avaliando MÓDULO: MIGRAÇÃO CELULAR")
    um_per_pixel = 0.5
    for case in migration_cases:
        img = cv2.imread(str(case["path"]), cv2.IMREAD_GRAYSCALE)
        gt_mask = case["gt_mask"]
        gt_area = case["gt_area_px"]

        print(f"\n[Cenário: {case['scenario']}] (GT Área: {gt_area} px)")

        for pipe_name, kwargs in pipelines:
            bench = run_benchmark_rigorous(migration, img, um_per_pixel=um_per_pixel, **kwargs)

            # Extração da máscara binária predita
            pred_mask = segment(img, **kwargs)
            pred_mask_path = masks_dir / f"migration_{case['scenario']}_{pipe_name}_pred.png"
            cv2.imwrite(str(pred_mask_path), pred_mask)

            dice = round(calculate_dice(pred_mask, gt_mask), 4)
            iou = round(calculate_iou(pred_mask, gt_mask), 4)
            pred_area_px = int(np.count_nonzero(pred_mask))
            abs_area_err_px = abs(pred_area_px - gt_area)
            rel_area_err_pct = round((abs_area_err_px / gt_area) * 100.0, 2)

            record = {
                "module": "migration",
                "scenario": case["scenario"],
                "pipeline": pipe_name,
                "dice": dice,
                "iou": iou,
                "gt_area_px": gt_area,
                "pred_area_px": pred_area_px,
                "abs_area_error_px": abs_area_err_px,
                "rel_area_error_pct": rel_area_err_pct,
                "wall_time_ms_mean": bench["wall_time_ms"]["mean"],
                "wall_time_ms_std": bench["wall_time_ms"]["std"],
                "cpu_time_ms_mean": bench["cpu_time_ms"]["mean"],
                "cpu_time_ms_std": bench["cpu_time_ms"]["std"],
                "rss_peak_mb": bench["memory_rss_mb"]["peak"],
                "rss_delta_mb": bench["memory_rss_mb"]["delta"],
                "pred_mask_file": str(pred_mask_path.name)
            }
            all_records.append(record)
            csv_rows.append(record)

            print(f"  • {pipe_name:<14} | Dice: {dice:.4f} | IoU: {iou:.4f} | Err Rel: {rel_area_err_pct:>6.2f}% | Wall: {bench['wall_time_ms']['mean']:>6.2f} ± {bench['wall_time_ms']['std']:>4.2f} ms | RSS Pico: {bench['memory_rss_mb']['peak']:>5.1f} MB")

    # Execução para Diâmetro de Fibras
    print("\n>>> Avaliando MÓDULO: DIÂMETRO DE FIBRAS")
    for case in fiber_cases:
        img = cv2.imread(str(case["path"]), cv2.IMREAD_GRAYSCALE)
        gt_mask = case["gt_mask"]
        gt_diam_px = case["gt_diameter_px"]
        gt_diam_um = gt_diam_px * um_per_pixel

        print(f"\n[Cenário: {case['scenario']}] (GT Diâmetro: {gt_diam_px:.1f} px / {gt_diam_um:.2f} um)")

        for pipe_name, kwargs in pipelines:
            bench = run_benchmark_rigorous(fiber, img, um_per_pixel=um_per_pixel, **kwargs)

            pred_mask = segment(img, **kwargs)
            pred_mask_path = masks_dir / f"fiber_{case['scenario']}_{pipe_name}_pred.png"
            cv2.imwrite(str(pred_mask_path), pred_mask)

            dice = round(calculate_dice(pred_mask, gt_mask), 4)
            iou = round(calculate_iou(pred_mask, gt_mask), 4)

            mean_d_um = bench["metrics"]["mean_local_diameter_um"]
            if mean_d_um is not None:
                abs_diam_err_um = round(abs(mean_d_um - gt_diam_um), 3)
                rel_diam_err_pct = round((abs_diam_err_um / gt_diam_um) * 100.0, 2)
            else:
                abs_diam_err_um = None
                rel_diam_err_pct = None

            record = {
                "module": "fiber",
                "scenario": case["scenario"],
                "pipeline": pipe_name,
                "dice": dice,
                "iou": iou,
                "gt_diameter_um": round(gt_diam_um, 3),
                "pred_mean_diameter_um": mean_d_um,
                "pred_std_diameter_um": bench["metrics"]["std_local_diameter_um"],
                "abs_diam_error_um": abs_diam_err_um,
                "rel_diam_error_pct": rel_diam_err_pct,
                "wall_time_ms_mean": bench["wall_time_ms"]["mean"],
                "wall_time_ms_std": bench["wall_time_ms"]["std"],
                "cpu_time_ms_mean": bench["cpu_time_ms"]["mean"],
                "cpu_time_ms_std": bench["cpu_time_ms"]["std"],
                "rss_peak_mb": bench["memory_rss_mb"]["peak"],
                "rss_delta_mb": bench["memory_rss_mb"]["delta"],
                "pred_mask_file": str(pred_mask_path.name)
            }
            all_records.append(record)
            csv_rows.append(record)

            diam_str = f"{mean_d_um:.2f} um" if mean_d_um else "N/A"
            err_str = f"{rel_diam_err_pct:.1f}%" if rel_diam_err_pct is not None else "N/A"
            print(f"  • {pipe_name:<14} | Dice: {dice:.4f} | Diâm Pred: {diam_str:>8} | Err Rel: {err_str:>6} | Wall: {bench['wall_time_ms']['mean']:>6.2f} ± {bench['wall_time_ms']['std']:>4.2f} ms | RSS Pico: {bench['memory_rss_mb']['peak']:>5.1f} MB")

    # Gravação do JSON com ambiente
    json_output = {
        "environment": env_meta,
        "protocol": {
            "warmup_runs": 5,
            "timed_repeats": 30,
            "calibration_um_per_pixel": um_per_pixel,
        },
        "records": all_records
    }
    json_path = out_dir / "benchmark_v02_results.json"
    json_path.write_text(json.dumps(json_output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Gravação dos CSVs
    migration_rows = [r for r in csv_rows if r["module"] == "migration"]
    fiber_rows = [r for r in csv_rows if r["module"] == "fiber"]

    if migration_rows:
        mig_csv = out_dir / "benchmark_v02_migration.csv"
        with open(mig_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(migration_rows[0].keys()))
            w.writeheader()
            w.writerows(migration_rows)

    if fiber_rows:
        fib_csv = out_dir / "benchmark_v02_fiber.csv"
        with open(fib_csv, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fiber_rows[0].keys()))
            w.writeheader()
            w.writerows(fiber_rows)

    csv_path = out_dir / "benchmark_v02_summary.csv"
    if csv_rows:
        union_fieldnames = []
        for r in csv_rows:
            for k in r.keys():
                if k not in union_fieldnames:
                    union_fieldnames.append(k)
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=union_fieldnames, restval="")
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 90)
    print(f"Resultados de Benchmark V0.2 exportados com sucesso:")
    print(f"  JSON: {json_path}")
    print(f"  CSV Sumário:   {csv_path}")
    print(f"  CSV Migração:  {out_dir / 'benchmark_v02_migration.csv'}")
    print(f"  CSV Fibras:    {out_dir / 'benchmark_v02_fiber.csv'}")
    print(f"  Máscaras auditáveis salvas em: {masks_dir}")
    print("=" * 90)


if __name__ == "__main__":
    execute_benchmark_v02()
