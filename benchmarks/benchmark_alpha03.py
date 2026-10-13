"""
SwiftMev — Alpha 0.3: Benchmark Completo de Validação Geométrica

Diferenças em relação à Alpha 0.2:
  1. Usa o pipeline modular (segmentação executada 1× por amostra)
  2. Jitter estocástico real (centro, ângulo, comprimento)
  3. W_rast medida na região central da fibra
  4. Benchmark de migração celular (scratch assay)
  5. Compara D_est contra D_geom e W_rast_central separadamente
  6. Métricas de diâmetro em pixels (mean, median, std)

Configuração:
  - 6 Espessuras: 3, 6, 10, 16, 20, 30 px
  - 5 Orientações: 0°, 30°, 45°, 60°, 90°
  - 4 Condições: ideal, noise, gradient, low_contrast
  - 5 amostras independentes por configuração (com jitter)
  - 3 Pipelines: Otsu, Otsu+CLAHE, Sauvola
  - Monitoramento de RSS por amostragem contínua
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil

# Garante acesso aos módulos
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(_ROOT / "engine"))

from benchmarks.geometric_generator import generate_geometric_fiber, generate_scratch_assay
from benchmarks.rss_monitor import RSSSamplingMonitor
from swiftmev.segmentation import segment
from swiftmev.measurements import measure_fibers, measure_migration
from swiftmev.validation import DegenerateMaskError


# ─────────────────────────────────────────
# Métricas de validação
# ─────────────────────────────────────────

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    return float(2.0 * inter / total) if total > 0 else 1.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else 1.0


# ─────────────────────────────────────────
# Benchmark de Fibras
# ─────────────────────────────────────────

def run_fiber_benchmark(
    samples_per_config: int = 5,
    base_seed: int = 42,
    jitter_center: float = 3.0,
    jitter_angle_deg: float = 2.0,
    jitter_length: float = 5.0,
    save_masks: bool = True,
) -> dict:
    """
    Executa a validação geométrica completa de fibras com a arquitetura Alpha 0.3.
    """
    diameters = [3.0, 6.0, 10.0, 16.0, 20.0, 30.0]
    angles = [0.0, 30.0, 45.0, 60.0, 90.0]
    conditions = ["ideal", "noise", "gradient", "low_contrast"]
    pipelines = [
        ("Otsu", {"method": "otsu", "use_clahe": False}),
        ("Otsu_CLAHE", {"method": "otsu", "use_clahe": True}),
        ("Sauvola", {"method": "sauvola", "use_clahe": False}),
    ]

    out_dir = Path("outputs/alpha03_validation").resolve()
    masks_dir = out_dir / "masks"
    if save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total_configs = len(diameters) * len(angles) * len(conditions)

    print("=" * 100)
    print(f"SWIFTMEV ALPHA 0.3 — VALIDAÇÃO GEOMÉTRICA DE FIBRAS")
    print(f"Configurações: {total_configs} | Amostras/config: {samples_per_config}")
    print(f"Jitter: centro sigma={jitter_center}px, angulo sigma={jitter_angle_deg} deg, comprimento sigma={jitter_length}px")
    print("=" * 100)

    config_idx = 0
    for d_geom in diameters:
        for angle in angles:
            for cond in conditions:
                config_idx += 1
                config_label = f"D{int(d_geom)}_A{int(angle)}_{cond}"

                # Gerar lote com jitter estocástico
                batch = []
                for s_idx in range(samples_per_config):
                    seed = base_seed + (config_idx * 1000) + s_idx
                    img, gt_mask, meta = generate_geometric_fiber(
                        size=256,
                        nominal_diameter=d_geom,
                        angle_deg=angle,
                        length=180.0,
                        condition=cond,
                        seed=seed,
                        jitter_center=jitter_center,
                        jitter_angle_deg=jitter_angle_deg,
                        jitter_length=jitter_length,
                    )
                    batch.append((img, gt_mask, meta))

                # Avaliar por pipeline
                for pipe_name, pipe_kwargs in pipelines:
                    dice_list, iou_list = [], []
                    d_est_list = []
                    w_rast_list = []
                    err_geom_list, err_rast_list = [], []
                    wall_ms_list, cpu_ms_list = [], []
                    degenerate_count = 0

                    with RSSSamplingMonitor(interval_ms=2.0) as rss_mon:
                        for s_idx, (img, gt_mask, meta) in enumerate(batch):
                            t_wall = time.perf_counter()
                            t_cpu = time.process_time()

                            # 1. Segmentação (executada uma única vez)
                            pred_mask = segment(img, **pipe_kwargs)

                            # 2. Métricas de segmentação
                            dice_list.append(dice(pred_mask, gt_mask))
                            iou_list.append(iou(pred_mask, gt_mask))

                            # 3. Medição de diâmetro (reutiliza a máscara)
                            try:
                                fiber_metrics = measure_fibers(pred_mask, um_per_pixel=1.0)
                                d_est = fiber_metrics["median_diameter_px"]
                            except DegenerateMaskError:
                                d_est = None
                                degenerate_count += 1

                            wall_ms_list.append((time.perf_counter() - t_wall) * 1000)
                            cpu_ms_list.append((time.process_time() - t_cpu) * 1000)

                            w_rast = meta["raster_width_central_px"]
                            w_rast_list.append(w_rast)

                            if d_est is not None:
                                d_est_list.append(d_est)
                                err_geom_list.append(d_est - d_geom)
                                err_rast_list.append(d_est - w_rast)

                            # Salvar máscara da 1ª repetição
                            if save_masks and s_idx == 0:
                                cv2.imwrite(
                                    str(masks_dir / f"{config_label}_{pipe_name}_pred.png"),
                                    pred_mask,
                                )
                                gt_path = masks_dir / f"{config_label}_gt.png"
                                if not gt_path.exists():
                                    cv2.imwrite(str(gt_path), gt_mask)

                    rss = rss_mon.get_summary()
                    n = len(dice_list)
                    w_rast_mean = float(np.mean(w_rast_list))

                    entry = {
                        "config_id": config_idx,
                        "nominal_diameter_px": d_geom,
                        "angle_deg": angle,
                        "condition": cond,
                        "pipeline": pipe_name,
                        "n_samples": n,
                        "n_valid": len(d_est_list),
                        "n_degenerate": degenerate_count,
                        "w_rast_central_mean_px": round(w_rast_mean, 3),
                        "rasterization_bias_px": round(w_rast_mean - d_geom, 3),
                        "dice_mean": round(float(np.mean(dice_list)), 4),
                        "dice_std": round(float(np.std(dice_list, ddof=1)), 4) if n > 1 else 0.0,
                        "iou_mean": round(float(np.mean(iou_list)), 4),
                    }

                    if d_est_list:
                        arr_geom = np.array(err_geom_list)
                        arr_rast = np.array(err_rast_list)
                        entry.update({
                            "d_est_mean_px": round(float(np.mean(d_est_list)), 3),
                            "d_est_std_px": round(float(np.std(d_est_list, ddof=1)), 3) if len(d_est_list) > 1 else 0.0,
                            "bias_vs_geom_px": round(float(np.mean(arr_geom)), 3),
                            "mae_vs_geom_px": round(float(np.mean(np.abs(arr_geom))), 3),
                            "rmse_vs_geom_px": round(float(np.sqrt(np.mean(arr_geom ** 2))), 3),
                            "bias_vs_rast_px": round(float(np.mean(arr_rast)), 3),
                            "mae_vs_rast_px": round(float(np.mean(np.abs(arr_rast))), 3),
                            "rmse_vs_rast_px": round(float(np.sqrt(np.mean(arr_rast ** 2))), 3),
                        })
                    else:
                        entry.update({
                            "d_est_mean_px": None, "d_est_std_px": None,
                            "bias_vs_geom_px": None, "mae_vs_geom_px": None, "rmse_vs_geom_px": None,
                            "bias_vs_rast_px": None, "mae_vs_rast_px": None, "rmse_vs_rast_px": None,
                        })

                    entry.update({
                        "wall_time_mean_ms": round(float(np.mean(wall_ms_list)), 2),
                        "cpu_time_mean_ms": round(float(np.mean(cpu_ms_list)), 2),
                        "rss_peak_mb": rss["rss_peak_observed_mb"],
                        "rss_delta_mb": rss["rss_delta_mb"],
                    })
                    summary_rows.append(entry)

                if config_idx % 20 == 0 or config_idx == total_configs:
                    print(f"  Fibras: {config_idx:3d}/{total_configs} configurações concluídas")

    return _save_results(summary_rows, out_dir, "fiber", samples_per_config,
                         jitter_center, jitter_angle_deg, jitter_length)


# ─────────────────────────────────────────
# Benchmark de Migração
# ─────────────────────────────────────────

def run_migration_benchmark(
    samples_per_config: int = 5,
    base_seed: int = 42,
) -> dict:
    """Validação geométrica de scratch assay."""
    fractions = [0.15, 0.25, 0.30, 0.40, 0.50]
    conditions = ["ideal", "noise", "gradient", "low_contrast"]
    pipelines = [
        ("Otsu", {"method": "otsu", "use_clahe": False, "invert": True}),
        ("Otsu_CLAHE", {"method": "otsu", "use_clahe": True, "invert": True}),
        ("Sauvola", {"method": "sauvola", "use_clahe": False, "invert": True}),
    ]

    out_dir = Path("outputs/alpha03_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    total = len(fractions) * len(conditions)
    print("\n" + "=" * 100)
    print(f"SWIFTMEV ALPHA 0.3 — VALIDAÇÃO DE MIGRAÇÃO CELULAR (SCRATCH ASSAY)")
    print(f"Configurações: {total} | Amostras/config: {samples_per_config}")
    print("=" * 100)

    config_idx = 0
    for frac in fractions:
        for cond in conditions:
            config_idx += 1

            batch = []
            for s_idx in range(samples_per_config):
                seed = base_seed + 50000 + (config_idx * 1000) + s_idx
                img, gt_mask, meta = generate_scratch_assay(
                    size=(256, 256),
                    scratch_fraction=frac,
                    condition=cond,
                    seed=seed,
                    jitter_width=3.0,
                )
                batch.append((img, gt_mask, meta))

            for pipe_name, pipe_kwargs in pipelines:
                dice_list, iou_list = [], []
                frac_err_list = []
                wall_ms_list = []
                degenerate_count = 0

                with RSSSamplingMonitor(interval_ms=2.0) as rss_mon:
                    for img, gt_mask, meta in batch:
                        t0 = time.perf_counter()
                        pred_mask = segment(img, **pipe_kwargs)
                        dice_list.append(dice(pred_mask, gt_mask))
                        iou_list.append(iou(pred_mask, gt_mask))

                        try:
                            mig = measure_migration(pred_mask, um_per_pixel=1.0, image_shape=img.shape[:2])
                            pred_frac = mig["empty_area_fraction"]
                            frac_err_list.append(pred_frac - meta["scratch_fraction_actual"])
                        except DegenerateMaskError:
                            degenerate_count += 1

                        wall_ms_list.append((time.perf_counter() - t0) * 1000)

                rss = rss_mon.get_summary()
                n = len(dice_list)

                entry = {
                    "scratch_fraction": frac,
                    "condition": cond,
                    "pipeline": pipe_name,
                    "n_samples": n,
                    "n_degenerate": degenerate_count,
                    "dice_mean": round(float(np.mean(dice_list)), 4),
                    "iou_mean": round(float(np.mean(iou_list)), 4),
                }
                if frac_err_list:
                    arr = np.array(frac_err_list)
                    entry.update({
                        "fraction_bias": round(float(np.mean(arr)), 4),
                        "fraction_mae": round(float(np.mean(np.abs(arr))), 4),
                    })
                else:
                    entry.update({"fraction_bias": None, "fraction_mae": None})

                entry.update({
                    "wall_time_mean_ms": round(float(np.mean(wall_ms_list)), 2),
                    "rss_peak_mb": rss["rss_peak_observed_mb"],
                })
                summary_rows.append(entry)

            if config_idx % 5 == 0 or config_idx == total:
                print(f"  Migração: {config_idx:3d}/{total} configurações concluídas")

    # Salvar CSV
    csv_path = out_dir / "migration_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    json_path = out_dir / "migration_results.json"
    output = {
        "metadata": {
            "benchmark": "SwiftMev Alpha 0.3 — Migration Validation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "samples_per_config": samples_per_config,
        },
        "records": summary_rows,
    }
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Migração salvo em: {csv_path}")
    return output


# ─────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────

def _save_results(rows, out_dir, prefix, samples_per_config,
                  jc, ja, jl) -> dict:
    csv_path = out_dir / f"{prefix}_geometry_summary.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    json_path = out_dir / f"{prefix}_geometry_results.json"
    output = {
        "metadata": {
            "benchmark": f"SwiftMev Alpha 0.3 — {prefix.title()} Geometric Validation",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "samples_per_config": samples_per_config,
            "jitter_center_sigma_px": jc,
            "jitter_angle_sigma_deg": ja,
            "jitter_length_sigma_px": jl,
        },
        "records": rows,
    }
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  {prefix.title()} salvo em:")
    print(f"    CSV:  {csv_path}")
    print(f"    JSON: {json_path}")
    return output


# ─────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SwiftMev Alpha 0.3 — Benchmark de Validação Completo"
    )
    parser.add_argument("--samples", type=int, default=5,
                        help="Amostras independentes por configuração")
    parser.add_argument("--skip-migration", action="store_true",
                        help="Pular benchmark de migração")
    parser.add_argument("--skip-fibers", action="store_true",
                        help="Pular benchmark de fibras")
    parser.add_argument("--no-masks", action="store_true",
                        help="Não salvar máscaras para auditoria")
    args = parser.parse_args()

    t_total = time.perf_counter()

    if not args.skip_fibers:
        fiber_results = run_fiber_benchmark(
            samples_per_config=args.samples,
            save_masks=not args.no_masks,
        )

    if not args.skip_migration:
        migration_results = run_migration_benchmark(
            samples_per_config=args.samples,
        )

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 100}")
    print(f"BENCHMARK COMPLETO em {elapsed:.1f}s")
    print(f"{'=' * 100}")
