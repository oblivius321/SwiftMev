"""
SwiftMev — Alpha 0.2: Validação Geométrica Rigorosa de Fibras
Testa:
- 6 Espessuras: 3, 6, 10, 16, 20 e 30 pixels
- 5 Orientações: 0°, 30°, 45°, 60° e 90°
- 4 Condições: Ideal, Ruído, Gradiente de Iluminação e Baixo Contraste
- Amostras independentes com sementes reprodutíveis
- Avaliação contra D_geom (teórico) e W_rast (medido na máscara rasterizada)
- Monitoramento de RSS por amostragem contínua (~2 ms)
- Métricas: Dice, IoU, MAE, RMSE, Viés e Intervalo de Confiança 95%
"""
from __future__ import annotations
import argparse
import csv
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
from pathlib import Path

# Garante acesso aos módulos de engine e benchmarks independentemente de PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(_ROOT / "engine"))

from benchmarks.geometric_generator import generate_geometric_fiber
from benchmarks.rss_monitor import RSSSamplingMonitor
from cellscope.processing import segment, fiber


def calculate_dice(p_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = p_mask > 0
    g = gt_mask > 0
    intersection = np.logical_and(p, g).sum()
    total = p.sum() + g.sum()
    return float((2.0 * intersection) / total) if total > 0 else 1.0


def calculate_iou(p_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    p = p_mask > 0
    g = gt_mask > 0
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(intersection / union) if union > 0 else 1.0


def run_fiber_evaluation(
    samples_per_config: int = 5,
    base_seed: int = 42,
    save_masks: bool = True
) -> dict:
    diameters = [3.0, 6.0, 10.0, 16.0, 20.0, 30.0]
    angles = [0.0, 30.0, 45.0, 60.0, 90.0]
    conditions = ["ideal", "noise", "gradient", "low_contrast"]
    pipelines = [
        ("Otsu", {"method": "otsu", "use_clahe": False}),
        ("Otsu_CLAHE", {"method": "otsu", "use_clahe": True}),
        ("Sauvola", {"method": "sauvola", "use_clahe": False}),
    ]

    out_dir = Path("outputs/alpha02_validation").resolve()
    masks_dir = out_dir / "masks"
    if save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    summary_rows = []

    total_configs = len(diameters) * len(angles) * len(conditions)
    print("=" * 95)
    print(f"SWIFTMEV ALPHA 0.2 — VALIDAÇÃO GEOMÉTRICA DE FIBRAS (SAMPLES={samples_per_config}/CONFIG)")
    print(f"Total de configurações: {total_configs} | Total de imagens por pipeline: {total_configs * samples_per_config}")
    print("=" * 95)

    config_idx = 0
    for d_geom in diameters:
        for angle in angles:
            for cond in conditions:
                config_idx += 1
                config_label = f"D{int(d_geom)}_A{int(angle)}_{cond}"

                # Lote de imagens estatisticamente independentes para esta configuração
                images_batch = []
                for s_idx in range(samples_per_config):
                    seed = base_seed + (config_idx * 1000) + s_idx
                    img, gt_mask, meta = generate_geometric_fiber(
                        size=256,
                        nominal_diameter=d_geom,
                        angle_deg=angle,
                        length=180.0,
                        condition=cond,
                        seed=seed
                    )
                    images_batch.append((img, gt_mask, meta))

                # Avaliação por pipeline
                for pipe_name, pipe_kwargs in pipelines:
                    dice_list = []
                    iou_list = []
                    pred_diam_list = []
                    w_rast_list = []
                    err_geom_list = []
                    err_rast_list = []
                    wall_time_list = []
                    cpu_time_list = []

                    with RSSSamplingMonitor(interval_ms=2.0) as rss_mon:
                        for s_idx, (img, gt_mask, meta) in enumerate(images_batch):
                            t_wall_start = time.perf_counter()
                            t_cpu_start = time.process_time()

                            # Chamada do motor de análise (calibração 1.0 px -> medido diretamente em pixels)
                            metrics, _ = fiber(img, um_per_pixel=1.0, **pipe_kwargs)

                            t_wall = (time.perf_counter() - t_wall_start) * 1000.0
                            t_cpu = (time.process_time() - t_cpu_start) * 1000.0

                            wall_time_list.append(t_wall)
                            cpu_time_list.append(t_cpu)

                            pred_mask = segment(img, **pipe_kwargs)
                            d_pred = metrics["median_local_diameter_um"]  # na calibração 1.0, equivale a pixels

                            w_rast = meta["raster_width_px"]
                            w_rast_list.append(w_rast)

                            dice = calculate_dice(pred_mask, gt_mask)
                            iou = calculate_iou(pred_mask, gt_mask)
                            dice_list.append(dice)
                            iou_list.append(iou)

                            if d_pred is not None:
                                pred_diam_list.append(d_pred)
                                err_geom_list.append(d_pred - d_geom)
                                err_rast_list.append(d_pred - w_rast)

                            # Salva máscara da 1ª repetição para auditoria visual
                            if save_masks and s_idx == 0:
                                mask_path = masks_dir / f"{config_label}_{pipe_name}_pred.png"
                                cv2.imwrite(str(mask_path), pred_mask)
                                gt_path = masks_dir / f"{config_label}_gt.png"
                                if not gt_path.exists():
                                    cv2.imwrite(str(gt_path), gt_mask)

                    rss_summary = rss_mon.get_summary()

                    # Estatísticas sobre as N amostras independentes
                    n = len(dice_list)
                    dice_mean = float(np.mean(dice_list))
                    dice_std = float(np.std(dice_list, ddof=1)) if n > 1 else 0.0
                    dice_ci95 = float(1.96 * dice_std / np.sqrt(n)) if n > 1 else 0.0

                    iou_mean = float(np.mean(iou_list))
                    iou_std = float(np.std(iou_list, ddof=1)) if n > 1 else 0.0

                    w_rast_mean = float(np.mean(w_rast_list))
                    rast_err_mean = float(w_rast_mean - d_geom)

                    if pred_diam_list:
                        d_pred_mean = float(np.mean(pred_diam_list))
                        d_pred_std = float(np.std(pred_diam_list, ddof=1)) if len(pred_diam_list) > 1 else 0.0

                        # Erro contra D_geom
                        mae_geom = float(np.mean(np.abs(err_geom_list)))
                        rmse_geom = float(np.sqrt(np.mean(np.array(err_geom_list) ** 2)))
                        bias_geom = float(np.mean(err_geom_list))

                        # Erro contra W_rast
                        mae_rast = float(np.mean(np.abs(err_rast_list)))
                        rmse_rast = float(np.sqrt(np.mean(np.array(err_rast_list) ** 2)))
                        bias_rast = float(np.mean(err_rast_list))
                    else:
                        d_pred_mean = d_pred_std = mae_geom = rmse_geom = bias_geom = mae_rast = rmse_rast = bias_rast = None

                    summary_entry = {
                        "config_id": config_idx,
                        "nominal_diameter_px": d_geom,
                        "angle_deg": angle,
                        "condition": cond,
                        "pipeline": pipe_name,
                        "n_samples": n,
                        "raster_width_px_mean": round(w_rast_mean, 2),
                        "rasterization_bias_px": round(rast_err_mean, 2),
                        "dice_mean": round(dice_mean, 4),
                        "dice_std": round(dice_std, 4),
                        "dice_ci95": round(dice_ci95, 4),
                        "iou_mean": round(iou_mean, 4),
                        "d_pred_mean_px": round(d_pred_mean, 2) if d_pred_mean else None,
                        "d_pred_std_px": round(d_pred_std, 2) if d_pred_std else None,
                        "bias_vs_geom_px": round(bias_geom, 2) if bias_geom is not None else None,
                        "mae_vs_geom_px": round(mae_geom, 2) if mae_geom is not None else None,
                        "rmse_vs_geom_px": round(rmse_geom, 2) if rmse_geom is not None else None,
                        "bias_vs_rast_px": round(bias_rast, 2) if bias_rast is not None else None,
                        "mae_vs_rast_px": round(mae_rast, 2) if mae_rast is not None else None,
                        "rmse_vs_rast_px": round(rmse_rast, 2) if rmse_rast is not None else None,
                        "wall_time_mean_ms": round(float(np.mean(wall_time_list)), 2),
                        "cpu_time_mean_ms": round(float(np.mean(cpu_time_list)), 2),
                        "rss_peak_observed_mb": rss_summary["rss_peak_observed_mb"],
                        "rss_delta_mb": rss_summary["rss_delta_mb"],
                    }
                    summary_rows.append(summary_entry)

                if config_idx % 10 == 0 or config_idx == total_configs:
                    print(f"Progresso: Configuração {config_idx:3d}/{total_configs} concluída...")

    # Exportação dos arquivos de validação científica
    csv_path = out_dir / "fiber_geometry_summary.csv"
    if summary_rows:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    json_path = out_dir / "fiber_geometry_results.json"
    meta_env = {
        "benchmark_name": "SwiftMev Alpha 0.2 — Experimental Validation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "samples_per_config": samples_per_config,
        "total_configurations": total_configs,
        "diameters_tested": diameters,
        "angles_tested": angles,
        "conditions_tested": conditions,
        "sampling_rate_ms": 2.0,
    }
    json_output = {
        "metadata": meta_env,
        "records": summary_rows
    }
    json_path.write_text(json.dumps(json_output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 95)
    print(f"Sucesso! Relatórios salvos em:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Máscaras auditáveis salvas em: {masks_dir}")
    print("=" * 95)
    return json_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwiftMev Alpha 0.2 Geometric Fiber Validation")
    parser.add_argument("--samples", type=int, default=5, help="Número de amostras independentes por configuração")
    args = parser.parse_args()
    run_fiber_evaluation(samples_per_config=args.samples)
