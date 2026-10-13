"""
SwiftMev Benchmark Suite
Avaliação empírica de latência (tempo), pico de memória (RAM) e concordância de métricas
entre métodos de segmentação clássicos (Otsu vs Sauvola, com e sem CLAHE) em diferentes resoluções.
"""
from __future__ import annotations
import gc
import json
import time
import tracemalloc
from pathlib import Path
import cv2
import numpy as np
from cellscope.processing import migration, fiber


def generate_synthetic_migration(size: int) -> np.ndarray:
    """Gera amostra sintética de ensaio de migração na resolução size x size."""
    np.random.seed(42)
    img = np.random.randint(140, 200, (size, size), dtype=np.uint8)
    # Fenda central (ferida/scratch)
    gap_start = int(size * 0.35)
    gap_end = int(size * 0.65)
    img[:, gap_start:gap_end] = np.random.randint(225, 255, (size, gap_end - gap_start), dtype=np.uint8)
    return img


def generate_synthetic_fiber(size: int) -> np.ndarray:
    """Gera amostra sintética de fibras na resolução size x size."""
    np.random.seed(42)
    img = np.random.randint(10, 30, (size, size), dtype=np.uint8)
    step = max(30, size // 6)
    thickness = max(4, size // 30)
    for y in range(step, size - step, step):
        cv2.line(img, (int(size * 0.05), y), (int(size * 0.95), y), 230, thickness=thickness)
    return img


def benchmark_single(func, img: np.ndarray, um_per_pixel: float, **kwargs) -> dict:
    """Mede tempo de execução (ms) e pico de memória alocada (MB)."""
    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    metrics, _ = func(img, um_per_pixel=um_per_pixel, **kwargs)

    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    return {
        "elapsed_ms": round(t_elapsed_ms, 2),
        "peak_ram_mb": round(peak_mb, 2),
        "metrics": metrics
    }


def run_benchmarks() -> list[dict]:
    resolutions = [512, 1024, 2048]
    methods = [
        ("Otsu", {"method": "otsu", "use_clahe": False}),
        ("Otsu + CLAHE", {"method": "otsu", "use_clahe": True}),
        ("Sauvola", {"method": "sauvola", "use_clahe": False}),
        ("Sauvola + CLAHE", {"method": "sauvola", "use_clahe": True}),
    ]

    results = []
    print("=" * 80)
    print("SWIFTMEV — BENCHMARK EMPÍRICO DE PERFORMANCE E MEMÓRIA")
    print("=" * 80)

    # 1. Módulo de Migração
    print("\n--- [MÓDULO: MIGRAÇÃO CELULAR] ---")
    for res in resolutions:
        img = generate_synthetic_migration(res)
        print(f"\nResolução: {res}x{res} ({round(img.size / 1e6, 2)} MP):")
        for label, kwargs in methods:
            bench = benchmark_single(migration, img, um_per_pixel=0.5, **kwargs)
            res_entry = {
                "module": "migration",
                "resolution": f"{res}x{res}",
                "pixels_mp": round(img.size / 1e6, 2),
                "pipeline": label,
                "elapsed_ms": bench["elapsed_ms"],
                "peak_ram_mb": bench["peak_ram_mb"],
                "metric_area_um2": bench["metrics"]["empty_area_um2"],
                "metric_fraction": bench["metrics"]["empty_area_fraction"]
            }
            results.append(res_entry)
            print(f"  • {label:<16}: {bench['elapsed_ms']:>7.2f} ms | RAM Pico: {bench['peak_ram_mb']:>5.2f} MB | Área Livre: {bench['metrics']['empty_area_fraction']*100:>5.1f}%")

    # 2. Módulo de Fibras
    print("\n--- [MÓDULO: DIÂMETRO DE FIBRAS] ---")
    for res in resolutions:
        img = generate_synthetic_fiber(res)
        print(f"\nResolução: {res}x{res} ({round(img.size / 1e6, 2)} MP):")
        for label, kwargs in methods:
            bench = benchmark_single(fiber, img, um_per_pixel=0.5, **kwargs)
            mean_d = bench["metrics"]["mean_local_diameter_um"]
            std_d = bench["metrics"]["std_local_diameter_um"]
            res_entry = {
                "module": "fiber",
                "resolution": f"{res}x{res}",
                "pixels_mp": round(img.size / 1e6, 2),
                "pipeline": label,
                "elapsed_ms": bench["elapsed_ms"],
                "peak_ram_mb": bench["peak_ram_mb"],
                "mean_diameter_um": mean_d,
                "std_diameter_um": std_d
            }
            results.append(res_entry)
            mean_str = f"{mean_d:.2f} ± {std_d:.2f} um" if mean_d else "N/A"
            print(f"  • {label:<16}: {bench['elapsed_ms']:>7.2f} ms | RAM Pico: {bench['peak_ram_mb']:>5.2f} MB | Diâmetro: {mean_str}")

    return results


if __name__ == "__main__":
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_data = run_benchmarks()
    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(bench_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório de benchmark salvo em: {json_path}")
