from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import psutil
from cellscope.processing import load_gray_with_meta, migration, fiber


def main() -> int:
    parser = argparse.ArgumentParser(description="SwiftMev / CellScope motor local de análise científica")
    parser.add_argument("--module", choices=["migration", "fiber"], required=True, help="Módulo de análise")
    parser.add_argument("--input", required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--output", required=True, help="Diretório de saída")
    parser.add_argument("--um-per-pixel", type=float, required=True, help="Fator de calibração micrometros/pixel")
    parser.add_argument("--method", choices=["otsu", "sauvola"], default="otsu", help="Método de segmentação (default: otsu)")
    parser.add_argument("--clahe", action="store_true", default=False, help="Ativar equalização adaptativa CLAHE no pré-processamento")
    parser.add_argument("--invert", action="store_true", default=False, help="Inverter polaridade da máscara binária")
    parser.add_argument("--sauvola-window", type=int, default=35, help="Tamanho da janela local para Sauvola (default: 35)")
    parser.add_argument("--sauvola-k", type=float, default=0.2, help="Parâmetro k para Sauvola (default: 0.2)")

    args = parser.parse_args()
    try:
        if args.um_per_pixel <= 0:
            raise ValueError("um-per-pixel deve ser maior que zero")

        proc = psutil.Process()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        img, img_meta = load_gray_with_meta(args.input)
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=True)

        func = migration if args.module == "migration" else fiber
        metrics, annotated = func(
            img,
            um_per_pixel=args.um_per_pixel,
            method=args.method,
            use_clahe=args.clahe,
            invert=args.invert,
            sauvola_window=args.sauvola_window,
            sauvola_k=args.sauvola_k
        )

        t_wall_elapsed_ms = round((time.perf_counter() - t_wall_start) * 1000, 2)
        t_cpu_elapsed_ms = round((time.process_time() - t_cpu_start) * 1000, 2)
        rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

        suffix = f"{args.module}_{args.method}"
        if args.clahe:
            suffix += "_clahe"
        path = out / f"{Path(args.input).stem}_{suffix}_overlay.png"

        if not cv2.imwrite(str(path), annotated):
            raise RuntimeError("Não foi possível salvar a imagem de saída")

        result = {
            "version": "0.2.0",
            "status": "experimental",
            "module": args.module,
            "method": args.method,
            "clahe_applied": args.clahe,
            "input": str(Path(args.input).resolve()),
            "overlay": str(path),
            "calibration_um_per_pixel": args.um_per_pixel,
            "latency": {
                "wall_clock_time_ms": t_wall_elapsed_ms,
                "cpu_time_ms": t_cpu_elapsed_ms,
                "execution_time_ms": t_wall_elapsed_ms  # compatibilidade com v0.1
            },
            "memory": {
                "process_rss_mb": rss_mb
            },
            "input_metadata": img_meta,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }

        json_path = out / f"{Path(args.input).stem}_{suffix}_metrics.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
