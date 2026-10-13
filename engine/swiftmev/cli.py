"""
SwiftMev — CLI (Alpha 0.3)

Ponto de entrada para a interface desktop (C# WPF).
Comunicação via JSON no stdout; erros no stderr.

Mudanças em relação à Alpha 0.2:
  - Importa do pacote swiftmev (modular) em vez de cellscope
  - Pipeline unificado: segmentação executada uma única vez
  - Validação obrigatória de um_per_pixel (positivo, finito, não-NaN)
  - Retorna a máscara junto com as métricas (para inspeção do pesquisador)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import psutil

from swiftmev import __version__
from swiftmev.imaging import load_gray_with_meta
from swiftmev.pipeline import (
    run_migration,
    run_fiber,
    run_fiber_from_mask,
    run_migration_from_mask,
)
from swiftmev.validation import CalibrationError, DegenerateMaskError
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SwiftMev motor local de análise científica"
    )
    parser.add_argument(
        "--module", choices=["migration", "fiber"],
        required=True, help="Módulo de análise"
    )
    parser.add_argument(
        "--input", required=True,
        help="Caminho da imagem de entrada"
    )
    parser.add_argument(
        "--output", required=True,
        help="Diretório de saída"
    )
    parser.add_argument(
        "--um-per-pixel", type=float, required=True,
        help="Fator de calibração micrometros/pixel"
    )
    parser.add_argument(
        "--method", choices=["otsu", "sauvola"], default="otsu",
        help="Método de segmentação (default: otsu)"
    )
    parser.add_argument(
        "--clahe", action="store_true", default=False,
        help="Ativar equalização adaptativa CLAHE no pré-processamento"
    )
    parser.add_argument(
        "--invert", action="store_true", default=False,
        help="Inverter polaridade da máscara binária"
    )
    parser.add_argument(
        "--sauvola-window", type=int, default=35,
        help="Tamanho da janela local para Sauvola (default: 35)"
    )
    parser.add_argument(
        "--sauvola-k", type=float, default=0.2,
        help="Parâmetro k para Sauvola (default: 0.2)"
    )
    parser.add_argument(
        "--save-mask", action="store_true", default=False,
        help="Salvar a máscara binária como arquivo PNG separado"
    )
    parser.add_argument(
        "--mask", default=None,
        help="Caminho para máscara binária pré-existente ou editada manualmente"
    )

    args = parser.parse_args()
    try:
        proc = psutil.Process()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        img, img_meta = load_gray_with_meta(args.input)
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=True)

        if args.mask:
            mask_raw, _ = load_gray_with_meta(args.mask)
            mask = np.where(mask_raw > 127, 255, 0).astype(np.uint8)
            if args.module == "migration":
                metrics, annotated = run_migration_from_mask(
                    mask,
                    um_per_pixel=args.um_per_pixel,
                    image_shape=img.shape[:2],
                    img_for_overlay=img,
                )
            else:
                metrics, annotated = run_fiber_from_mask(
                    mask,
                    um_per_pixel=args.um_per_pixel,
                    img_for_overlay=img,
                )
            metrics["mask_source"] = "manual_or_external"
            suffix = f"{args.module}_edited_mask"
        else:
            seg_kwargs = dict(
                method=args.method,
                use_clahe=args.clahe,
                invert=args.invert,
                sauvola_window=args.sauvola_window,
                sauvola_k=args.sauvola_k,
            )
            if args.module == "migration":
                metrics, mask, annotated = run_migration(
                    img, um_per_pixel=args.um_per_pixel, **seg_kwargs
                )
            else:
                metrics, mask, annotated = run_fiber(
                    img, um_per_pixel=args.um_per_pixel, **seg_kwargs
                )
            suffix = f"{args.module}_{args.method}"
            if args.clahe:
                suffix += "_clahe"

        t_wall_elapsed_ms = round((time.perf_counter() - t_wall_start) * 1000, 2)
        t_cpu_elapsed_ms = round((time.process_time() - t_cpu_start) * 1000, 2)
        rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

        stem = Path(args.input).stem

        overlay_path = out / f"{stem}_{suffix}_overlay.png"
        if not cv2.imwrite(str(overlay_path), annotated):
            raise RuntimeError("Não foi possível salvar a imagem de saída")

        # Salvar máscara se solicitado
        mask_path_str = None
        if args.save_mask:
            mask_path = out / f"{stem}_{suffix}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            mask_path_str = str(mask_path)

        result = {
            "version": __version__,
            "status": "experimental",
            "module": args.module,
            "method": args.method,
            "clahe_applied": args.clahe,
            "input": str(Path(args.input).resolve()),
            "overlay": str(overlay_path),
            "mask": mask_path_str,
            "calibration_um_per_pixel": args.um_per_pixel,
            "latency": {
                "wall_clock_time_ms": t_wall_elapsed_ms,
                "cpu_time_ms": t_cpu_elapsed_ms,
            },
            "memory": {
                "process_rss_mb": rss_mb,
            },
            "input_metadata": img_meta,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }

        json_path = out / f"{stem}_{suffix}_metrics.json"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0

    except (CalibrationError, DegenerateMaskError) as exc:
        print(
            json.dumps(
                {"status": "validation_error", "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2

    except Exception as exc:
        print(
            json.dumps(
                {"status": "error", "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
