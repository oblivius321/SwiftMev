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
    standalone_flags = [
        "--list-models",
        "--studio-scan",
        "--studio-split",
        "--studio-qc",
        "--studio-export",
    ]
    is_standalone = any(flag in sys.argv for flag in standalone_flags)

    parser.add_argument(
        "--module", choices=["migration", "fiber"],
        required=not is_standalone,
        help="Módulo de análise"
    )
    parser.add_argument(
        "--input", required=not is_standalone and ("--studio-qc" not in sys.argv),
        help="Caminho da imagem de entrada"
    )
    parser.add_argument(
        "--output", required=not is_standalone,
        help="Diretório de saída"
    )
    parser.add_argument(
        "--um-per-pixel", type=float, required=not is_standalone,
        help="Fator de calibração micrometros/pixel"
    )
    parser.add_argument(
        "--method", choices=["otsu", "sauvola", "ai"], default="otsu",
        help="Método de segmentação (default: otsu)"
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Identificador do modelo de IA a ser utilizado (quando --method=ai)"
    )
    parser.add_argument(
        "--models-dir", default=None,
        help="Caminho customizado para o diretório de modelos de IA"
    )
    parser.add_argument(
        "--list-models", action="store_true", default=False,
        help="Listar modelos de IA locais disponíveis e encerrar"
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
    # Dataset Studio Arguments
    parser.add_argument(
        "--studio-scan", default=None,
        help="Executar varredura e reconciliação de inventário do Dataset Studio no diretório indicado"
    )
    parser.add_argument(
        "--studio-split", default=None,
        help="Calcular particionamento disjunto e sem vazamento para o diretório de micrografias"
    )
    parser.add_argument(
        "--studio-qc", action="store_true", default=False,
        help="Inspecionar qualidade e anomalias de uma máscara anotada (--mask obrigatório)"
    )
    parser.add_argument(
        "--studio-export", action="store_true", default=False,
        help="Exportar pacote de release versionado do Dataset Studio"
    )
    parser.add_argument(
        "--studio-input-dir", default=None,
        help="Diretório de origem das micrografias para exportação do Studio"
    )
    parser.add_argument(
        "--studio-output-dir", default=None,
        help="Diretório de destino da release do Studio"
    )
    parser.add_argument(
        "--studio-dataset-name", default="swiftmev_fiber_dataset",
        help="Nome do dataset a ser exportado (default: swiftmev_fiber_dataset)"
    )
    parser.add_argument(
        "--studio-version", default="0.1.0",
        help="Versão do dataset (default: 0.1.0)"
    )
    parser.add_argument(
        "--studio-train-ratio", type=float, default=0.70,
        help="Fração de imagens no conjunto de treino (default: 0.70)"
    )
    parser.add_argument(
        "--studio-val-ratio", type=float, default=0.15,
        help="Fração de imagens no conjunto de validação (default: 0.15)"
    )
    parser.add_argument(
        "--studio-test-ratio", type=float, default=0.15,
        help="Fração de imagens no conjunto de teste (default: 0.15)"
    )
    parser.add_argument(
        "--studio-require-reviewed", action="store_true", default=False,
        help="Exigir que todas as anotações exportadas estejam com status 'reviewed' ou 'approved'"
    )
    parser.add_argument(
        "--studio-overwrite", action="store_true", default=False,
        help="Permitir sobrescrever uma release existente com mesma versão"
    )

    args = parser.parse_args()

    if args.list_models:
        try:
            from swiftmev.ai import list_available_models, is_ai_available
            models = list_available_models(models_dir=args.models_dir)
            print(json.dumps({
                "ai_available": is_ai_available(),
                "models": models
            }, indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
            return 1

    if args.studio_scan:
        try:
            from swiftmev.studio import reconcile_inventory
            res = reconcile_inventory(args.studio_scan)
            res["micrographs"] = [m.to_dict() for m in res["micrographs"]]
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
            return 1

    if args.studio_split:
        try:
            from swiftmev.studio import scan_directory, partition_groups_disjoint
            recs = scan_directory(args.studio_split)
            split = partition_groups_disjoint(
                recs,
                train_ratio=args.studio_train_ratio,
                val_ratio=args.studio_val_ratio,
                test_ratio=args.studio_test_ratio,
            )
            print(json.dumps(split.to_dict(), indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
            return 1

    if args.studio_qc:
        try:
            if not args.mask:
                print(json.dumps({"status": "error", "message": "O parâmetro --mask é obrigatório para --studio-qc."}), file=sys.stderr)
                return 1
            from swiftmev.studio import inspect_annotation_quality
            mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
            micro_gray = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE) if args.input else None
            alerts = inspect_annotation_quality(mask_img, micro_gray)
            print(json.dumps({"qc_alerts": [a.to_dict() for a in alerts]}, indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
            return 1

    if args.studio_export:
        try:
            if not args.studio_input_dir or not args.studio_output_dir:
                print(json.dumps({"status": "error", "message": "--studio-input-dir e --studio-output-dir são obrigatórios para exportação."}), file=sys.stderr)
                return 1
            from swiftmev.studio import scan_directory, partition_groups_disjoint, export_dataset_release
            recs = scan_directory(args.studio_input_dir)
            split = partition_groups_disjoint(
                recs,
                train_ratio=args.studio_train_ratio,
                val_ratio=args.studio_val_ratio,
                test_ratio=args.studio_test_ratio,
            )
            manifest = export_dataset_release(
                destination_dir=args.studio_output_dir,
                dataset_name=args.studio_dataset_name,
                dataset_version=args.studio_version,
                micrographs=recs,
                split_record=split,
                source_root=args.studio_input_dir,
                require_reviewed_annotations=args.studio_require_reviewed,
                overwrite=args.studio_overwrite,
            )
            print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
            return 1

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
            if args.method == "ai":
                seg_kwargs["model_id"] = args.model_id
                seg_kwargs["models_dir"] = args.models_dir

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

        if "ai_model" in metrics:
            result["ai_model_info"] = metrics["ai_model"]

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
        # Erros específicos do subsistema de IA recebem código de saída 3
        from swiftmev.ai.exceptions import AIError
        if isinstance(exc, AIError):
            print(
                json.dumps(
                    {
                        "status": "ai_error",
                        "error_code": exc.error_code,
                        "message": str(exc)
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 3

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
