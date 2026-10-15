"""
SwiftMev AI — Manifesto de Modelos (Alpha 0.6A)

Contrato rigoroso de metadados, validação de tensores, proveniência científica
e integridade criptográfica SHA-256.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
from swiftmev.ai.exceptions import ModelManifestError


REQUIRED_FIELDS = [
    "model_id",
    "name",
    "task",
    "architecture",
    "model_version",
    "input_shape",
    "input_dtype",
    "normalization",
    "output_type",
    "threshold",
    "license",
    "provenance",
    "scientific_limitations",
    "checksum_sha256",
    "onnx_filename",
]


@dataclass
class ModelManifest:
    model_id: str
    name: str
    task: str
    architecture: str
    model_version: str
    input_shape: list[int]
    input_dtype: str
    normalization: dict
    output_type: str
    threshold: float
    license: str
    provenance: str
    scientific_limitations: str
    checksum_sha256: str
    onnx_filename: str
    tile_overlap: int = 64
    manifest_path: Path | None = None

    def validate(self) -> None:
        """Valida a conformidade científica e estrutural do manifesto."""
        for req in REQUIRED_FIELDS:
            val = getattr(self, req, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ModelManifestError(f"Campo obrigatório ausente ou vazio no manifesto: '{req}'")

        if len(self.input_shape) != 4:
            raise ModelManifestError(
                f"input_shape deve possuir 4 dimensões (N, C, H, W). Recebido: {self.input_shape}"
            )
        if self.input_shape[0] != 1:
            raise ModelManifestError(
                f"Batch size no input_shape deve ser 1 para inferência em CPU. Recebido: {self.input_shape[0]}"
            )
        if self.input_shape[1] not in (1, 3):
            raise ModelManifestError(
                f"Canais de entrada devem ser 1 (grayscale) ou 3 (RGB). Recebido: {self.input_shape[1]}"
            )
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ModelManifestError(
                f"Threshold de binarização deve estar no intervalo [0.0, 1.0]. Recebido: {self.threshold}"
            )
        if len(self.checksum_sha256) != 64 or not all(c in "0123456789abcdefABCDEF" for c in self.checksum_sha256):
            raise ModelManifestError(
                f"checksum_sha256 inválido: deve ser string hexadecimal de 64 caracteres."
            )

    @classmethod
    def from_dict(cls, data: dict, manifest_path: Path | None = None) -> ModelManifest:
        for req in REQUIRED_FIELDS:
            if req not in data:
                raise ModelManifestError(f"Chave obrigatória '{req}' ausente no manifesto JSON.")

        manifest = cls(
            model_id=str(data["model_id"]),
            name=str(data["name"]),
            task=str(data["task"]),
            architecture=str(data["architecture"]),
            model_version=str(data["model_version"]),
            input_shape=list(data["input_shape"]),
            input_dtype=str(data["input_dtype"]),
            normalization=dict(data["normalization"]),
            output_type=str(data["output_type"]),
            threshold=float(data["threshold"]),
            license=str(data["license"]),
            provenance=str(data["provenance"]),
            scientific_limitations=str(data["scientific_limitations"]),
            checksum_sha256=str(data["checksum_sha256"]).lower(),
            onnx_filename=str(data["onnx_filename"]),
            tile_overlap=int(data.get("tile_overlap", 64)),
            manifest_path=manifest_path,
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, file_path: str | Path) -> ModelManifest:
        p = Path(file_path).resolve()
        if not p.exists():
            raise ModelManifestError(f"Arquivo de manifesto não encontrado: {p}")
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError as exc:
            raise ModelManifestError(f"Erro de sintaxe JSON em '{p}': {exc}")
        return cls.from_dict(data, manifest_path=p)

    def to_dict(self) -> dict:
        d = asdict(self)
        if "manifest_path" in d:
            d["manifest_path"] = str(d["manifest_path"]) if d["manifest_path"] else None
        return d
