"""
SwiftMev AI — Exceções Especializadas (Alpha 0.6A)

Separação estrita entre erros técnicos de runtime/modelo e rejeições metrológicas científicas.
"""
from __future__ import annotations


class AIError(Exception):
    """Classe base para erros do subsistema de inteligência artificial."""
    def __init__(self, message: str, error_code: str = "AI_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class AINotAvailableError(AIError):
    """Emitido quando a biblioteca onnxruntime não está instalada no ambiente."""
    def __init__(self, message: str = "Inferência por IA indisponível: o pacote 'onnxruntime' não está instalado."):
        super().__init__(message, error_code="AI_DEPENDENCY_MISSING")


class AIModelNotFoundError(AIError):
    """Emitido quando o modelo ONNX ou seu manifesto não foi encontrado no disco."""
    def __init__(self, model_id: str, search_path: str = ""):
        msg = f"Modelo de IA '{model_id}' não encontrado no repositório local ({search_path})."
        super().__init__(msg, error_code="MODEL_NOT_FOUND")
        self.model_id = model_id


class ModelManifestError(AIError):
    """Emitido quando o manifesto manifest.json está ausente, mal formatado ou com campos faltando."""
    def __init__(self, message: str):
        super().__init__(message, error_code="MANIFEST_INVALID")


class ChecksumMismatchError(AIError):
    """Emitido quando o hash SHA-256 do arquivo ONNX difere do registrado no manifesto."""
    def __init__(self, expected: str, actual: str, file_path: str):
        msg = (
            f"Falha de integridade no modelo '{file_path}': hash SHA-256 divergente.\n"
            f"Esperado: {expected}\n"
            f"Obtido:   {actual}"
        )
        super().__init__(msg, error_code="CHECKSUM_MISMATCH")
        self.expected = expected
        self.actual = actual


class TaskIncompatibilityError(AIError):
    """Emitido quando o modelo é solicitado para uma tarefa diferente da declarada (ex: modelo de fibras para scratch)."""
    def __init__(self, model_task: str, requested_task: str):
        msg = (
            f"Incompatibilidade de tarefa científica: o modelo selecionado é restrito a '{model_task}', "
            f"mas a análise solicitada foi '{requested_task}'."
        )
        super().__init__(msg, error_code="TASK_INCOMPATIBLE")
        self.model_task = model_task
        self.requested_task = requested_task


class TensorMismatchError(AIError):
    """Emitido quando os tensores reais do arquivo ONNX não coincidem com as especificações do manifesto."""
    def __init__(self, message: str):
        super().__init__(message, error_code="TENSOR_MISMATCH")
