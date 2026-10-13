"""
CellScope — DEPRECATED. Use swiftmev em vez disso.

Este pacote redireciona todas as importações para swiftmev.*
para manter compatibilidade com código existente durante a transição.
"""
__version__ = "0.3.0"

import warnings
warnings.warn(
    "O pacote 'cellscope' foi renomeado para 'swiftmev'. "
    "Atualize seus imports. Este redirecionamento será removido na v0.4.",
    DeprecationWarning,
    stacklevel=2,
)
