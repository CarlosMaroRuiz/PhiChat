from __future__ import annotations

from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

import re
import warnings

def _to_snake_case(name: str) -> str:
    """Convierte un nombre a snake_case según los estándares de LangChain."""
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
    return re.sub(r'\W+', '_', name).strip('_')

def create_tool(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    return_direct: bool = False,
) -> BaseTool:
    """Crea una BaseTool optimizada para phi4-mini y compatible con LangChain.

    La descripcion es lo unico que phi4-mini lee para decidir si usa la tool.
    Debe describir *cuando* usarla, no solo que hace.

    Args:
        func: Funcion Python que implementa la herramienta. Debe tener type hints.
        name: Nombre de la herramienta. Por defecto usa ``func.__name__``. Se convertirá a snake_case.
        description: Descripcion legible. Por defecto usa el docstring de ``func``.
        args_schema: Schema Pydantic para los argumentos. Mejora la precision.
        return_direct: Si es ``True``, el resultado se devuelve directamente al usuario.

    Returns:
        Una instancia de ``BaseTool`` lista para usar con ``bind_tools`` o ``run_tool_loop``.
    """
    if not func.__annotations__:
        warnings.warn(
            f"La función '{func.__name__}' no tiene type hints. LangChain exige type hints "
            "para generar correctamente el esquema de la herramienta.",
            UserWarning
        )

    raw_name = name or func.__name__
    safe_name = _to_snake_case(raw_name)

    return StructuredTool.from_function(
        func=func,
        name=safe_name,
        description=description or func.__doc__ or f"Ejecuta {safe_name}",
        args_schema=args_schema,
        return_direct=return_direct,
    )
