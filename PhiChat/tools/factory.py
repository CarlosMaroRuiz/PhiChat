from __future__ import annotations

from typing import Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

def create_tool(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
    return_direct: bool = False,
) -> BaseTool:
    """Crea una BaseTool optimizada para phi4-mini.

    La descripcion es lo unico que phi4-mini lee para decidir si usa la tool.
    Debe describir *cuando* usarla, no solo que hace.

    Args:
        func: Funcion Python que implementa la herramienta.
        name: Nombre de la herramienta. Por defecto usa ``func.__name__``.
        description: Descripcion legible. Por defecto usa el docstring de ``func``.
        args_schema: Schema Pydantic para los argumentos. Mejora la precision.
        return_direct: Si es ``True``, el resultado se devuelve directamente al usuario.

    Returns:
        Una instancia de ``BaseTool`` lista para usar con ``bind_tools`` o ``run_tool_loop``.
    """
    return StructuredTool.from_function(
        func=func,
        name=name or func.__name__,
        description=description or func.__doc__ or f"Ejecuta {func.__name__}",
        args_schema=args_schema,
        return_direct=return_direct,
    )
