from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX


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


def _extract_inline_tool_calls(content: str) -> list[dict]:
    """Extrae tool calls cuando phi4-mini las escribe como bloques JSON numerados en el texto.

    Phi-4 a veces no usa el formato de array unico sino bloques separados:
    ``1. [{"name": "tool_a", ...}]  2. [{"name": "tool_b", ...}]``

    Args:
        content: Texto del mensaje del modelo.

    Returns:
        Lista de dicts con las claves ``name``, ``args`` e ``id``,
        o lista vacia si no se encontraron bloques validos.
    """
    pattern = re.compile(r'\[\s*\{.*?"(?:name|type)".*?\}\s*\]', re.DOTALL)
    result: list[dict[str, Any]] = []

    for match in pattern.findall(content):
        try:
            raw = json.loads(match)
            calls = raw if isinstance(raw, list) else [raw]
            for c in calls:
                name = c.get("name") or c.get("type") or ""
                if not name:
                    continue
                result.append({
                    "name": name,
                    "args": c.get("arguments") or c.get("parameters") or {},
                    "id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                })
        except (json.JSONDecodeError, AttributeError):
            continue

    return result


def run_tool_loop(
    messages: list[BaseMessage],
    tools: list[BaseTool],
    *,
    model: str = "phi4-mini:latest",
    temperature: float = 0.7,
    num_ctx: int = 8192,
    keep_alive: int = 0,
    tool_system_suffix: str = _PHI4_TOOL_SYSTEM_SUFFIX,
    examples: str | None = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> AIMessage:
    """Ejecuta el loop agentico: LLM → tool calls → resultados → LLM.

    Repite hasta obtener una respuesta sin tool calls o alcanzar ``max_iterations``.
    Admite tool calls estructuradas y bloques JSON escritos inline en el texto.

    Args:
        messages: Historial de mensajes de la conversacion.
        tools: Lista de herramientas disponibles para el modelo.
        model: Nombre del modelo Ollama a usar.
        temperature: Temperatura de muestreo. 0 para respuestas deterministas.
        num_ctx: Tamano del contexto en tokens. 8192 recomendado con tools.
        keep_alive: Tiempo en segundos para mantener el modelo en memoria.
        tool_system_suffix: Sufijo inyectado en el system prompt para activar tool calling.
        examples: Ejemplos de few-shot para mejorar la precision.
        max_iterations: Numero maximo de pasos antes de abortar.
        verbose: Si es ``True``, imprime cada paso y resultado de tool.

    Returns:
        El ultimo ``AIMessage`` del modelo.
    """
    from PhiChat.model import ChatPhi

    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    bound_llm = ChatPhi(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        tool_system_suffix=tool_system_suffix,
    ).bind_tools(tools, examples=examples)

    current_messages = list(messages)

    for step in range(max_iterations):
        response: AIMessage = bound_llm.invoke(current_messages)

        if not response.tool_calls:
            tool_calls = _extract_inline_tool_calls(response.content or "")
        else:
            tool_calls = response.tool_calls

        if not tool_calls:
            return response

        if verbose:
            print(f"[step {step + 1}] {[tc['name'] for tc in tool_calls]}")

        current_messages.append(response)

        for tc in tool_calls:
            call_id = tc["id"]
            tool_name = tc["name"]
            call_args = tc["args"]
            
            selected_tool = tool_map.get(tool_name)
            if not selected_tool:
                tool_result = f"Error: Tool '{tool_name}' no encontrada."
            else:
                if verbose:
                    print(f"  -> {tool_name}({call_args})", end="... ", flush=True)
                
                try:
                    import asyncio
                    try:
                        tool_result = selected_tool.invoke(call_args)
                    except Exception as e:
                        if "does not support sync invocation" in str(e) or "coroutine" in str(e):
                            try:
                                tool_result = asyncio.run(selected_tool.ainvoke(call_args))
                            except RuntimeError:
                                # Si ya hay un loop corriendo (caso del test async), usamos este truco
                                import nest_asyncio
                                nest_asyncio.apply()
                                tool_result = asyncio.run(selected_tool.ainvoke(call_args))
                        else:
                            raise e
                    
                    if verbose: print(f"{tool_result}")
                except Exception as e:
                    tool_result = f"Error en '{tool_name}': {e}"
                    if verbose: print(f"ERROR: {e}")
            
            current_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

    return AIMessage(content="[Limite de iteraciones alcanzado]")


async def arun_tool_loop(
    messages: list[BaseMessage],
    tools: list[BaseTool],
    *,
    model: str = "phi4-mini:latest",
    temperature: float = 0.7,
    num_ctx: int = 8192,
    keep_alive: int = 0,
    tool_system_suffix: str = _PHI4_TOOL_SYSTEM_SUFFIX,
    examples: str | None = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> AIMessage:
    """
    Version asincrona nativa del loop agentico.
    
    Recomendada para aplicaciones web, servidores MCP y entornos de alta concurrencia.
    """
    from PhiChat.model import ChatPhi

    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
    bound_llm = ChatPhi(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        tool_system_suffix=tool_system_suffix,
    ).bind_tools(tools, examples=examples)

    current_messages = list(messages)

    for step in range(max_iterations):
        response: AIMessage = await bound_llm.ainvoke(current_messages)

        if not response.tool_calls:
            tool_calls = _extract_inline_tool_calls(response.content or "")
        else:
            tool_calls = response.tool_calls

        if not tool_calls:
            return response

        if verbose:
            print(f"[step {step + 1} async] {[tc['name'] for tc in tool_calls]}")

        current_messages.append(response)

        for tc in tool_calls:
            call_id = tc["id"]
            tool_name = tc["name"]
            call_args = tc["args"]
            
            selected_tool = tool_map.get(tool_name)
            if not selected_tool:
                tool_result = f"Error: Tool '{tool_name}' no encontrada."
            else:
                if verbose:
                    print(f"  -> {tool_name}({call_args})", end="... ", flush=True)
                
                try:
                    # En la version asincrona siempre usamos ainvoke
                    tool_result = await selected_tool.ainvoke(call_args)
                    if verbose: print(f"{tool_result}")
                except Exception as e:
                    tool_result = f"Error en '{tool_name}': {e}"
                    if verbose: print(f"ERROR: {e}")
            
            current_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

    return AIMessage(content="[Limite de iteraciones alcanzado]")