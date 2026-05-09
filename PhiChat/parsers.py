from __future__ import annotations

import json
import uuid

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX, _TOOL_CALL_PATTERNS


from typing import Any

def parse_phi4_tool_calls(response: AIMessage) -> list[dict[str, Any]]:
    """
    Extrae llamadas a herramientas (tool calls) del contenido del mensaje.
    
    Esta funcion es necesaria cuando el modelo phi4-mini genera las llamadas
    como texto plano con etiquetas en lugar de usar el campo nativo de Ollama.
    
    Args:
        response: El objeto AIMessage devuelto por el modelo.
        
    Returns:
        Una lista de diccionarios con el formato estandar de LangChain: 
        {"name": str, "args": dict, "id": str}.
    """
    if response.tool_calls:
        return list(response.tool_calls)

    content = response.content or ""
    if not content:
        return []

    for pattern in _TOOL_CALL_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        try:
            raw = json.loads(match.group(1))
            calls = raw if isinstance(raw, list) else [raw]
            return [
                {
                    "name": c.get("name") or c.get("type") or c.get("function", {}).get("name", ""),
                    "args": (
                        c.get("arguments")
                        or c.get("args")
                        or c.get("parameters")
                        or c.get("function", {}).get("arguments", {})
                        or c.get("function", {}).get("args", {})
                        or {}
                    ),
                    "id": c.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                }
                for c in calls
                if c.get("name") or c.get("type") or c.get("function", {}).get("name")
            ]
        except (json.JSONDecodeError, AttributeError):
            continue

    return []


def inject_tool_system_message(
    messages: list[BaseMessage],
    suffix: str = _PHI4_TOOL_SYSTEM_SUFFIX,
) -> list[BaseMessage]:
    """Agrega el sufijo al SystemMessage existente o inserta uno nuevo si no hay."""
    result = list(messages)
    for i, msg in enumerate(result):
        if isinstance(msg, SystemMessage):
            if suffix.strip() not in (msg.content or ""):
                result[i] = SystemMessage(content=(msg.content or "") + suffix)
            return result
    result.insert(0, SystemMessage(content="Eres un asistente util." + suffix))
    return result
