from __future__ import annotations

import json
import uuid

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX, _TOOL_CALL_PATTERNS


from typing import Any

def normalize_tool_call(c: dict[str, Any]) -> dict[str, Any] | None:
    """Normaliza un dict de llamada a herramienta al formato estandar de LangChain."""
    name = str(c.get("name") or c.get("type") or c.get("function", {}).get("name", ""))
    if not name:
        return None
    
    raw_args = (
        c.get("arguments")
        or c.get("args")
        or c.get("parameters")
        or c.get("function", {}).get("arguments", {})
        or c.get("function", {}).get("args", {})
        or {}
    )
    
    # Si los argumentos vienen como string (JSON), los limpiamos y parseamos
    if isinstance(raw_args, str):
        # Limpieza agresiva de ruidos comunes en modelos pequeños
        clean_args = raw_args.strip()
        if clean_args.endswith("}}}"): clean_args = clean_args[:-2]
        elif clean_args.endswith("}}"): clean_args = clean_args[:-1]
        
        try:
            args = json.loads(clean_args)
        except:
            # Si falla el parseo de un string que parece JSON, intentamos extraer el primer objeto
            obj_match = re.search(r"\{.*\}", clean_args, re.DOTALL)
            if obj_match:
                try:
                    args = json.loads(obj_match.group(0))
                except:
                    args = {"raw_input": raw_args}
            else:
                args = {"raw_input": raw_args}
    else:
        args = raw_args
        
    return {
        "name": name,
        "args": args if isinstance(args, dict) else {"value": args},
        "id": str(c.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
    }

def parse_phi4_tool_calls(response: AIMessage) -> list[dict[str, Any]]:
    """
    Extrae llamadas a herramientas (tool calls) del contenido del mensaje.
    """
    if response.tool_calls:
        results = [normalize_tool_call(tc) for tc in response.tool_calls]
        return [r for r in results if r is not None]

    content = response.content or ""
    if not content:
        return []

    for pattern in _TOOL_CALL_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        
        raw_str = match.group(1).strip()
        
        def try_parse(s: str) -> list[dict] | None:
            try:
                data = json.loads(s)
                return data if isinstance(data, list) else [data]
            except:
                return None

        # Intento 1: Tal cual viene
        calls = try_parse(raw_str)
        
        # Intento 2: Si no termina en ], intentamos cerrar el array y posibles objetos internos
        if not calls:
            clean_str = raw_str.strip()
            if clean_str.endswith("]"):
                clean_str = clean_str[:-1].strip()
            
            for suffix in ["]", "}]", "}}]"]:
                calls = try_parse(clean_str + suffix)
                if calls: break

        # Intento 3: Busqueda regresiva agresiva
        if not calls:
            for i in range(len(raw_str), 0, -1):
                sub = raw_str[:i].strip()
                for suffix in ["]", "}]", "}}]"]:
                    res = try_parse(sub + suffix)
                    if res:
                        calls = res
                        break
                if calls: break
        
        if not calls:
            continue

        try:
            results = [normalize_tool_call(c) for c in calls]
            return [r for r in results if r is not None]
        except Exception:
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
