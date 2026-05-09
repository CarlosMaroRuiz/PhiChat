from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX

def _extract_inline_tool_calls(content: str) -> list[dict]:
    """Extrae tool calls cuando phi4-mini las escribe como bloques JSON numerados en el texto."""
    from PhiChat.utils.parsers import normalize_tool_call
    pattern = re.compile(r'\[\s*\{.*?"(?:name|type|function)".*?\}\s*\]', re.DOTALL)
    result: list[dict[str, Any]] = []

    for match in pattern.findall(content):
        try:
            raw = json.loads(match)
            calls = raw if isinstance(raw, list) else [raw]
            for c in calls:
                normalized = normalize_tool_call(c)
                if normalized:
                    result.append(normalized)
        except (json.JSONDecodeError, AttributeError):
            continue

    return result


def run_tool_loop(
    messages: list[BaseMessage],
    tools: list[BaseTool],
    *,
    model: str = "phi4",
    temperature: float = 0.7,
    num_ctx: int = 8192,
    keep_alive: int = 0,
    tool_system_suffix: str = _PHI4_TOOL_SYSTEM_SUFFIX,
    examples: str | None = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> AIMessage:
    """Ejecuta el loop agentico: LLM → tool calls → resultados → LLM."""
    from PhiChat.models.phi_model import ChatPhi

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
    model: str = "phi4",
    temperature: float = 0.7,
    num_ctx: int = 8192,
    keep_alive: int = 0,
    tool_system_suffix: str = _PHI4_TOOL_SYSTEM_SUFFIX,
    examples: str | None = None,
    max_iterations: int = 10,
    verbose: bool = False,
) -> AIMessage:
    """Version asincrona nativa del loop agentico."""
    from PhiChat.models.phi_model import ChatPhi

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
                    tool_result = await selected_tool.ainvoke(call_args)
                    if verbose: print(f"{tool_result}")
                except Exception as e:
                    tool_result = f"Error en '{tool_name}': {e}"
                    if verbose: print(f"ERROR: {e}")
            
            current_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

    return AIMessage(content="[Limite de iteraciones alcanzado]")
