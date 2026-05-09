import re

_PHI4_TOOL_SYSTEM_SUFFIX = (
    "\n\nYou have access to tools. If a user request requires a tool, respond ONLY with the tool call in this format: <|tool_call|>[{\"name\": \"tool_name\", \"arguments\": {...}}]<|/tool_call|>. "
    "If the request is a general question or doesn't need tools, respond with normal text WITHOUT any tool call tags. "
    "NEVER mix normal text and <|tool_call|> tags in the same response."
)

_TOOL_CALL_PATTERNS = [
    # Formato oficial con cierre
    re.compile(r"<\|tool_calls?\|>\s*(\[.*?\])\s*<\|/tool_calls?\|>", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\[.*?\])\s*<\|/tool_call\|>", re.DOTALL),
    # Formato oficial con posible corte (sin cierre)
    re.compile(r"<\|tool_calls?\|>\s*(\[.*)", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\[.*)", re.DOTALL),
    # Formatos alternativos comunes en Phi-4
    re.compile(r"functools\s*(\[.*)", re.DOTALL),
    # Solo bloque JSON si tiene estructura de herramienta evidente
    re.compile(r"(\[\s*\{\s*\"name\":\s*\"[^\"]+\".*\])", re.DOTALL),
]