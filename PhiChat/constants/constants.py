import re

_PHI4_TOOL_SYSTEM_SUFFIX = (
    "\n\nYou have access to external tools. "
    "If the user's request requires using a tool to be fulfilled, you MUST use a tool call in this exact format: <|tool_call|>[{\"name\": \"tool_name\", \"arguments\": {...}}]<|/tool_call|>. "
    "If the request is a general question that you can answer without tools, respond with normal text. "
    "NEVER refuse to use an available tool if it is necessary for the task."
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