import re

_PHI4_TOOL_SYSTEM_SUFFIX = (
    "\n\nYou are a tool-calling assistant. When a user asks a question that requires external data, you MUST respond with a tool call in the exact format: <|tool_call|>[{\"name\": \"tool_name\", \"arguments\": {...}}]<|/tool_call|>. "
    "DO NOT PROVIDE ANY PROSE, EXPLANATIONS, OR PREAMBLE. JUST THE TOOL CALL."
)

_TOOL_CALL_PATTERNS = [
    re.compile(r"<\|tool_calls?\|>\s*(\[.*?(?:\]|$))\s*(?:<\|/tool_calls?\|>|$)", re.DOTALL),
    re.compile(r"<\|tool_call\|>\s*(\[.*?(?:\]|$))\s*(?:<\|/tool_call\|>|$)", re.DOTALL),
    re.compile(r"functools\s*(\[.*?(?:\]|$))", re.DOTALL),
    # Captura mas agresiva de cualquier bloque que empiece con [ {
    re.compile(r"(\[\s*\{.*(?:\"name\"|\"arguments\").*)", re.DOTALL),
]