"""
PhiChat
-------
Wrapper LangChain para modelos de la familia Phi-4 via Ollama con soporte robusto de tool calling.
"""

from PhiChat.models import ChatPhi
from PhiChat.tools import create_tool, run_tool_loop, arun_tool_loop
from PhiChat.utils import parse_phi4_tool_calls, inject_tool_system_message
from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX, _TOOL_CALL_PATTERNS

__all__ = [
    "ChatPhi",
    "create_tool",
    "run_tool_loop",
    "arun_tool_loop",
    "parse_phi4_tool_calls",
    "inject_tool_system_message",
    "_PHI4_TOOL_SYSTEM_SUFFIX",
    "_TOOL_CALL_PATTERNS",
]
