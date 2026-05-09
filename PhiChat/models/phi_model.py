from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, PrivateAttr

from PhiChat.constants import _PHI4_TOOL_SYSTEM_SUFFIX, _TOOL_CALL_PATTERNS
from PhiChat.utils.parsers import inject_tool_system_message, parse_phi4_tool_calls


class ChatPhi(BaseChatModel):
    """
    Wrapper de BaseChatModel para Phi-4 vía Ollama con soporte de Tool Calling robusto.
    
    Implementa la normalización de etiquetas propietarias de Phi-4 (<|tool_call|>)
    al formato estándar de LangChain, facilitando su uso en agentes y grafos.
    """

    model: str = Field(default="phi4")
    temperature: float = Field(default=0.7)
    num_ctx: int = Field(default=8192)
    keep_alive: int = Field(default=0)
    tool_system_suffix: str = Field(default=_PHI4_TOOL_SYSTEM_SUFFIX)

    # Ejemplos few-shot opcionales; se setean vía bind_tools(examples=...)
    _examples: str | None = PrivateAttr(default=None)
    # Tools bindeadas actualmente
    _bound_tools: list[BaseTool] = PrivateAttr(default_factory=list)
    # Delegado Ollama (no serializado por Pydantic)
    _ollama: ChatOllama = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Inicializa el delegado ChatOllama con los parámetros del modelo."""
        self._ollama = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
        )

    @property
    def _llm_type(self) -> str:
        return "chat-phi4"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }

    def _patch_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Prepara el historial para Phi-4: re-inyecta tool_calls y tool_responses."""
        patched_messages = []
        from langchain_core.messages import ToolMessage, HumanMessage
        
        # 1. Normalizar el historial para que Phi-4 reconozca llamadas y respuestas
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # Re-inyectar bloque <|tool_call|> si no está ya en el texto
                if not any(tag in (msg.content or "") for tag in ("<|tool_call", "functools[")):
                    # IMPORTANTE: Phi-4 espera 'arguments' en lugar de 'args'
                    tc_phi4 = []
                    for tc in msg.tool_calls:
                        tc_phi4.append({
                            "name": tc["name"],
                            "arguments": tc["args"]
                        })
                    tool_json = json.dumps(tc_phi4, ensure_ascii=False)
                    new_content = f"<|tool_call|>{tool_json}<|/tool_call|>"
                    if msg.content:
                        new_content = f"{msg.content}\n{new_content}"
                    patched_messages.append(AIMessage(content=new_content, tool_calls=msg.tool_calls))
                else:
                    patched_messages.append(msg)
            
            elif isinstance(msg, ToolMessage):
                # Usamos una instrucción más clara para que Phi-4 reconozca el resultado
                content = f"<|tool_response|>{msg.content}<|/tool_response|>\nCon base en el resultado anterior, responde al usuario."
                patched_messages.append(HumanMessage(content=content))
            
            else:
                patched_messages.append(msg)

        if not self._bound_tools:
            return patched_messages

        # 2. Generar definiciones JSON de las herramientas actuales
        tool_defs = [convert_to_openai_tool(t) for t in self._bound_tools]
        tools_json = json.dumps(tool_defs, indent=2, ensure_ascii=False)

        content = f"{self.tool_system_suffix}\n\n[AVAILABLE TOOLS]\n{tools_json}"

        if self._examples:
            content += f"\n\n[EJEMPLOS]\n{self._examples}"
        
        return inject_tool_system_message(patched_messages, content)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Delega en ChatOllama y normaliza tool calls antes de devolver."""
        effective_stop = list(stop or [])
        if self._bound_tools:
            effective_stop += ["<|/tool_call|>", "<|/tool_calls|>"]

        # _generate devuelve un ChatResult
        chat_result = self._ollama._generate(
            self._patch_messages(messages),
            stop=effective_stop or None,
            run_manager=None,
        )
        response = chat_result.generations[0].message
        normalized = self._normalize_static(response)
        return ChatResult(generations=[ChatGeneration(message=normalized)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream filtrando tokens propietarios y emitiendo tool_call_chunks al final."""
        effective_stop = list(stop or [])
        if self._bound_tools:
            effective_stop += ["<|/tool_call|>", "<|/tool_calls|>"]

        raw_chunks: list[AIMessageChunk] = []
        full_content = ""
        in_tool_call = False

        for chunk in self._ollama._stream(
            self._patch_messages(messages),
            stop=effective_stop or None,
            run_manager=None,
        ):
            # En _stream nativo, el chunk es de tipo ChatGenerationChunk
            msg_chunk = chunk.message
            raw_chunks.append(msg_chunk)
            content = msg_chunk.content or ""
            full_content += content

            # Detección robusta: si ya detectamos que estamos en una tool call, paramos de emitir.
            if not in_tool_call:
                # Comprobación de contenido completo para detectar etiquetas en medio de un chunk
                if any(tag in full_content for tag in ("<|tool_call", "<|tool_calls", "functools[")):
                    in_tool_call = True
            
            if not in_tool_call:
                # Filtrado preventivo de prefijos: si el chunk actual o el final del contenido 
                # acumulado parecen el inicio de una etiqueta, no emitimos.
                potential_tags = ("<", "<|", "<|t", "<|to", "<|too", "<|tool", "func", "functools")
                if not any(full_content.endswith(p) for p in potential_tags):
                    yield chunk

        # Post-proceso final: detecta tool calls en el contenido completo acumulado
        normalized = self._normalize_static(AIMessage(content=full_content))
        if normalized.tool_calls:
            final_msg = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"]),
                        "id": tc["id"],
                        "index": i,
                    }
                    for i, tc in enumerate(normalized.tool_calls)
                ],
            )
            yield ChatGenerationChunk(message=final_msg)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream asíncrono nativo filtrando tokens propietarios."""
        effective_stop = list(stop or [])
        if self._bound_tools:
            effective_stop += ["<|/tool_call|>", "<|/tool_calls|>"]

        raw_chunks: list[AIMessageChunk] = []
        full_content = ""
        in_tool_call = False

        async for chunk in self._ollama._astream(
            self._patch_messages(messages),
            stop=effective_stop or None,
            run_manager=None,
        ):
            # En _astream nativo, el chunk es de tipo ChatGenerationChunk
            msg_chunk = chunk.message
            raw_chunks.append(msg_chunk)
            content = msg_chunk.content or ""
            full_content += content

            if not in_tool_call:
                if any(tag in full_content for tag in ("<|tool_call", "functools[")):
                    in_tool_call = True
            
            if not in_tool_call:
                potential_tags = ("<", "<|", "<|t", "<|to", "<|too", "<|tool", "func", "functools")
                if not any(full_content.endswith(p) for p in potential_tags):
                    yield chunk

        normalized = self._normalize_static(AIMessage(content=full_content))
        if normalized.tool_calls:
            final_msg = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"]),
                        "id": tc["id"],
                        "index": i,
                    }
                    for i, tc in enumerate(normalized.tool_calls)
                ],
            )
            yield ChatGenerationChunk(message=final_msg)

    def bind_tools(
        self,
        tools: Sequence[BaseTool | Callable | dict],
        *,
        tool_choice: str | None = None,
        examples: str | None = None,
        **kwargs: Any,
    ) -> "ChatPhi":
        """Bindea herramientas y devuelve una nueva instancia de ChatPhi con tools activas.

        A diferencia del enfoque anterior (_PhiBoundModel), devuelve el mismo tipo
        ChatPhi, lo que garantiza compatibilidad total con create_react_agent, ToolNode,
        y cualquier componente que inspeccione el tipo del modelo.

        Args:
            tools: Lista de herramientas (BaseTool, @tool, o dicts de schema).
            tool_choice: No recomendado para phi4. Ignorado internamente.
            examples: Ejemplos few-shot para mejorar la precisión del modelo.
            **kwargs: Ignorados, presentes por compatibilidad.

        Returns:
            Nueva instancia de ChatPhi con las tools y ejemplos configurados.
        """
        # Normaliza a BaseTool si vienen como callables o dicts
        normalized: list[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                normalized.append(t)
            elif callable(t):
                # @tool decorados u otros callables con .name/.description
                normalized.append(t)  # type: ignore[arg-type]

        # Crea una nueva instancia para no mutar el estado del modelo original
        bound = ChatPhi(
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
            tool_system_suffix=self.tool_system_suffix,
        )
        bound._bound_tools = normalized
        bound._examples = examples or self._examples

        # IMPORTANTE: NO llamamos a self._ollama.bind_tools(tools) para evitar el error 400
        # El esquema se inyecta manualmente via _patch_messages.
        bound._ollama = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
        )

        return bound

    def run_tool_loop(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> AIMessage:
        """Ejecuta el loop agéntico completo: LLM → tools → LLM."""
        from PhiChat.tools import run_tool_loop

        return run_tool_loop(
            messages=messages,
            tools=tools,
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
            tool_system_suffix=self.tool_system_suffix,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    async def arun_tool_loop(
        self,
        messages: list[BaseMessage],
        tools: list[BaseTool],
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> AIMessage:
        """
        Ejecuta el loop agéntico de forma asíncrona (LLM -> tools -> LLM).
        """
        from PhiChat.tools import arun_tool_loop

        return await arun_tool_loop(
            messages=messages,
            tools=tools,
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
            tool_system_suffix=self.tool_system_suffix,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def with_structured_output(
        self,
        schema: type[BaseModel] | dict,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Extrae salida estructurada forzando formato JSON y schemas en el prompt."""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_json = schema.model_json_schema()
            parser = JsonOutputParser(pydantic_object=schema)
        else:
            schema_json = schema
            parser = JsonOutputParser()

        def _invoke_with_schema(messages: Any) -> Any:
            schema_instruction = (
                f"\n\nResponde ÚNICAMENTE con un objeto JSON válido que siga este schema:\n"
                f"{json.dumps(schema_json, ensure_ascii=False)}\n"
                f"No incluyas explicaciones, solo el JSON."
            )
            
            # Usamos el delegado con format="json"
            self._ollama.format = "json"
            try:
                patched = inject_tool_system_message(messages, schema_instruction)
                response = self.invoke(patched)
                # Limpiar posibles bloques markdown si el modelo los pone a pesar de format="json"
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(content)
                if include_raw:
                    return {"raw": response, "parsed": parsed}
                return parsed
            finally:
                self._ollama.format = None # Restauramos

        return RunnableLambda(_invoke_with_schema)

    @staticmethod
    def _normalize_static(response: AIMessage) -> AIMessage:
        """Extrae tool calls de content y limpia el contenido residual."""
        tool_calls = parse_phi4_tool_calls(response)
        if not tool_calls:
            return response

        clean_content = response.content or ""
        for pattern in _TOOL_CALL_PATTERNS:
            clean_content = pattern.sub("", clean_content)

        return AIMessage(
            content=clean_content.strip(),
            tool_calls=tool_calls,
            response_metadata=response.response_metadata,
            id=response.id,
        )