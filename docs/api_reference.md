# PhiChat — Referencia de API (API Reference)

Este documento detalla todas las clases, funciones y módulos que componen la librería **PhiChat**. Es el mapa de referencia para cualquier desarrollo o modificación interna.

---

## Módulo: `PhiChat.model`
Contiene el núcleo del wrapper de LangChain para Ollama y Phi-4.

### Clase `ChatPhi`
Hereda de `BaseChatModel` de LangChain. Sobrescribe los métodos principales para inyectar y extraer el uso de herramientas de un modelo puramente conversacional.

- **`model_post_init(self, __context: Any) -> None`**
  - **Uso**: Inicializa internamente el objeto `ChatOllama` delegado.
- **`_patch_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]`**
  - **Uso**: Transforma el historial de mensajes al formato esperado por Phi-4. Re-inyecta las respuestas de las herramientas con la etiqueta `<|tool_response|>` y añade el JSON string de las herramientas disponibles al `SystemMessage`.
- **`bind_tools(self, tools: Sequence, *, examples: str = None) -> ChatPhi`**
  - **Uso**: Enlaza las herramientas al modelo. **Importante**: No delega en `ChatOllama.bind_tools` para evitar el error `400 Bad Request`. Devuelve una nueva instancia inmutable.
- **`invoke(self, messages) -> AIMessage`**
  - **Uso**: Ejecuta el modelo bloqueando el hilo de ejecución. Extrae los "tool calls" mediante `_normalize_static`.
- **`stream(self, messages) -> Iterator[ChatGenerationChunk]`**
  - **Uso**: Devuelve los tokens uno a uno.
  - **Detalle Técnico**: Mantiene un búfer para detectar si el modelo empieza a escupir una etiqueta `<|tool_call|>`. Si la detecta, detiene la emisión de tokens para no mostrar código técnico al usuario.
- **`run_tool_loop(self, messages, tools, max_iterations=10, verbose=False) -> AIMessage`**
  - **Uso**: Método auxiliar para ejecutar un ciclo reactivo (ReAct) LLM -> Tools -> LLM sincrónico.
- **`arun_tool_loop(self, ...)`**
  - **Uso**: Versión asíncrona (awaitable) del `run_tool_loop`.
- **`with_structured_output(self, schema: BaseModel | dict) -> Runnable`**
  - **Uso**: Fuerza al modelo a responder estrictamente en un formato JSON definido por el schema proporcionado.
- **`_normalize_static(response: AIMessage) -> AIMessage`**
  - **Uso**: Método estático que procesa la respuesta en bruto, elimina los tokens de control mediante las regex de `constants.py` y puebla el atributo `tool_calls` estandarizado de LangChain.

---

## Módulo: `PhiChat.parsers`
Contiene la "magia" de resiliencia: la capacidad de arreglar lo que el modelo rompe.

### Función `parse_phi4_tool_calls(response: AIMessage) -> list[dict]`
- **Uso**: Búsqueda en el contenido generado de las llamadas a herramientas. 
- **Resiliencia (Smart & Surgical JSON Repair)**:
  - Intento 1: Parsea el JSON directamente.
  - Intento 2: Si el modelo omitió corchetes o llaves de cierre, hace limpieza de los bordes del texto y prueba añadir sufijos (`]`, `}]`, `}}]`).
  - Intento 3: Si el JSON tiene basura al final, retrocede carácter por carácter intentando buscar un punto válido donde cerrar los objetos.
- **Retorno**: Una lista de `dict` con el formato: `{"name": str, "args": dict, "id": str}`.

### Función `inject_tool_system_message(messages: list[BaseMessage], suffix: str) -> list[BaseMessage]`
- **Uso**: Añade las instrucciones obligatorias de formateo de herramientas al `SystemMessage`. Si no hay `SystemMessage` en el historial, crea uno en la posición `0`.

---

## Módulo: `PhiChat.tools`
Manejo, conversión y orquestación de herramientas y bucles de ejecución.

### Función `create_tool(func, name, description, args_schema, return_direct) -> BaseTool`
- **Uso**: Creador optimizado. Convierte cualquier función Python en una herramienta compatible (una `StructuredTool` de LangChain). Si no se proveen nombre o descripción, los toma de los metadatos de la función (`__name__`, `__doc__`).

### Función `_extract_inline_tool_calls(content: str) -> list[dict]`
- **Uso**: Mecanismo de contingencia. Extrae llamadas si el modelo en lugar de usar un array global, enumera el JSON (`1. [{"name":...}]`).
- **Retorno**: Lista de herramientas parseadas.

### Función `run_tool_loop` y `arun_tool_loop`
- **Uso**: Motores de ejecución agéntica sincrónica y asincrónica respectivamente.
- **Lógica**: Invocan al modelo, detectan si devolvió `tool_calls`, ejecutan de manera segura la función Python correspondiente, capturan su `stdout`/`return` y devuelven el resultado al modelo como un `ToolMessage`. Se repite hasta alcanzar el máximo de iteraciones.

---

## Módulo: `PhiChat.constants`
### `_PHI4_TOOL_SYSTEM_SUFFIX`
- Cadena de texto inyectada como prompt del sistema para ordenar al modelo que devuelva el formato estricto `<|tool_call|>[...]`.

### `_TOOL_CALL_PATTERNS`
- Lista de expresiones regulares compiladas (`re.compile`).
- **Importancia Crítica**: Incluyen patrones que permiten extraer arrays incompletos (e.g. JSONs que nunca cerraron) para entregarlos al algoritmo de *Surgical Repair* en `parsers.py`.
