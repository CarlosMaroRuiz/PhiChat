# ChatPhi: Integración con flujos LangChain

ChatPhi hereda de `BaseChatModel` y es compatible con cualquier componente del ecosistema
LangChain. Este documento describe cómo integrarlo en los patrones más comunes, optimizado para **Phi-4**.

---

## Estructura del paquete

```
PhiChat/
├── __init__.py        # Exporta ChatPhi, create_tool, run_tool_loop y helpers
├── model.py           # Clase ChatPhi (ChatOllama + fixes phi4)
├── parsers.py         # parse_phi4_tool_calls, inject_tool_system_message
├── tools.py           # create_tool, run_tool_loop, arun_tool_loop
└── constants/
    ├── __init__.py    # Exporta las constantes del subpaquete
    └── constants.py   # _PHI4_TOOL_SYSTEM_SUFFIX, _TOOL_CALL_PATTERNS
```

---

| Problema en Phi-4                             | Solución en ChatPhi                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------------ |
| Ollama da Error 400 al usar `tools` nativo  | `bind_tools` realiza un bypass e inyecta los schemas manualmente en el prompt      |
| El modelo olvida sus propias llamadas pasadas | `_patch_messages` re-inyecta bloques `<|tool_call|>` en el historial                 |
| Las respuestas de herramientas son ignoradas  | Se encapsulan en etiquetas `<|tool_response|>` nativas                               |
| Los tokens de control ensucian el stream      | El filtro de streaming robusto bloquea prefijos fragmentados (ej. `<|tool`)          |
| `with_structured_output` impreciso          | Override personalizado que usa `format="json"` de Ollama con inyección de esquema |

> **Punto Crítico**: La descripción de la tool es lo único que Phi-4 lee para decidir si usarla. Debe ser clara, en lenguaje natural, y mencionar **cuándo** usarla, no solo qué hace.

---

## Importación

```python
# Importación directa del paquete (recomendado)
from PhiChat import ChatPhi, create_tool, run_tool_loop

# O por módulo específico
from PhiChat.model import ChatPhi
from PhiChat.tools import create_tool, run_tool_loop
from PhiChat.parsers import parse_phi4_tool_calls
```

---

## 1. Pipe básico (LCEL)

El patrón más simple. `ChatPhi` se usa como cualquier `ChatModel` en una chain.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PhiChat import ChatPhi

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en {tema}."),
    ("human", "{pregunta}"),
])

chain = prompt | ChatPhi() | StrOutputParser()

result = chain.invoke({
    "tema": "física cuántica",
    "pregunta": "Explica el entrelazamiento cuántico en 2 líneas.",
})
```

---

## 2. Tool calling con bind_tools

### Definir tools con @tool

La descripción de cada tool es crítica. Phi-4 decide si usarla basándose exclusivamente en ese texto.

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from PhiChat import ChatPhi

@tool
def convertir_moneda(monto: float, de: str, a: str) -> str:
    """
    Convierte un monto entre divisas.
    Úsala cuando el usuario pregunte sobre conversiones de moneda o tipos de cambio.
    """
    tasas = {"USD": 1.0, "MXN": 17.2, "EUR": 0.92}
    resultado = monto * (tasas[a] / tasas[de])
    return f"{monto} {de} = {resultado:.2f} {a}"

llm = ChatPhi(temperature=0.3)
chain = llm.bind_tools([convertir_moneda])

result = chain.invoke([HumanMessage(content="¿Cuánto es 500 MXN en EUR?")])
# result.tool_calls -> [{"name": "convertir_moneda", "args": {...}, "id": "..."}]
```

### Definir tools con args_schema (Recomendado)

Phi-4 genera argumentos más precisos cuando el esquema tiene `Field(description=...)` explícito en cada campo.

```python
from pydantic import BaseModel, Field
from PhiChat import ChatPhi, create_tool

class BusquedaInput(BaseModel):
    query: str = Field(description="Término a buscar en la base de datos")
    limite: int = Field(default=5, description="Número máximo de resultados a retornar")

def buscar_productos(query: str, limite: int = 5) -> str:
    return f"Resultados para '{query}' (max {limite})"

tool_busqueda = create_tool(
    func=buscar_productos,
    description=(
        "Busca productos en el catálogo por nombre o categoría. "
        "Úsala cuando el usuario pregunte por disponibilidad o precios."
    ),
    args_schema=BusquedaInput,
)

chain = ChatPhi().bind_tools([tool_busqueda])
```

---

## 3. Mejores Prácticas de Ingeniería (Verificadas)

Para garantizar una librería de calidad profesional, sigue estos patrones validados en LangChain v0.3:

### A. Manejo de Errores y Auto-Corrección
No permitas que las herramientas lancen excepciones que rompan el flujo. Captura el error y devuélvelo como un mensaje para que Phi-4 pueda razonar y corregir su llamada.

```python
@tool
def mi_herramienta_robusta(param: int) -> str:
    """Ejecuta una acción sensible."""
    try:
        # Lógica de la herramienta
        return "Éxito"
    except Exception as e:
        # Devolver el error al modelo para que intente otra estrategia
        return f"Error en la herramienta: {str(e)}. Intenta con un valor diferente."
```

### B. Validación de Esquemas con Pydantic
Usa validadores de Pydantic para capturar errores de tipo antes de que lleguen a la lógica de negocio.

```python
from pydantic import BaseModel, Field, field_validator

class CalculoInput(BaseModel):
    valor: int = Field(description="Un número positivo")

    @field_validator("valor")
    @classmethod
    def mayor_que_cero(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("El valor debe ser mayor que cero")
        return v
```

### C. Idempotencia y Transacciones Atómicas
Diseña herramientas que realicen operaciones atómicas. Si un agente reintenta una llamada debido a un timeout, la herramienta debe ser capaz de manejarlo sin duplicar efectos secundarios.

---

## 4. Orquestación con LangGraph: Conceptos Avanzados

### A. Super-steps y Tolerancia a Fallos
LangGraph organiza la ejecución en "super-steps" (ticks del grafo). 
- **Checkpoints**: Se guarda un `StateSnapshot` al final de cada super-step.
- **Task-level Writes**: Si un nodo falla dentro de un super-step paralelo, LangGraph guarda los resultados de los nodos que sí terminaron. Al reanudar, **solo se re-ejecutan los nodos fallidos**, optimizando el uso de tokens y tiempo.

### B. Estructura de un StateSnapshot
Cuando llamas a `agent.get_state(config)`, obtienes un objeto con:
- `values`: Valores actuales del estado (mensajes, variables).
- `next`: Tupla con los nombres de los nodos que se ejecutarán a continuación (vacía si el grafo terminó).
- `metadata`: Incluye el `source` (input, loop, update) y el `step` (contador de super-steps).
- `parent_config`: Referencia al checkpoint anterior, permitiendo navegar hacia atrás.

### C. Time Travel y Replay
Puedes re-ejecutar el grafo desde cualquier punto pasado utilizando un `checkpoint_id`:

```python
# Buscar un checkpoint específico en el historial
history = list(agent.get_state_history(config))
old_checkpoint = history[5] # Por ejemplo, 5 pasos atrás

# Reanudar desde ese punto
agent.invoke(None, old_checkpoint.config)
```

---

## 5. Mejores Prácticas para la Interfaz de Usuario (UI)
Si estás construyendo un chat basado en `PhiChat`, sigue estas guías para renderizar herramientas:

1. **Estados del Ciclo de Vida**: Maneja siempre los tres estados: `pending` (ejecutando), `completed` (éxito) y `error` (fallo).
2. **Streaming Reactivo**: Usa el `tool_call_chunks` emitido por `ChatPhi` para mostrar un "Loading Card" instantáneo antes de que la herramienta termine de ejecutarse.
3. **Parseo Seguro**: Los resultados de las herramientas llegan como strings. Usa bloques `try/except` al parsear JSON en el frontend y ofrece una vista genérica (ej. JSON colapsable) si no tienes un componente visual específico para esa herramienta.

---

## 6. Salida estructurada

Override del método base. Usa `format="json"` de Ollama internamente e inyecta el esquema en el system prompt de forma automática.

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from PhiChat import ChatPhi

class Resumen(BaseModel):
    titulo: str = Field(description="Título del texto en max. 5 palabras")
    ideas_clave: list[str] = Field(description="Lista de 3 ideas principales")
    sentimiento: str = Field(description="Positivo, Negativo o Neutro")
    puntaje: float = Field(description="Puntaje de 0.0 a 10.0")

llm = ChatPhi().with_structured_output(Resumen)

result = llm.invoke([HumanMessage(content="Analiza este texto: ...")])
# result -> Resumen(titulo="...", ideas_clave=[...], ...)
```

---

## 4. Loop agéntico sin AgentExecutor

Útil para ejecutar múltiples llamadas a herramientas de forma secuencial sin la sobrecarga de un motor de agentes completo.

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from PhiChat import ChatPhi

@tool
def buscar_producto(nombre: str) -> str:
    """Busca productos en el catálogo."""
    return f"Detalles de {nombre}..."

llm = ChatPhi(temperature=0.2)
messages = [HumanMessage(content="¿Tienen laptop?")]

# Loop: LLM -> tool_call -> resultado -> LLM -> respuesta final
result = llm.run_tool_loop(
    messages=messages,
    tools=[buscar_producto],
    verbose=True
)
```

> **Nota**: Para aplicaciones asíncronas (web/servidores), utiliza `await llm.arun_tool_loop(...)`.

---

## 5. Streaming

El streaming filtra automáticamente los tokens de control y emite chunks limpios.

```python
llm = ChatPhi()

for chunk in llm.stream("Explica la relatividad"):
    print(chunk.content, end="", flush=True)
```

---

## 6. Integración con LangGraph (Recomendado)

PhiChat es totalmente compatible con `create_react_agent` y persistencia de estado.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from PhiChat import ChatPhi

llm = ChatPhi(temperature=0)
memory = MemorySaver()
agent = create_react_agent(llm, tools=[mi_tool], checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}
result = agent.invoke({"messages": [("user", "Hola")]}, config)

# Inspección y Manipulación del Estado
# Recupera el estado actual de la conversación
state = agent.get_state(config)
print(state.values["messages"])

# Actualizar el estado manualmente (inyectar información)
agent.update_state(config, {"messages": [HumanMessage(content="Nota: El sistema está en mantenimiento")]})
```

### Human-in-the-Loop (Interrupción y Aprobación)
Puedes interrumpir el grafo antes de ejecutar herramientas críticas para validación humana.

```python
agent = create_react_agent(
    llm, 
    tools=[herramientas_sensibles], 
    checkpointer=memory,
    interrupt_before=["tools"] # Se detiene justo antes de llamar a la herramienta
)
```

---

## 7. Integración con Model Context Protocol (MCP)

PhiChat soporta herramientas remotas de servidores MCP de forma nativa.

```python
from PhiChat import ChatPhi
from langchain_mcp_adapters.client import MultiServerMCPClient

async def run_mcp():
    async with MultiServerMCPClient({"db": {"transport": "stdio", "command": "python", "args": ["server.py"]}}) as client:
        tools = await client.get_tools()
        llm = ChatPhi()
        # El loop agéntico maneja la ejecución asíncrona de MCP automáticamente
        res = await llm.arun_tool_loop(messages=[...], tools=tools)
```

---

## Configuración del modelo

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `model` | `phi4` | Modelo Ollama a usar |
| `temperature` | `0.7` | Creatividad (0.0 para herramientas) |
| `num_ctx` | `8192` | Ventana de contexto |
| `keep_alive` | `0` | Tiempo de retención en RAM |

```python
llm = ChatPhi(model="phi4:latest", temperature=0, num_ctx=16384)
```
