# PhiChat

Wrapper de LangChain para **Phi-4** (14B) y la familia Phi de Microsoft via Ollama, con soporte robusto y nativo de tool calling.

Phi-4 genera y reconoce tool calls y responses mediante etiquetas propietarias (`<|tool_call|>` y `<|tool_response|>`). `PhiChat` normaliza este protocolo automáticamente para que cualquier flujo LCEL, agente o grafo funcione de forma transparente.

---

## Instalacion

Requiere [Ollama](https://ollama.com) corriendo localmente.

```bash
ollama pull phi4
```

Dependencias (recomendado con `uv`):

```bash
# Inicializar y sincronizar dependencias
uv sync
```

O via pip:

```bash
pip install langchain-ollama langchain-core langchain-mcp-adapters mcp pydantic nest-asyncio
```

---

## Inicio rapido

```python
from PhiChat import ChatPhi
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Obtiene el clima de una ciudad."""
    return f"Soleado en {city}, 25 grados."

llm = ChatPhi()
chain = llm.bind_tools([get_weather])

response = chain.invoke("Como esta el clima en Madrid?")
print(response.content)
print(response.tool_calls)
```

---

## Modelos recomendados

| `phi4` (default) | 10 GB | Estado del arte, excelente razonamiento y herramientas |
| `phi4-mini`     | 4 GB  | Más rápido, ideal para tareas simples                |
| `phi3.5-mini`   | 4 GB  | Muy rápido, pero propenso a errores en herramientas  |

Cambiar de modelo:

```python
llm = ChatPhi(model="phi4:latest")
```

---

## Estructura del paquete

```text
PhiChat/
├── pyproject.toml      # Configuración del proyecto
├── PhiChat/            # Código fuente (Paquete)
│   ├── __init__.py
│   ├── model.py
│   ├── tools.py
│   ├── parsers.py
│   └── constants/
└── test/               # Suite de pruebas
```

---

## Compatibilidad LCEL

`ChatPhi` hereda de `ChatOllama` y es compatible con la interfaz estandar de LangChain.

### Pipe estandar

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente util."),
    ("human", "{input}")
])

chain = prompt | ChatPhi() | StrOutputParser()
result = chain.invoke({"input": "Hola"})
```

### bind_tools (LCEL completo)

`bind_tools` devuelve `_PhiBoundModel`, un `Runnable` propio que:

- Inyecta el system prompt y las definiciones JSON de las herramientas.
- Normaliza las tool calls al campo estándar `tool_calls` y filtra los tokens de control del stream.
- Re-inyecta el historial (llamadas y respuestas) para evitar bucles agénticos.
- Soporta herramientas asíncronas de forma nativa (ideal para servidores **MCP**).
- Soporta `.invoke()`, `.stream()`, `.batch()`, `.ainvoke()`, `.astream()`.

```python
llm = ChatPhi()
bound = llm.bind_tools([get_weather])

# invoke
response = bound.invoke("Clima en Roma?")

# stream
for chunk in bound.stream("Clima en Tokio?"):
    print(chunk.content, end="")

# async
response = await bound.ainvoke("Clima en Paris?")
```

### with_structured_output

```python
from pydantic import BaseModel, Field

class Ciudad(BaseModel):
    nombre: str = Field(description="Nombre de la ciudad")
    pais: str = Field(description="Pais donde esta la ciudad")

structured = ChatPhi().with_structured_output(Ciudad)
result = structured.invoke("Capital de Francia")
# result es un dict con las llaves del schema
```

---

## Tool Calling

### Definir una herramienta (estandar LangChain)

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="Nombre de la ciudad")

@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Obtiene el clima actual de una ciudad.
    Usa esta herramienta cuando el usuario pregunte por el clima o temperatura.
    """
    return f"Clima en {city}: soleado, 25 grados."
```

> **Nota**: La descripcion de la herramienta es lo mas importante para phi4-mini.
> Debe describir *cuando* usarla, no solo que hace.

### Loop agentico con run_tool_loop

Para flujos que requieren multiples pasos (LLM -> tool -> LLM):

```python
llm = ChatPhi()
response = llm.run_tool_loop(
    messages=[{"role": "user", "content": "Clima en Madrid y en Paris?"}],
    tools=[get_weather],
    verbose=True,       # imprime cada paso
    max_iterations=5
)
print(response.content)
```

### Crear herramientas con create_tool

```python
from PhiChat import create_tool

def buscar_producto(nombre: str) -> str:
    return f"Producto encontrado: {nombre}"

tool = create_tool(
    func=buscar_producto,
    description="Busca un producto en el catalogo. Usa cuando el usuario quiera encontrar un articulo."
)
```

---

## Integracion con LangGraph

`ChatPhi.bind_tools()` devuelve un `Runnable` compatible con `create_react_agent`:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatPhi(model="phi4:latest", temperature=0)
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=MemorySaver()
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Clima en Berlin?"}]},
    config={"configurable": {"thread_id": "1"}}
)
print(result["messages"][-1].content)
```

---

## Configuracion

| Parametro              | Default              | Descripcion                                             |
| ---------------------- | -------------------- | ------------------------------------------------------- |
| `model`              | `phi4-mini:latest` | Modelo Ollama a usar                                    |
| `temperature`        | `0.7`              | Temperatura de muestreo                                 |
| `num_ctx`            | `8192`             | Tokens de contexto. Necesario aumentar con muchas tools |
| `keep_alive`         | `0`                | Tiempo en segundos para mantener el modelo en memoria   |
| `tool_system_suffix` | (interno)            | Sufijo inyectado en el system prompt para activar tools |

```python
llm = ChatPhi(
    model="phi4:latest",
    temperature=0,
    num_ctx=16384,
    keep_alive=300
)
```

---

## Por que es necesario PhiChat

| Comportamiento                                           | Solución en PhiChat                                    |
| -------------------------------------------------------- | ------------------------------------------------------ |
| Error 400 en Ollama (Native Tooling)                     | Bypass nativo e inyección manual de schemas en prompt  |
| Tool calls en `content` en lugar de `tool_calls`         | `parse_phi4_tool_calls()` extrae y normaliza el JSON   |
| Olvida su propio historial de herramientas               | Re-inyección automática de `<|tool_call|>` en historia |
| Resultados de herramientas ignorados                     | Inyección de etiquetas `<|tool_response|>` nativas     |
| Tokens `<|tool_call|>` aparecen en el stream             | Filtro robusto de prefijos y etiquetas en el stream    |
| Salida estructurada inconsistente                        | Override de `with_structured_output` con modo JSON     |
| Herramientas MCP asíncronas                              | Soporte nativo para ejecución async en `run_tool_loop` |

## Integración con MCP (Model Context Protocol)

PhiChat es totalmente compatible con servidores MCP a través de `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from PhiChat import ChatPhi

async def main():
    # Conectar a servidores MCP
    client = MultiServerMCPClient({"my_server": {"transport": "stdio", "command": "python", "args": ["server.py"]}})
    mcp_tools = await client.get_tools()

    # Integrar con PhiChat
    llm = ChatPhi(model="phi4")
    agent = llm.bind_tools(mcp_tools)
    
    # ¡Listo! Phi-4 usará las herramientas del servidor MCP
    res = await agent.ainvoke("Usa la herramienta del servidor MCP para...")
```

---

## Ingeniería y Mejores Prácticas

El proyecto `PhiChat` sigue estándares estrictos de ingeniería de software para garantizar la mantenibilidad y robustez:

- **Estándares Python**: Adherencia a PEP 8 (estilo), PEP 484 (tipado estático) y PEP 257 (documentación).
- **Validación Continua**: Es obligatorio que cualquier cambio pase la suite de pruebas completa (`uv run pytest`) para asegurar la integridad del protocolo de etiquetas de Phi-4.
- **Tipado Estricto**: Uso de `Type Hints` en todas las funciones y clases para facilitar el análisis estático.
- **Resiliencia**: Implementación de patrones de reintento y manejo de excepciones específicas para interactuar con modelos locales de forma segura.
- **Ciclo de Vida (SDLC)**: Proceso definido desde el análisis del impacto en el parsing hasta la validación formal de integración.

---

## Testing

La suite de pruebas garantiza la integridad de los protocolos de comunicación y la ejecución agéntica.

```bash
# Ejecutar todos los tests
uv run pytest

# Ejecutar un test específico (ej. los parsers)
uv run pytest test/test_parsers.py
```

Cubre: Parsing nativo de protocolos Phi-4, chat simple, selección de herramientas, salida estructurada, streaming con filtrado de tokens y ejecución de herramientas asíncronas (MCP).
