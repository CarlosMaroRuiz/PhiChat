# PhiChat

Wrapper de LangChain para **Phi-4 (14B)** y la familia Phi de Microsoft vía Ollama, con soporte robusto y nativo de tool calling.

Phi-4 genera y reconoce tool calls y responses mediante etiquetas propietarias (`<|tool_call|>` y `<|tool_response|>`). **PhiChat** normaliza este protocolo automáticamente para que cualquier flujo LCEL, agente o grafo funcione de forma transparente.

## Instalación

Requiere [Ollama](https://ollama.com/) corriendo localmente.

```bash
ollama pull phi4
```

### Vía Pip
```bash
pip install phichat
```

> [!NOTE]
> Aunque está optimizado para **Phi-4**, este wrapper es compatible con otros modelos de la familia Phi (ej. `phi4-mini`, `phi3.5-mini`) simplemente cambiando el parámetro `model` al inicializar.

### Con uv (Recomendado)
```bash
uv add phichat
```

## Inicio rápido

```python
from PhiChat import ChatPhi
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Obtiene el clima de una ciudad."""
    return f"Soleado en {city}, 25 grados."

llm = ChatPhi()
chain = llm.bind_tools([get_weather])

response = chain.invoke("¿Cómo está el clima en Madrid?")
print(response.content)
print(response.tool_calls)
```

> [!TIP]
> Para una guía detallada sobre cómo integrar PhiChat con **LangGraph**, **MCP** y mejores prácticas de ingeniería, consulta nuestra [Guía de Integración Completa](INTEGRATION_GUIDE.md).

## Modelos recomendados

| Modelo | Tamaño | Características |
| :--- | :--- | :--- |
| **phi4** (default) | 10 GB | Estado del arte, excelente razonamiento y herramientas |
| **phi4-mini** | 4 GB | Más rápido, ideal para tareas simples |
| **phi3.5-mini** | 4 GB | Muy rápido, pero propenso a errores en herramientas |

Para cambiar de modelo:
```python
llm = ChatPhi(model="phi4-mini:latest")
```

## Características principales

*   **Bypass de Ollama 400**: Soluciona el error de "Native Tooling" mediante inyección manual de esquemas.
*   **Normalización de Tool Calls**: Extrae y convierte el JSON de Phi-4 al formato estándar `tool_calls`.
*   **Ejecución Paralela**: `arun_tool_loop` utiliza concurrencia nativa (`asyncio.gather`) para invocar múltiples herramientas simultáneamente.
*   **Soporte return_direct**: Soporte nativo para herramientas que devuelven resultados crudos sin pasar de nuevo por el modelo.
*   **Memoria Agéntica**: Re-inyección automática del historial para evitar bucles infinitos.
*   **Salida Estructurada**: Soporte nativo para `with_structured_output` en modo JSON.
*   **Streaming**: Filtrado de tokens de control (`<|tool_call|>`) para streams limpios.

## Integración con LangGraph

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from PhiChat import ChatPhi

llm = ChatPhi(model="phi4", temperature=0)
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    checkpointer=MemorySaver()
)

config = {"configurable": {"thread_id": "1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "¿Clima en Berlín?"}]},
    config=config
)
```

## Ingeniería y Mejores Prácticas

El proyecto PhiChat sigue estándares estrictos de ingeniería de software:

*   **Estándares Python**: Adherencia a PEP 8, PEP 484 (tipado estático) y PEP 257 (documentación).
*   **Validación Continua**: Suite de pruebas completa para asegurar la integridad del protocolo de etiquetas.
*   **Tipado Estricto**: Uso de Type Hints en todas las funciones y clases.
*   **Resiliencia**: Manejo de excepciones específicas para modelos locales.

## Testing

Garantiza la integridad de los protocolos de comunicación y la ejecución agéntica.

```bash
# Ejecutar todos los tests
uv run pytest

# Ejecutar un test específico (ej. los parsers)
uv run pytest test/test_parsers.py
```

## Proyectos que usan PhiChat

Descubre cómo otros desarrolladores están integrando `PhiChat` en sus arquitecturas:

*   [Terminal Phi](https://github.com/CarlosMaroRuiz/terminal_phi): Un asistente de terminal autónomo que utiliza Phi-4 para la gestión de archivos y ejecución segura de comandos en Windows (Human-In-The-Loop).

## Desarrollo con Skills (Phi Protocol)

Este proyecto utiliza un sistema de **Skills** para estandarizar la ingeniería de software. Puedes encontrar la guía completa de operación en [skills/SKILL.md](skills/SKILL.md).

### Mandato de Calidad
Cualquier contribución debe cumplir con el **Protocolo Phi**:
1.  **Verificación Total**: `uv run pytest`
2.  **Estándares**: Adherencia estricta a PEP 8, 484 y 257.

---
Desarrollado para optimizar el uso de modelos Microsoft Phi en entornos de producción con LangChain.
