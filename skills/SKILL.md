---
name: phi
description: Estandar de ingenieria y operacion para el ecosistema PhiChat. Define los protocolos de desarrollo, integracion con LangChain/LangGraph y mantenimiento de la calidad del software.
---

## Mandato de Calidad e Integridad
Cualquier cambio en el repositorio debe cumplir con:
1. **Validacion de Tests**: Ejecucion obligatoria de `uv run pytest`. No se aceptan regresiones.
2. **Estandar Python**: Adherencia a PEP 8 (estilo), PEP 484 (tipado estatico) y PEP 257 (docstrings).
3. **Compatibilidad**: Mantenimiento de la compatibilidad con LangChain v0.3+ y LangGraph.

Comando de verificacion total: !`uv run pytest && uv run python example.py`

## Estado y Diagnostico de Ingenieria
- **Entorno de Ejecucion**: !`python --version`
- **Gestor de Paquetes**: !`uv --version`
- **Infraestructura LLM**: !`ollama --version`
- **Inventario de Modelos**: !`ollama list | grep phi`

## Guia de Ingenieria de Software

### 1. Desarrollo Pythonic y Tipado
Todo nuevo componente debe utilizar anotaciones de tipo completas para facilitar el analisis estatico y mejorar la legibilidad.
```python
from typing import Sequence, Optional
from langchain_core.messages import BaseMessage

def procesar_historial(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """
    Normaliza el historial de mensajes para el protocolo Phi-4.
    
    Args:
        messages: Secuencia de mensajes de LangChain.
        
    Returns:
        Lista de mensajes con etiquetas de control inyectadas.
    """
    # Implementacion...
```

### 2. Arquitectura de Agentes (LangGraph)
Seguir el principio de responsabilidad unica. Separar la logica de las herramientas de la orquestacion del grafo.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Definir el estado del agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Orquestacion limpia
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
```

### 3. Gestion de Errores y Resiliencia
No permitir que excepciones de bajo nivel (ej. fallos de red en Ollama) rompan el loop del agente. Implementar reintentos y capturas especificas.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_with_retry(model, prompt):
    try:
        return model.invoke(prompt)
    except Exception as e:
        logger.error(f"Error en invocacion: {e}")
        raise
```

## Protocolo de Integracion MCP
Para integrar herramientas MCP de forma segura:
1. Validar el esquema de la herramienta antes del bind.
2. Manejar la naturaleza asincrona de las conexiones stdio/sse.
3. Asegurar que `nest_asyncio` esté configurado si se ejecutan loops dentro de entornos ya asincronos.

## Resolucion de Problemas Tecnicos

| Sintoma | Analisis de Ingenieria | Accion Correctiva |
|----------|------------------------|-------------------|
| **Mismatch de Tipos** | Incompatibilidad entre esquemas Pydantic y Tool Calling. | Validar `args_schema` en la definicion de la `@tool`. |
| **Fuga de Memoria** | Contextos de Ollama no liberados o sesiones largas. | Configurar `keep_alive` adecuadamente y monitorear `num_ctx`. |
| **Alucinacion de Tool** | El modelo genera JSON mal formado o etiquetas invalidas. | Refinar el `tool_system_suffix` en `constants.py`. |

## Ciclo de Vida del Desarrollo (SDLC)
1. **Analisis de Requisitos**: Determinar si el cambio afecta el parsing nativo.
2. **Diseño de Interfaz**: Definir tipos de entrada y salida (Pydantic).
3. **Implementacion**: Codificar siguiendo PEP 8 y usando `uv` para gestion de dependencias.
4. **Testing**:
   - Unitarios: `test_parsers.py` (logica sin LLM).
   - Integracion: `test_tools_llm.py` (flujo completo con Ollama).
5. **Revision**: Verificar que `uv.lock` este actualizado.

---
*Este skill garantiza que PhiChat sea un proyecto de grado de produccion, mantenible y escalable.*
