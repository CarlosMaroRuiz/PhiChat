# PhiChat — Documentación Técnica del Proyecto

## 1. Visión General
PhiChat es un wrapper avanzado de **LangChain** diseñado específicamente para el modelo **Phi-4** (y versiones mini). Su objetivo principal es resolver las limitaciones nativas de Ollama en el manejo de **Tool Calling** y proporcionar una integración robusta con el ecosistema de **MCP (Model Context Protocol)**.

## 2. Arquitectura de Resiliencia (JSON Smart Repair)
El núcleo de la librería reside en su capacidad para manejar alucinaciones de formato comunes en modelos de lenguaje pequeños:
- **Regex Flexible**: Detecta bloques de herramientas incluso si el modelo olvida las etiquetas `<|tool_call|>`.
- **Surgical Repair (v0.1.7+)**: Algoritmo que reconstruye JSONs rotos, cerrando automáticamente llaves `{` y corchetes `]` internos que el modelo deja abiertos por error.
- **Filtrado de Control**: Elimina tokens de control y preámbulos innecesarios para garantizar que solo el contenido útil llegue al usuario o al ejecutor de herramientas.

## 3. Clase Principal: `ChatPhi`
La clase `ChatPhi` hereda de `BaseChatModel` de LangChain, lo que la hace compatible con todo el ecosistema (LangGraph, Cadenas, Agentes).

### Métodos Clave:
- **`bind_tools(tools)`**: Inyecta dinámicamente los esquemas de herramientas en el System Prompt para forzar al modelo a usarlas sin depender de la API de Ollama (que es limitada en Phi-4).
- **`invoke(messages)`**: Ejecución síncrona. Pasa los mensajes por la capa de limpieza y el parser de herramientas.
- **`stream(messages)`**: Implementa streaming real-token-by-token. Filtra etiquetas de herramientas en tiempo real para una UI limpia.
- **`run_tool_loop(messages, tools)`**: (Helper) Ejecuta un ciclo automático de pensamiento -> acción -> observación hasta alcanzar una respuesta final.

## 4. Estructura de Archivos
- `PhiChat/model.py`: Lógica principal del modelo y parcheo de mensajes.
- `PhiChat/parsers.py`: El cerebro del "Smart JSON Repair".
- `PhiChat/constants/constants.py`: Regex y Prompts de sistema estandarizados.
- `laboratory/`: Scripts de prueba y agentes de QA con LangGraph.

## 5. Guía para Futuros Desarrolladores
- **Modificar el Parser**: Si el modelo empieza a alucinar un formato nuevo, ajusta `_TOOL_CALL_PATTERNS` en `constants.py`.
- **Integración MCP**: La librería está preparada para usar `langchain-mcp-adapters`. Simplemente pasa las herramientas obtenidas del servidor MCP a `bind_tools`.
- **Streaming**: El método `_stream` utiliza un búfer preventivo para evitar que el usuario vea etiquetas técnicas como `<|tool_call|>`.

## 6. Variables de Entorno
- `LANGSMITH_TRACING=true`: (Opcional) Recomendado para auditar los ciclos de herramientas en producción.

---
*Este documento sirve como base para la evolución del proyecto. Mantener actualizado con cada cambio en el parser.*
