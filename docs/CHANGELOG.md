# Historial de Cambios (Changelog) — PhiChat

## [0.2.1] - 2026-05-09
### Añadido
- **Suavizado de Prompt**: Ajuste en `constants.py` para permitir respuestas de texto natural en preguntas generales, evitando que el modelo alucine herramientas cuando no son necesarias.
- **Robustez de Parser**: Implementación de limpieza agresiva de ruidos (como `}}}`) en argumentos JSON antes del parseo para evitar fallos por alucinaciones.

## [0.2.0] - 2026-05-09
### Añadido
- **Soporte Async Nativo**: Exportación de `arun_tool_loop` para flujos de agentes asíncronos de alta concurrencia.
- **Normalización de Herramientas**: Nueva lógica para manejar automáticamente diferentes formatos de argumentos (`args`, `arguments`, `parameters`).

### Cambiado
- **Arquitectura Modular**: Gran refactorización del paquete en sub-módulos profesionales: `models/`, `tools/`, `utils/` y `constants/`.
- **Mejora de Streaming**: Migración a métodos privados (`_astream`) para eliminar por completo la duplicación de tokens ("eco").

### Corregido
- **AttributeError en Streaming**: Solucionado el fallo al acceder a propiedades en `ChatGenerationChunk` durante el flujo asíncrono.

## [0.1.7] - 2026-05-09
### Añadido
- **Surgical JSON Repair**: Nueva lógica capaz de reparar JSONs con llaves internas `{` faltantes (común en `phi4-mini`).
- **Regex de Captura Agresiva**: El parser ahora detecta bloques JSON incluso si el modelo omite las etiquetas `<|tool_call|>`.
- **Sincronización de Disco**: Uso de `f.flush()` y `os.fsync()` en las herramientas de laboratorio para asegurar la escritura inmediata de reportes.
- **Dynamic Steering**: Implementación en LangGraph para obligar a los agentes a finalizar tareas (como generar reportes) antes de terminar el flujo.

### Corregido
- Error donde los modelos pequeños imprimían el JSON crudo en la terminal en lugar de ejecutar la herramienta.
- Error de Unicode en terminales Windows al usar emojis en los logs.

## [0.1.6] - 2026-05-09
### Añadido
- **Smart JSON Repair**: Primera versión del algoritmo de reparación para cerrar corchetes `]` faltantes.
- Soporte para **LangGraph** en el laboratorio.

## [0.1.4] - 2026-05-08
### Añadido
- Soporte inicial para **Tool Calling** mediante inyección de sistema.
- Integración básica con **Ollama** y filtrado de tokens de control.

---
*Nota: A partir de la v0.1.7, la librería se considera estable para flujos agénticos complejos con modelos de la familia Phi-4.*
